import os
import re
import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemReader, state_dict_loader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
import argparse
import pickle
import json
from typing import Any, Dict
from pathlib import Path
import sys

# ==========================================
# 1. 核心工具：绕过 Megatron 依赖加载器
# ==========================================
class UnpicklerWrapper(pickle.Unpickler):
    """
    自定义 Unpickler，当遇到 megatron 或 glm 开头的类时，
    返回一个空对象，防止因缺少库而报错。
    """
    def find_class(self, mod_name, name):
        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            class DummyClass:
                def __init__(self, *args, **kwargs): pass
            return DummyClass
        return super().find_class(mod_name, name)

import io

def _ensure_tensor(v):
    if isinstance(v, torch.Tensor):
        return v
    if hasattr(v, "read"):
        if hasattr(v, "seek"):
            v.seek(0)
        data = v.read()
        if len(data) == 0:
            raise EOFError("Empty stream when loading tensor; lazy checkpoint may use shared stream.")
        result = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if isinstance(result, dict):
            raise ValueError(
                "Lazy checkpoint appears to use shared stream; got full state dict when loading one value. "
                "Try running conversion in an environment where Megatron can be imported."
            )
        return result
    return v


class WrappedStorageReader(FileSystemReader):
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


# ==========================================
# 2. 加载普通 torch 格式 (model_optim_rng.pt)
# ==========================================
def load_torch_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    从 model_optim_rng.pt 加载模型权重，将按层存储的 key 转为与 dist 一致的堆叠格式。
    """
    print(f"Loading torch checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_sd = checkpoint.get("model")
    if model_sd is None:
        model_sd = checkpoint.get("model0")
    if model_sd is None:
        raise KeyError("Checkpoint must contain 'model' or 'model0'.")

    # 匹配 decoder.layers.<i>.<suffix>
    layer_pattern = re.compile(r"^decoder\.layers\.(\d+)\.(.+)$")
    layer_indices_by_suffix = {}  # suffix -> set of layer indices

    for key in model_sd.keys():
        m = layer_pattern.match(key)
        if m:
            layer_idx = int(m.group(1))
            suffix = m.group(2)
            if suffix not in layer_indices_by_suffix:
                layer_indices_by_suffix[suffix] = set()
            layer_indices_by_suffix[suffix].add(layer_idx)

    if not layer_indices_by_suffix:
        raise ValueError(
            "No keys matching 'decoder.layers.<i>.*' found. "
            "Ensure this is an MCore (decoder) torch checkpoint."
        )

    num_layers = max(max(indices) for indices in layer_indices_by_suffix.values()) + 1

    raw_state_dict = {}

    # 非按层 key：直接拷贝
    for key in list(model_sd.keys()):
        if not layer_pattern.match(key):
            raw_state_dict[key] = _ensure_tensor(model_sd[key])

       # 按层 key：堆叠为 decoder.layers.<suffix>
    for suffix, indices in layer_indices_by_suffix.items():
        if len(indices) != num_layers:
            raise ValueError(
                f"Layer count mismatch for suffix '{suffix}': "
                f"expected {num_layers} layers, got {sorted(indices)}."
            )
        tensors = [
            _ensure_tensor(model_sd[f"decoder.layers.{i}.{suffix}"])
            for i in range(num_layers)
        ]
        if any(t is None for t in tensors):
            continue  # 该 suffix 部分层为 None（如可选 bias），跳过不堆叠
        stacked = torch.stack(tensors)
        raw_state_dict[f"decoder.layers.{suffix}"] = stacked

    print(f"Loaded {len(raw_state_dict)} keys from torch checkpoint (num_layers={num_layers}).")
    return raw_state_dict


def load_dist_checkpoint(input_dir: str) -> Dict[str, torch.Tensor]:
    """加载 torch_dist 格式 checkpoint 目录。"""
    print(f"Loading distributed checkpoint from: {input_dir}")
    raw_state_dict = {}
    state_dict_loader._load_state_dict(
        raw_state_dict,
        storage_reader=WrappedStorageReader(input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"Loaded {len(raw_state_dict)} raw tensors from distcp.")
    return raw_state_dict


# ==========================================
# 3. 核心逻辑：Megatron -> HF 映射（统一入口）
# ==========================================
def convert_checkpoint(args):
    if args.checkpoint_file:
        raw_state_dict = load_torch_checkpoint(args.checkpoint_file)
    else:
        raw_state_dict = load_dist_checkpoint(args.input_dir)

    # --- 自动推断模型参数 ---
    if "embedding.word_embeddings.weight" in raw_state_dict:
        vocab_size, hidden_size = raw_state_dict["embedding.word_embeddings.weight"].shape
    else:
        vocab_size, hidden_size = raw_state_dict["model.language_model.embedding.word_embeddings.weight"].shape

    qkv_key = "decoder.layers.self_attention.linear_qkv.weight"
    if qkv_key not in raw_state_dict:
        qkv_key = "model.language_model.encoder.layers.self_attention.linear_qkv.weight"
        if qkv_key not in raw_state_dict:
            raise ValueError("Could not find linear_qkv weight to infer num_layers.")

    num_layers, qkv_dim, _ = raw_state_dict[qkv_key].shape
    print(f"Detected Arch: Layers={num_layers}, Hidden={hidden_size}, Vocab={vocab_size} (padded)")

    # --- 开始转换 ---
    hf_state_dict = {}

    # 1. Embedding
    print("Converting Embeddings...")
    emb_weight = raw_state_dict.get("embedding.word_embeddings.weight")
    if emb_weight is None:
        emb_weight = raw_state_dict.get("model.language_model.embedding.word_embeddings.weight")

    if args.vocab_size and emb_weight.shape[0] > args.vocab_size:
        print(f"Trimming vocab from {emb_weight.shape[0]} to {args.vocab_size}")
        emb_weight = emb_weight[: args.vocab_size, :]

    hf_state_dict["model.embed_tokens.weight"] = emb_weight
    output_layer_weight = raw_state_dict.get("output_layer.weight")
    if output_layer_weight is not None:
        lm_head_weight = output_layer_weight.clone()
        if args.vocab_size and lm_head_weight.shape[0] > args.vocab_size:
            lm_head_weight = lm_head_weight[: args.vocab_size, :]
        hf_state_dict["lm_head.weight"] = lm_head_weight
        print("Using checkpoint output_layer as lm_head (untied).")
        _emb = emb_weight.float()
        _lm = lm_head_weight.float()
        if _emb.shape == _lm.shape:
            cos_per_row = torch.nn.functional.cosine_similarity(_emb, _lm, dim=1)
            c = cos_per_row.double()
            print(
                f"  embed_tokens vs lm_head (per-row): mean_cos={c.mean().item():.6f}, "
                f"min={c.min().item():.6f}, max={c.max().item():.6f}"
            )
            q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=c.dtype, device=c.device)
            quantiles = torch.quantile(c, q).tolist()
            print(
                f"  quantiles (0/25/50/75/100%): "
                f"[{quantiles[0]:.4f}, {quantiles[1]:.4f}, {quantiles[2]:.4f}, {quantiles[3]:.4f}, {quantiles[4]:.4f}]"
            )
            hist = torch.histc(c, bins=4, min=-1.0, max=1.0)
            print(
                f"  hist [-1,-0.5): {hist[0].int().item()}, [-0.5,0): {hist[1].int().item()}, "
                f"[0,0.5): {hist[2].int().item()}, [0.5,1]: {hist[3].int().item()}"
            )
        else:
            print(f"  embed_tokens shape {_emb.shape} vs lm_head shape {_lm.shape} (cannot compare)")
    else:
        hf_state_dict["lm_head.weight"] = emb_weight
        print("Using embedding as lm_head (tied).")

    # 2. Final Norm
    print("Converting Final Norm...")
    final_norm = raw_state_dict.get("decoder.final_layernorm.weight")
    if final_norm is None:
        final_norm = raw_state_dict.get("model.language_model.encoder.final_layernorm.weight")
    hf_state_dict["model.norm.weight"] = final_norm

    # 3. Layers
    print("Converting Layers...")
    head_dim = 128
    total_qkv_dim = qkv_dim
    num_heads = args.num_attention_heads
    num_kv_heads = args.num_key_value_heads if args.num_key_value_heads else num_heads
    calc_dim = (num_heads + 2 * num_kv_heads) * head_dim
    if calc_dim != total_qkv_dim:
        head_dim = total_qkv_dim // (num_heads + 2 * num_kv_heads)
        print(f"Adjusted head_dim to {head_dim}")
    num_query_groups = num_kv_heads
    value_num_per_group = num_heads // num_query_groups

    def get_tensor(name_suffix):
        key = f"decoder.layers.{name_suffix}"
        if key in raw_state_dict:
            return raw_state_dict[key]
        key = f"model.language_model.encoder.layers.{name_suffix}"
        if key in raw_state_dict:
            return raw_state_dict[key]
        return None

    layers_qkv = get_tensor("self_attention.linear_qkv.weight")
    layers_qkv_bias = get_tensor("self_attention.linear_qkv.bias")
    layers_o = get_tensor("self_attention.linear_proj.weight")
    layers_mlp_fc1 = get_tensor("mlp.linear_fc1.weight")
    layers_mlp_fc2 = get_tensor("mlp.linear_fc2.weight")
    layers_input_norm = get_tensor("self_attention.linear_qkv.layer_norm_weight")
    layers_post_norm = get_tensor("mlp.linear_fc1.layer_norm_weight")
    layers_q_norm = get_tensor("self_attention.q_layernorm.weight")
    layers_k_norm = get_tensor("self_attention.k_layernorm.weight")

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        hf_state_dict[f"{prefix}.input_layernorm.weight"] = layers_input_norm[i]
        hf_state_dict[f"{prefix}.post_attention_layernorm.weight"] = layers_post_norm[i]

        qkv = layers_qkv[i]
        qkv_reshaped = qkv.view(num_query_groups, -1, head_dim, hidden_size)
        q, k, v = torch.split(qkv_reshaped, [value_num_per_group, 1, 1], dim=1)
        hf_state_dict[f"{prefix}.self_attn.q_proj.weight"] = q.reshape(-1, hidden_size)
        hf_state_dict[f"{prefix}.self_attn.k_proj.weight"] = k.reshape(-1, hidden_size)
        hf_state_dict[f"{prefix}.self_attn.v_proj.weight"] = v.reshape(-1, hidden_size)

        if layers_qkv_bias is not None:
            b = layers_qkv_bias[i]
            b_reshaped = b.view(num_query_groups, -1, head_dim)
            qb, kb, vb = torch.split(b_reshaped, [value_num_per_group, 1, 1], dim=1)
            hf_state_dict[f"{prefix}.self_attn.q_proj.bias"] = qb.reshape(-1)
            hf_state_dict[f"{prefix}.self_attn.k_proj.bias"] = kb.reshape(-1)
            hf_state_dict[f"{prefix}.self_attn.v_proj.bias"] = vb.reshape(-1)

        hf_state_dict[f"{prefix}.self_attn.o_proj.weight"] = layers_o[i]
        if layers_q_norm is not None:
            hf_state_dict[f"{prefix}.self_attn.q_norm.weight"] = layers_q_norm[i]
        if layers_k_norm is not None:
            hf_state_dict[f"{prefix}.self_attn.k_norm.weight"] = layers_k_norm[i]

        fc1 = layers_mlp_fc1[i]
        gate, up = torch.chunk(fc1, 2, dim=0)
        hf_state_dict[f"{prefix}.mlp.gate_proj.weight"] = gate
        hf_state_dict[f"{prefix}.mlp.up_proj.weight"] = up
        hf_state_dict[f"{prefix}.mlp.down_proj.weight"] = layers_mlp_fc2[i]

    # --- 保存 ---
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "pytorch_model.bin")
    print(f"Saving HF checkpoint to {save_path}...")
    torch.save(hf_state_dict, save_path)

    config = {
        "architectures": ["Qwen3ForCausalLM"] if layers_q_norm is not None else ["LlamaForCausalLM"],
        "model_type": "qwen3",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": hf_state_dict["model.layers.0.mlp.gate_proj.weight"].shape[0],
        "max_position_embeddings": 4096,
        "max_sequence_length": 1024,
        "max_window_layers": num_layers,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": args.vocab_size if args.vocab_size else vocab_size,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Megatron-LM checkpoint (torch_dist or single .pt) to HuggingFace format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to Megatron torch_dist checkpoint directory (use for dist format).",
    )
    group.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        help="Path to single file checkpoint model_optim_rng.pt (use for torch format).",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output HF directory")
    parser.add_argument("--num_attention_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--num_key_value_heads", type=int, default=None, help="Number of KV heads (GQA)")
    parser.add_argument("--vocab_size", type=int, default=None, help="Real vocab size to trim padding")

    args = parser.parse_args()
    convert_checkpoint(args)