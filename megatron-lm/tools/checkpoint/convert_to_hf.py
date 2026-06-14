import os
import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import FileSystemReader, state_dict_loader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
import argparse
import pickle
import json
from typing import Any
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

class WrappedStorageReader(FileSystemReader):
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            # 使用自定义 Unpickler
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
            # 跳过优化器状态和非权重数据
            if "optimizer" in k or "_state" in k:
                continue
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)

# ==========================================
# 2. 核心逻辑：Megatron -> HF 映射
# ==========================================
def convert_checkpoint(args):
    print(f"Loading distributed checkpoint from: {args.input_dir}")
    
    # --- 加载原始权重 ---
    raw_state_dict = {}
    state_dict_loader._load_state_dict(
        raw_state_dict,
        storage_reader=WrappedStorageReader(args.input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"Loaded {len(raw_state_dict)} raw tensors from distcp.")

    # --- 自动推断模型参数 ---
    # 根据 Embedding 层推断
    if "embedding.word_embeddings.weight" in raw_state_dict:
        vocab_size, hidden_size = raw_state_dict["embedding.word_embeddings.weight"].shape
    else:
        # 兼容旧命名
        vocab_size, hidden_size = raw_state_dict["model.language_model.embedding.word_embeddings.weight"].shape

    # 根据 QKV 权重推断层数
    qkv_key = "decoder.layers.self_attention.linear_qkv.weight"
    # 兼容旧命名
    if qkv_key not in raw_state_dict:
        qkv_key = "model.language_model.encoder.layers.self_attention.linear_qkv.weight"
        if qkv_key not in raw_state_dict:
             # 如果是单层存储的旧格式，这里逻辑需要调整，但本脚本主要针对 mcore 合并格式
             raise ValueError("Could not find linear_qkv weight to infer num_layers.")
             
    num_layers, qkv_dim, _ = raw_state_dict[qkv_key].shape
    
    print(f"Detected Arch: Layers={num_layers}, Hidden={hidden_size}, Vocab={vocab_size} (padded)")

    # --- 开始转换 ---
    hf_state_dict = {}
    
    # 1. 处理 Embedding
    print("Converting Embeddings...")
    # emb_weight = raw_state_dict.get("embedding.word_embeddings.weight") or \
    #              raw_state_dict.get("model.language_model.embedding.word_embeddings.weight")
    emb_weight = raw_state_dict.get("embedding.word_embeddings.weight")
    if emb_weight is None:
        emb_weight = raw_state_dict.get("model.language_model.embedding.word_embeddings.weight")
    
    # 裁剪 Padding (例如从 152064 -> 151936)
    if args.vocab_size and emb_weight.shape[0] > args.vocab_size:
        print(f"Trimming vocab from {emb_weight.shape[0]} to {args.vocab_size}")
        emb_weight = emb_weight[:args.vocab_size, :]
        
    hf_state_dict["model.embed_tokens.weight"] = emb_weight
    # 通常 lm_head 共享 embedding，但也可能独立。如果 MCore 没存 lm_head，则复用 embedding
    # 若使用了 --untie-embeddings-and-output-weights，checkpoint 中有独立的 output_layer，必须用它做 lm_head
    output_layer_weight = raw_state_dict.get("output_layer.weight")
    if output_layer_weight is not None:
        lm_head_weight = output_layer_weight.clone()
        if args.vocab_size and lm_head_weight.shape[0] > args.vocab_size:
            lm_head_weight = lm_head_weight[: args.vocab_size, :]
        hf_state_dict["lm_head.weight"] = lm_head_weight
        print("Using checkpoint output_layer as lm_head (untied).")
        # 顺带比较 embed_tokens 与 lm_head：按行对应算 cosine_sim + 分布统计
        _emb = emb_weight.float()
        _lm = lm_head_weight.float()
        if _emb.shape == _lm.shape:
            cos_per_row = torch.nn.functional.cosine_similarity(_emb, _lm, dim=1)
            c = cos_per_row.double()
            print(f"  embed_tokens vs lm_head (per-row): mean_cos={c.mean().item():.6f}, "
                  f"min={c.min().item():.6f}, max={c.max().item():.6f}")
            q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=c.dtype, device=c.device)
            quantiles = torch.quantile(c, q).tolist()
            print(f"  quantiles (0/25/50/75/100%): [{quantiles[0]:.4f}, {quantiles[1]:.4f}, {quantiles[2]:.4f}, {quantiles[3]:.4f}, {quantiles[4]:.4f}]")
            hist = torch.histc(c, bins=4, min=-1.0, max=1.0)
            print(f"  hist [-1,-0.5): {hist[0].int().item()}, [-0.5,0): {hist[1].int().item()}, [0,0.5): {hist[2].int().item()}, [0.5,1]: {hist[3].int().item()}")
        else:
            print(f"  embed_tokens shape {_emb.shape} vs lm_head shape {_lm.shape} (cannot compare)")
    else:
        hf_state_dict["lm_head.weight"] = emb_weight
        print("Using embedding as lm_head (tied).")
    
    

    # 2. 处理 Final Norm
    print("Converting Final Norm...")
    # final_norm = raw_state_dict.get("decoder.final_layernorm.weight") or \
    #              raw_state_dict.get("model.language_model.encoder.final_layernorm.weight")
    final_norm = raw_state_dict.get("decoder.final_layernorm.weight")
    if final_norm is None:
        final_norm = raw_state_dict.get("model.language_model.encoder.final_layernorm.weight")
    hf_state_dict["model.norm.weight"] = final_norm

    # 3. 循环处理每一层
    print("Converting Layers...")
    
    # 预计算 GQA 参数
    # 注意：这里需要用户手动指定 head 数量，或者我们尝试猜测
    # 假设 QKV = [num_layers, (num_heads + 2*num_kv_heads) * head_dim, hidden]
    # 如果不知道具体配置，通常假设 head_dim = 128 (Llama/Qwen 标准)
    head_dim = 128
    total_qkv_dim = qkv_dim
    # 简化的 GQA 逻辑：如果 total_qkv 刚好是 hidden 的几倍？
    # 通常 Q=hidden, K,V 取决于 GQA。
    # 为了脚本通用，我们使用参数传入
    num_heads = args.num_attention_heads
    num_kv_heads = args.num_key_value_heads if args.num_key_value_heads else num_heads
    
    # 校验维度
    calc_dim = (num_heads + 2 * num_kv_heads) * head_dim
    # 如果算出来不对，可能 head_dim 不是 128，尝试反推
    if calc_dim != total_qkv_dim:
        head_dim = total_qkv_dim // (num_heads + 2 * num_kv_heads)
        print(f"Adjusted head_dim to {head_dim}")

    num_query_groups = num_kv_heads
    value_num_per_group = num_heads // num_query_groups
    
    # 获取所有层的权重 (避免在循环里反复查找)
    # 使用 .get 兼容 naming
    def get_tensor(name_suffix):
        # 尝试 decoder.layers...
        key = f"decoder.layers.{name_suffix}"
        if key in raw_state_dict: return raw_state_dict[key]
        # 尝试 model.language_model.encoder.layers...
        key = f"model.language_model.encoder.layers.{name_suffix}"
        if key in raw_state_dict: return raw_state_dict[key]
        return None

    layers_qkv = get_tensor("self_attention.linear_qkv.weight")
    layers_qkv_bias = get_tensor("self_attention.linear_qkv.bias") # 可能没有
    layers_o = get_tensor("self_attention.linear_proj.weight")
    
    layers_mlp_fc1 = get_tensor("mlp.linear_fc1.weight") # Gate + Up
    layers_mlp_fc2 = get_tensor("mlp.linear_fc2.weight") # Down
    
    layers_input_norm = get_tensor("self_attention.linear_qkv.layer_norm_weight")
    layers_post_norm = get_tensor("mlp.linear_fc1.layer_norm_weight")
    
    # Qwen 特有的 Q/K Norm
    layers_q_norm = get_tensor("self_attention.q_layernorm.weight")
    layers_k_norm = get_tensor("self_attention.k_layernorm.weight")

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        
        # --- LayerNorms ---
        hf_state_dict[f"{prefix}.input_layernorm.weight"] = layers_input_norm[i]
        hf_state_dict[f"{prefix}.post_attention_layernorm.weight"] = layers_post_norm[i]
        
        # --- QKV Split ---
        qkv = layers_qkv[i] # [4096, hidden]
        # Reshape for GQA split: [Groups, Q_per_group + K + V, head_dim, hidden]
        qkv_reshaped = qkv.view(num_query_groups, -1, head_dim, hidden_size)
        # split dim 1: [value_num_per_group, 1, 1]
        q, k, v = torch.split(qkv_reshaped, [value_num_per_group, 1, 1], dim=1)
        hf_state_dict[f"{prefix}.self_attn.q_proj.weight"] = q.reshape(-1, hidden_size)

        hf_state_dict[f"{prefix}.self_attn.k_proj.weight"] = k.reshape(-1, hidden_size)
        hf_state_dict[f"{prefix}.self_attn.v_proj.weight"] = v.reshape(-1, hidden_size)
        
        # Handle QKV Bias if exists
        if layers_qkv_bias is not None:
             # bias 处理逻辑类似，只是维度少一维
             b = layers_qkv_bias[i]
             b_reshaped = b.view(num_query_groups, -1, head_dim)
             qb, kb, vb = torch.split(b_reshaped, [value_num_per_group, 1, 1], dim=1)
             hf_state_dict[f"{prefix}.self_attn.q_proj.bias"] = qb.reshape(-1)
             hf_state_dict[f"{prefix}.self_attn.k_proj.bias"] = kb.reshape(-1)
             hf_state_dict[f"{prefix}.self_attn.v_proj.bias"] = vb.reshape(-1)

        # --- O Proj ---
        hf_state_dict[f"{prefix}.self_attn.o_proj.weight"] = layers_o[i]

        # --- Q/K Norm (Qwen/Llama with norms) ---
        if layers_q_norm is not None:
            hf_state_dict[f"{prefix}.self_attn.q_norm.weight"] = layers_q_norm[i]
        if layers_k_norm is not None:
            hf_state_dict[f"{prefix}.self_attn.k_norm.weight"] = layers_k_norm[i]

        # --- MLP (SwiGLU: Gate + Up) ---
        fc1 = layers_mlp_fc1[i] # [2*intermediate, hidden]
        gate, up = torch.chunk(fc1, 2, dim=0)
        hf_state_dict[f"{prefix}.mlp.gate_proj.weight"] = gate
        hf_state_dict[f"{prefix}.mlp.up_proj.weight"] = up
        
        # --- MLP Down ---
        hf_state_dict[f"{prefix}.mlp.down_proj.weight"] = layers_mlp_fc2[i]

    # --- 保存 ---
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "pytorch_model.bin")
    print(f"Saving HF checkpoint to {save_path}...")
    torch.save(hf_state_dict, save_path)
    
    # --- 生成 Config ---
    # 这是一个基础的 config，可能需要你根据具体训练参数调整
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
        "torch_dtype": "bfloat16"
    }
    
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to Megatron distcp folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output HF directory")
    # 下面这几个参数需要你知道训练时的设置
    parser.add_argument("--num_attention_heads", type=int, default=16, help="e.g. 16 for 0.6B?")
    parser.add_argument("--num_key_value_heads", type=int, default=None, help="If GQA, set this")
    parser.add_argument("--vocab_size", type=int, default=None, help="Real vocab size to trim padding")
    
    args = parser.parse_args()
    convert_checkpoint(args)