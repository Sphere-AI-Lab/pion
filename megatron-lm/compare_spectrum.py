"""
对比两个 HF 检查点中 2D 参数矩阵的奇异值谱（排除 embedding 和 lm_head）。
q/k/v/o 按 head 切分后分别做 SVD；head_dim 由 --hidden-size 与 --num-attention-heads 推导。
用法: python compare_spectrum.py <dir1> [dir2] [--num-attention-heads 16] [--hidden-size 1024] [--out spectra.png]
"""
import argparse
import os
import re

import torch
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def load_state_dict(path):
    base = path
    st = os.path.join(base, "model.safetensors")
    if os.path.isfile(st):
        from safetensors.torch import load_file
        return load_file(st)
    pt = os.path.join(base, "pytorch_model.bin")
    if os.path.isfile(pt):
        return torch.load(pt, map_location="cpu", weights_only=True)
    for f in os.listdir(base):
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(os.path.join(base, f))
        if f.endswith(".bin"):
            return torch.load(os.path.join(base, f), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No state dict found in {base}")


EXCLUDE_PREFIX = ("model.embed_tokens.", "embed_tokens.", "lm_head.")


def is_2d_and_not_embed_or_head(name):
    if any(name.startswith(p) for p in EXCLUDE_PREFIX):
        return False
    if "embed_tokens" in name or "lm_head" in name:
        return False
    return True


def get_param_type(name):
    for part in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if part in name:
            return part
    return None


def svd_spectrum(mat):
    """返回奇异值（降序），numpy 一维数组。"""
    if mat.dim() != 2:
        return None
    mat_f = mat.float()
    k = min(min(mat.shape[0], mat.shape[1]), 512)
    try:
        U, S, Vh = torch.linalg.svd(mat_f, full_matrices=False)
        s = S.cpu().numpy()
    except Exception:
        return None
    return s


def svd_spectrum_per_head(param, head_dim, proj_type):
    """
    按 head 切分后对每个 head 的矩阵做 SVD，返回 list of (head_id, spectrum)。
    HF 约定：q/k/v_proj (num_heads*head_dim, hidden_size)，o_proj (hidden_size, num_heads*head_dim)。
    GQA 下 k/v 的 head 数可能少于 q，head_dim 不变。
    """
    if param.dim() != 2:
        return None
    param = param.float()
    out, in_dim = param.shape
    if proj_type in ("q_proj", "k_proj", "v_proj"):
        if out % head_dim != 0:
            return None
        num_heads = out // head_dim
        heads = param.view(num_heads, head_dim, in_dim)
        spectra = []
        for h in range(num_heads):
            s = svd_spectrum(heads[h])
            if s is not None:
                spectra.append((h, s))
        return spectra
    if proj_type == "o_proj":
        if in_dim % head_dim != 0:
            return None
        num_heads = in_dim // head_dim
        heads = param.view(out, num_heads, head_dim).permute(1, 0, 2)
        spectra = []
        for h in range(num_heads):
            s = svd_spectrum(heads[h])
            if s is not None:
                spectra.append((h, s))
        return spectra
    return None


def collect_spectra_by_type(state_dict, num_attention_heads=16, hidden_size=1024):
    """
    返回 dict: param_type -> list of (layer_id, spectrum array)。
    head_dim = hidden_size // num_attention_heads。
    q/k/v/o 按 head 切分，每个 head 单独 SVD；down/gate/up 整矩阵 SVD；gate+up 同层 concat 记为 gate_up_proj。
    """
    head_dim = hidden_size // num_attention_heads

    by_layer_type = {}
    for name, param in state_dict.items():
        if param.dim() != 2:
            continue
        if not is_2d_and_not_embed_or_head(name):
            continue
        ptype = get_param_type(name)
        if ptype is None:
            continue
        layer_match = re.search(r"layers\.(\d+)", name)
        layer_id = int(layer_match.group(1)) if layer_match else 0
        by_layer_type[(layer_id, ptype)] = (name, param)

    max_layer_id = max(layer_id for (layer_id, _) in by_layer_type) if by_layer_type else 0
    print_layers = (0, max_layer_id)

    by_type = {}

    # 1) q_proj, k_proj, v_proj, o_proj：按 head 切分，每个 head 单独 SVD
    for ptype in ("q_proj", "k_proj", "v_proj", "o_proj"):
        for (layer_id, t), (name, param) in by_layer_type.items():
            if t != ptype:
                continue
            if layer_id in print_layers:
                print(f"[SVD 前] layer={layer_id} {ptype}: name={name!r} shape={tuple(param.shape)}")
            per_head = svd_spectrum_per_head(param, head_dim, ptype)
            if not per_head:
                continue
            for head_id, s in per_head:
                if layer_id in print_layers and head_id == 0:
                    print(f"[SVD 后] layer={layer_id} {ptype} head={head_id}: 最大奇异值(谱范数)={s[0]:.6e}")
                if ptype not in by_type:
                    by_type[ptype] = []
                by_type[ptype].append((layer_id, s))

    # 2) down_proj, gate_proj, up_proj：整矩阵 SVD
    for ptype in ("down_proj", "gate_proj", "up_proj"):
        for (layer_id, t), (name, param) in by_layer_type.items():
            if t != ptype:
                continue
            if layer_id in print_layers:
                print(f"[SVD 前] layer={layer_id} {ptype}: name={name!r} shape={tuple(param.shape)}")
            s = svd_spectrum(param)
            if s is None:
                continue
            if layer_id in print_layers:
                print(f"[SVD 后] layer={layer_id} {ptype}: 最大奇异值(谱范数)={s[0]:.6e}")
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append((layer_id, s))

    # 3) gate_proj 与 up_proj：同层沿 dim=0 concat 后做 SVD，记为 gate_up_proj
    layers_gate = set(layer_id for (layer_id, t) in by_layer_type if t == "gate_proj")
    layers_up = set(layer_id for (layer_id, t) in by_layer_type if t == "up_proj")
    for layer_id in sorted(layers_gate & layers_up):
        _, gate_param = by_layer_type[(layer_id, "gate_proj")]
        _, up_param = by_layer_type[(layer_id, "up_proj")]
        gate = gate_param.float()
        up = up_param.float()
        if gate.shape[1] != up.shape[1]:
            continue
        concat = torch.cat([gate, up], dim=0)
        if layer_id in print_layers:
            print(f"[SVD 前] layer={layer_id} gate_up_proj(concat): shape={tuple(concat.shape)}")
        s = svd_spectrum(concat)
        if s is None:
            continue
        if layer_id in print_layers:
            print(f"[SVD 后] layer={layer_id} gate_up_proj(concat): 最大奇异值(谱范数)={s[0]:.6e}")
        if "gate_up_proj" not in by_type:
            by_type["gate_up_proj"] = []
        by_type["gate_up_proj"].append((layer_id, s))

    for ptype in by_type:
        by_type[ptype].sort(key=lambda x: x[0])
    return by_type


def main():
    parser = argparse.ArgumentParser(
        description="Compare 2D param spectrum (SVD) of HF checkpoints. Q/K/V/O per-head SVD."
    )
    parser.add_argument("dir1", help="HF checkpoint dir (e.g. iter_0010000_hf)")
    parser.add_argument("dir2", nargs="?", default=None, help="Optional second HF dir for comparison")
    parser.add_argument("--out", default="spectrum_compare.png", help="Output figure path")
    parser.add_argument("--top_k", type=int, default=512, help="Max singular value index to plot")
    parser.add_argument("--no_plot", action="store_true", help="Only dump spectra to .npz, no plot")
    parser.add_argument("--num-attention-heads", type=int, default=16, help="Number of attention heads (for head_dim)")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden size (head_dim = hidden_size // num_attention_heads)")
    args = parser.parse_args()

    print("Loading state dict(s)...")
    sd1 = load_state_dict(args.dir1)
    single_mode = args.dir2 is None
    head_dim = args.hidden_size // args.num_attention_heads
    print(f"head_dim = {args.hidden_size} // {args.num_attention_heads} = {head_dim}")

    if single_mode:
        by_type_1 = collect_spectra_by_type(
            sd1,
            num_attention_heads=args.num_attention_heads,
            hidden_size=args.hidden_size,
        )
        types = sorted(by_type_1.keys())
        by_type_2 = None
    else:
        sd2 = load_state_dict(args.dir2)
        print("Computing spectra for dir1...")
        by_type_1 = collect_spectra_by_type(
            sd1,
            num_attention_heads=args.num_attention_heads,
            hidden_size=args.hidden_size,
        )
        print("Computing spectra for dir2...")
        by_type_2 = collect_spectra_by_type(
            sd2,
            num_attention_heads=args.num_attention_heads,
            hidden_size=args.hidden_size,
        )
        types = sorted(set(by_type_1.keys()) & set(by_type_2.keys()))

    if not types:
        print("No 2D param types found (excluding embed & lm_head).")
        return 1
    print(f"Param types: {types}")

    out_npz = args.out.rsplit(".", 1)[0] + ".npz"
    save_dict = {}

    def pad(s, length):
        if len(s) >= length:
            return np.array(s[:length])
        return np.concatenate([np.array(s), np.zeros(length - len(s))])

    if single_mode:
        for t in types:
            L1 = [s for _, s in by_type_1[t]]
            max_len = max(len(s) for s in L1)
            M1 = np.stack([pad(s, max_len) for s in L1])
            save_dict[f"{t}_single"] = M1
    else:
        for t in types:
            L1 = [s for _, s in by_type_1[t]]
            L2 = [s for _, s in by_type_2[t]]
            max_len = max(len(s) for s in L1 + L2)
            M1 = np.stack([pad(s, max_len) for s in L1])
            M2 = np.stack([pad(s, max_len) for s in L2])
            save_dict[f"{t}_iter1"] = M1
            save_dict[f"{t}_iter2"] = M2

    np.savez_compressed(out_npz, **save_dict)
    print(f"Saved spectra to {out_npz}")

    if args.no_plot:
        return 0

    if not HAS_PLOT:
        print("matplotlib not available, skip plotting.")
        return 0

    ntypes = len(types)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for idx, ptype in enumerate(types):
        ax = axes[idx]
        L1 = [s for _, s in by_type_1[ptype]]
        k = min(args.top_k, min(len(s) for s in L1))
        x = np.arange(1, k + 1)
        mean1 = np.mean([s[:k] for s in L1], axis=0)
        std1 = np.std([s[:k] for s in L1], axis=0)
        ax.fill_between(x, mean1 - std1, mean1 + std1, alpha=0.3, color="C0")
        ax.plot(x, mean1, color="C0", label=args.dir1.split("/")[-1][:12])
        if not single_mode:
            L2 = [s for _, s in by_type_2[ptype]]
            k = min(args.top_k, k, min(len(s) for s in L2))
            x = np.arange(1, k + 1)
            mean2 = np.mean([s[:k] for s in L2], axis=0)
            std2 = np.std([s[:k] for s in L2], axis=0)
            ax.fill_between(x, mean2 - std2, mean2 + std2, alpha=0.3, color="C1")
            ax.plot(x, mean2, color="C1", label=args.dir2.split("/")[-1][:12])
        ax.set_title(ptype)
        ax.set_xlabel("Singular value index")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_yscale("log")
    for j in range(ntypes, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")
    return 0


if __name__ == "__main__":
    exit(main())