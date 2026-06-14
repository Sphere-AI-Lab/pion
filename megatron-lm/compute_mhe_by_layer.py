"""
对 HF 检查点中每层的 2D 参数矩阵（排除 embedding 和 lm_head）计算 MHE 能量。
用法: python compute_mhe_by_layer.py <dir1> [dir2 ...] [--out mhe.csv] [--s 2]
"""
import argparse
import os
import re

import torch


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


def is_2d_and_not_embed_or_head(name):
    if "embed_tokens" in name or "lm_head" in name:
        return False
    return True


def get_param_type(name):
    for part in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if part in name:
            return part
    return None


def compute_mhe_loss(weight_matrix, s=2):
    """MHE 能量：归一化行后按原始对 + 镜像对 Riesz s-能量，再除以 C_{2n,2}。"""
    with torch.no_grad():
        w = torch.nn.functional.normalize(weight_matrix.data.to(torch.float64), p=2, dim=1)
        n = w.shape[0]
        device = w.device

        cos_sim = torch.mm(w, w.t())
        eps = 1e-7
        energy_orig = torch.pow(torch.clamp(2.0 - 2.0 * cos_sim, min=eps), -s / 2)
        energy_mirr = torch.pow(torch.clamp(2.0 + 2.0 * cos_sim, min=eps), -s / 2)

        total_energy = 2.0 * (torch.sum(energy_orig) + torch.sum(energy_mirr))
        n_total = 2 * n
        self_energy = n_total * torch.pow(torch.tensor(eps, device=device, dtype=torch.float64), -s / 2)
        total_energy -= self_energy
        cnt = n_total * (n_total - 1)
        return (total_energy / cnt).item()

def compute_svd_entropy(weight_matrix, eps=1e-12):
    """SVD 熵：p_i = s_i^2 / sum(s_j^2)，entropy = -sum(p_i * log(p_i))。"""
    with torch.no_grad():
        w = weight_matrix.data.to(torch.float64)
        if w.shape[0] < 2 or w.shape[1] < 2:
            return float("nan")
        S = torch.linalg.svdvals(w)
        s_sq = S * S
        total = s_sq.sum().item()
        if total <= 0:
            return float("nan")
        p = (s_sq / total).clamp(min=eps)
        entropy = -(p * torch.log(p)).sum().item()
        return entropy

def collect_mhe_by_layer(state_dict, s=2):
    """返回 list of (layer_id, param_type, mhe_value, svd_entropy_value)。"""
    results = []
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
        try:
            mhe = compute_mhe_loss(param, s=s)
            svd_ent = compute_svd_entropy(param)
            print(name,"param shape:", param.shape, "mhe:", mhe, "svd_entropy:", svd_ent)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        results.append((layer_id, ptype, mhe, svd_ent))
    results.sort(key=lambda x: (x[0], x[1]))
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute MHE per layer for HF checkpoints.")
    parser.add_argument("dirs", nargs="+", help="HF model dir(s), e.g. iter_0010000_hf iter_0019073_hf")
    parser.add_argument("--out", default="mhe_by_layer.csv", help="Output CSV path")
    parser.add_argument("--s", type=float, default=2, help="Riesz exponent for MHE")
    args = parser.parse_args()

    all_rows = []
    for d in args.dirs:
        label = os.path.basename(d.rstrip("/"))
        print(f"Loading {label}...")
        sd = load_state_dict(d)
        print(f"Computing MHE (s={args.s}) and SVD entropy for {label}...")
        rows = collect_mhe_by_layer(sd, s=args.s)
        for layer_id, ptype, mhe, svd_ent in rows:
            all_rows.append({"dir": label, "layer": layer_id, "param_type": ptype, "mhe": mhe, "svd_entropy": svd_ent})

    from collections import defaultdict
    key_to_vals = defaultdict(dict)
    for r in all_rows:
        key = (r["layer"], r["param_type"])
        key_to_vals[key][r["dir"]] = {"mhe": r["mhe"], "svd_entropy": r["svd_entropy"]}

    dirs_ordered = [os.path.basename(d.rstrip("/")) for d in args.dirs]
    with open(args.out, "w") as f:
        header = ["layer", "param_type"]
        for d in dirs_ordered:
            header.append(f"mhe_{d}")
        for d in dirs_ordered:
            header.append(f"svd_entropy_{d}")
        f.write(",".join(header) + "\n")
        for (layer_id, ptype) in sorted(key_to_vals.keys()):
            vals = key_to_vals[(layer_id, ptype)]
            row = [str(layer_id), ptype]
            for d in dirs_ordered:
                row.append(str(vals.get(d, {}).get("mhe", "")))
            for d in dirs_ordered:
                row.append(str(vals.get(d, {}).get("svd_entropy", "")))
            f.write(",".join(row) + "\n")
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    exit(main())