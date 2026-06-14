"""
对比两个 Hugging Face 格式模型目录：结构 + 参数值是否一致。
用法: python compare_hf_models.py <dir1> <dir2>
"""
import argparse
import json
import os

import torch


def load_state_dict(path):
    """支持 .safetensors 和 pytorch_model.bin / model.safetensors"""
    base = path
    # 优先 safetensors
    st = os.path.join(base, "model.safetensors")
    if os.path.isfile(st):
        from safetensors.torch import load_file
        return load_file(st)
    pt = os.path.join(base, "pytorch_model.bin")
    if os.path.isfile(pt):
        return torch.load(pt, map_location="cpu", weights_only=True)
    # 兼容单文件在根目录
    for f in os.listdir(base):
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(os.path.join(base, f))
        if f.endswith(".bin"):
            return torch.load(os.path.join(base, f), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No state dict found in {base}")


def load_config(path):
    p = os.path.join(path, "config.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def compare_config(c1, c2):
    if c1 is None and c2 is None:
        return True, "both have no config"
    if c1 is None or c2 is None:
        return False, "one has config, the other does not"
    # 只比和结构/形状相关的
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "intermediate_size"):
        if key in c1 and key in c2 and c1[key] != c2[key]:
            return False, f"config.{key}: {c1[key]} vs {c2[key]}"
    return True, "config structure ok"


def compare_state_dicts(sd1, sd2, rtol=1e-5, atol=1e-8):
    """比较两个 state_dict：keys、shapes、数值。"""
    keys1, keys2 = set(sd1.keys()), set(sd2.keys())
    only1 = keys1 - keys2
    only2 = keys2 - keys1
    if only1:
        return False, f"Keys only in dir1: {sorted(only1)[:5]}{'...' if len(only1)>5 else ''}"
    if only2:
        return False, f"Keys only in dir2: {sorted(only2)[:5]}{'...' if len(only2)>5 else ''}"

    for k in keys1:
        t1, t2 = sd1[k], sd2[k]
        if t1.shape != t2.shape:
            return False, f"Shape mismatch '{k}': {t1.shape} vs {t2.shape}"
        if t1.dtype != t2.dtype:
            # 允许 float32 vs bfloat16 等，统一转 float32 再比
            t1 = t1.float() if t1.is_floating_point() else t1
            t2 = t2.float() if t2.is_floating_point() else t2
        if t1.is_floating_point():
            if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
                diff = (t1 - t2).abs()
                return False, f"Value mismatch '{k}': max_diff={diff.max().item():.6e}, mean_diff={diff.float().mean().item():.6e}"
        else:
            if not torch.equal(t1, t2):
                return False, f"Value mismatch (int) '{k}'"
    return True, "all keys match in shape and value"


def main():
    parser = argparse.ArgumentParser(description="Compare two HF-format model dirs (structure + params).")
    parser.add_argument("dir1", help="First HF model directory")
    parser.add_argument("dir2", help="Second HF model directory")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for float compare")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for float compare")
    args = parser.parse_args()

    dir1 = args.dir1.rstrip("/")
    dir2 = args.dir2.rstrip("/")
    print(f"Dir1: {dir1}")
    print(f"Dir2: {dir2}")

    # config
    c1, c2 = load_config(dir1), load_config(dir2)
    ok, msg = compare_config(c1, c2)
    print(f"Config: {msg}")
    if not ok:
        print("FAIL: config differs.")
        return

    # state dict
    try:
        sd1 = load_state_dict(dir1)
        sd2 = load_state_dict(dir2)
    except Exception as e:
        print(f"Load state dict error: {e}")
        return
    print(f"Dir1 keys: {len(sd1)}, Dir2 keys: {len(sd2)}")
    ok, msg = compare_state_dicts(sd1, sd2, rtol=args.rtol, atol=args.atol)
    print(f"State dict: {msg}")
    if ok:
        print("PASS: structure and parameter values are consistent.")
    else:
        print("FAIL: structure or parameter values differ.")
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())