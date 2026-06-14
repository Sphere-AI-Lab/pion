# Pion

This repository extends [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) with the Pion optimizer and the training scripts used in our paper.

## Implementation

### `pion.py`

Path: `megatron/core/optimizer/pion.py`

Pion applies a bilateral Lie update to 2D weight matrices. Vector parameters, embeddings, and the output layer still use AdamW. The matrix update is:

```
W ← W exp(A_in) + exp(A_out) W − W
```

The truncated matrix exponential is implemented with a Taylor expansion (see `--pion-degree`).

Two momentum geometries from the paper are included:

| Geometry | Flag |
|----------|------|
| Lie–Lie | `--pion-momentum lie_lie` |
| Transported ambient + ambient | `--pion-momentum transported_ambient_ambient` |

Use `--optimizer pion` to enable it. Other commonly used flags:

- `--pion-update-side {both, alternate}` — bilateral vs. alternating unilateral updates
- `--pion-scaling {rms, none}` — update scaling (`rms` uses `--pion-rms`, default 0.2)
- `--pion-use-second-momentum` — optional second-moment normalization
- `--pion-degree` — Taylor order for the matrix exponential (default: 2)

### `pion_msign.py`

Path: `megatron/core/optimizer/pion_msign.py`

This variant keeps Pion's update structure but orthogonalizes gradients with a Muon-style msign step (Newton–Schulz). Enable it with `--optimizer pion_msign`.

Relevant flags include `--pion-lr`, `--pion-min-lr`, `--pion-msign-lambda`, `--num-ns-steps`, and `--coefficient-type`. See `opt_llama_60M_pion_msign.sh` for an example.

## Setup

1. Install dependencies following the main [README.md](./README.md).
2. Prepare C4 data in Megatron `.bin` format and a HuggingFace tokenizer. Override `TRAIN_BASE_PATH` and `VALID_BASE_PATH` in the launch scripts if your paths differ.
3. From the repo root:

```bash
bash opt_llama_60M_pion.sh
```

## Reproducing paper experiments

### Exploration runs (Llama 60M, C4, ~9.6B tokens)

```bash
bash opt_llama_60M_pion.sh
```

Default setup: 2 GPUs, `global_batch_size=512`, `seq_length=256`, RMSNorm, `--optimizer pion`.

Sweep the main Pion knobs via environment variables at the top of the script or directly in `TRAINING_ARGS`:

| Knob | Options |
|------|---------|
| `PION_MOMENTUM` | `lie_lie`, `transported_ambient_ambient` |
| `PION_UPDATE_SIDE` | `both`, `alternate` |
| `--pion-scaling` | `rms`, `none` |
| `USE_SECOND_MOMENTUM=1` | adds `--pion-use-second-momentum` |

Examples:

```bash
# Lie–Lie, bilateral updates
PION_MOMENTUM=lie_lie PION_UPDATE_SIDE=both bash opt_llama_60M_pion.sh

# Transported ambient, alternating sides, with second momentum
PION_MOMENTUM=transported_ambient_ambient \
PION_UPDATE_SIDE=alternate \
USE_SECOND_MOMENTUM=1 \
bash opt_llama_60M_pion.sh
```

### BF16 pretraining (Llama 1.3B, ~54B tokens, 8 GPUs)

```bash
bash opt_llama_1.3B_adamw.sh   # AdamW
bash opt_llama_1.3B_muon.sh    # Muon
bash opt_llama_1.3B_pion.sh    # Pion
```

These scripts use pure bf16 optimizer training (`--pure-bf16-optimizer`). Update `REPO_PATH`, data paths, and GPU settings for your cluster before launching.

### Normalization-free training (Llama 60M, `--normalization NoNorm`)

```bash
bash opt_llama_60M_adamw_no_norm.sh
bash opt_llama_60M_muon_no_norm.sh
bash opt_llama_60M_pion_no_norm.sh
bash opt_llama_60M_pion_msign.sh
```

## Launch scripts

| Script | Model | Optimizer | Notes |
|--------|-------|-----------|-------|
| `opt_llama_60M_pion.sh` | 60M | Pion | Main entry for paper ablations |
| `opt_llama_60M_pion_no_norm.sh` | 60M | Pion | NoNorm |
| `opt_llama_60M_pion_msign.sh` | 60M | Pion msign | Muon orthogonalization + Pion |
| `opt_llama_60M_adamw_no_norm.sh` | 60M | AdamW | NoNorm baseline |
| `opt_llama_60M_muon_no_norm.sh` | 60M | Muon | NoNorm baseline |
| `opt_llama_1.3B_pion.sh` | 1.3B | Pion | BF16 main result |
| `opt_llama_1.3B_adamw.sh` | 1.3B | AdamW | BF16 baseline |
| `opt_llama_1.3B_muon.sh` | 1.3B | Muon | BF16 baseline |

## Hyperparameters

**`--optimizer pion`**

| Flag | Default | Description |
|------|---------|-------------|
| `--pion-degree` | 2 | Taylor order for matrix exponential |
| `--pion-momentum` | `none` (→ `lie_lie`) | Momentum geometry |
| `--pion-update-side` | `both` | `both` or `alternate` |
| `--pion-scaling` | `rms` | `rms` or `none` in unified `pion.py` |
| `--pion-rms` | 0.2 | RMS scale when `--pion-scaling rms` |
| `--pion-beta1` / `--pion-beta2` | 0.9 / 0.999 | Pion momentum betas |
| `--pion-use-second-momentum` | off | Second-moment normalization |
| `--pion-no-split-qkv` | split on | Disable Q/K/V split |
| `--pion-no-split-qkv-per-head` | per-head on | Disable per-head Q split |
| `--pion-qkv-split-granularity` | auto | `head`, `qkv`, or `group` |

**`--optimizer pion_msign`**

| Flag | Description |
|------|-------------|
| `--pion-lr` / `--pion-min-lr` | Separate LR schedule for matrix params |
| `--pion-msign-lambda` | msign step-size multiplier |
| `--num-ns-steps` | Newton–Schulz iterations |
| `--coefficient-type` | NS polynomial (e.g. `quintic`) |

**Limitations of unified `pion.py`**

- Distributed optimizer is not supported.
- Matrix updates require bf16 (fp16 is not supported).
- Scaling modes `fnorm`, `brb`, and `spectral` listed in some CLI help strings are not implemented in the unified file; use `rms` or `none`.

## Code layout

```
megatron/core/optimizer/
├── pion.py              # lie_lie & transported_ambient_ambient
├── pion_msign.py        # msign + Pion
└── optimizer_config.py  # pion_* config fields
```

Training runs through `pretrain_gpt.py`. Optimizers are wired in `megatron/training/training.py` via `get_megatron_pion_optimizer` and `get_megatron_pion_ortho_exp_optimizer`.

## Citation

If you use this code, please cite the Pion paper (replace with the final BibTeX when available):

```bibtex
@article{pion2025,
  title   = {Pion: ...},
  author  = {...},
  journal = {...},
  year    = {2025}
}
```

## Acknowledgments

Built on NVIDIA Megatron-LM / Megatron Core. See [README.md](./README.md) for upstream installation and usage.
