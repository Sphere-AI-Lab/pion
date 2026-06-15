# Pion: A Spectrum-Preserving Optimizer via Orthogonal Equivalence Transformation

This repository is the official PyTorch implementation of Pion Optimizer, by Kexuan Shi, Hanxuan Li,  Zeju Qiu, Yandong Wen, Simon Buchholz, Weiyang Liu.

The code is coming soon. Stay tuned.  :)

We have open-sourced two core implementations of Pion mentioned in our paper: `transported_ambient_ambient` and `lie_lie`. The specific code can be found in [`pion.py`](https://github.com/Sphere-AI-Lab/pion/blob/main/megatron-lm/megatron/core/optimizer/pion.py). Additionally, we further explored the gradient uniformization of muon under Pion's update rule, and we proposed a `pion_msign.py` version.

### Exploration
For the exploration experiments mentioned in the follow-up paper, please run: 
```bash
cd pion/Megatron-LM
bash opt_llama_60M_pion.sh
```
Modify the `pion-update-side`, `pion-momentum`, and `pion-scaling` parameters in the training script to conduct explorations.
### Pretraining Experiments
To reproduce the bf16 pretraining experiments in the paper, please use: 
```bash
bash opt_llama_1.3B_adamw.sh # AdamW
bash opt_llama_1.3B_muon.sh # Muon
bash opt_llama_1.3B_pion.sh # Pion
```

For reproducing the Normalization-free experiments in the paper, please use:
```bash
bash opt_llama_60M_adamw_no_norm.sh # AdamW
bash opt_llama_60M_muon_no_norm.sh # Muon
bash opt_llama_60M_pion_no_norm.sh # Pion
```

## Running RL Experiments

### Environment Setup

The RL experiments are built on top of [verl](https://github.com/volcengine/verl). Please follow the installation instructions in [verl/README.md](verl/README.md) to set up the environment.

Before running, you need to edit the scripts and replace the placeholder paths:
- `/path/to/your/dataset/` — path to the preprocessed dataset (see [verl data preparation](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html))
- `/path/to/your/model` — path to the pretrained model.

### Running GRPO Training with Pion Optimizer

We provide a ready-to-use script for training Qwen3-1.7B on the DeepMath dataset using GRPO with the Pion optimizer:

```bash
cd verl
bash examples/grpo_trainer/run_qwen3_1.7b_pion_deepmath.sh # for Qwen3-1.7B
bash examples/grpo_trainer/run_distilled_pion_deepmath.sh # for DeepSeek-R1-Distilled-Qwen-1.5B
```

To run baseline comparisons with AdamW and Muon:

```bash
# Qwen3-1.7B
bash examples/grpo_trainer/run_qwen3_1.7b_adamw_deepmath.sh   # AdamW
bash examples/grpo_trainer/run_qwen3_1.7b_muon_deepmath.sh    # Muon

# DeepSeek-R1-Distilled-Qwen-1.5B
bash examples/grpo_trainer/run_distilled_adamw_deepmath.sh    # AdamW
bash examples/grpo_trainer/run_distilled_muon_deepmath.sh     # Muon
```
