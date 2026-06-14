# Megatron 与 lm-eval-harness 集成指南

本指南说明如何将 Megatron 的 `eval_utils.py` 与 `lm-eval-harness` 结合使用，在不同任务上评测模型。

## 目录

1. [安装依赖](#安装依赖)
2. [基本使用方法](#基本使用方法)
3. [与 eval_utils.py 集成](#与-eval_utilspy-集成)
4. [高级用法](#高级用法)
5. [常见问题](#常见问题)

## 安装依赖

首先安装 `lm-eval-harness`：

```bash
pip install lm-eval
```

或者从源码安装：

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## 基本使用方法

### 方法1: 使用适配器类直接评测

```python
from tasks.lm_eval_adapter import MegatronLMAdapter, create_megatron_adapter_from_checkpoint
from megatron.core.enums import ModelType
from your_model import model_provider
from your_tokenizer import get_tokenizer

# 创建适配器
tokenizer = get_tokenizer()
adapter = create_megatron_adapter_from_checkpoint(
    checkpoint_path="/path/to/checkpoint",
    model_provider_func=model_provider,
    tokenizer=tokenizer,
    model_type=ModelType.encoder_or_decoder
)

# 使用 lm-eval-harness 评测
from lm_eval import simple_evaluate

results = simple_evaluate(
    model=adapter,
    tasks=["hellaswag", "arc", "mmlu"],
    batch_size=8,
    device="cuda"
)

print(results)
```

### 方法2: 使用便捷函数

```python
from tasks.lm_eval_adapter import use_megatron_with_lm_eval_harness
from your_model import model_provider
from your_tokenizer import get_tokenizer

tokenizer = get_tokenizer()
results = use_megatron_with_lm_eval_harness(
    model_provider_func=model_provider,
    checkpoint_path="/path/to/checkpoint",
    tokenizer=tokenizer,
    tasks=["hellaswag", "arc", "mmlu"],
    batch_size=8
)
```

### 方法3: 命令行使用

```bash
python tasks/example_lm_eval_usage.py \
    --checkpoint-path /path/to/checkpoint \
    --tasks hellaswag,arc,mmlu \
    --batch-size 8 \
    --output-path results.json
```

## 与 eval_utils.py 集成

### 在训练过程中同时使用两种评测方式

你可以将 `eval_utils.py` 的 `accuracy_func_provider` 与 `lm-eval-harness` 结合使用：

```python
from tasks.eval_utils import accuracy_func_provider
from tasks.lm_eval_adapter import accuracy_func_provider_with_lm_eval
from tasks.finetune_utils import finetune

# 定义数据集提供函数（用于 eval_utils.py）
def single_dataset_provider(datapath):
    # 返回你的数据集
    return your_dataset(datapath)

# 创建结合了两种评测方式的函数
def end_of_epoch_callback_provider():
    # 原生 Megatron 评测
    native_metrics_func = accuracy_func_provider(single_dataset_provider)
    
    # 结合 lm-eval-harness 评测
    combined_func = accuracy_func_provider_with_lm_eval(
        single_dataset_provider,
        lm_eval_tasks=["hellaswag", "arc"]  # 可选
    )
    
    return combined_func

# 在训练中使用
finetune(
    train_valid_datasets_provider=...,
    model_provider=...,
    end_of_epoch_callback_provider=end_of_epoch_callback_provider
)
```

### 仅评测模式（不训练）

```python
from tasks.lm_eval_adapter import use_megatron_with_lm_eval_harness
from megatron.training import get_args

# 设置仅评测模式
args = get_args()
args.epochs = 0  # 不训练，只评测

# 运行评测
results = use_megatron_with_lm_eval_harness(
    model_provider_func=model_provider,
    checkpoint_path=args.load,
    tokenizer=tokenizer,
    tasks=["hellaswag", "arc", "mmlu", "winogrande"]
)
```

## 高级用法

### 自定义评测任务

如果你需要评测自定义任务，可以：

1. **使用 eval_utils.py 评测自定义数据集**：

```python
from tasks.eval_utils import accuracy_func_provider

def my_custom_dataset_provider(datapath):
    # 实现你的数据集
    class CustomDataset:
        def __init__(self, datapath):
            self.datapath = datapath
            self.dataset_name = "custom_task"
            # ... 加载数据
        
        def __getitem__(self, idx):
            return {
                'text': ...,
                'types': ...,
                'label': ...,
                'padding_mask': ...,
                'uid': ...
            }
    
    return CustomDataset(datapath)

# 使用
metrics_func = accuracy_func_provider(my_custom_dataset_provider)
metrics_func(model, epoch=0)
```

2. **使用 lm-eval-harness 评测自定义任务**：

参考 [lm-eval-harness 文档](https://github.com/EleutherAI/lm-evaluation-harness) 创建自定义任务。

### 批量评测多个检查点

```python
import os
from tasks.lm_eval_adapter import use_megatron_with_lm_eval_harness

checkpoint_dir = "/path/to/checkpoints"
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("iter_")]

all_results = {}
for checkpoint in sorted(checkpoints):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    print(f"评测检查点: {checkpoint}")
    
    results = use_megatron_with_lm_eval_harness(
        model_provider_func=model_provider,
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        tasks=["hellaswag", "arc"]
    )
    
    all_results[checkpoint] = results

# 保存所有结果
import json
with open("all_checkpoint_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
```

### 分布式评测

适配器支持 Megatron 的分布式设置。确保在使用前正确初始化分布式环境：

```python
from megatron.training import initialize_megatron

# 初始化 Megatron（包括分布式设置）
initialize_megatron(extra_args_provider=None)

# 然后使用适配器
adapter = create_megatron_adapter_from_checkpoint(...)
```

## 常见问题

### Q1: 如何获取模型提供函数？

模型提供函数应该返回你的模型实例。示例：

```python
def model_provider(pre_process=True, post_process=True):
    from megatron.model import GPTModel
    args = get_args()
    model = GPTModel(
        config=args,
        num_tokentypes=0,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process
    )
    return model
```

### Q2: Tokenizer 不匹配怎么办？

确保使用与训练时相同的 tokenizer。如果使用 HuggingFace tokenizer：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-model-name")
# 或者从文件加载
tokenizer = AutoTokenizer.from_pretrained("/path/to/tokenizer")
```

### Q3: 内存不足怎么办？

- 减小 `batch_size`
- 使用梯度检查点（如果支持）
- 使用更少的并行度

### Q4: 如何只评测特定任务？

在 `tasks` 参数中只指定需要的任务：

```python
results = use_megatron_with_lm_eval_harness(
    ...,
    tasks=["hellaswag"]  # 只评测 hellaswag
)
```

### Q5: 如何查看支持的评测任务？

```bash
lm-eval --tasks list
```

或者：

```python
from lm_eval import tasks
print(tasks.ALL_TASKS)
```

## 支持的评测任务

lm-eval-harness 支持大量评测任务，包括但不限于：

- **常识推理**: HellaSwag, ARC, WinoGrande
- **数学**: GSM8K, MATH
- **阅读理解**: SQuAD, RACE
- **知识**: MMLU, TruthfulQA
- **代码**: HumanEval, MBPP
- **多语言**: XNLI, XQuAD

完整列表请参考 [lm-evaluation-harness 文档](https://github.com/EleutherAI/lm-evaluation-harness)。

## 注意事项

1. **模型格式**: 确保检查点格式与你的模型定义匹配
2. **Tokenizer**: 使用与训练时相同的 tokenizer
3. **分布式**: 如果使用分布式训练，评测时也需要相应的分布式设置
4. **内存**: 某些任务（如 MMLU）可能需要较大内存
5. **设备**: 确保有足够的 GPU 内存

## 示例脚本

完整的使用示例请参考：
- `tasks/example_lm_eval_usage.py` - 基本使用示例
- `tasks/lm_eval_adapter.py` - 适配器实现

## 参考资源

- [lm-evaluation-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [lm-evaluation-harness 文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)
- [Megatron-LM 文档](https://github.com/NVIDIA/Megatron-LM)
