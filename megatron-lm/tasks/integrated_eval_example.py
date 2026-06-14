#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
"""
集成示例：同时使用 eval_utils.py 和 lm-eval-harness 进行评测

这个示例展示了如何：
1. 使用 eval_utils.py 评测自定义数据集
2. 使用 lm-eval-harness 评测标准任务
3. 在训练过程中同时运行两种评测
"""

import os
import sys
import torch

# 添加 Megatron 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron.training import get_args, print_rank_last
from megatron.core.enums import ModelType
from tasks.eval_utils import accuracy_func_provider
from tasks.lm_eval_adapter import (
    create_megatron_adapter_from_checkpoint,
    use_megatron_with_lm_eval_harness
)
from tasks.finetune_utils import finetune


def create_integrated_eval_callback(
    single_dataset_provider,
    model_provider_func,
    tokenizer,
    lm_eval_tasks=None,
    lm_eval_checkpoint_path=None
):
    """
    创建集成的评测回调函数
    
    这个函数结合了：
    - eval_utils.py 的 accuracy_func_provider（用于自定义数据集）
    - lm-eval-harness（用于标准评测任务）
    
    Args:
        single_dataset_provider: 数据集提供函数（用于 eval_utils.py）
        model_provider_func: 模型提供函数
        tokenizer: tokenizer 对象
        lm_eval_tasks: lm-eval-harness 要评测的任务列表
        lm_eval_checkpoint_path: 用于 lm-eval 的检查点路径（如果与训练检查点不同）
    
    Returns:
        评测回调函数
    """
    # 创建原生 Megatron 评测函数
    native_metrics_func = accuracy_func_provider(single_dataset_provider)
    
    def integrated_metrics_func(model, epoch, output_predictions=False):
        """
        集成的评测函数
        
        这个函数会：
        1. 运行原生 Megatron 评测（使用 eval_utils.py）
        2. 如果指定了 lm-eval 任务，也运行它们
        """
        print_rank_last("=" * 80)
        print_rank_last(f"开始评测 Epoch {epoch}")
        print_rank_last("=" * 80)
        
        # 1. 运行原生 Megatron 评测
        print_rank_last("\n[1/2] 运行 Megatron 原生评测 (eval_utils.py)...")
        native_metrics_func(model, epoch, output_predictions)
        
        # 2. 运行 lm-eval-harness 评测（如果指定了任务）
        if lm_eval_tasks:
            print_rank_last(f"\n[2/2] 运行 lm-eval-harness 评测 (任务: {lm_eval_tasks})...")
            
            try:
                from lm_eval import simple_evaluate
                
                # 确定检查点路径
                args = get_args()
                checkpoint_path = lm_eval_checkpoint_path or args.load
                
                if checkpoint_path is None:
                    print_rank_last("警告: 未指定检查点路径，跳过 lm-eval 评测")
                    return
                
                # 创建适配器
                print_rank_last(f"加载检查点: {checkpoint_path}")
                adapter = create_megatron_adapter_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model_provider_func=model_provider_func,
                    tokenizer=tokenizer,
                    model_type=ModelType.encoder_or_decoder
                )
                
                # 运行评测
                print_rank_last("开始 lm-eval-harness 评测...")
                results = simple_evaluate(
                    model=adapter,
                    tasks=lm_eval_tasks,
                    batch_size=getattr(args, 'eval_batch_size', 8),
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # 打印结果
                print_rank_last("\n=== lm-eval-harness 评测结果 ===")
                if 'results' in results:
                    for task_name, task_results in results['results'].items():
                        print_rank_last(f"\n任务: {task_name}")
                        if isinstance(task_results, dict):
                            for metric, value in task_results.items():
                                if isinstance(value, (int, float)):
                                    print_rank_last(f"  {metric}: {value:.4f}")
                                else:
                                    print_rank_last(f"  {metric}: {value}")
                        else:
                            print_rank_last(f"  结果: {task_results}")
                
                print_rank_last("=" * 80)
                
            except ImportError:
                print_rank_last("警告: lm-eval-harness 未安装，跳过 lm-eval 评测")
                print_rank_last("安装方法: pip install lm-eval")
            except Exception as e:
                print_rank_last(f"lm-eval 评测出错: {e}")
                import traceback
                traceback.print_exc()
    
    return integrated_metrics_func


def example_usage():
    """
    使用示例
    
    这个函数展示了如何在实际训练脚本中使用集成评测
    """
    
    # 示例1: 定义数据集提供函数
    def my_dataset_provider(datapath):
        """
        你的数据集提供函数
        
        需要返回一个数据集对象，包含：
        - dataset_name 属性
        - __getitem__ 方法，返回包含 'text', 'types', 'label', 'padding_mask' 的字典
        """
        # 这里需要根据你的实际数据集实现
        # 示例：
        # from your_dataset import YourDataset
        # return YourDataset(datapath)
        raise NotImplementedError("请实现你的数据集提供函数")
    
    # 示例2: 定义模型提供函数
    def model_provider(pre_process=True, post_process=True):
        """
        你的模型提供函数
        """
        # 这里需要根据你的实际模型实现
        # 示例：
        # from megatron.model import GPTModel
        # args = get_args()
        # model = GPTModel(...)
        # return model
        raise NotImplementedError("请实现你的模型提供函数")
    
    # 示例3: 获取 tokenizer
    def get_tokenizer():
        """
        获取 tokenizer
        """
        from transformers import AutoTokenizer
        args = get_args()
        tokenizer_path = getattr(args, 'tokenizer_path', 'gpt2')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    # 示例4: 创建集成评测回调
    tokenizer = get_tokenizer()
    integrated_callback = create_integrated_eval_callback(
        single_dataset_provider=my_dataset_provider,
        model_provider_func=model_provider,
        tokenizer=tokenizer,
        lm_eval_tasks=["hellaswag", "arc"],  # 可选：指定要评测的 lm-eval 任务
        lm_eval_checkpoint_path=None  # 如果为 None，使用训练检查点
    )
    
    # 示例5: 在训练中使用
    def end_of_epoch_callback_provider():
        return integrated_callback
    
    # 在 finetune 函数中使用
    # finetune(
    #     train_valid_datasets_provider=...,
    #     model_provider=model_provider,
    #     end_of_epoch_callback_provider=end_of_epoch_callback_provider
    # )


def standalone_eval_example():
    """
    独立评测示例（不训练，只评测）
    
    这个示例展示了如何仅运行评测，不进行训练
    """
    print("=" * 80)
    print("独立评测示例")
    print("=" * 80)
    
    # 设置参数（仅评测模式）
    args = get_args()
    args.epochs = 0  # 不训练
    
    # 定义你的函数
    def model_provider(pre_process=True, post_process=True):
        # 实现你的模型
        pass
    
    def get_tokenizer():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained('gpt2')
    
    def dataset_provider(datapath):
        # 实现你的数据集
        pass
    
    tokenizer = get_tokenizer()
    checkpoint_path = args.load
    
    # 方法1: 使用原生 eval_utils.py
    print("\n方法1: 使用 eval_utils.py 评测自定义数据集")
    metrics_func = accuracy_func_provider(dataset_provider)
    # metrics_func(model, epoch=-1, output_predictions=False)
    
    # 方法2: 使用 lm-eval-harness
    print("\n方法2: 使用 lm-eval-harness 评测标准任务")
    results = use_megatron_with_lm_eval_harness(
        model_provider_func=model_provider,
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        tasks=["hellaswag", "arc", "mmlu"],
        batch_size=8
    )
    print(f"评测结果: {results}")
    
    # 方法3: 同时使用两种方法
    print("\n方法3: 同时使用两种评测方法")
    integrated_callback = create_integrated_eval_callback(
        single_dataset_provider=dataset_provider,
        model_provider_func=model_provider,
        tokenizer=tokenizer,
        lm_eval_tasks=["hellaswag", "arc"]
    )
    # integrated_callback(model, epoch=-1)


if __name__ == '__main__':
    print("这是一个示例脚本，展示了如何集成使用 eval_utils.py 和 lm-eval-harness")
    print("请根据你的实际需求修改和实现相关函数")
    print("\n主要功能：")
    print("1. create_integrated_eval_callback() - 创建集成评测回调")
    print("2. example_usage() - 在训练中使用集成评测")
    print("3. standalone_eval_example() - 独立评测示例")
