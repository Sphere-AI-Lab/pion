#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
"""
示例：如何使用 lm-eval-harness 评测 Megatron 模型

使用方法：
1. 安装依赖：
   pip install lm-eval

2. 运行示例：
   python tasks/example_lm_eval_usage.py \
     --checkpoint-path /path/to/checkpoint \
     --tasks hellaswag,arc,mmlu \
     --batch-size 8
"""

import argparse
import os
import sys
import torch

# 添加 Megatron 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron.training import get_args, initialize_megatron
from megatron.core.enums import ModelType
from tasks.lm_eval_adapter import (
    MegatronLMAdapter,
    create_megatron_adapter_from_checkpoint,
    use_megatron_with_lm_eval_harness
)


def get_model_provider():
    """
    定义你的模型提供函数
    
    这个函数应该返回一个未初始化的模型实例
    示例（需要根据你的实际模型调整）：
    """
    from megatron.model import GPTModel
    
    def model_provider(pre_process=True, post_process=True):
        """构建模型"""
        args = get_args()
        
        model = GPTModel(
            config=args,
            num_tokentypes=0,
            parallel_output=False,
            pre_process=pre_process,
            post_process=post_process
        )
        
        return model
    
    return model_provider


def get_tokenizer():
    """
    获取 tokenizer
    
    需要根据你的实际 tokenizer 实现
    """
    from transformers import AutoTokenizer
    
    args = get_args()
    
    # 从 args 获取 tokenizer 路径或名称
    tokenizer_path = getattr(args, 'tokenizer_path', None)
    if tokenizer_path is None:
        # 默认使用 GPT2 tokenizer
        tokenizer_path = "gpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 设置 pad token（如果需要）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="使用 lm-eval-harness 评测 Megatron 模型")
    
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        required=True,
        help='模型检查点路径'
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        default='hellaswag,arc',
        help='要评测的任务，用逗号分隔（例如: hellaswag,arc,mmlu）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='批次大小'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备类型 (cuda/cpu)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='结果输出路径（可选）'
    )
    
    # Megatron 参数（需要根据实际情况添加）
    parser.add_argument(
        '--megatron-config',
        type=str,
        default=None,
        help='Megatron 配置文件路径'
    )
    
    args, unknown_args = parser.parse_known_args()
    
    # 初始化 Megatron（如果需要）
    # 注意：这需要根据你的实际设置调整
    # initialize_megatron(extra_args_provider=None, args_defaults={})
    
    # 解析任务列表
    tasks = [task.strip() for task in args.tasks.split(',')]
    
    print(f"开始评测任务: {tasks}")
    print(f"检查点路径: {args.checkpoint_path}")
    
    try:
        # 获取模型提供函数和 tokenizer
        model_provider = get_model_provider()
        tokenizer = get_tokenizer()
        
        # 方法1: 使用便捷函数
        print("\n=== 使用方法1: 便捷函数 ===")
        results = use_megatron_with_lm_eval_harness(
            model_provider_func=model_provider,
            checkpoint_path=args.checkpoint_path,
            tokenizer=tokenizer,
            tasks=tasks,
            model_type=ModelType.encoder_or_decoder,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # 打印结果
        print("\n=== 评测结果 ===")
        for task_name, task_results in results['results'].items():
            print(f"\n任务: {task_name}")
            for metric, value in task_results.items():
                print(f"  {metric}: {value:.4f}")
        
        # 保存结果
        if args.output_path:
            import json
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {args.output_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
