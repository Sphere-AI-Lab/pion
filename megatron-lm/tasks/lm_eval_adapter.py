# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
"""
适配器：将 Megatron 模型与 lm-eval-harness 集成

使用方法：
1. 安装 lm-eval-harness: pip install lm-eval
2. 使用此适配器运行评测：
   python -m lm_eval --model megatron \
     --model_args checkpoint_path=/path/to/checkpoint,config_path=/path/to/config \
     --tasks hellaswag,arc \
     --device cuda
"""

import os
import torch
from typing import List, Optional, Tuple, Union
from functools import partial

from megatron.training import get_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.training import get_model, setup_model_and_optimizer
from megatron.core.enums import ModelType
from megatron.core import mpu
from tasks.finetune_utils import process_batch
from tasks.eval_utils import accuracy_func_provider, calculate_correct_answers


class MegatronLMAdapter:
    """
    Megatron 模型适配器，用于 lm-eval-harness
    
    这个适配器将 Megatron 模型包装成 lm-eval-harness 可以使用的格式
    """
    
    def __init__(
        self,
        model_provider_func,
        checkpoint_path: Optional[str] = None,
        model_type: ModelType = ModelType.encoder_or_decoder,
        tokenizer=None,
        max_length: int = 2048,
        device: str = "cuda"
    ):
        """
        初始化适配器
        
        Args:
            model_provider_func: Megatron 模型提供函数
            checkpoint_path: 模型检查点路径
            model_type: 模型类型
            tokenizer: tokenizer 对象
            max_length: 最大序列长度
            device: 设备类型
        """
        self.model_provider_func = model_provider_func
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        
        # 初始化模型
        self._setup_model()
        
    def _setup_model(self):
        """设置和加载模型"""
        # 获取模型
        self.model = get_model(
            self.model_provider_func,
            self.model_type,
            wrap_with_ddp=False
        )
        
        # 加载检查点
        if self.checkpoint_path:
            args = get_args()
            original_load = args.load
            args.load = self.checkpoint_path
            original_rng = args.no_load_rng
            args.no_load_rng = True
            
            load_checkpoint(self.model, None, None)
            
            args.load = original_load
            args.no_load_rng = original_rng
        
        # 设置为评估模式
        for m in self.model:
            m.eval()
    
    def _tokenize(self, text: Union[str, List[str]]) -> dict:
        """
        Tokenize 输入文本
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            tokenized 结果
        """
        if isinstance(text, str):
            text = [text]
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer 未设置")
        
        # 使用 tokenizer 进行编码
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return encoded
    
    def generate(
        self,
        context: Union[str, List[str]],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        生成文本（用于生成类任务）
        
        Args:
            context: 输入上下文
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling 参数
            
        Returns:
            生成的文本列表
        """
        # 注意：Megatron 的生成逻辑可能需要根据具体模型实现
        # 这里提供一个基础框架
        raise NotImplementedError(
            "生成功能需要根据具体的 Megatron 模型实现。"
            "如果使用 GPT 模型，请参考 megatron/text_generation_utils.py"
        )
    
    def loglikelihood(
        self,
        requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """
        计算对数似然（用于多项选择等任务）
        
        Args:
            requests: [(context, continuation)] 列表
            
        Returns:
            [(loglikelihood, is_greedy)] 列表
        """
        results = []
        
        with torch.no_grad():
            for context, continuation in requests:
                # 构建完整文本
                full_text = context + continuation
                
                # Tokenize
                encoded = self._tokenize([full_text])
                tokens = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 获取 context 的长度
                context_encoded = self._tokenize([context])
                context_len = context_encoded['input_ids'].shape[1]
                
                # 前向传播
                output = self.model[0](tokens, attention_mask)
                
                # 计算 logits
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                # 计算 continuation 的对数似然
                # 注意：这里需要根据实际模型输出格式调整
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                # 提取 continuation 部分的 log 概率
                continuation_tokens = tokens[0, context_len:]
                continuation_log_probs = log_probs[0, context_len-1:-1]
                
                # 计算总的对数似然
                loglikelihood = 0.0
                for i, token_id in enumerate(continuation_tokens):
                    if i < continuation_log_probs.shape[0]:
                        loglikelihood += continuation_log_probs[i, token_id].item()
                
                # 判断是否为贪婪选择（简化处理）
                is_greedy = True
                
                results.append((loglikelihood, is_greedy))
        
        return results
    
    def loglikelihood_rolling(
        self,
        requests: List[str]
    ) -> List[float]:
        """
        计算滚动对数似然（用于 perplexity 等任务）
        
        Args:
            requests: 文本列表
            
        Returns:
            对数似然列表
        """
        results = []
        
        with torch.no_grad():
            for text in requests:
                # Tokenize
                encoded = self._tokenize([text])
                tokens = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 前向传播
                output = self.model[0](tokens, attention_mask)
                
                # 计算 logits
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                # 计算对数似然
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                # 计算总的对数似然
                loglikelihood = 0.0
                for i in range(tokens.shape[1] - 1):
                    token_id = tokens[0, i + 1]
                    loglikelihood += log_probs[0, i, token_id].item()
                
                results.append(loglikelihood)
        
        return results


def create_megatron_adapter_from_checkpoint(
    checkpoint_path: str,
    model_provider_func,
    tokenizer,
    model_type: ModelType = ModelType.encoder_or_decoder,
    **kwargs
) -> MegatronLMAdapter:
    """
    从检查点创建 Megatron 适配器
    
    Args:
        checkpoint_path: 检查点路径
        model_provider_func: 模型提供函数
        tokenizer: tokenizer 对象
        model_type: 模型类型
        **kwargs: 其他参数
        
    Returns:
        MegatronLMAdapter 实例
    """
    return MegatronLMAdapter(
        model_provider_func=model_provider_func,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        tokenizer=tokenizer,
        **kwargs
    )


def use_megatron_with_lm_eval_harness(
    model_provider_func,
    checkpoint_path: str,
    tokenizer,
    tasks: List[str],
    model_type: ModelType = ModelType.encoder_or_decoder,
    **kwargs
):
    """
    使用 lm-eval-harness 评测 Megatron 模型的便捷函数
    
    示例:
        from tasks.lm_eval_adapter import use_megatron_with_lm_eval_harness
        from your_model import model_provider
        from your_tokenizer import get_tokenizer
        
        tokenizer = get_tokenizer()
        results = use_megatron_with_lm_eval_harness(
            model_provider_func=model_provider,
            checkpoint_path="/path/to/checkpoint",
            tokenizer=tokenizer,
            tasks=["hellaswag", "arc", "mmlu"]
        )
        print(results)
    
    Args:
        model_provider_func: Megatron 模型提供函数
        checkpoint_path: 检查点路径
        tokenizer: tokenizer 对象
        tasks: 要评测的任务列表
        model_type: 模型类型
        **kwargs: 其他参数
        
    Returns:
        评测结果字典
    """
    try:
        from lm_eval import simple_evaluate
    except ImportError:
        raise ImportError(
            "请先安装 lm-eval-harness: pip install lm-eval"
        )
    
    # 创建适配器
    adapter = create_megatron_adapter_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_provider_func=model_provider_func,
        tokenizer=tokenizer,
        model_type=model_type,
        **kwargs
    )
    
    # 运行评测
    results = simple_evaluate(
        model=adapter,
        tasks=tasks,
        **kwargs
    )
    
    return results


# 与 eval_utils.py 集成的函数
def accuracy_func_provider_with_lm_eval(
    single_dataset_provider,
    lm_eval_tasks: Optional[List[str]] = None
):
    """
    结合 accuracy_func_provider 和 lm-eval-harness 的评测函数
    
    这个函数可以在训练过程中同时使用 Megatron 原生的评测和 lm-eval-harness
    
    Args:
        single_dataset_provider: 数据集提供函数
        lm_eval_tasks: 要运行的 lm-eval-harness 任务列表
        
    Returns:
        评测函数
    """
    # 获取原生的 accuracy 函数
    native_metrics_func = accuracy_func_provider(single_dataset_provider)
    
    def combined_metrics_func(model, epoch, output_predictions=False):
        # 运行原生评测
        native_metrics_func(model, epoch, output_predictions)
        
        # 如果指定了 lm-eval 任务，也运行它们
        if lm_eval_tasks:
            try:
                from lm_eval import simple_evaluate
                
                # 注意：这里需要将 model 适配成 lm-eval 格式
                # 实际实现可能需要根据具体情况调整
                print_rank_last("运行 lm-eval-harness 评测...")
                # results = simple_evaluate(...)
                # 打印结果
                
            except ImportError:
                print_rank_last("警告: lm-eval-harness 未安装，跳过 lm-eval 评测")
    
    return combined_metrics_func
