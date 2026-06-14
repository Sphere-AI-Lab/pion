#!/usr/bin/env python3
"""
示例：手动重现 lm-evaluation-harness 的模型加载流程

这个脚本展示了 lm-evaluation-harness 如何从 model_args 中读取配置并构建模型。
你可以运行这个脚本来理解整个流程。
"""

import sys
import os

# 添加 lm-evaluation-harness 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lm-evaluation-harness'))

import torch
import transformers
from lm_eval import utils
from lm_eval.api.registry import get_model


def step1_parse_args_string():
    """步骤1: 解析 model_args 字符串"""
    print("=" * 80)
    print("步骤1: 解析 model_args 字符串")
    print("=" * 80)
    
    # 模拟命令行参数: --model_args "pretrained=EleutherAI/gpt-j-6B,dtype=float,revision=main"
    model_args_string = "pretrained=EleutherAI/gpt-j-6B,dtype=float,revision=main"
    
    print(f"输入字符串: {model_args_string}")
    
    # 使用 lm-eval 的解析函数
    parsed_args = utils.simple_parse_args_string(model_args_string)
    
    print(f"解析后的字典: {parsed_args}")
    print()
    
    return parsed_args


def step2_get_model_class():
    """步骤2: 获取模型类"""
    print("=" * 80)
    print("步骤2: 获取模型类")
    print("=" * 80)
    
    # 模拟命令行参数: --model hf
    model_name = "hf"
    
    print(f"模型名称: {model_name}")
    
    # 从注册表获取模型类
    model_class = get_model(model_name)
    
    print(f"获取到的模型类: {model_class}")
    print(f"类名: {model_class.__name__}")
    print()
    
    return model_class


def step3_load_config(pretrained, revision="main", trust_remote_code=False):
    """步骤3: 加载模型配置"""
    print("=" * 80)
    print("步骤3: 加载模型配置")
    print("=" * 80)
    
    print(f"模型路径/名称: {pretrained}")
    print(f"Revision: {revision}")
    print(f"Trust remote code: {trust_remote_code}")
    
    # 使用 transformers.AutoConfig.from_pretrained 加载配置
    # 这是 lm-eval 在 _get_config() 中做的事情
    config = transformers.AutoConfig.from_pretrained(
        pretrained,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    
    print(f"\n配置类型: {type(config)}")
    print(f"模型类型: {config.model_type}")
    print(f"词汇表大小: {config.vocab_size}")
    print(f"隐藏层大小: {getattr(config, 'n_embd', getattr(config, 'hidden_size', 'N/A'))}")
    print(f"层数: {getattr(config, 'n_layer', getattr(config, 'num_hidden_layers', 'N/A'))}")
    print(f"注意力头数: {getattr(config, 'n_head', getattr(config, 'num_attention_heads', 'N/A'))}")
    
    # 显示配置文件的路径
    if hasattr(config, '_name_or_path'):
        print(f"\n配置来源: {config._name_or_path}")
    
    print()
    return config


def step4_load_tokenizer(pretrained, revision="main", trust_remote_code=False):
    """步骤4: 加载 tokenizer"""
    print("=" * 80)
    print("步骤4: 加载 Tokenizer")
    print("=" * 80)
    
    print(f"模型路径/名称: {pretrained}")
    
    # 使用 transformers.AutoTokenizer.from_pretrained 加载 tokenizer
    # 这是 lm-eval 在 _create_tokenizer() 中做的事情
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    
    print(f"Tokenizer 类型: {type(tokenizer)}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"特殊 token:")
    print(f"  - BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  - EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  - PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    print()
    return tokenizer


def step5_load_model(pretrained, config, dtype="float", revision="main", trust_remote_code=False):
    """步骤5: 加载模型"""
    print("=" * 80)
    print("步骤5: 加载模型")
    print("=" * 80)
    
    print(f"模型路径/名称: {pretrained}")
    print(f"数据类型: {dtype}")
    
    # 转换 dtype 字符串为 torch.dtype
    if dtype == "float":
        torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    print(f"Torch dtype: {torch_dtype}")
    
    # 根据配置确定模型类
    # 这是 lm-eval 在 _get_backend() 和 _create_model() 中做的事情
    model_type = config.model_type
    
    # 选择对应的模型类
    if model_type in ["gpt2", "gptj", "gpt_neox", "llama", "mistral", "mixtral"]:
        model_class = transformers.AutoModelForCausalLM
        print(f"使用模型类: AutoModelForCausalLM (因果语言模型)")
    elif model_type in ["t5", "bart"]:
        model_class = transformers.AutoModelForSeq2SeqLM
        print(f"使用模型类: AutoModelForSeq2SeqLM (序列到序列模型)")
    else:
        model_class = transformers.AutoModelForCausalLM
        print(f"使用默认模型类: AutoModelForCausalLM")
    
    print(f"\n开始加载模型（这可能需要一些时间）...")
    
    # 使用 from_pretrained 加载模型
    # 这是 lm-eval 在 _create_model() 中做的事情
    try:
        model = model_class.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map="auto",  # 自动分配设备
        )
        
        print(f"模型类型: {type(model)}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 显示模型设备
        if hasattr(model, 'device'):
            print(f"模型设备: {model.device}")
        elif hasattr(model, 'hf_device_map'):
            print(f"模型设备映射: {model.hf_device_map}")
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("注意: 这可能是由于内存不足或模型文件不存在")
        model = None
    
    print()
    return model


def step6_create_hflm_instance():
    """步骤6: 使用 lm-eval 创建完整的 HFLM 实例"""
    print("=" * 80)
    print("步骤6: 使用 lm-eval 创建完整的 HFLM 实例")
    print("=" * 80)
    
    # 模拟完整的命令行参数
    model_name = "hf"
    model_args_string = "pretrained=EleutherAI/gpt-j-6B,dtype=float"
    
    print(f"模型名称: {model_name}")
    print(f"模型参数: {model_args_string}")
    
    # 获取模型类
    model_class = get_model(model_name)
    
    # 使用 create_from_arg_string 创建实例（这是 lm-eval 实际使用的方法）
    print("\n使用 lm-eval 的 create_from_arg_string 创建模型实例...")
    
    try:
        # 注意: 实际创建模型需要时间和资源
        # 这里只是展示流程，不实际加载大模型
        print("（跳过实际模型加载，因为这需要大量内存和时间）")
        print("实际代码:")
        print(f"  lm = {model_class.__name__}.create_from_arg_string(")
        print(f"      '{model_args_string}',")
        print(f"      {{'batch_size': 8, 'device': 'cuda'}}")
        print(f"  )")
        
    except Exception as e:
        print(f"创建实例时出错: {e}")
    
    print()


def main():
    """主函数：演示完整的模型加载流程"""
    print("\n" + "=" * 80)
    print("lm-evaluation-harness 模型加载流程演示")
    print("=" * 80 + "\n")
    
    # 步骤1: 解析参数
    parsed_args = step1_parse_args_string()
    
    # 步骤2: 获取模型类
    model_class = step2_get_model_class()
    
    # 步骤3: 加载配置
    pretrained = parsed_args.get("pretrained", "EleutherAI/gpt-j-6B")
    revision = parsed_args.get("revision", "main")
    trust_remote_code = parsed_args.get("trust_remote_code", False)
    
    print("\n注意: 以下步骤需要网络连接来下载模型文件")
    print("如果模型已缓存，将使用缓存版本\n")
    
    try:
        config = step3_load_config(pretrained, revision, trust_remote_code)
        
        # 步骤4: 加载 tokenizer
        tokenizer = step4_load_tokenizer(pretrained, revision, trust_remote_code)
        
        # 步骤5: 加载模型（可选，需要大量内存）
        dtype = parsed_args.get("dtype", "float")
        
        print("\n是否加载完整模型？这需要大量内存和时间。")
        print("（在实际使用中，lm-eval 会自动执行这一步）")
        load_full_model = False  # 设置为 True 来实际加载模型
        
        if load_full_model:
            model = step5_load_model(pretrained, config, dtype, revision, trust_remote_code)
        else:
            print("跳过完整模型加载（设置 load_full_model=True 来实际加载）")
            print()
        
        # 步骤6: 展示如何使用 lm-eval 创建实例
        step6_create_hflm_instance()
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("这可能是因为:")
        print("1. 网络连接问题（无法下载模型）")
        print("2. 模型名称不存在")
        print("3. 缺少必要的依赖")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    print("\n关键要点:")
    print("1. model_args 字符串被解析为字典")
    print("2. 根据 --model 参数获取对应的模型类（如 HFLM）")
    print("3. 使用 AutoConfig.from_pretrained() 加载 config.json")
    print("4. 使用 AutoTokenizer.from_pretrained() 加载 tokenizer")
    print("5. 使用 AutoModelForCausalLM.from_pretrained() 加载模型权重")
    print("\n详细说明请参考: LM_EVAL_MODEL_LOADING.md")


if __name__ == "__main__":
    main()
