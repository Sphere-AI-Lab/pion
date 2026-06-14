#!/bin/bash
# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
#
# 在 Singularity 容器中离线安装 lm-evaluation-harness
#
# 使用方法:
#   1. 在宿主机上准备依赖包（可选，如果需要离线安装依赖）:
#      bash install_lm_eval_singularity.sh --prepare-deps
#
#   2. 在 Singularity 容器中安装:
#      singularity exec your_container.sif bash install_lm_eval_singularity.sh --install
#
#   3. 或者直接在容器内运行:
#      singularity shell your_container.sif
#      bash /path/to/install_lm_eval_singularity.sh --install

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LM_EVAL_DIR="$PROJECT_ROOT/lm-evaluation-harness"
DEPS_DIR="$PROJECT_ROOT/lm_eval_deps"  # 依赖包下载目录

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否在 Singularity 容器中
check_singularity() {
    if [ -n "${SINGULARITY_CONTAINER:-}" ] || [ -f /.singularity.d/runscript ]; then
        print_info "检测到 Singularity 容器环境"
        return 0
    else
        print_warn "未检测到 Singularity 容器环境，继续执行..."
        return 1
    fi
}

# 准备依赖包（在宿主机上运行）
prepare_dependencies() {
    print_info "准备依赖包..."
    
    if [ ! -d "$LM_EVAL_DIR" ]; then
        print_error "找不到 lm-evaluation-harness 目录: $LM_EVAL_DIR"
        exit 1
    fi
    
    mkdir -p "$DEPS_DIR"
    
    print_info "下载依赖包到: $DEPS_DIR"
    
    # 进入 lm-evaluation-harness 目录
    cd "$LM_EVAL_DIR"
    
    # 使用 pip download 下载所有依赖
    print_info "下载依赖包..."
    pip download -d "$DEPS_DIR" \
        -r <(python -c "
import sys
sys.path.insert(0, '.')
from pyproject.toml import tomllib
with open('pyproject.toml', 'rb') as f:
    config = tomllib.load(f)
deps = config['project']['dependencies']
for dep in deps:
    print(dep)
" 2>/dev/null || python3 -c "
import sys
sys.path.insert(0, '.')
try:
    import tomli as tomllib
except ImportError:
    print('需要安装 tomli: pip install tomli', file=sys.stderr)
    sys.exit(1)
with open('pyproject.toml', 'rb') as f:
    config = tomllib.load(f)
deps = config['project']['dependencies']
for dep in deps:
    print(dep)
") 2>/dev/null || {
        # 如果上面的方法失败，使用简单的方法
        print_warn "无法自动解析依赖，使用手动方式..."
        print_info "请手动下载以下依赖包:"
        cat <<EOF
accelerate>=0.26.0
evaluate>=0.4.0
datasets>=2.16.0
jsonlines
numexpr
peft>=0.2.0
pybind11>=2.6.2
pytablewriter
rouge-score>=0.0.4
sacrebleu>=1.5.0
scikit-learn>=0.24.1
sqlitedict
torch>=1.8
tqdm-multiprocess
transformers>=4.1
zstandard
dill
word2number
more_itertools
EOF
    }
    
    print_info "依赖包准备完成"
    print_info "依赖包位置: $DEPS_DIR"
}

# 在容器中安装
install_in_container() {
    print_info "开始在容器中安装 lm-evaluation-harness..."
    
    # 检查目录
    if [ ! -d "$LM_EVAL_DIR" ]; then
        print_error "找不到 lm-evaluation-harness 目录: $LM_EVAL_DIR"
        print_error "请确保 lm-evaluation-harness 代码在: $LM_EVAL_DIR"
        exit 1
    fi
    
    # 检查 Python
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        print_error "未找到 Python"
        exit 1
    fi
    
    PYTHON_CMD=$(command -v python3 || command -v python)
    print_info "使用 Python: $PYTHON_CMD"
    print_info "Python 版本: $($PYTHON_CMD --version)"
    
    # 检查 pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        print_error "未找到 pip"
        exit 1
    fi
    
    PIP_CMD=$(command -v pip3 || command -v pip)
    print_info "使用 pip: $PIP_CMD"
    
    # 进入 lm-evaluation-harness 目录
    cd "$LM_EVAL_DIR"
    print_info "工作目录: $(pwd)"
    
    # 如果有依赖包目录，先安装依赖
    if [ -d "$DEPS_DIR" ] && [ "$(ls -A $DEPS_DIR 2>/dev/null)" ]; then
        print_info "从本地目录安装依赖包: $DEPS_DIR"
        $PIP_CMD install --no-index --find-links "$DEPS_DIR" -r <(echo ".")
    else
        print_warn "未找到本地依赖包目录，将尝试从网络安装依赖"
        print_warn "如果容器无法访问网络，请先运行 --prepare-deps"
    fi
    
    # 安装 lm-evaluation-harness
    print_info "安装 lm-evaluation-harness..."
    
    # 方法1: 使用 editable install (推荐，便于开发)
    if [ "${EDITABLE_INSTALL:-1}" = "1" ]; then
        print_info "使用 editable 模式安装 (pip install -e .)"
        $PIP_CMD install -e . --no-deps || {
            print_warn "editable 安装失败，尝试普通安装..."
            $PIP_CMD install . --no-deps || {
                print_error "安装失败"
                exit 1
            }
        }
    else
        # 方法2: 普通安装
        print_info "使用普通模式安装 (pip install .)"
        $PIP_CMD install . --no-deps || {
            print_error "安装失败"
            exit 1
        }
    fi
    
    # 安装依赖（如果之前没有安装）
    print_info "安装依赖包..."
    $PIP_CMD install -e . || {
        print_warn "部分依赖可能安装失败，继续..."
    }
    
    # 验证安装
    print_info "验证安装..."
    if $PYTHON_CMD -c "import lm_eval; print('lm_eval version:', lm_eval.__version__ if hasattr(lm_eval, '__version__') else 'unknown')" 2>/dev/null; then
        print_info "✓ lm-evaluation-harness 安装成功！"
        
        # 检查命令行工具
        if command -v lm-eval &> /dev/null || command -v lm_eval &> /dev/null; then
            print_info "✓ 命令行工具可用"
        fi
        
        # 显示版本信息
        $PYTHON_CMD -c "
try:
    import lm_eval
    print('安装路径:', lm_eval.__file__ if hasattr(lm_eval, '__file__') else 'unknown')
except Exception as e:
    print('无法获取详细信息:', e)
" 2>/dev/null || true
        
    else
        print_error "安装验证失败"
        exit 1
    fi
    
    print_info "安装完成！"
    print_info "可以使用以下命令测试:"
    print_info "  python -m lm_eval --help"
    print_info "  lm-eval --help"
}

# 显示使用说明
show_usage() {
    cat <<EOF
在 Singularity 容器中离线安装 lm-evaluation-harness

使用方法:
  $0 [选项]

选项:
  --prepare-deps    在宿主机上准备依赖包（需要网络连接）
  --install         在容器中安装 lm-evaluation-harness
  --help           显示此帮助信息

示例:
  # 1. 在宿主机上准备依赖（可选）
  bash $0 --prepare-deps

  # 2. 在容器中安装
  singularity exec container.sif bash $0 --install
  
  # 或者进入容器后运行
  singularity shell container.sif
  bash $0 --install

环境变量:
  EDITABLE_INSTALL  是否使用 editable 安装 (默认: 1)
                    设置为 0 使用普通安装

EOF
}

# 主函数
main() {
    case "${1:-}" in
        --prepare-deps)
            prepare_dependencies
            ;;
        --install)
            check_singularity || true
            install_in_container
            ;;
        --help|-h)
            show_usage
            ;;
        "")
            print_warn "未指定操作，显示帮助信息"
            show_usage
            print_info "运行 '$0 --install' 开始安装"
            ;;
        *)
            print_error "未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
