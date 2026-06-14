# 在 Singularity 容器中离线安装 lm-evaluation-harness

本指南说明如何在 Singularity 容器中离线安装 lm-evaluation-harness。

## 前提条件

1. **已有 lm-evaluation-harness 代码**
   - 代码应该在项目根目录下的 `lm-evaluation-harness/` 目录中
   - 如果代码在其他位置，请修改脚本中的 `LM_EVAL_DIR` 变量

2. **Singularity 容器**
   - 容器中需要安装 Python 和 pip
   - 建议 Python 版本 >= 3.9

## 安装方法

### 方法1: 直接安装（推荐，如果容器可以访问网络）

如果容器可以访问网络，可以直接安装：

```bash
# 进入容器
singularity shell your_container.sif

# 进入 lm-evaluation-harness 目录
cd /path/to/lm-evaluation-harness

# 安装
pip install -e .
```

### 方法2: 完全离线安装

如果容器无法访问网络，需要先在宿主机上准备依赖包：

#### 步骤1: 在宿主机上准备依赖包（需要网络）

```bash
# 在宿主机上运行
bash Megatron-LM/tasks/install_lm_eval_singularity.sh --prepare-deps
```

这会下载所有依赖包到 `lm_eval_deps/` 目录。

#### 步骤2: 在容器中安装

```bash
# 方法A: 使用 singularity exec
singularity exec your_container.sif \
    bash /path/to/install_lm_eval_singularity.sh --install

# 方法B: 进入容器后运行
singularity shell your_container.sif
bash /path/to/install_lm_eval_singularity.sh --install
```

### 方法3: 手动安装

如果脚本不适用，可以手动安装：

```bash
# 1. 进入容器
singularity shell your_container.sif

# 2. 进入 lm-evaluation-harness 目录
cd /path/to/lm-evaluation-harness

# 3. 如果有本地依赖包目录
pip install --no-index --find-links /path/to/lm_eval_deps -e .

# 4. 或者直接安装（如果容器可以访问网络）
pip install -e .
```

## 验证安装

安装完成后，验证是否成功：

```bash
# 在容器中运行
python -c "import lm_eval; print('安装成功')"

# 或者测试命令行工具
lm-eval --help
# 或
python -m lm_eval --help
```

## 常见问题

### Q1: 找不到 lm-evaluation-harness 目录

**问题**: 脚本报错找不到目录

**解决**: 
1. 检查代码是否在正确位置：`/path/to/project/lm-evaluation-harness/`
2. 修改脚本中的 `LM_EVAL_DIR` 变量指向正确路径
3. 或者使用绝对路径运行脚本

### Q2: pip 安装失败，提示缺少依赖

**问题**: 安装时提示某些包找不到

**解决**:
1. 如果容器可以访问网络，让 pip 自动安装依赖：
   ```bash
   pip install -e .  # 不使用 --no-deps
   ```

2. 如果完全离线，确保已运行 `--prepare-deps` 下载所有依赖

3. 手动安装缺失的依赖：
   ```bash
   pip install --no-index --find-links /path/to/lm_eval_deps package_name
   ```

### Q3: 安装后无法导入 lm_eval

**问题**: `import lm_eval` 失败

**解决**:
1. 检查安装路径：
   ```bash
   python -c "import sys; print(sys.path)"
   ```

2. 检查是否在正确的 Python 环境中：
   ```bash
   which python
   which pip
   ```

3. 重新安装：
   ```bash
   pip uninstall lm_eval -y
   pip install -e /path/to/lm-evaluation-harness
   ```

### Q4: 权限问题

**问题**: 安装时提示权限不足

**解决**:
1. 使用 `--user` 选项安装到用户目录：
   ```bash
   pip install -e . --user
   ```

2. 或者使用虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

### Q5: 依赖版本冲突

**问题**: 某些依赖包版本与现有包冲突

**解决**:
1. 查看冲突的包：
   ```bash
   pip check
   ```

2. 升级或降级冲突的包

3. 使用虚拟环境隔离

## 依赖列表

lm-evaluation-harness 的主要依赖包括：

- accelerate>=0.26.0
- evaluate>=0.4.0
- datasets>=2.16.0
- jsonlines
- numexpr
- peft>=0.2.0
- pybind11>=2.6.2
- pytablewriter
- rouge-score>=0.0.4
- sacrebleu>=1.5.0
- scikit-learn>=0.24.1
- sqlitedict
- torch>=1.8
- tqdm-multiprocess
- transformers>=4.1
- zstandard
- dill
- word2number
- more_itertools

完整依赖列表请查看 `lm-evaluation-harness/pyproject.toml`。

## 使用虚拟环境（推荐）

为了避免依赖冲突，建议在容器中使用虚拟环境：

```bash
# 在容器中
python -m venv /path/to/venv
source /path/to/venv/bin/activate

# 安装
cd /path/to/lm-evaluation-harness
pip install -e .
```

## 与 Megatron 集成

安装完成后，可以使用之前创建的适配器：

```python
from tasks.lm_eval_adapter import use_megatron_with_lm_eval_harness

# 现在可以正常使用了
results = use_megatron_with_lm_eval_harness(...)
```

## 脚本选项

安装脚本支持以下选项：

- `--prepare-deps`: 在宿主机上准备依赖包（需要网络）
- `--install`: 在容器中安装
- `--help`: 显示帮助信息

环境变量：
- `EDITABLE_INSTALL`: 是否使用 editable 安装（默认: 1）

## 示例：完整安装流程

```bash
# === 在宿主机上 ===

# 1. 准备依赖包（可选，如果需要完全离线）
cd /ssd/wenyandongLab/shikexuan
bash Megatron-LM/tasks/install_lm_eval_singularity.sh --prepare-deps

# === 在容器中 ===

# 2. 安装
singularity exec your_container.sif \
    bash /ssd/wenyandongLab/shikexuan/Megatron-LM/tasks/install_lm_eval_singularity.sh --install

# 3. 验证
singularity exec your_container.sif \
    python -c "import lm_eval; print('Success!')"
```

## 注意事项

1. **路径问题**: 确保脚本中的路径在容器内可访问
2. **网络访问**: 如果容器可以访问网络，直接安装会更简单
3. **Python 版本**: 确保 Python >= 3.9
4. **权限**: 如果遇到权限问题，使用 `--user` 或虚拟环境
5. **依赖冲突**: 如果与现有包冲突，考虑使用虚拟环境

## 故障排除

如果遇到问题，可以：

1. 检查脚本输出中的错误信息
2. 手动运行安装命令查看详细错误
3. 检查 Python 和 pip 版本
4. 查看 `lm-evaluation-harness/` 目录是否完整
5. 检查容器内的文件系统权限

## 参考

- [lm-evaluation-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [Singularity 文档](https://docs.sylabs.io/)
- [pip 离线安装文档](https://pip.pypa.io/en/stable/user_guide/#installing-from-local-packages)
