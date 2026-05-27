ROOT_DIR="/lustre/fast/fast/wliu/hli"
SINGULARITY_BIN="/usr/bin/singularity"
IMAGE_PATH="/lustre/fast/fast/wliu/hli/verl_vllm018.dev1.sif"

# 设置自定义 Python 包路径（持久化安装的 transformers）
# export SINGULARITYENV_PYTHONPATH="/workspace/pion_usage/python_packages:\$PYTHONPATH"

# 设置 CUDA 库路径
# export SINGULARITYENV_LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:\$LDx_LIBRARY_PATH"

$SINGULARITY_BIN shell --nv --writable-tmpfs \
  --bind ${ROOT_DIR}:/workspace \
  $IMAGE_PATH

 