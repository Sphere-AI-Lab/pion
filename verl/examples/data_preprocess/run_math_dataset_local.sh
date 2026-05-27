#!/usr/bin/env bash
# 使用本地 MATH 数据预处理为 verl 训练用 parquet（调用 math_dataset.py）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

LOCAL_DATASET_PATH="/home/lihanxuan/verl_pack/dataset/math/data"
LOCAL_SAVE_DIR="/home/lihanxuan/verl_pack/dataset/processed"

mkdir -p "${LOCAL_SAVE_DIR}"

exec python3 "${SCRIPT_DIR}/math_dataset.py" \
  --local_dataset_path "${LOCAL_DATASET_PATH}" \
  --local_save_dir "${LOCAL_SAVE_DIR}"
