#!/usr/bin/env bash
# 使用本地 DeepMath-103K parquet 预处理为 verl 训练用 parquet（调用 deepmath_dataset.py）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

LOCAL_DATASET_PATH="/home/lihanxuan/verl_pack/dataset/deepmath/data"
LOCAL_SAVE_DIR="/home/lihanxuan/verl_pack/dataset/processed_verl/deepmath"

mkdir -p "${LOCAL_SAVE_DIR}"

exec python3 "${SCRIPT_DIR}/deepmath_dataset.py" \
  --local_dataset_path "${LOCAL_DATASET_PATH}" \
  --local_save_dir "${LOCAL_SAVE_DIR}"
