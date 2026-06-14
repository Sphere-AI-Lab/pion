#!/bin/bash
set -o pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1 # 这个是命令是强制通信的
# --- 屏蔽干扰日志 ---
export WANDB_SILENT=true               # WandB 静默模式
export TORCH_CPP_LOG_LEVEL=ERROR       # 抑制 C++ 端的分布式警告
export NCCL_ML_DISABLE=1               # 减少某些环境下的 NCCL 冗余告警
export NCCL_NVLS_ENABLE=1
export WANDB_MODE=offline

PORT=$((29500 + $$ % 100))


# 确保 libcuda.so 符号链接存在
mkdir -p /usr/local/cuda/compat/lib 2>/dev/null || true
ln -sf /.singularity.d/libs/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so.1 2>/dev/null || true
ln -sf /.singularity.d/libs/libcuda.so /usr/local/cuda/compat/lib/libcuda.so 2>/dev/null || true
export LD_LIBRARY_PATH=/workspace/cuda_fix/compat/lib:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=true

# Total tokens, global batch size, and training iterations
# Strategy for parallelization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 1: 1.2; 2: 2.4; 4: 4.8; 8: 9.6;
TOKEN=9.6
# bash arithmetic only supports integer
# Convert TOKEN like 1.2 -> 12 (x10), then scale back.
if [[ "$TOKEN" == *.* ]]; then
    TOKEN_X10=${TOKEN/./}
    TOTAL_TOKENS=$((TOKEN_X10 * 10**8))
else
    TOTAL_TOKENS=$((TOKEN * 10**9))
fi
GLOBAL_BATCH=512
TRAIN_ITER=$((TOTAL_TOKENS / GLOBAL_BATCH / 256))

NNODES=1
NUM_GPUS=8
ACCUMULATION_STEPS=1
MICRO_BATCH_SIZE=$((GLOBAL_BATCH / NUM_GPUS / ACCUMULATION_STEPS))
WORLD_SIZE=$((NUM_GPUS * $NNODES))


TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}

PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH}
)

DISTRIBUTED_ARGS=(
    --nproc_per_node $NUM_GPUS
    --nnodes $NNODES
)       
LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-1e-5}
PION_LR=${PION_LR:-1e-2}
PION_MIN_LR=${PION_MIN_LR:-1e-4}
PION_MOMENTUM=${PION_MOMENTUM:-transported_ambient_ambient}
# Pretrain Script
PRETRAIN_SCRIPT="pretrain_gpt.py"
# Experiment Name and Saving Path
JOB_NAME=llama_60M_pion_msign_lr_${PION_LR}_min_${PION_MIN_LR}
REPO_PATH="/data/people/kshi/results/${JOB_NAME}"
TENSORBOARD_PATH="${REPO_PATH}/tensorboard/${JOB_NAME}"
CHECKPOINT_PATH="${REPO_PATH}/checkpoints/${JOB_NAME}"
WANDB_PATH="${REPO_PATH}/wandb/${JOB_NAME}"


TRAINING_ARGS=(
    # --pure-bf16-optimizer
    --use-same-init-for-output-layers
    --lr ${LR}
    --min-lr ${MIN_LR}
    --pion-lr ${PION_LR}
    --pion-min-lr ${PION_MIN_LR}
    --lr-warmup-iters 0
    --lr-decay-style cosine
    --lr-decay-iters $TRAIN_ITER
    --adam-beta1 0.9
    --adam-beta2 0.95
    --pion-beta1 0.95
    --adam-eps 1e-8
    --optimizer pion_msign
    --pion-msign-lambda 1.0
    --num-ns-steps 5
    --coefficient-type quintic
    --weight-decay 0.1
    --clip-grad 1.0
    --no-gradient-accumulation-fusion
    # --pion-no-split-qkv
    # --pion-no-split-qkv-per-head # Cancel this flag to use split qkv per head
    # --pion-no-split-gate
)

LOG_DIR="${REPO_PATH}/logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"
mkdir -p $TENSORBOARD_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $WANDB_PATH

TRAIN_DATA_NAME="c4-megatron/train"
TRAIN_BASE_PATH="${TRAIN_BASE_PATH:-/data/shared/pion_usage/${TRAIN_DATA_NAME}}"

# 验证数据路径
VALID_DATA_NAME="c4-megatron/val" 
VALID_BASE_PATH="${VALID_BASE_PATH:-/data/shared/pion_usage/${VALID_DATA_NAME}}"

# 构建训练数据路径
DATA_PATH=""
while IFS= read -r file; do
    common_prefix=${file%".bin"}
    DATA_PATH+="1 ${common_prefix} "  # 训练集可以带权重
done < <(find "$TRAIN_BASE_PATH" -type f -path "**.bin" 2>/dev/null)

# 构建验证数据路径
VALID_DATA_PATH=""
while IFS= read -r file; do
    common_prefix=${file%".bin"}
    VALID_DATA_PATH+="${common_prefix} "  # 验证集不带权重（用于 full_validation）
done < <(find "$VALID_BASE_PATH" -type f -path "**.bin" 2>/dev/null)

# The path to cache the data
DATA_PATH_CACHE="/data/shared/pion_usage/${TRAIN_DATA_NAME}_cache"

# 提供valida-data-path和full-validation，在训练结束时，就会自动验证。
DATA_ARGS=(
    --tokenizer-model /data/shared/pion_usage/tokenizer_t5
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-hf-use-fast
    --seq-length 256
    --train-data-path $DATA_PATH
    --valid-data-path $VALID_DATA_PATH
    --full-validation
    --data-cache-path ${DATA_PATH_CACHE}
    --train-iters $TRAIN_ITER 
    --num-dataset-builder-threads 8
    --num-workers 4
    # --no-mmap-bin-files
    --distributed-timeout-minutes 240
    --eval-interval 10000
)

# 这边的seq-length,在模型中是必须要要设置的。ball那个工作的4096也是在此处设置的。那么Megatron-LM中，这边在数据集分割的时候
# 应该默认就是用MODEL_ARGS中的seq-length来分割的。 这边的Seq-length实际上就是data的seq-length.
# 如果使用layernorm，我们去掉no-persist-layer-norm可以提升模型的运行速度。
# 但是rms norm，并没有关系

MODEL_ARGS=(
    --use-same-init-for-output-layers
    --num-layers 8
    --hidden-size 512
    --ffn-hidden-size 1376
    --num-attention-heads 8
    --norm-epsilon 1e-6
    --kv-channels 64
    --max-position-embeddings 1024
    --attention-dropout 0
    --hidden-dropout 0
    --bf16
    --use-rotary-position-embeddings
    --rotary-base 10000
    --swiglu
    --untie-embeddings-and-output-weights
    --normalization RMSNorm
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --transformer-impl transformer_engine
    --attention-backend fused
    --init-method-std 0.02
    --no-persist-layer-norm
    --use-cpu-initialization
)
# 训练结束后，会保存最后一次迭代得到的checkpoint。
CKPT_ARGS=(
    --load ${CHECKPOINT_PATH}
    --ckpt-format "torch"
    --save-interval 10000
    --save $CHECKPOINT_PATH
    --save-initial-checkpoint
)

LOGGER_ARGS=(
    --log-params-norm
    --log-throughput
    --log-interval 100
    --log-params-norm
    --log-num-zeros-in-grad
    --log-validation-ppl-to-tensorboard
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --tensorboard-dir ${TENSORBOARD_PATH}
)

WANDB_ARGS=(
    --wandb-project test
    --wandb-exp-name $JOB_NAME
    --wandb-save-dir ${WANDB_PATH}
)

# Auto-restart settings for transient crash (e.g. Bus error).
MAX_RESTARTS=${MAX_RESTARTS:-20}
RESTART_SLEEP_SECONDS=${RESTART_SLEEP_SECONDS:-30}
attempt=0

while true; do
    attempt=$((attempt + 1))
    echo "[$(date '+%F %T')] launch attempt ${attempt}/${MAX_RESTARTS}" | tee -a "$LOG_FILE"

    {
        PYTHONWARNINGS=ignore torchrun --master_port $PORT \
            ${DISTRIBUTED_ARGS[@]} \
            $PRETRAIN_SCRIPT \
            ${DATA_ARGS[@]} \
            ${MODEL_ARGS[@]} \
            ${TRAINING_ARGS[@]} \
            ${PARALLEL_ARGS[@]} \
            ${CKPT_ARGS[@]} \
            ${LOGGER_ARGS[@]} \
            ${WANDB_ARGS[@]}
    } 2>&1 | grep --line-buffered -v -E "(Warning|DeprecationWarning|UserWarning|FutureWarning|WARNING|Deprecated)" | tee -a "$LOG_FILE"

    run_status=${PIPESTATUS[0]}
    if [[ ${run_status} -eq 0 ]]; then
        echo "[$(date '+%F %T')] training finished successfully." | tee -a "$LOG_FILE"
        break
    fi

    if [[ ${attempt} -ge ${MAX_RESTARTS} ]]; then
        echo "[$(date '+%F %T')] reached MAX_RESTARTS=${MAX_RESTARTS}, exit code=${run_status}." | tee -a "$LOG_FILE"
        exit ${run_status}
    fi

    if tail -n 300 "$LOG_FILE" | grep -qi "Fatal Python error: Bus error\|Bus error"; then
        echo "[$(date '+%F %T')] detected Bus error, sleep ${RESTART_SLEEP_SECONDS}s then restart." | tee -a "$LOG_FILE"
        sleep "${RESTART_SLEEP_SECONDS}"
    else
        echo "[$(date '+%F %T')] non-Bus-error failure (exit=${run_status}), stop auto-restart." | tee -a "$LOG_FILE"
        exit ${run_status}
    fi
done