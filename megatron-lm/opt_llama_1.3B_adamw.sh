#!/bin/bash
set -o pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_SILENT=true               
export TORCH_CPP_LOG_LEVEL=ERROR       
export NCCL_ML_DISABLE=1
export WANDB_MODE=offline

PORT=$((29500 + $$ % 100))


mkdir -p /usr/local/cuda/compat/lib 2>/dev/null || true
ln -sf /.singularity.d/libs/libcuda.so.1 /usr/local/cuda/compat/lib/libcuda.so.1 2>/dev/null || true
ln -sf /.singularity.d/libs/libcuda.so /usr/local/cuda/compat/lib/libcuda.so 2>/dev/null || true
export LD_LIBRARY_PATH=/workspace/cuda_fix/compat/lib:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=true

# Total tokens, global batch size, and training iterations
# Strategy for parallelization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 1: 1.2; 2: 2.4; 4: 4.8; 8: 9.6;
TOKEN=54
# bash arithmetic only supports integers.
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

LR=5e-4
MIN_LR=5e-6

# Pretrain Script
PRETRAIN_SCRIPT="pretrain_gpt.py"
# Experiment Name and Saving Path
JOB_NAME=llama_adamw_${TOKEN}B_1.3B_lr_${LR}_min_${MIN_LR}
REPO_PATH="/workspace/results2/${JOB_NAME}"
TENSORBOARD_PATH="${REPO_PATH}/tensorboard/${JOB_NAME}"
CHECKPOINT_PATH="${REPO_PATH}/checkpoints/${JOB_NAME}"
WANDB_PATH="${REPO_PATH}/wandb/${JOB_NAME}"


TRAINING_ARGS=(
    --pure-bf16-optimizer
    --lr ${LR}
    --min-lr ${MIN_LR}
    --lr-warmup-iters 500
    --lr-decay-style cosine
    --lr-decay-iters $TRAIN_ITER
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --weight-decay 0.1
    --clip-grad 1.0
    --no-gradient-accumulation-fusion
)

LOG_DIR="${REPO_PATH}/logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"
mkdir -p $TENSORBOARD_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $WANDB_PATH

TRAIN_DATA_NAME="c4-megatron/train"
TRAIN_BASE_PATH="${TRAIN_BASE_PATH:-/workspace/${TRAIN_DATA_NAME}}"

VALID_DATA_NAME="c4-megatron/val" 
VALID_BASE_PATH="${VALID_BASE_PATH:-/workspace/${VALID_DATA_NAME}}"

DATA_PATH=""
while IFS= read -r file; do
    common_prefix=${file%".bin"}
    DATA_PATH+="1 ${common_prefix} "
done < <(find "$TRAIN_BASE_PATH" -type f -path "**.bin" 2>/dev/null)

VALID_DATA_PATH=""
while IFS= read -r file; do
    common_prefix=${file%".bin"}
    VALID_DATA_PATH+="${common_prefix} "
done < <(find "$VALID_BASE_PATH" -type f -path "**.bin" 2>/dev/null)

# The path to cache the data
DATA_PATH_CACHE="/workspace/${TRAIN_DATA_NAME}_cache"

DATA_ARGS=(
    --tokenizer-model /ssd/wenyandongLab/shikexuan/t5-base
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-hf-use-fast
    --seq-length 256
    --train-data-path $DATA_PATH
    --valid-data-path $VALID_DATA_PATH
    --full-validation
    --data-cache-path ${DATA_PATH_CACHE}
    --train-iters $TRAIN_ITER 
    --num-dataset-builder-threads 128
    --num-workers 32
    --no-mmap-bin-files
    --distributed-timeout-minutes 60
    --eval-interval 10000
)


MODEL_ARGS=(
    --use-same-init-for-output-layers
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 5460
    --num-attention-heads 32
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
CKPT_ARGS=(
    --load ${CHECKPOINT_PATH}
    --ckpt-format "torch_dist"
    --save-interval 10000
    --save $CHECKPOINT_PATH
    --save-initial-checkpoint
)

LOGGER_ARGS=(
    --log-params-norm
    --log-throughput
    --log-interval 50
    --log-params-norm
    --log-num-zeros-in-grad
    # --log-max-attention-logit 1,12,24
    # --log-linear-layer-norms 1,12,24
    # --log-max-attention-prob 1,12,24
    # --log-large-attention-logit-ratio
    # --large-attention-logit-threshold 100
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

    if [[ ${run_status} -eq 135 ]] || tail -n 500 "$LOG_FILE" | grep -Eqi "Bus error|SIGBUS|Signal 7|exitcode[[:space:]]*:[[:space:]]*-7"; then
        echo "[$(date '+%F %T')] detected SIGBUS/Bus error, sleep ${RESTART_SLEEP_SECONDS}s then restart." | tee -a "$LOG_FILE"
        sleep "${RESTART_SLEEP_SECONDS}"
    else
        echo "[$(date '+%F %T')] non-Bus-error failure (exit=${run_status}), stop auto-restart." | tee -a "$LOG_FILE"
        exit ${run_status}
    fi
done