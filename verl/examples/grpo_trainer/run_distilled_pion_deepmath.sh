#!/usr/bin/env bash

set -x

deepmath_train_path=/path/to/your/dataset/train.parquet
deepmath_test_path=/path/to/your/dataset/test.parquet
# train_files="['$deepmath_train_path']"
# test_files="['$deepmath_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$deepmath_train_path" \
    data.val_files="$deepmath_test_path" \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.train_max_samples=50000 \
    data.seed=42 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/path/to/your/model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.optimizer=PionOptimizer \
    actor_rollout_ref.actor.optim.optimizer_impl=verl.custom_optimizer.pion \
    +actor_rollout_ref.actor.optim.override_optimizer_config.betas=[0.9,0.999] \
    +actor_rollout_ref.actor.optim.override_optimizer_config.eps=1e-8 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.degree=2 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.head_dim=128 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=12 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='distilled_deepmath_grpo' \
    trainer.experiment_name='distilled_deepmath_grpo_pion' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@
