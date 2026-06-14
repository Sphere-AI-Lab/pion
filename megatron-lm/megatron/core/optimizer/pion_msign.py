# pyright: reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false
# pyright: reportAttributeAccessIssue=false, reportIncompatibleMethodOverride=false
# pyright: reportOperatorIssue=false
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pion with alternating unilateral orthogonal updates via truncated matrix exponential.

Per step for each matrix shard:
  1. EMA on raw gradient G.
  2. Bilateral projection:
     G_in = skew(W^T G_ema),  G_out = skew(G_ema W^T).
  3. msign on the active side via polar orthogonal factor (Newton–Schulz).
  4. Build orthogonal factor via truncated matrix exponential.
  5. Alternating unilateral update (truncated exp integrated on W):
     - odd steps:  W <- W + delta_in
     - even steps: W <- W + delta_out
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torch.optim import Optimizer

from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank
from megatron.core import parallel_state

from . import _get_param_groups, get_megatron_optimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey

try:
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz
    from emerging_optimizers.utils import fp32_matmul_precision

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False

    def newton_schulz(*args, **kwargs):
        raise ImportError("emerging_optimizers is required for pion_msign.")

    def fp32_matmul_precision(*args, **kwargs):
        raise ImportError("emerging_optimizers is required for pion_msign.")

logger = logging.getLogger(__name__)


def _matrix_param_groups(
    model_chunks: List[MegatronModule],
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
    matrix_params: List[torch.nn.Parameter],
) -> List[Dict[str, Any]]:
    """Mirror pion.py's matrix-only grouping without mutating requires_grad."""
    matrix_param_ids = {id(p) for p in matrix_params}
    groups: List[Dict[str, Any]] = []
    for group in _get_param_groups(model_chunks, config, config_overrides):
        params = [p for p in group["params"] if id(p) in matrix_param_ids]
        if params:
            new_group = dict(group)
            new_group["params"] = params
            groups.append(new_group)
    return groups


def _matrix_exp_truncated_integrated(
    A: torch.Tensor,
    p_data: torch.Tensor,
    side: str,
    group: Dict[str, Any],
    state: Dict[str, Any],
) -> torch.Tensor:
    """Truncated matrix exponential integrated on W (no alpha scaling)."""
    if side == "in":
        powers = p_data @ A
    else:
        powers = A @ p_data

    degree = group.get("degree", 2)
    out = powers.clone()
    for i in range(2, degree + 1):
        inv_i = 1.0 / i
        if side == "in":
            powers = (powers @ A).mul_(inv_i)
        else:
            powers = (A @ powers).mul_(inv_i)
        out.add_(powers)
    return out


def _msign_bilateral(
    G: torch.Tensor,
    num_ns_steps: int,
    coefficient_type: str = "quintic",
    fp32_matmul_prec: str = "medium",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton–Schulz msign; bf16 internal iter when fp32_matmul_prec is medium (same as Muon)."""
    with fp32_matmul_precision(cast(Any, fp32_matmul_prec)):
        return newton_schulz(
            G,
            steps=num_ns_steps,
            coefficient_type=cast(Any, coefficient_type),
            eps=eps,
        )


def _ortho_exp_alt_update(
    w: torch.Tensor,
    g_ema: torch.Tensor,
    group: Dict[str, Any],
    state: Dict[str, Any],
) -> torch.Tensor:
    """Alternating unilateral update with truncated matrix exponential."""
    w32 = w.float()
    g32 = g_ema.float()

    num_ns_steps = group.get("num_ns_steps", 5)
    coefficient_type = group.get("coefficient_type", "quintic")
    fp32_matmul_prec = group.get("fp32_matmul_prec", "medium")
    lr = group.get("lr", group.get("max_lr", 1e-4))
    lam = group.get("pion_msign_lambda", 1.0)
    eta = float(lr * lam)

    if "step" not in state:
        state["step"] = 0
    state["step"] += 1

    if state["step"] % 2 == 1:
        grad_in = w32.t() @ g32
        skew_in = 0.5 * (grad_in - grad_in.t())
        u_in = _msign_bilateral(
            skew_in, num_ns_steps, coefficient_type, fp32_matmul_prec=fp32_matmul_prec
        )
        delta = _matrix_exp_truncated_integrated(-eta * u_in, w32, "in", group, state)
        return w32 + delta
    else:
        grad_out = g32 @ w32.t()
        skew_out = 0.5 * (grad_out - grad_out.t())
        u_out = _msign_bilateral(
            skew_out, num_ns_steps, coefficient_type, fp32_matmul_prec=fp32_matmul_prec
        )
        delta = _matrix_exp_truncated_integrated(-eta * u_out, w32, "out", group, state)
        return w32 + delta


class PionOrthoExpOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
        pion_msign_lambda: float = 1.0,
        num_ns_steps: int = 5,
        coefficient_type: str = "quintic",
        split_qkv: bool = True,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[Tuple[int, int, int]] = None,
        split_fc1_up_gate: bool = True,
        is_fc1_up_gate_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        split_qkv_per_head: bool = True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            pion_msign_lambda=pion_msign_lambda,
            num_ns_steps=num_ns_steps,
            coefficient_type=coefficient_type,
        )
        super().__init__(params, defaults)
        self.split_qkv = split_qkv and (qkv_split_shapes is not None) and (is_qkv_fn is not None)
        self.is_qkv_fn = is_qkv_fn if is_qkv_fn is not None else (lambda p: False)
        self.qkv_split_shapes = tuple(qkv_split_shapes) if qkv_split_shapes else (0, 0, 0)
        self.split_fc1_up_gate = split_fc1_up_gate and (is_fc1_up_gate_fn is not None)
        self.is_fc1_up_gate_fn = is_fc1_up_gate_fn if is_fc1_up_gate_fn is not None else (lambda p: False)
        self.split_qkv_per_head = split_qkv_per_head

    def _ema_grad(self, grad: torch.Tensor, state: Dict[str, Any], p: torch.Tensor, beta1: float) -> torch.Tensor:
        if "exp_avg_g" not in state:
            state["exp_avg_g"] = torch.zeros_like(grad, dtype=torch.float32, device=p.device)
        state["exp_avg_g"].mul_(beta1).add_(grad.float(), alpha=1.0 - beta1)
        return state["exp_avg_g"]

    def _pion_update_for_matrix(self, p: torch.Tensor, grad: torch.Tensor, group: Dict[str, Any], state: Dict[str, Any]) -> None:
        beta1 = group.get("pion_beta1")
        if beta1 is None:
            betas = group.get("betas", self.defaults["betas"])
            beta1 = betas[0] if isinstance(betas, (tuple, list)) and len(betas) >= 1 else betas
        beta1 = float(beta1)

        p_data = p.data.float() if p.dtype != torch.float32 else p.data
        grad_f = grad.float() if grad.dtype != torch.float32 else grad
        out_dim, in_dim = p_data.shape

        is_qkv = self.split_qkv and self.is_qkv_fn(p)
        if is_qkv:
            total = sum(self.qkv_split_shapes)
            num_query_groups = out_dim // total
            q_per_group, k_per_group, v_per_group = self.qkv_split_shapes
            log_single_rank(logger, logging.DEBUG, f"pion_ortho_exp qkv shape {p_data.shape}, split {self.qkv_split_shapes}")
            if self.split_qkv_per_head:
                num_heads_per_group = q_per_group // k_per_group
                head_dim = k_per_group
                view = p_data.view(num_query_groups, total, in_dim)
                g_view = grad_f.view(num_query_groups, total, in_dim)
                w_q_heads: List[List[torch.Tensor]] = []
                for g in range(num_query_groups):
                    w_q_g = view[g, :q_per_group, :]
                    w_q_heads.append([w_q_g[h * head_dim : (h + 1) * head_dim, :].clone() for h in range(num_heads_per_group)])
                w_k_list = [view[g, q_per_group:q_per_group + k_per_group, :].clone() for g in range(num_query_groups)]
                w_v_list = [view[g, q_per_group + k_per_group:total, :].clone() for g in range(num_query_groups)]
                for g in range(num_query_groups):
                    for h in range(num_heads_per_group):
                        idx = g * num_heads_per_group + h
                        q_state_key = f"q_{idx}"
                        if q_state_key not in state:
                            state[q_state_key] = {}
                        g_slice = g_view[g, h * head_dim : (h + 1) * head_dim, :]
                        g_ema_slice = self._ema_grad(g_slice, state[q_state_key], p, beta1)
                        w_q_heads[g][h] = _ortho_exp_alt_update(w_q_heads[g][h], g_ema_slice, group, state[q_state_key])
                    k_state_key = f"k_{g}"
                    v_state_key = f"v_{g}"
                    if k_state_key not in state:
                        state[k_state_key] = {}
                    if v_state_key not in state:
                        state[v_state_key] = {}
                    g_k = g_view[g, q_per_group:q_per_group + k_per_group, :]
                    g_v = g_view[g, q_per_group + k_per_group:total, :]
                    g_ema_k = self._ema_grad(g_k, state[k_state_key], p, beta1)
                    g_ema_v = self._ema_grad(g_v, state[v_state_key], p, beta1)
                    w_k_list[g] = _ortho_exp_alt_update(w_k_list[g], g_ema_k, group, state[k_state_key])
                    w_v_list[g] = _ortho_exp_alt_update(w_v_list[g], g_ema_v, group, state[v_state_key])
                new_p = torch.cat([torch.cat([torch.cat(w_q_heads[g], dim=0), w_k_list[g], w_v_list[g]], dim=0) for g in range(num_query_groups)], dim=0)
            else:
                view = p_data.view(num_query_groups, total, in_dim)
                g_view = grad_f.view(num_query_groups, total, in_dim)
                q_blocks = [view[g, :q_per_group, :].clone() for g in range(num_query_groups)]
                k_blocks = [
                    view[g, q_per_group : q_per_group + k_per_group, :].clone()
                    for g in range(num_query_groups)
                ]
                v_blocks = [
                    view[g, q_per_group + k_per_group : total, :].clone()
                    for g in range(num_query_groups)
                ]
                q_grad_blocks = [g_view[g, :q_per_group, :] for g in range(num_query_groups)]
                k_grad_blocks = [
                    g_view[g, q_per_group : q_per_group + k_per_group, :]
                    for g in range(num_query_groups)
                ]
                v_grad_blocks = [
                    g_view[g, q_per_group + k_per_group : total, :]
                    for g in range(num_query_groups)
                ]

                q_all = torch.cat(q_blocks, dim=0)
                k_all = torch.cat(k_blocks, dim=0)
                v_all = torch.cat(v_blocks, dim=0)
                q_grad_all = torch.cat(q_grad_blocks, dim=0)
                k_grad_all = torch.cat(k_grad_blocks, dim=0)
                v_grad_all = torch.cat(v_grad_blocks, dim=0)

                for state_key in ("q", "k", "v"):
                    if state_key not in state:
                        state[state_key] = {}
                q_all = _ortho_exp_alt_update(
                    q_all,
                    self._ema_grad(q_grad_all, state["q"], p, beta1),
                    group,
                    state["q"],
                )
                k_all = _ortho_exp_alt_update(
                    k_all,
                    self._ema_grad(k_grad_all, state["k"], p, beta1),
                    group,
                    state["k"],
                )
                v_all = _ortho_exp_alt_update(
                    v_all,
                    self._ema_grad(v_grad_all, state["v"], p, beta1),
                    group,
                    state["v"],
                )

                q_chunks = list(q_all.split(q_per_group, dim=0))
                k_chunks = list(k_all.split(k_per_group, dim=0))
                v_chunks = list(v_all.split(v_per_group, dim=0))
                new_p = torch.cat(
                    [
                        torch.cat([q_chunks[g], k_chunks[g], v_chunks[g]], dim=0)
                        for g in range(num_query_groups)
                    ],
                    dim=0,
                )
        elif self.split_fc1_up_gate and self.is_fc1_up_gate_fn(p):
            half = out_dim // 2
            w_up = p_data[:half].clone()
            w_gate = p_data[half:].clone()
            g_up = grad_f[:half]
            g_gate = grad_f[half:]
            up_state_key = "fc1_up"
            gate_state_key = "fc1_gate"
            if up_state_key not in state:
                state[up_state_key] = {}
            if gate_state_key not in state:
                state[gate_state_key] = {}
            g_ema_up = self._ema_grad(g_up, state[up_state_key], p, beta1)
            g_ema_gate = self._ema_grad(g_gate, state[gate_state_key], p, beta1)
            w_up = _ortho_exp_alt_update(w_up, g_ema_up, group, state[up_state_key])
            w_gate = _ortho_exp_alt_update(w_gate, g_ema_gate, group, state[gate_state_key])
            new_p = torch.cat([w_up, w_gate], dim=0)
        else:
            g_ema = self._ema_grad(grad_f, state, p, beta1)
            new_p = _ortho_exp_alt_update(p_data, g_ema, group, state)

        if new_p.dtype != p.data.dtype:
            new_p = new_p.to(p.data.dtype)
        p.data.copy_(new_p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._pion_update_for_matrix(p, p.grad.data, group, self.state[p])
        return loss


def get_megatron_pion_ortho_exp_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    config.optimizer = "adam"

    assert HAVE_EMERGING_OPTIMIZERS, "Emerging Optimizers is not installed."

    if config.use_distributed_optimizer:
        raise Exception("Pion-ortho-exp with distributed optimizer is not supported.")

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f"Setting up Pion-ortho-exp optimizer with config {config}")

    matrix_params: List[torch.nn.Parameter] = []
    qkv_split_shapes: Optional[Tuple[int, int, int]] = None
    split_fc1_up_gate = False

    for model_chunk in model_chunks:
        num_attention_heads = getattr(model_chunk.config, "num_attention_heads", None)
        num_query_groups = getattr(model_chunk.config, "num_query_groups", None)
        kv_channels = getattr(model_chunk.config, "kv_channels", None)
        if num_attention_heads is not None and num_query_groups is not None and kv_channels is not None:
            qkv_split_shapes = (
                num_attention_heads // num_query_groups * kv_channels,
                kv_channels,
                kv_channels,
            )
        gated_linear_unit = getattr(model_chunk.config, "gated_linear_unit", False)
        split_fc1_up_gate = gated_linear_unit and getattr(config, "pion_split_gate", True)
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 2 and not getattr(param, "is_embedding_or_output_parameter", False):
                setattr(param, "_pion_param_name", name)
                if "linear_qkv.weight" in name:
                    param.is_qkv = True
                if "linear_fc1.weight" in name and split_fc1_up_gate:
                    param.is_fc1_up_gate = True
                matrix_params.append(param)

    matrix_param_groups = _matrix_param_groups(model_chunks, config, config_overrides, matrix_params)
    adam_lr = float(config.lr if config.lr is not None else 1e-4)
    adam_min_lr = float(config.min_lr if config.min_lr is not None else 0.0)
    pion_lr = getattr(config, "pion_lr", None)
    pion_min_lr = getattr(config, "pion_min_lr", None)
    matrix_lr = float(pion_lr if pion_lr is not None else adam_lr)
    matrix_min_lr = float(pion_min_lr if pion_min_lr is not None else adam_min_lr)
    if not matrix_param_groups:
        matrix_param_groups = [{
            "params": matrix_params,
            "max_lr": matrix_lr,
            "min_lr": matrix_min_lr,
            "wd_mult": 1.0,
            "lr_mult": 1.0,
            "is_expert_parallel": False,
            "default_config": True,
        }]

    degree = getattr(config, "pion_degree", 2)
    pion_msign_lambda = getattr(config, "pion_msign_lambda", 1.0)
    num_ns_steps = getattr(config, "muon_num_ns_steps", 5)
    coefficient_type = getattr(config, "muon_coefficient_type", "quintic")
    fp32_matmul_prec = getattr(config, "muon_fp32_matmul_prec", "medium")
    pion_beta1_cfg = getattr(config, "pion_beta1", config.adam_beta1) # pion_beta1字段不存在时，退回到config.adam_beta1 #所以没有问题
    matrix_beta1 = pion_beta1_cfg if pion_beta1_cfg and pion_beta1_cfg > 0 else config.adam_beta1
    for g in matrix_param_groups:
        g["degree"] = degree
        g["pion_msign_lambda"] = pion_msign_lambda
        g["num_ns_steps"] = num_ns_steps
        g["coefficient_type"] = coefficient_type
        g["fp32_matmul_prec"] = fp32_matmul_prec
        g["pion_beta1"] = matrix_beta1
        g["max_lr"] = matrix_lr
        g["lr"] = matrix_lr
        g["min_lr"] = matrix_min_lr

    log_single_rank(
        logger,
        logging.INFO,
        f"Pion-ortho-exp matrix EMA beta1={matrix_beta1} "
        f"(pion_beta1={pion_beta1_cfg}, adam_beta1={config.adam_beta1}), "
        f"msign fp32_matmul_prec={fp32_matmul_prec}",
    )

    pion_optimizer = PionOrthoExpOptimizer(
        matrix_param_groups,
        lr=matrix_lr,
        betas=(matrix_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
        pion_msign_lambda=pion_msign_lambda,
        num_ns_steps=num_ns_steps,
        coefficient_type=coefficient_type,
        split_qkv=getattr(config, "pion_split_qkv", True),
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=qkv_split_shapes,
        split_fc1_up_gate=split_fc1_up_gate,
        is_fc1_up_gate_fn=lambda p: getattr(p, "is_fc1_up_gate", False),
        split_qkv_per_head=getattr(config, "pion_split_qkv_per_head", True),
    )

    def pion_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                if len(opt.state[p]) == 0:
                    opt.state[p]["step"] = 0
                    opt.state[p]["exp_avg_g"] = torch.zeros(p.shape, device=p.device, dtype=torch.float32)

    if config.fp16:
        raise Exception("Pion-ortho-exp with fp16 is not supported; use bf16.")
    if config.bf16:
        optimizer = Float16OptimizerWithFloat16Params(
            pion_optimizer,
            config,
            None,
            pion_init_state_fn,  # pyright: ignore[reportArgumentType]
        )
    else:
        optimizer = FP32Optimizer(pion_optimizer, config, pion_init_state_fn)

    optimizers: List[MegatronOptimizer] = [optimizer]

    for p in matrix_params:
        p.requires_grad = False

    chained_adam = cast(
        ChainedOptimizer,
        get_megatron_optimizer(
            config,
            model_chunks,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        ),
    ) 
    for p in matrix_params:
        p.requires_grad = True

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                if len(opt.state[p]) == 0:
                    opt.state[p]["exp_avg"] = torch.zeros_like(p.data)
                    opt.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)

    optimizers += chained_adam.chained_optimizers
    setattr(optimizer, "grad_stats_parallel_group", parallel_state.get_model_parallel_group())
    tp_group = pg_collection.tp if (pg_collection and hasattr(pg_collection, "tp")) else parallel_state.get_tensor_model_parallel_group()
    setattr(optimizer, "tp_group", tp_group)
    return ChainedOptimizer(optimizers)


__all__ = ["PionOrthoExpOptimizer", "get_megatron_pion_ortho_exp_optimizer"]
