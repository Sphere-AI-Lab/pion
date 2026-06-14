# pyright: reportMissingImports=false
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unified Megatron Pion optimizer for open source.

Pion is used for matrix parameters while AdamW is kept for vector, embedding,
and output parameters.  Matrix updates are expressed as two Lie generators:

    W <- W exp(A_in) + exp(A_out) W - W

where both exponentials are approximated by a truncated Taylor expansion.
The momentum policy is selected per param group through Pion hyperparameters.
"""

import csv
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torch.optim import Optimizer

from megatron.core import parallel_state
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey


logger = logging.getLogger(__name__)


def _matrix_exp_truncated(
    A: torch.Tensor,
    degree: int,
    left_mult: Optional[torch.Tensor] = None,
    right_mult: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the non-identity part of exp(A), optionally multiplied by W."""
    powers = A.clone()
    if left_mult is not None:
        powers = left_mult @ powers
    if right_mult is not None:
        powers = powers @ right_mult

    out = powers.clone()
    for i in range(2, degree + 1):
        if right_mult is not None:
            powers = A @ powers / i
        else:
            powers = powers @ A / i
        out.add_(powers)
    return out


def _matrix_exp_truncated_identity(A: torch.Tensor, degree: int) -> torch.Tensor:
    """Return I + A + A^2/2! + ... + A^degree/degree! in fp32."""
    A = A.float()
    n = A.shape[0]
    out = torch.eye(n, device=A.device, dtype=torch.float32)
    term = torch.eye(n, device=A.device, dtype=torch.float32)
    for i in range(1, degree + 1):
        term = term @ A / float(i)
        out.add_(term)
    return out


def _skew_grad_in_out(w: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lie factors from Euclidean grad G."""
    wt_g = w.t() @ g
    g_in = wt_g - wt_g.t()
    g_wt = g @ w.t()
    g_out = g_wt - g_wt.t()
    return g_in, g_out


def _scale_update_matrix_rms(
    A_in: torch.Tensor,
    A_out: torch.Tensor,
    p_data: torch.Tensor,
    group: Dict[str, Any],
    update_side: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale the induced ambient update to lr * pion_rms RMS."""
    if update_side == "in":
        base = p_data @ A_in
    elif update_side == "out":
        base = A_out @ p_data
    else:
        base = p_data @ A_in + A_out @ p_data
    m, n = p_data.shape
    fro_norm = base.norm(p="fro")
    lr = group.get("lr", group.get("max_lr", 1e-4))
    rms = float(group.get("pion_rms", 0.2))
    alpha = float(lr) * rms * math.sqrt(m * n) / (fro_norm + 1e-12)
    return alpha * A_in, alpha * A_out


def _scale_biside_generators(
    A_in: torch.Tensor,
    A_out: torch.Tensor,
    p_block: torch.Tensor,
    group: Dict[str, Any],
    update_side: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scaling_mode = group.get("pion_scaling", "rms")
    if scaling_mode in ("rms", "rms_0.2"):
        return _scale_update_matrix_rms(A_in, A_out, p_block, group, update_side)
    if scaling_mode in ("none", None):
        return A_in, A_out
    raise ValueError(f"Unsupported pion_scaling for unified pion.py: {scaling_mode}")


def _configured_update_side(group: Dict[str, Any]) -> str:
    side = str(group.get("pion_update_side", "both") or "both")
    aliases = {
        "bilateral": "both",
        "biside": "both",
        "alternate_update": "alternate",
        "alternating": "alternate",
        "left": "out",
        "output": "out",
        "right": "in",
        "input": "in",
    }
    side = aliases.get(side, side)
    if side not in ("both", "alternate", "in", "out"):
        raise ValueError(
            f"Unsupported pion_update_side: {side}. Expected both, alternate, in, or out."
        )
    return side


def _effective_update_side(group: Dict[str, Any], state: Dict[str, Any]) -> str:
    side = _configured_update_side(group)
    if side != "alternate":
        return side
    step = int(state.get("step", 0))
    return "in" if (step % 2 == 1) else "out"


def _qkv_split_granularity(group: Dict[str, Any], default: str) -> str:
    granularity = str(group.get("pion_qkv_split_granularity", default) or default)
    aliases = {
        "per_head": "head",
        "split_head": "head",
        "split_qkv": "qkv",
        "q_k_v": "qkv",
        "per_group": "group",
    }
    granularity = aliases.get(granularity, granularity)
    if granularity not in ("head", "qkv", "group"):
        raise ValueError(
            "Unsupported pion_qkv_split_granularity: "
            f"{granularity}. Expected head, qkv, or group."
        )
    return granularity


def _momentum_mode(group: Dict[str, Any]) -> str:
    """Resolve hyperparameters to one of the two Pion momentum geometries."""
    explicit = group.get("pion_momentum", None)
    joint = group.get("pion_12_momentum", None)
    if explicit not in (None, "", "none"):
        mode = str(explicit)
    elif joint not in (None, "", "none"):
        mode = str(joint)
    else:
        first = str(group.get("pion_first_momentum", "none") or "none")
        if first in ("ambient", "ambient_transport", "transported_ambient"):
            mode = "transported_ambient_ambient"
        else:
            mode = "lie_lie"

    aliases = {
        "lie": "lie_lie",
        "lie+lie": "lie_lie",
        "lie-lie": "lie_lie",
        "ambient": "transported_ambient_ambient",
        "ambient_transport": "transported_ambient_ambient",
        "transported_ambient": "transported_ambient_ambient",
        "transported_ambient+ambient": "transported_ambient_ambient",
        "transported_ambient_ambient_space": "transported_ambient_ambient",
        "ambient_ambient_transport": "transported_ambient_ambient",
        "ambient_ambient": "transported_ambient_ambient",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("lie_lie", "transported_ambient_ambient"):
        raise ValueError(
            "Unsupported Pion momentum geometry. Expected lie_lie or "
            f"transported_ambient_ambient, got: {mode}"
        )
    return mode


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def _use_second_momentum(group: Dict[str, Any]) -> bool:
    explicit = group.get("pion_use_second_momentum", None)
    if explicit is not None:
        return _str_to_bool(explicit)

    joint = group.get("pion_12_momentum", None)
    if joint not in (None, "", "none"):
        return True

    second = group.get("pion_second_momentum", "none")
    return second not in (None, "", "none")


def _pion_betas(group: Dict[str, Any]) -> Tuple[float, float]:
    """Resolve Pion beta coefficients independently from AdamW when configured."""
    betas = group.get("betas", (0.9, 0.999))
    default_beta1 = float(betas[0]) if isinstance(betas, (tuple, list)) else 0.9
    default_beta2 = (
        float(betas[1]) if isinstance(betas, (tuple, list)) and len(betas) > 1 else 0.999
    )
    beta1 = group.get("pion_beta1", default_beta1)
    beta2 = group.get("pion_beta2", default_beta2)
    if beta1 is None:
        beta1 = default_beta1
    if beta2 is None:
        beta2 = default_beta2
    return float(beta1), float(beta2)


def _lie_lie_generators(
    w: torch.Tensor,
    g: torch.Tensor,
    group: Dict[str, Any],
    state: Dict[str, Any],
    key_prefix: str,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[str]]:
    beta1, beta2 = _pion_betas(group)
    eps = float(group.get("adam_eps", 1e-8))
    use_second = _use_second_momentum(group)

    g_in, g_out = _skew_grad_in_out(w, g)
    k_mi = f"{key_prefix}_lie_m_in"
    k_mo = f"{key_prefix}_lie_m_out"
    k_vi = f"{key_prefix}_lie_v_in"
    k_vo = f"{key_prefix}_lie_v_out"
    if k_mi not in state:
        state[k_mi] = torch.zeros_like(g_in, dtype=torch.float32, device=w.device)
        state[k_mo] = torch.zeros_like(g_out, dtype=torch.float32, device=w.device)
    if use_second and k_vi not in state:
        state[k_vi] = torch.zeros_like(g_in, dtype=torch.float32, device=w.device)
        state[k_vo] = torch.zeros_like(g_out, dtype=torch.float32, device=w.device)

    state[k_mi].mul_(beta1).add_(g_in.float(), alpha=1.0 - beta1)
    state[k_mo].mul_(beta1).add_(g_out.float(), alpha=1.0 - beta1)
    A_in = state[k_mi]
    A_out = state[k_mo]
    if use_second:
        state[k_vi].mul_(beta2).add_(g_in.float().square(), alpha=1.0 - beta2)
        state[k_vo].mul_(beta2).add_(g_out.float().square(), alpha=1.0 - beta2)
        A_in = A_in / (state[k_vi].sqrt() + eps)
        A_out = A_out / (state[k_vo].sqrt() + eps)
    return (-A_in).to(out_dtype), (-A_out).to(out_dtype), None


def _transported_ambient_ambient_generators(
    w: torch.Tensor,
    g: torch.Tensor,
    group: Dict[str, Any],
    state: Dict[str, Any],
    key_prefix: str,
    out_dtype: torch.dtype,
    update_side: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[str]]:
    beta1, beta2 = _pion_betas(group)
    eps = float(group.get("adam_eps", 1e-8))
    use_second = _use_second_momentum(group)

    k_m = f"{key_prefix}_amb_m"
    k_v = f"{key_prefix}_amb_v"
    if k_m not in state:
        state[k_m] = torch.zeros(w.shape[0], w.shape[1], dtype=torch.float32, device=w.device)
    if use_second and k_v not in state:
        state[k_v] = torch.zeros(w.shape[0], w.shape[1], dtype=torch.float32, device=w.device)

    gf = g.float()
    m = state[k_m]
    m.mul_(beta1).add_(gf, alpha=1.0 - beta1)
    tilde_g = m
    if use_second:
        v = state[k_v]
        v.mul_(beta2).add_(m.square(), alpha=1.0 - beta2)
        tilde_g = m / (v.sqrt() + eps)

    if update_side == "in":
        wt_g = w.t() @ tilde_g
        g_in = wt_g - wt_g.t()
        g_out = torch.zeros(w.shape[0], w.shape[0], dtype=torch.float32, device=w.device)
        return (-g_in).to(out_dtype), g_out.to(out_dtype), k_m

    if update_side == "out":
        g_wt = tilde_g @ w.t()
        g_out = g_wt - g_wt.t()
        g_in = torch.zeros(w.shape[1], w.shape[1], dtype=torch.float32, device=w.device)
        return g_in.to(out_dtype), (-g_out).to(out_dtype), k_m

    g_in, g_out = _skew_grad_in_out(w, tilde_g)
    return (-g_in).to(out_dtype), (-g_out).to(out_dtype), k_m


def _build_update_generators(
    w: torch.Tensor,
    g: torch.Tensor,
    group: Dict[str, Any],
    state: Dict[str, Any],
    key_prefix: str,
    out_dtype: torch.dtype,
    update_side: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[str]]:
    mode = _momentum_mode(group)
    if mode == "lie_lie":
        return _lie_lie_generators(w, g, group, state, key_prefix, out_dtype)
    if mode == "transported_ambient_ambient":
        return _transported_ambient_ambient_generators(
            w, g, group, state, key_prefix, out_dtype, update_side
        )
    raise ValueError(
        "Unsupported Pion momentum mode. Expected lie_lie or "
        f"transported_ambient_ambient. Got: {mode}"
    )


def _apply_biside_update(
    p_block: torch.Tensor,
    A_in: torch.Tensor,
    A_out: torch.Tensor,
    group: Dict[str, Any],
    state: Dict[str, Any],
    transport_key: Optional[str] = None,
) -> None:
    degree = int(group.get("degree", 2))
    exp_map = group.get("pion_exp_map", "exp_truncated")
    if exp_map not in ("exp_truncated", "taylor"):
        raise ValueError(f"Unified Pion only supports exp_truncated, got: {exp_map}")

    update_side = _effective_update_side(group, state)
    scaled_in, scaled_out = _scale_biside_generators(A_in, A_out, p_block, group, update_side)
    if transport_key is not None:
        m_buf = state[transport_key]
        if update_side == "in":
            E_in = _matrix_exp_truncated_identity(scaled_in, degree)
            p_block.copy_((p_block.float() @ E_in).to(dtype=p_block.dtype))
            m_buf.copy_((m_buf.float() @ E_in).to(dtype=m_buf.dtype))
            return
        if update_side == "out":
            E_out = _matrix_exp_truncated_identity(scaled_out, degree)
            p_block.copy_((E_out @ p_block.float()).to(dtype=p_block.dtype))
            m_buf.copy_((E_out @ m_buf.float()).to(dtype=m_buf.dtype))
            return
        
        # Update method 1:
        # E_in = _matrix_exp_truncated_identity(scaled_in, degree)
        # E_out = _matrix_exp_truncated_identity(scaled_out, degree)
        # p_block.copy_((E_out @ p_block.float()).to(dtype=p_block.dtype))
        # p_block.copy_((p_block.float() @ E_in).to(dtype=p_block.dtype))
        # m_buf.copy_((E_out @ m_buf.float()).to(dtype=m_buf.dtype))
        # m_buf.copy_((m_buf.float() @ E_in).to(dtype=m_buf.dtype))

        # Update method 2:
        p_block.add_(_matrix_exp_truncated(scaled_in, degree, left_mult=p_block))
        p_block.add_(_matrix_exp_truncated(scaled_out, degree, right_mult=p_block))
        m_buf.add_(_matrix_exp_truncated(scaled_in, degree, left_mult=m_buf))
        m_buf.add_(_matrix_exp_truncated(scaled_out, degree, right_mult=m_buf))

        # p_block.copy_((E_out @ p_block.float() @ E_in).to(dtype=p_block.dtype))
        # m_buf.copy_((E_out @ m_buf.float() @ E_in).to(dtype=m_buf.dtype))
        return

    if update_side in ("both", "in"):
        p_block.add_(_matrix_exp_truncated(scaled_in, degree, left_mult=p_block))
    if update_side in ("both", "out"):
        p_block.add_(_matrix_exp_truncated(scaled_out, degree, right_mult=p_block))


class PionOptimizer(Optimizer):
    """Pion optimizer for 2D matrix parameters."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        degree: int = 2,
        split_qkv: bool = True,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        qkv_split_shapes: Optional[Tuple[int, int, int]] = None,
        split_fc1_up_gate: bool = True,
        is_fc1_up_gate_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        split_qkv_per_head: bool = True,
        qkv_split_granularity: Optional[str] = None,
        pion_scaling: str = "rms",
        pion_rms: float = 0.2,
        pion_momentum: str = "none",
        pion_update_side: str = "both",
        pion_beta1: Optional[float] = None,
        pion_beta2: Optional[float] = None,
    ):
        resolved_qkv_granularity = (
            qkv_split_granularity if qkv_split_granularity is not None
            else ("head" if split_qkv_per_head else "qkv")
        )
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            degree=degree,
            pion_scaling=pion_scaling,
            pion_rms=pion_rms,
            pion_momentum=pion_momentum,
            pion_update_side=pion_update_side,
            pion_beta1=pion_beta1,
            pion_beta2=pion_beta2,
            pion_qkv_split_granularity=resolved_qkv_granularity,
            pion_use_second_momentum=None,
            pion_12_momentum="none",
            pion_first_momentum="none",
            pion_second_momentum="none",
            pion_exp_map="exp_truncated",
            adam_eps=1e-8,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["pion_update_side"] = pion_update_side
        self.split_qkv = split_qkv and (qkv_split_shapes is not None) and (is_qkv_fn is not None)
        self.is_qkv_fn = is_qkv_fn if is_qkv_fn is not None else (lambda p: False)
        self.qkv_split_shapes = tuple(qkv_split_shapes) if qkv_split_shapes else (0, 0, 0)
        self.split_fc1_up_gate = split_fc1_up_gate and (is_fc1_up_gate_fn is not None)
        self.is_fc1_up_gate_fn = (
            is_fc1_up_gate_fn if is_fc1_up_gate_fn is not None else (lambda p: False)
        )
        self.split_qkv_per_head = split_qkv_per_head
        self.qkv_split_granularity = resolved_qkv_granularity

        first_group = self.param_groups[0] if len(self.param_groups) > 0 else {}
        self.pion_update_side = _configured_update_side(first_group)
        log_single_rank(
            logger,
            logging.INFO,
            f"Pion update side configured as {self.pion_update_side}",
        )
    
        self._pion_update_csv = first_group.get("pion_update_csv", None)
        self._pion_update_csv_interval = max(
            1, int(first_group.get("pion_update_csv_interval", 1))
        )
        self._optimizer_step = 0
        self._pending_update_rows: List[
            Tuple[int, int, str, str, int, float, float, float, float, float, float, str, str]
        ] = []
        if self._pion_update_csv and self._is_log_rank():
            csv_dir = os.path.dirname(self._pion_update_csv)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            if (not os.path.exists(self._pion_update_csv)) or os.path.getsize(self._pion_update_csv) == 0:
                with open(self._pion_update_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "optimizer_step",
                            "param_step",
                            "param_name",
                            "unit_name",
                            "unit_numel",
                            "update_fro_norm",
                            "update_max_abs",
                            "a_in_fro_norm",
                            "a_out_fro_norm",
                            "a_in_spec_norm",
                            "a_out_spec_norm",
                            "momentum_mode",
                            "update_side",
                        ]
                    )

    def _is_log_rank(self) -> bool:
        return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0

    def _should_log_for_param_step(self, param_step: int) -> bool:
        return (
            self._pion_update_csv is not None
            and self._is_log_rank()
            and (param_step % self._pion_update_csv_interval == 0)
        )

    def _queue_update_row(
        self,
        param_step: int,
        param_name: str,
        unit_name: str,
        before: torch.Tensor,
        after: torch.Tensor,
        A_in: torch.Tensor,
        A_out: torch.Tensor,
        group: Dict[str, Any],
        update_side: str,
    ) -> None:
        lr = group.get("lr", group.get("max_lr", 1e-4))
        delta = (after - before).float() / (abs(float(lr)) + 1e-12)
        a_in_f = A_in.float()
        a_out_f = A_out.float()
        self._pending_update_rows.append(
            (
                self._optimizer_step,
                int(param_step),
                param_name,
                unit_name,
                int(after.numel()),
                float(delta.norm(p="fro").item()),
                float(delta.abs().max().item()),
                float(a_in_f.norm(p="fro").item()),
                float(a_out_f.norm(p="fro").item()),
                float(torch.linalg.matrix_norm(a_in_f, ord=2).item()),
                float(torch.linalg.matrix_norm(a_out_f, ord=2).item()),
                _momentum_mode(group),
                update_side,
            )
        )

    def _flush_update_rows(self) -> None:
        if (not self._pending_update_rows) or (self._pion_update_csv is None) or (not self._is_log_rank()):
            self._pending_update_rows = []
            return
        with open(self._pion_update_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self._pending_update_rows)
        self._pending_update_rows = []

    def _update_block(
        self,
        block: torch.Tensor,
        grad_block: torch.Tensor,
        group: Dict[str, Any],
        state: Dict[str, Any],
        key_prefix: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 构造更新矩阵
        update_side = _effective_update_side(group, state)
        A_in, A_out, transport_key = _build_update_generators(
            block, grad_block, group, state, key_prefix, block.dtype, update_side
        )
        # 应用更新矩阵
        _apply_biside_update(block, A_in, A_out, group, state, transport_key)
        return A_in, A_out

    def _update_and_maybe_log(
        self,
        block: torch.Tensor,
        grad_block: torch.Tensor,
        group: Dict[str, Any],
        state: Dict[str, Any],
        key_prefix: str,
        param_name: str,
        should_log: bool,
    ) -> None:
        before = block.clone() if should_log else None
        A_in, A_out = self._update_block(block, grad_block, group, state, key_prefix)
        if should_log and before is not None:
            self._queue_update_row(
                state["step"],
                param_name,
                key_prefix,
                before,
                block,
                A_in,
                A_out,
                group,
                _effective_update_side(group, state),
            )

    def _pion_update_for_matrix(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        group: Dict[str, Any],
        state: Dict[str, Any],
    ) -> None:
        p_data = p.data.float() if p.dtype != torch.float32 else p.data
        grad_f = grad.float() if grad.dtype != torch.float32 else grad
        out_dim, in_dim = p_data.shape

        if "step" not in state:
            state["step"] = 0
        state["step"] += 1
        should_log = self._should_log_for_param_step(state["step"])
        param_name = getattr(p, "_pion_param_name", "unknown_param")

        if self.split_qkv and self.is_qkv_fn(p):
            total = sum(self.qkv_split_shapes)
            num_query_groups = out_dim // total
            q_per_group, k_per_group, v_per_group = self.qkv_split_shapes
            view = p_data.view(num_query_groups, total, in_dim)
            grad_view = grad_f.view(num_query_groups, total, in_dim)
            qkv_granularity = _qkv_split_granularity(group, self.qkv_split_granularity)

            if qkv_granularity == "qkv":
                q_blocks = [view[g_idx, :q_per_group, :].clone() for g_idx in range(num_query_groups)]
                k_blocks = [
                    view[g_idx, q_per_group : q_per_group + k_per_group, :].clone()
                    for g_idx in range(num_query_groups)
                ]
                v_blocks = [
                    view[
                        g_idx,
                        q_per_group + k_per_group : q_per_group + k_per_group + v_per_group,
                        :,
                    ].clone()
                    for g_idx in range(num_query_groups)
                ]
                q_grad_blocks = [
                    grad_view[g_idx, :q_per_group, :] for g_idx in range(num_query_groups)
                ]
                k_grad_blocks = [
                    grad_view[g_idx, q_per_group : q_per_group + k_per_group, :]
                    for g_idx in range(num_query_groups)
                ]
                v_grad_blocks = [
                    grad_view[
                        g_idx,
                        q_per_group + k_per_group : q_per_group + k_per_group + v_per_group,
                        :,
                    ]
                    for g_idx in range(num_query_groups)
                ]

                q_all = torch.cat(q_blocks, dim=0)
                k_all = torch.cat(k_blocks, dim=0)
                v_all = torch.cat(v_blocks, dim=0)
                q_grad_all = torch.cat(q_grad_blocks, dim=0)
                k_grad_all = torch.cat(k_grad_blocks, dim=0)
                v_grad_all = torch.cat(v_grad_blocks, dim=0)
                self._update_and_maybe_log(q_all, q_grad_all, group, state, "q", param_name, should_log)
                self._update_and_maybe_log(k_all, k_grad_all, group, state, "k", param_name, should_log)
                self._update_and_maybe_log(v_all, v_grad_all, group, state, "v", param_name, should_log)

                q_chunks = list(q_all.split(q_per_group, dim=0))
                k_chunks = list(k_all.split(k_per_group, dim=0))
                v_chunks = list(v_all.split(v_per_group, dim=0))
                new_p = torch.cat(
                    [
                        torch.cat([q_chunks[g_idx], k_chunks[g_idx], v_chunks[g_idx]], dim=0)
                        for g_idx in range(num_query_groups)
                    ],
                    dim=0,
                )
                p.data.copy_(new_p.to(dtype=p.data.dtype) if new_p.dtype != p.data.dtype else new_p)
                return

            new_groups: List[torch.Tensor] = []
            for g_idx in range(num_query_groups):
                group_view = view[g_idx]
                grad_group = grad_view[g_idx]
                q_block = group_view[:q_per_group]
                q_grad = grad_group[:q_per_group]
                k_block = group_view[q_per_group : q_per_group + k_per_group].clone()
                k_grad = grad_group[q_per_group : q_per_group + k_per_group]
                v_block = group_view[
                    q_per_group + k_per_group : q_per_group + k_per_group + v_per_group
                ].clone()
                v_grad = grad_group[
                    q_per_group + k_per_group : q_per_group + k_per_group + v_per_group
                ]

                if qkv_granularity == "head" and q_per_group % k_per_group == 0:
                    head_dim = k_per_group
                    q_heads: List[torch.Tensor] = []
                    for h_idx in range(q_per_group // head_dim):
                        q_head = q_block[h_idx * head_dim : (h_idx + 1) * head_dim].clone()
                        q_head_grad = q_grad[h_idx * head_dim : (h_idx + 1) * head_dim]
                        self._update_and_maybe_log(
                            q_head,
                            q_head_grad,
                            group,
                            state,
                            f"q_g{g_idx}_h{h_idx}",
                            param_name,
                            should_log,
                        )
                        q_heads.append(q_head)
                    q_new = torch.cat(q_heads, dim=0)
                else:
                    q_new = q_block.clone()
                    self._update_and_maybe_log(
                        q_new, q_grad, group, state, f"q_g{g_idx}", param_name, should_log
                    )

                self._update_and_maybe_log(
                    k_block, k_grad, group, state, f"k_g{g_idx}", param_name, should_log
                )
                self._update_and_maybe_log(
                    v_block, v_grad, group, state, f"v_g{g_idx}", param_name, should_log
                )
                new_groups.append(torch.cat([q_new, k_block, v_block], dim=0))

            new_p = torch.cat(new_groups, dim=0)
            p.data.copy_(new_p.to(dtype=p.data.dtype) if new_p.dtype != p.data.dtype else new_p)
            return

        if self.split_fc1_up_gate and self.is_fc1_up_gate_fn(p):
            half = out_dim // 2
            w_up = p_data[:half].clone()
            w_gate = p_data[half:].clone()
            g_up = grad_f[:half]
            g_gate = grad_f[half:]
            self._update_and_maybe_log(
                w_up, g_up, group, state, "fc1_up", param_name, should_log
            )
            self._update_and_maybe_log(
                w_gate, g_gate, group, state, "fc1_gate", param_name, should_log
            )
            new_p = torch.cat([w_up, w_gate], dim=0)
            p.data.copy_(new_p.to(dtype=p.data.dtype) if new_p.dtype != p.data.dtype else new_p)
            return

        main = p_data.clone() if p_data.data_ptr() == p.data.data_ptr() else p_data
        self._update_and_maybe_log(main, grad_f, group, state, "main", param_name, should_log)
        p.data.copy_(main.to(dtype=p.data.dtype) if main.dtype != p.data.dtype else main)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self._optimizer_step += 1
        self._pending_update_rows = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if len(p.shape) != 2:
                    continue
                self._pion_update_for_matrix(p, p.grad.data, group, self.state[p])
        self._flush_update_rows()
        return loss


def _matrix_param_groups(
    model_chunks: List[MegatronModule],
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
    matrix_params: List[torch.nn.Parameter],
) -> List[Dict[str, Any]]:
    matrix_param_ids = {id(p) for p in matrix_params}
    groups: List[Dict[str, Any]] = []
    for group in _get_param_groups(model_chunks, config, config_overrides):
        params = [p for p in group["params"] if id(p) in matrix_param_ids]
        if params:
            new_group = dict(group)
            new_group["params"] = params
            groups.append(new_group)
    return groups


def get_megatron_pion_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Build Pion for matrix params + AdamW for non-matrix params."""
    config.optimizer = "adam"

    if config.use_distributed_optimizer:
        raise Exception("Pion with distributed optimizer is not supported.")

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f"Setting up unified Pion optimizer with config {config}")

    matrix_params: List[torch.nn.Parameter] = []
    non_matrix_params: List[torch.nn.Parameter] = []
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
                log_single_rank(logger, logging.DEBUG, f"matrix_params: {name} {param.shape}")
            else:
                non_matrix_params.append(param)

    lr = float(config.lr if config.lr is not None else 1e-4)
    matrix_param_groups = _matrix_param_groups(model_chunks, config, config_overrides, matrix_params)
    if not matrix_param_groups:
        matrix_param_groups = [
            {
                "params": matrix_params,
                "max_lr": lr,
                "min_lr": config.min_lr,
                "wd_mult": 1.0,
                "lr_mult": 1.0,
                "is_expert_parallel": False,
                "default_config": True,
            }
        ]

    degree = getattr(config, "pion_degree", 2)
    pion_scaling = getattr(config, "pion_scaling", "rms")
    pion_rms = getattr(config, "pion_rms", 0.2)
    pion_momentum = getattr(config, "pion_momentum", "none")
    pion_use_second_momentum = getattr(config, "pion_use_second_momentum", None)
    pion_update_side = getattr(config, "pion_update_side", "both")
    pion_qkv_split_granularity = getattr(config, "pion_qkv_split_granularity", None)
    if pion_qkv_split_granularity is None:
        pion_qkv_split_granularity = (
            "head" if getattr(config, "pion_split_qkv_per_head", True) else "qkv"
        )
    pion_12_momentum = getattr(config, "pion_12_momentum", "none")
    pion_first_momentum = getattr(config, "pion_first_momentum", "none")
    pion_second_momentum = getattr(config, "pion_second_momentum", "none")
    pion_exp_map = getattr(config, "pion_exp_map", "exp_truncated")
    pion_update_csv = getattr(config, "pion_update_csv", None)
    pion_update_csv_interval = getattr(config, "pion_update_csv_interval", 1)
    adam_eps = getattr(config, "adam_eps", 1e-8)
    pion_beta1 = getattr(config, "pion_beta1", 0.9)
    pion_beta2 = getattr(config, "pion_beta2", 0.999)

    for group in matrix_param_groups:
        group["degree"] = degree
        group["pion_scaling"] = pion_scaling
        group["pion_rms"] = pion_rms
        group["pion_momentum"] = pion_momentum
        group["pion_use_second_momentum"] = pion_use_second_momentum
        group["pion_update_side"] = pion_update_side
        group["pion_qkv_split_granularity"] = pion_qkv_split_granularity
        group["pion_12_momentum"] = pion_12_momentum
        group["pion_first_momentum"] = pion_first_momentum
        group["pion_second_momentum"] = pion_second_momentum
        group["pion_exp_map"] = pion_exp_map
        group["pion_update_csv"] = pion_update_csv
        group["pion_update_csv_interval"] = pion_update_csv_interval
        group["adam_eps"] = adam_eps
        group["pion_beta1"] = pion_beta1
        group["pion_beta2"] = pion_beta2

    pion_optimizer = PionOptimizer(
        matrix_param_groups,
        lr=lr,
        betas=(pion_beta1, pion_beta2),
        weight_decay=config.weight_decay,
        degree=degree,
        split_qkv=getattr(config, "pion_split_qkv", True),
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=qkv_split_shapes,
        split_fc1_up_gate=split_fc1_up_gate,
        is_fc1_up_gate_fn=lambda p: getattr(p, "is_fc1_up_gate", False),
        split_qkv_per_head=getattr(config, "pion_split_qkv_per_head", True),
        qkv_split_granularity=pion_qkv_split_granularity,
        pion_scaling=pion_scaling,
        pion_rms=pion_rms,
        pion_momentum=pion_momentum,
        pion_update_side=pion_update_side,
        pion_beta1=pion_beta1,
        pion_beta2=pion_beta2,
    )

    def pion_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group["params"]:
                if len(opt.state[p]) == 0:
                    opt.state[p]["step"] = 0

    if config.fp16:
        raise Exception("Pion with fp16 is not supported; use bf16.")
    if config.bf16:
        optimizer = Float16OptimizerWithFloat16Params(
            pion_optimizer, config, None, pion_init_state_fn  # pyright: ignore[reportArgumentType]
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
    tp_group = (
        pg_collection.tp
        if (pg_collection and hasattr(pg_collection, "tp"))
        else parallel_state.get_tensor_model_parallel_group()
    )
    setattr(optimizer, "tp_group", tp_group)
    _ = non_matrix_params, adam_init_state_fn
    return ChainedOptimizer(optimizers)


__all__ = ["PionOptimizer", "get_megatron_pion_optimizer"]