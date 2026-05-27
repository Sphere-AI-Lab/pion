from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration for approximate orthogonalization."""
    assert g.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    if g.size(0) > g.size(1):
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    if g.size(0) > g.size(1):
        x = x.T
    return x


def _route_to_adamw_by_name(param_name: str) -> bool:
    if not param_name:
        return False
    n = param_name.lower()
    tokens = (
        "lm_head",
        "embed_tokens",
        "word_embeddings",
        "tok_embeddings",
        "token_embedding",
        "wte",
        "embedding",
    )
    return any(t in n for t in tokens)


def muon_optimizer_requires_use_orig_params(optimizer_name: str) -> bool:
    return (optimizer_name or "").strip().lower() == "muonoptimizer"


def muon_optimizer_selected_from_config(optim_config: Any) -> bool:
    if optim_config is None:
        return False
    name: Optional[str] = None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(optim_config):
            sel = OmegaConf.select(optim_config, "optimizer", default=None)
            name = str(sel) if sel is not None else None
    except Exception:
        name = None
    if name is None and isinstance(optim_config, dict):
        o = optim_config.get("optimizer")
        name = str(o) if o is not None else None
    if name is None:
        o = getattr(optim_config, "optimizer", None)
        if o is not None:
            name = str(o)
    if name is None:
        try:
            o = optim_config.get("optimizer", None)  # type: ignore[attr-defined]
            if o is not None:
                name = str(o)
        except Exception:
            pass
    if not name:
        return False
    return muon_optimizer_requires_use_orig_params(name)


def prepare_muon_module_tags(
    module: torch.nn.Module,
    *,
    optimizer_name: str,
    override_optimizer_config: Any = None,
) -> None:
    del override_optimizer_config
    if not muon_optimizer_requires_use_orig_params(optimizer_name):
        return
    for name, param in module.named_parameters():
        if param.ndim == 2:
            setattr(param, "_muon_param_name", name)


def assert_fsdp_orig_params_effective_for_muon(
    fsdp_module: torch.nn.Module,
    *,
    intended_use_orig: bool,
    role: str,
    optim_config: Any,
    fsdp_strategy: str,
    rank: int,
) -> None:
    if fsdp_strategy != "fsdp" or not intended_use_orig or role != "actor" or optim_config is None:
        return
    if not muon_optimizer_selected_from_config(optim_config):
        return
    try:
        from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel as FSDPCls
    except ImportError:
        return
    subs = FSDPCls.fsdp_modules(fsdp_module)
    bad_flags = [m for m in subs if getattr(m, "_use_orig_params", True) is False]
    if bad_flags:
        raise RuntimeError(
            f"FSDP use_orig_params=True was requested, but {len(bad_flags)} nested FSDP module(s) have "
            f"_use_orig_params=False (first: {bad_flags[0]!r})."
        )
    flat_named = [(n, tuple(p.shape)) for n, p in fsdp_module.named_parameters() if isinstance(p, FlatParameter)]
    if flat_named:
        raise RuntimeError(
            "FSDP use_orig_params=True but FlatParameter still appears in named_parameters (e.g. "
            f"{flat_named[0]})."
        )

    trainable = [(n, p) for n, p in fsdp_module.named_parameters() if p.requires_grad and p.numel() > 0]
    twod = [n for n, p in trainable if p.ndim == 2]
    ndim_hist: dict[int, int] = {}
    for _, p in trainable:
        ndim_hist[p.ndim] = ndim_hist.get(p.ndim, 0) + 1
    if trainable and not twod:
        raise RuntimeError(
            "Muon + FSDP1: use_orig_params=True was passed to FSDP(), but there are no 2D trainable "
            f"parameters under named_parameters (ndim histogram: {ndim_hist})."
        )
    if rank == 0:
        logger.info(
            "[FSDP/Muon] trainable=%s with_ndim==2=%s ndim_hist=%s",
            len(trainable),
            len(twod),
            ndim_hist,
        )


class MuonOptimizer(Optimizer):
    """Muon for 2D params + AdamW fallback for non-2D params."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        **kwargs: Any,
    ):
        import warnings

        kwargs.pop("max_lr", None)
        for _silent in (
            "fused",
            "foreach",
            "capturable",
            "maximize",
            "differentiable",
            "bf16_stochastic_round",
            "master_weights",
            "store_param_remainders",
            "exp_avg_dtype",
            "exp_avg_sq_dtype",
            "master_weight_dtype",
        ):
            kwargs.pop(_silent, None)
        if kwargs:
            warnings.warn(
                f"MuonOptimizer: ignoring unknown config keys {sorted(kwargs.keys())}",
                UserWarning,
                stacklevel=2,
            )

        param_list = list(params)
        muon_params = []
        adamw_params = []
        for p in param_list:
            if not p.requires_grad:
                continue
            param_name = str(getattr(p, "_muon_param_name", "") or "")
            force_adamw = _route_to_adamw_by_name(param_name)
            if p.ndim == 2 and not getattr(p, "_muon_skip", False) and not force_adamw:
                print(f"Muon matrix param {param_name} shape={tuple(p.shape)}")
                muon_params.append(p)
            else:
                print(f"Muon AdamW param {param_name} shape={tuple(p.shape)} ndim={p.ndim} type={type(p).__name__}")
                adamw_params.append(p)

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=betas,
            eps=eps,
            is_muon=False,
        )
        groups = []
        if muon_params:
            groups.append(
                {
                    "params": muon_params,
                    "is_muon": True,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "nesterov": nesterov,
                    "ns_steps": ns_steps,
                    "betas": betas,
                    "eps": eps,
                }
            )
        if adamw_params:
            groups.append(
                {
                    "params": adamw_params,
                    "is_muon": False,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "betas": betas,
                    "eps": eps,
                }
            )
        if not groups:
            raise ValueError("MuonOptimizer: no trainable parameters (requires_grad=True).")

        super().__init__(groups, defaults)

    @staticmethod
    def _adjust_lr_for_muon(lr: float, param_shape: tuple[int, ...]) -> float:
        a, b = param_shape[:2]
        return lr * (0.2 * math.sqrt(max(a, b)))

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        wd = group["weight_decay"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            if g.ndim > 2:
                g = g.view(g.size(0), -1)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            if nesterov:
                g = g.add(buf, alpha=momentum)
            else:
                g = buf
            u = zeropower_via_newtonschulz5(g, steps=ns_steps)
            adjusted_lr = self._adjust_lr_for_muon(lr, tuple(p.shape))

            p.data.mul_(1 - lr * wd)
            p.data.add_(u, alpha=-adjusted_lr)

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["moment1"] = torch.zeros_like(g)
                state["moment2"] = torch.zeros_like(g)
            state["step"] += 1
            step = state["step"]
            buf1 = state["moment1"]
            buf2 = state["moment2"]
            buf1.lerp_(g, 1 - beta1)
            buf2.lerp_(g.square(), 1 - beta2)
            g = buf1 / (eps + buf2.sqrt())
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            scale = bias_correction1 / bias_correction2**0.5
            p.data.mul_(1 - lr * wd)
            p.data.add_(g, alpha=-lr / scale)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("is_muon", False):
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss


__all__ = [
    "MuonOptimizer",
    "assert_fsdp_orig_params_effective_for_muon",
    "muon_optimizer_requires_use_orig_params",
    "muon_optimizer_selected_from_config",
    "prepare_muon_module_tags",
]
