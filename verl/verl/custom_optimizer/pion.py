from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def _route_to_adamw_by_name(param_name: str) -> bool:
    """Return True when a 2D parameter should use AdamW instead of Pion."""
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


def _matrix_exp_truncated_integrated(
    A: torch.Tensor,
    p_data: torch.Tensor,
    side: str,
    group: Dict[str, Any],
    state: Dict[str, Any],
) -> torch.Tensor:
    """Truncated matrix exponential for Pion update (in/out side)."""
    if side == 'in':
        powers = p_data @ A
    else:
        powers = A @ p_data

    m, n = powers.shape
    fro_norm = powers.norm(p="fro")
    lr = group.get('lr', group.get('max_lr', 1e-4))
    degree = group.get('degree', 2)
    alpha = (
        lr
        * 0.2
        * math.sqrt(m * n)
        / (fro_norm + 1e-12)
    )
    powers = powers.mul(alpha)
    out = powers.clone()

    scaled_A = A * alpha
    for i in range(2, degree + 1):
        inv_i = 1.0 / i
        if side == 'in':
            powers = (powers @ scaled_A).mul_(inv_i)
        else:
            powers = (scaled_A @ powers).mul_(inv_i)
        out.add_(powers)
    return out

def tag_parameters_for_pion(
    module: torch.nn.Module,
    *,
    head_dim: Optional[int] = None,
) -> None:
    """Mark HF-style attention row-parallel projections for per-block Pion updates.

    When ``head_dim`` is set (typically ``hidden_size // num_attention_heads``), 2D weights whose
    names look like separate ``q_proj`` / ``k_proj`` / ``v_proj`` (or ``.query`` / ``.key`` /
    ``.value``) get ``_pion_per_head=True``. Each optimizer step then applies Pion along output
    slices of size ``head_dim`` (per query head for Q, per KV head for K/V in GQA).

    Fused Megatron-style QKV / gate-up matrices are not tagged here; treat them as ordinary
    matrices or fork tagging patterns locally.

    Call once after the training module is built (before or after FSDP wrap; ``named_parameters``
    must include the actor weights). With FSDP1, training must use ``use_orig_params=True`` so
    these parameters stay 2D (see :func:`pion_optimizer_requires_use_orig_params`).
    """
    for name, param in module.named_parameters():
        if param.dim() != 2:
            continue
        setattr(param, "_pion_param_name", name)

        if head_dim is None or int(head_dim) <= 0:
            continue
        if not (name.endswith(".weight") or name.endswith(".weight_orig")):
            continue
        if any(
            token in name
            for token in (
                "q_proj",
                "k_proj",
                "v_proj",
                ".query.",
                ".key.",
                ".value.",
            )
        ):
            param._pion_per_head = True  # type: ignore[attr-defined]
            param._pion_head_dim = int(head_dim)  # type: ignore[attr-defined]


def pion_optimizer_requires_use_orig_params(optimizer_name: str) -> bool:
    """Whether the given optimizer needs FSDP1 ``use_orig_params=True`` (2D matrix updates).

    True for :class:`PionOptimizer`, :class:`verl.custom_optimizer.pion_ambient.PionAmbientOptimizer`,
    and :class:`verl.custom_optimizer.pion_ambient_v2.PionAmbientV2Optimizer`.
    PyTorch FSDP1 with ``use_orig_params=False`` registers flattened ``FlatParameter`` (1D) tensors
    with the optimizer, so Pion-style matrix updates never run. Call sites must force
    ``use_orig_params=True`` when this returns True.
    """
    n = (optimizer_name or "").strip().lower()
    return n in ("pionoptimizer", "pionambientoptimizer", "pionambientv2optimizer")


def assert_fsdp_orig_params_effective_for_pion(
    fsdp_module: torch.nn.Module,
    *,
    intended_use_orig: bool,
    role: str,
    optim_config: Any,
    fsdp_strategy: str,
    rank: int,
) -> None:
    """After FSDP1 wrap: if Pion + use_orig_params was requested, verify nested FSDP and 2D trainable weights.

    PyTorch does not always define ``_use_orig_params`` on every internal FSDP object; missing attribute
    must be treated as "OK" (default True). Only an explicit ``False`` indicates a mismatch.
    """
    if fsdp_strategy != "fsdp" or not intended_use_orig or role != "actor" or optim_config is None:
        return
    if not pion_optimizer_selected_from_config(optim_config):
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

    trainable = [
        (n, p)
        for n, p in fsdp_module.named_parameters()
        if p.requires_grad and p.numel() > 0
    ]
    twod = [n for n, p in trainable if p.ndim == 2]
    ndim_hist: Dict[int, int] = {}
    for _, p in trainable:
        ndim_hist[p.ndim] = ndim_hist.get(p.ndim, 0) + 1
    if trainable and not twod:
        raise RuntimeError(
            "Pion + FSDP1: use_orig_params=True was passed to FSDP(), but there are no 2D trainable "
            f"parameters under named_parameters (ndim histogram: {ndim_hist}). "
            "The optimizer will only see 1D tensors, so Pion cannot run. "
            "Check: (1) PyTorch version supports use_orig_params with your sharding setup; "
            "(2) actor.fsdp_config.fsdp_size=-1 or >= world_size so device_mesh is 1D (FULL_SHARD) — "
            "2D mesh uses HYBRID_SHARD which can differ; (3) the module passed to build_optimizer is the "
            "same FSDP root returned from FSDP(...)."
        )
    if rank == 0:
        sample = [
            (n, tuple(p.shape), type(p).__name__)
            for n, p in fsdp_module.named_parameters()
            if p.requires_grad and p.numel() > 0
        ][:6]
        logger.info(
            "[FSDP/Pion] trainable=%s with_ndim==2=%s ndim_hist=%s sample=%s",
            len(trainable),
            len(twod),
            ndim_hist,
            sample,
        )


def pion_optimizer_selected_from_config(optim_config: Any) -> bool:
    """True if ``optim_config`` selects :class:`PionOptimizer` (OmegaConf / dict / dataclass)."""
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
    return pion_optimizer_requires_use_orig_params(name)


def prepare_pion_module_tags(
    module: torch.nn.Module,
    *,
    optimizer_name: str,
    override_optimizer_config: Any = None,
) -> None:
    """If using PionOptimizer, resolve ``head_dim`` and run :func:`tag_parameters_for_pion`.

    ``head_dim`` comes from ``override_optimizer_config.head_dim`` when set, otherwise from
    ``hidden_size // num_attention_heads`` on the (unwrapped) HF ``config`` when available.
    """
    if not pion_optimizer_requires_use_orig_params(optimizer_name):
        return

    from omegaconf import OmegaConf

    head_dim: Optional[int] = None
    override_set = False
    if override_optimizer_config is not None:
        if OmegaConf.is_config(override_optimizer_config):
            if "head_dim" in override_optimizer_config:
                override_set = True
                head_dim = int(override_optimizer_config["head_dim"])
        elif isinstance(override_optimizer_config, dict):
            if "head_dim" in override_optimizer_config:
                override_set = True
                head_dim = int(override_optimizer_config["head_dim"])
        else:
            hd = getattr(override_optimizer_config, "head_dim", None)
            if hd is not None:
                override_set = True
                head_dim = int(hd)

    if override_set and (head_dim is None or head_dim <= 0):
        return

    if not override_set:
        head_dim = None
    if head_dim is None or head_dim <= 0:
        base = module
        while hasattr(base, "_fsdp_wrapped_module"):
            base = base._fsdp_wrapped_module
        cfg = getattr(base, "config", None)
        if cfg is not None:
            hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
            n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
            if hidden is not None and n_heads:
                head_dim = int(hidden // n_heads)

    if head_dim is not None and head_dim > 0:
        tag_parameters_for_pion(module, head_dim=head_dim)


class _PionMatrixCore:
    """Pion updates for 2D parameters, with optional per-head row blocks (for separate q/k/v)."""

    def __init__(
        self,
        *,
        head_dim: int = 0,
        per_head_fn: Optional[Callable[[torch.Tensor], bool]] = None,
    ):
        self.head_dim = int(head_dim)
        self.per_head_fn = per_head_fn if per_head_fn is not None else (lambda _: False)

    def _pion_update_output_row_blocks(
        self,
        p: torch.Tensor,
        grad_f: torch.Tensor,
        p_data: torch.Tensor,
        group: Dict[str, Any],
        state: Dict[str, Any],
        beta1: float,
        head_dim: int,
    ) -> None:
        """Apply Pion independently to each output slice of shape (head_dim, in_dim)."""
        out_dim, in_dim = p_data.shape
        num_blocks = out_dim // head_dim

        if "step" not in state:
            state["step"] = 0
            state["exp_avg_in_blocks"] = [
                torch.zeros((in_dim, in_dim), device=p.device, dtype=torch.float32)
                for _ in range(num_blocks)
            ]
            state["exp_avg_out_blocks"] = [
                torch.zeros((head_dim, head_dim), device=p.device, dtype=torch.float32)
                for _ in range(num_blocks)
            ]
        elif len(state["exp_avg_in_blocks"]) != num_blocks:
            state["step"] = 0
            state["exp_avg_in_blocks"] = [
                torch.zeros((in_dim, in_dim), device=p.device, dtype=torch.float32)
                for _ in range(num_blocks)
            ]
            state["exp_avg_out_blocks"] = [
                torch.zeros((head_dim, head_dim), device=p.device, dtype=torch.float32)
                for _ in range(num_blocks)
            ]

        state["step"] += 1
        update_side = "in" if (state["step"] % 2 == 1) else "out"

        view = p_data.view(num_blocks, head_dim, in_dim)
        grad_view = grad_f.view(num_blocks, head_dim, in_dim)
        w_list = [view[b].clone() for b in range(num_blocks)]
        g_list = [grad_view[b] for b in range(num_blocks)]

        for b in range(num_blocks):
            grad_in_b = w_list[b].t() @ g_list[b]
            grad_in_b = grad_in_b - grad_in_b.t()
            state["exp_avg_in_blocks"][b].mul_(beta1).add_(grad_in_b, alpha=1 - beta1)
            grad_out_b = g_list[b] @ w_list[b].t()
            grad_out_b = grad_out_b - grad_out_b.t()
            state["exp_avg_out_blocks"][b].mul_(beta1).add_(grad_out_b, alpha=1 - beta1)

        if update_side == "in":
            for b in range(num_blocks):
                A = (-state["exp_avg_in_blocks"][b]).to(p_data.dtype)
                w_list[b].add_(_matrix_exp_truncated_integrated(A, w_list[b], "in", group, state))
        else:
            for b in range(num_blocks):
                A = (-state["exp_avg_out_blocks"][b]).to(p_data.dtype)
                w_list[b].add_(_matrix_exp_truncated_integrated(A, w_list[b], "out", group, state))

        new_p = torch.cat(w_list, dim=0)
        if new_p.dtype != p.data.dtype:
            new_p = new_p.to(p.data.dtype)
        p.data.copy_(new_p)

    def pion_update_for_matrix(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        group: Dict[str, Any],
        state: Dict[str, Any],
    ) -> None:
        betas = group.get("betas", (0.9, 0.999))
        beta1 = betas[0] if isinstance(betas, (tuple, list)) else betas

        p_data = p.data.float() if p.dtype != torch.float32 else p.data
        grad_f = grad.float() if grad.dtype != torch.float32 else grad
        out_dim, in_dim = p_data.shape

        hd = self.head_dim
        use_row_blocks = hd > 0 and self.per_head_fn(p) and (out_dim % hd == 0)
        if use_row_blocks:
            logger.debug(
                "Pion per-head row blocks shape=%s head_dim=%s num_blocks=%s",
                tuple(p_data.shape),
                hd,
                out_dim // hd,
            )
            self._pion_update_output_row_blocks(p, grad_f, p_data, group, state, beta1, hd)
            return

        if "step" not in state:
            state["step"] = 0
            state["exp_avg_in"] = torch.zeros((in_dim, in_dim), device=p.device, dtype=torch.float32)
            state["exp_avg_out"] = torch.zeros((out_dim, out_dim), device=p.device, dtype=torch.float32)

        state["step"] += 1
        update_side = "in" if (state["step"] % 2 == 1) else "out"

        grad_in = p_data.t() @ grad_f
        grad_in = grad_in - grad_in.t()
        grad_out = grad_f @ p_data.t()
        grad_out = grad_out - grad_out.t()

        state["exp_avg_in"].mul_(beta1).add_(grad_in, alpha=1 - beta1)
        state["exp_avg_out"].mul_(beta1).add_(grad_out, alpha=1 - beta1)

        if update_side == "in":
            A = (-state["exp_avg_in"]).to(p_data.dtype)
        else:
            A = (-state["exp_avg_out"]).to(p_data.dtype)

        delta_p = _matrix_exp_truncated_integrated(A, p_data, update_side, group, state)
        if delta_p.dtype != p.data.dtype:
            delta_p = delta_p.to(p.data.dtype)
        p.data.add_(delta_p)


class PionOptimizer(Optimizer):
    """verl / PyTorch FSDP entrypoint: Pion for rank-2 weights, AdamW for other parameters.

    Compatible with ``verl.workers.config.optimizer.build_optimizer``::

        optimizer_impl: verl.custom_optimizer.pion
        optimizer: PionOptimizer

    Optional overrides (``override_optimizer_config``)::

        degree: 2
        head_dim: null  # e.g. hidden_size // num_attention_heads; enables per-head Pion on tagged q/k/v

    For Hugging Face models with separate ``q_proj`` / ``k_proj`` / ``v_proj``, call
    ``tag_parameters_for_pion(module, head_dim=...)`` once on the actor (FSDP engine does this when
    ``head_dim`` is set or can be inferred from ``module.config``). Other 2D weights use full-matrix
    Pion; ``gate_proj`` / ``up_proj`` need no special casing.

    Embedding / LM head: mark ``param._pion_skip = True`` to use AdamW instead of Pion.

    Deprecated (ignored with a warning): ``split_qkv``, ``qkv_split_shapes``, ``split_fc1_up_gate``,
    ``split_qkv_per_head``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        *,
        degree: int = 2,
        head_dim: Optional[int] = None,
        **kwargs: Any,
    ):
        import warnings

        _legacy = (
            "split_qkv",
            "qkv_split_shapes",
            "split_fc1_up_gate",
            "split_qkv_per_head",
        )
        if any(k in kwargs for k in _legacy):
            warnings.warn(
                "PionOptimizer: split_qkv / qkv_split_shapes / split_fc1_up_gate / split_qkv_per_head "
                "are deprecated and ignored. Use head_dim plus tag_parameters_for_pion(module, head_dim=...).",
                DeprecationWarning,
                stacklevel=2,
            )
        for k in _legacy:
            kwargs.pop(k, None)

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
                f"PionOptimizer: ignoring unknown config keys {sorted(kwargs.keys())}",
                UserWarning,
                stacklevel=2,
            )

        hd = int(head_dim) if head_dim is not None and int(head_dim) > 0 else 0

        param_list = list(params)
        matrix_params: List[torch.nn.Parameter] = []
        vector_params: List[torch.nn.Parameter] = []
        for p in param_list:
            if not p.requires_grad:
                continue
            param_name = str(getattr(p, "_pion_param_name", "") or "")
            force_adamw = _route_to_adamw_by_name(param_name)
            if p.ndim == 2 and not getattr(p, "_pion_skip", False) and not force_adamw:
                print(
                    f"Pion matrix param {param_name} shape={tuple(p.shape)}",
                )
                matrix_params.append(p)
            else:
                print(
                    f"Pion AdamW param {param_name} shape={tuple(p.shape)} ndim={p.ndim} type={type(p).__name__}",
                )
                vector_params.append(p)

        is_per_head_fn = lambda p: bool(getattr(p, "_pion_per_head", False))

        if hd == 0:
            for p in param_list:
                if not p.requires_grad or p.ndim != 2:
                    continue
                hdp = getattr(p, "_pion_head_dim", None)
                if hdp is not None and int(hdp) > 0:
                    hd = int(hdp)
                    break

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False,
            is_pion=False,
            degree=degree,
        )
        groups = []
        if matrix_params:
            groups.append(
                {
                    "params": matrix_params,
                    "is_pion": True,
                    "lr": lr,
                    "betas": betas,
                    "weight_decay": 0.0,
                    "degree": degree,
                }
            )
        if vector_params:
            groups.append(
                {
                    "params": vector_params,
                    "is_pion": False,
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "amsgrad": False,
                    "degree": degree,
                }
            )

        if not groups:
            raise ValueError(
                "PionOptimizer: no trainable parameters (requires_grad=True). "
                "Check parameter list or masks."
            )

        trainable_nonempty = [p for p in param_list if p.requires_grad and p.numel() > 0]
        if trainable_nonempty and not matrix_params:
            raise RuntimeError(
                "PionOptimizer: optimizer 参数列表里没有任何 2D 权重。请确认："
                "(1) 使用的是 FSDP1（actor.strategy=fsdp），且传给 torch.distributed.fsdp.FSDP 的 "
                "use_orig_params 为 True；"
                "(2) fsdp_config.use_orig_params 在 OmegaConf 合并后为布尔 true（不要用未加引号的奇怪字符串）；"
                "(3) 若仍失败，看训练日志里 [FSDP actor] 一行与嵌套 FSDP 的 _use_orig_params 校验报错。"
            )

        super().__init__(groups, defaults)

        self._pion_core: Optional[_PionMatrixCore] = None
        if matrix_params:
            self._pion_core = _PionMatrixCore(
                head_dim=hd,
                per_head_fn=is_per_head_fn,
            )

    def _adamw_step_group(self, group: Dict[str, Any]) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("PionOptimizer does not support sparse gradients")
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            st = state["step"]
            prev = int(st.item()) if isinstance(st, torch.Tensor) else int(st)
            state["step"] = prev + 1
            step = state["step"]

            if group["weight_decay"] != 0:
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = group["lr"] / bias_correction1

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("is_pion", False):
                assert self._pion_core is not None
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self._pion_core.pion_update_for_matrix(
                        p, p.grad.data, group, self.state[p]
                    )
            else:
                self._adamw_step_group(group)

        return loss


__all__ = [
    "PionOptimizer",
    "assert_fsdp_orig_params_effective_for_pion",
    "pion_optimizer_requires_use_orig_params",
    "pion_optimizer_selected_from_config",
    "prepare_pion_module_tags",
    "tag_parameters_for_pion",
    "spectral_norm_power",
]
