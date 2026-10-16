"""Backbone-independent torch primitives for AnyFlow v1.5."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

GatherResult = torch.Tensor | Sequence[torch.Tensor]
GatherFn = Callable[[torch.Tensor], GatherResult]


def apply_timestep_shift(normalized: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the FlowMatch/AnyFlow rational shift to normalized times."""
    if not torch.is_tensor(normalized):
        normalized = torch.as_tensor(normalized, dtype=torch.float32)
    if not torch.is_floating_point(normalized):
        normalized = normalized.to(torch.float32)
    if not isinstance(shift, (int, float)) or not torch.isfinite(torch.tensor(float(shift))):
        raise ValueError(f"shift must be finite, got {shift!r}")
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    if shift == 1.0:
        return normalized
    return shift * normalized / (1.0 + (shift - 1.0) * normalized)


def bounded_difference_timesteps(
    t: torch.Tensor,
    r: torch.Tensor,
    *,
    epsilon: float = 5.0,
    num_train_timesteps: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FAR v1.5 finite-difference window bounded by ``[r, N]``."""
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not isinstance(num_train_timesteps, int) or num_train_timesteps <= 0:
        raise ValueError(
            f"num_train_timesteps must be a positive integer, got {num_train_timesteps!r}"
        )
    t_tensor = torch.as_tensor(t)
    r_tensor = torch.as_tensor(r, device=t_tensor.device, dtype=t_tensor.dtype)
    try:
        t_tensor, r_tensor = torch.broadcast_tensors(t_tensor, r_tensor)
    except RuntimeError as exc:
        raise ValueError(
            f"t and r are not broadcastable: {tuple(t_tensor.shape)} vs {tuple(r_tensor.shape)}"
        ) from exc
    t_plus = (t_tensor + epsilon).clamp_max(num_train_timesteps)
    t_minus = torch.maximum(t_tensor - epsilon, r_tensor)
    return t_plus, t_minus


@torch.no_grad()
def central_difference_derivative(
    u_plus: torch.Tensor,
    u_minus: torch.Tensor,
    t_plus: torch.Tensor,
    t_minus: torch.Tensor,
) -> torch.Tensor:
    """Compute FAR v1.5's derivative using the actual bounded window width."""
    if u_plus.shape != u_minus.shape:
        raise ValueError(
            "u_plus and u_minus must have identical shapes, got "
            f"{tuple(u_plus.shape)} and {tuple(u_minus.shape)}"
        )
    plus = torch.as_tensor(t_plus, device=u_plus.device, dtype=torch.float32)
    minus = torch.as_tensor(t_minus, device=u_plus.device, dtype=torch.float32)
    try:
        plus, minus = torch.broadcast_tensors(plus, minus)
    except RuntimeError as exc:
        raise ValueError(
            f"t_plus and t_minus are not broadcastable: {tuple(plus.shape)} vs {tuple(minus.shape)}"
        ) from exc
    denominator = plus - minus
    denominator = _append_trailing_singletons(denominator.clamp_min(1e-6), u_plus.ndim)
    return ((u_plus.detach().float() - u_minus.detach().float()) / denominator).detach()


@torch.no_grad()
def gaussian_timestep_weights(
    shifted_raw_timesteps: torch.Tensor,
    *,
    shift: float,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Match FAR v1.5's mean-normalized Gaussian (``bsmntw``) grid."""
    if not torch.is_tensor(shifted_raw_timesteps):
        shifted_raw_timesteps = torch.as_tensor(shifted_raw_timesteps)
    if not isinstance(num_train_timesteps, int) or num_train_timesteps <= 0:
        raise ValueError(
            f"num_train_timesteps must be a positive integer, got {num_train_timesteps!r}"
        )
    device = shifted_raw_timesteps.device
    work_dtype = torch.float64 if shifted_raw_timesteps.dtype == torch.float64 else torch.float32
    normalized_grid = torch.linspace(
        1.0,
        0.0,
        num_train_timesteps + 1,
        dtype=work_dtype,
        device=device,
    )[:-1]
    shifted_grid = apply_timestep_shift(normalized_grid, shift)
    raw_grid = shifted_grid * num_train_timesteps
    unnormalized = torch.exp(
        -2.0 * ((raw_grid - num_train_timesteps / 2.0) / num_train_timesteps).square()
    )
    unnormalized = unnormalized - unnormalized.min()
    weight_sum = unnormalized.sum()
    if not bool(torch.isfinite(weight_sum)) or not bool(weight_sum > 0):
        raise ValueError("gaussian training-grid weights have invalid normalization")
    grid_weights = unnormalized * (num_train_timesteps / weight_sum)

    flat = shifted_raw_timesteps.to(dtype=work_dtype).reshape(-1)
    nearest = torch.argmin((raw_grid[:, None] - flat[None, :]).abs(), dim=0)
    return grid_weights[nearest].reshape(shifted_raw_timesteps.shape)


def adaptive_rescale_non_diffusion_losses(
    per_sample_loss: torch.Tensor,
    is_diffusion: torch.Tensor,
    *,
    gather_fn: GatherFn | None = None,
    process_group: object | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """AnyFlow adaptive rescaling with empty-class-safe logical-DP gather.

    Diffusion losses stay unchanged.  Each non-diffusion loss is scaled toward
    the finite logical-global mean diffusion loss, matching the reference
    ``mean_diffusion / (loss.detach() + eps)`` rule.  If the logical batch has
    no finite diffusion reference (or the local batch has no non-diffusion
    samples), this is an identity operation instead of producing NaN.

    ``process_group`` must be the logical DP group, excluding SP replicas.
    Tests and non-distributed callers may inject ``gather_fn(tensor)`` instead.
    """
    if gather_fn is not None and process_group is not None:
        raise ValueError("provide at most one of gather_fn and process_group")
    if per_sample_loss.ndim != 1 or is_diffusion.ndim != 1:
        raise ValueError("per_sample_loss and is_diffusion must both be 1-D")
    if per_sample_loss.shape != is_diffusion.shape:
        raise ValueError(
            "per_sample_loss and is_diffusion shapes differ: "
            f"{tuple(per_sample_loss.shape)} vs {tuple(is_diffusion.shape)}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    is_diffusion = is_diffusion.to(device=per_sample_loss.device, dtype=torch.bool)
    global_loss = _gather_flattened(
        per_sample_loss, gather_fn=gather_fn, process_group=process_group
    )
    global_is_diffusion = _gather_flattened(
        is_diffusion, gather_fn=gather_fn, process_group=process_group
    ).to(dtype=torch.bool)
    if global_loss.numel() != global_is_diffusion.numel():
        raise ValueError(
            "gathered loss and sample-type mask lengths differ: "
            f"{global_loss.numel()} vs {global_is_diffusion.numel()}"
        )

    with torch.no_grad():
        candidates = global_loss[global_is_diffusion]
        candidates = candidates[torch.isfinite(candidates)]
        if candidates.numel() == 0 or bool(is_diffusion.all()):
            return per_sample_loss
        reference = candidates.mean().to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
        denominator = per_sample_loss.detach() + eps
        scale = reference / denominator
        scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
    return torch.where(is_diffusion, per_sample_loss, per_sample_loss * scale)


def _append_trailing_singletons(value: torch.Tensor, ndim: int) -> torch.Tensor:
    if value.ndim > ndim:
        raise ValueError(f"cannot broadcast shape {tuple(value.shape)} to ndim={ndim}")
    return value.reshape(*value.shape, *((1,) * (ndim - value.ndim)))


def _gather_flattened(
    tensor: torch.Tensor,
    *,
    gather_fn: GatherFn | None,
    process_group: object | None,
) -> torch.Tensor:
    detached = tensor.detach()
    if gather_fn is not None:
        gathered = gather_fn(detached)
        if torch.is_tensor(gathered):
            return gathered.to(device=tensor.device).reshape(-1)
        if not isinstance(gathered, Sequence) or len(gathered) == 0:
            raise ValueError("gather_fn must return a tensor or a non-empty tensor sequence")
        return torch.cat([part.to(device=tensor.device).reshape(-1) for part in gathered], dim=0)

    if process_group is not None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "process_group was provided but torch.distributed is not initialized"
            )
        world_size = torch.distributed.get_world_size(group=process_group)
        parts = [torch.empty_like(detached) for _ in range(world_size)]
        torch.distributed.all_gather(parts, detached, group=process_group)
        return torch.cat([part.reshape(-1) for part in parts], dim=0)
    return detached.reshape(-1)
