"""V4 video patch packing shared by full and incremental execution."""

from __future__ import annotations

import numpy as np


def patchify_video(latents: object, patch_size: tuple[int, int, int] = (1, 2, 2)) -> object:
    """Pack ``[B,C,T,H,W]`` latents into ``[B,N,C*pt*ph*pw]`` rows."""

    is_torch = type(latents).__module__.startswith("torch")
    if is_torch:
        import torch

        if not isinstance(latents, torch.Tensor):
            raise TypeError("unrecognized torch-like latent value")
        values = latents
    else:
        values = np.asarray(latents)
    if values.ndim != 5:
        raise ValueError(f"latents must be [B,C,T,H,W], got {values.shape}")
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = values.shape
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"latents shape {values.shape} is not divisible by {patch_size}")
    rows = values.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    if is_torch:
        rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return rows.reshape(batch, -1, channels * patch_t * patch_h * patch_w).contiguous()
    rows = rows.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return np.ascontiguousarray(rows.reshape(batch, -1, channels * patch_t * patch_h * patch_w))


def unpatchify_video(
    rows: object,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    channels: int | None = None,
) -> object:
    """Invert :func:`patchify_video` into ``[B,C,T,H,W]`` latents."""

    is_torch = type(rows).__module__.startswith("torch")
    if is_torch:
        import torch

        if not isinstance(rows, torch.Tensor):
            raise TypeError("unrecognized torch-like row value")
        values = rows
    else:
        values = np.asarray(rows)
    if values.ndim != 3:
        raise ValueError(f"rows must be [B,N,D], got {values.shape}")
    patch_t, patch_h, patch_w = patch_size
    if num_latent_frames % patch_t or latent_height % patch_h or latent_width % patch_w:
        raise ValueError("requested latent grid is not divisible by patch_size")
    patch_volume = patch_t * patch_h * patch_w
    if channels is None:
        if values.shape[-1] % patch_volume:
            raise ValueError("row width is not divisible by patch volume")
        channels = values.shape[-1] // patch_volume
    expected_rows = (
        (num_latent_frames // patch_t) * (latent_height // patch_h) * (latent_width // patch_w)
    )
    if values.shape[1:] != (expected_rows, channels * patch_volume):
        raise ValueError(
            f"rows shape {values.shape} does not match expected "
            f"N={expected_rows}, D={channels * patch_volume}"
        )
    batch = values.shape[0]
    latents = values.reshape(
        batch,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    if is_torch:
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return latents.reshape(
            batch, channels, num_latent_frames, latent_height, latent_width
        ).contiguous()
    latents = latents.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return np.ascontiguousarray(
        latents.reshape(batch, channels, num_latent_frames, latent_height, latent_width)
    )


__all__ = ["patchify_video", "unpatchify_video"]
