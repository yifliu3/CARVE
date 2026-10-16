"""Native H3 three-axis rotary position grids used by the v4 token layout."""

from __future__ import annotations

import math

import numpy as np


def spatial_position_axis(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    """Build the provider-compatible float64 endpoint-excluded spatial axis."""

    if dim <= 0 or patch <= 0 or dim % patch:
        raise ValueError(f"dim={dim} must be positive and divisible by patch={patch}")
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * 32


def frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int = 2,
    patch_w: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened ``(h,w)`` coordinates and the width axis for a frame."""

    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = spatial_position_axis(latent_height, patch_h, sqrt_area)
    width_grid = spatial_position_axis(latent_width, patch_w, sqrt_area)
    grid_h, grid_w = np.meshgrid(height_grid, width_grid, indexing="ij")
    return np.stack((grid_h.reshape(-1), grid_w.reshape(-1)), axis=-1), width_grid


def temporal_position_grid(num_latent_frames: int, origin: float = 0.0) -> np.ndarray:
    """Return H3's non-uniform ``5/3*(1,4,4,4,4)`` rotary-time grid."""

    if num_latent_frames < 0:
        raise ValueError(f"num_latent_frames must be non-negative, got {num_latent_frames}")
    spans = np.asarray(
        [(5.0 / 3.0) * (1, 4, 4, 4, 4)[index % 5] for index in range(num_latent_frames)],
        dtype=np.float64,
    )
    if num_latent_frames == 0:
        return spans
    result = np.empty(num_latent_frames, dtype=np.float64)
    result[0] = float(origin)
    if num_latent_frames > 1:
        result[1:] = float(origin) + np.cumsum(spans[:-1])
    return result


def native_video_position_grid(
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    origin: float = 0.0,
) -> np.ndarray:
    """Return flattened native H3 ``(t,h,w)`` IDs in frame-major order."""

    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise ValueError("H3 MM-RoPE layout requires temporal patch_size=1")
    frame_grid, _ = frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    times = temporal_position_grid(num_latent_frames, origin)
    result = np.empty((num_latent_frames, frame_grid.shape[0], 3), dtype=np.float64)
    result[:, :, 0] = times[:, None]
    result[:, :, 1:] = frame_grid[None]
    return result.reshape(-1, 3)
