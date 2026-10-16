"""One token layout shared by bidirectional, TF, replay and incremental execution."""

from dataclasses import dataclass

import torch

from solarwm.backends.minimax_h3.geometry import native_video_position_grid
from solarwm.backends.minimax_h3.layout import patchify_video, unpatchify_video


@dataclass(frozen=True)
class Run:
    role: str
    chunk: int
    start: int
    stop: int


@dataclass
class Layout:
    positions: torch.Tensor
    tags: torch.Tensor
    time_ids: torch.Tensor
    text_indices: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    runs: tuple[Run, ...]
    output_slice: slice
    origin: float
    height: int
    width: int
    chunk_size: int

    @property
    def length(self):
        return self.positions.shape[0]

    def kwargs(self):
        return dict(
            position_ids=self.positions,
            token_tags=self.tags,
            timestep_indices=self.time_ids,
            text_indices=self.text_indices,
            video_indices=self.video_indices,
            audio_indices=self.audio_indices,
        )


def build_layout(
    tags,
    references,
    audio_rows,
    *,
    chunks,
    height,
    width,
    chunk_size=5,
    mode="tf",
    current_chunk=None,
    proxy_height=None,
    proxy_width=None,
):
    """Static prefix; proxy rows; optional clean history; noisy target rows.

    Coordinates stored here are chunk-local. Attention assigns each visible
    chunk its position inside the *query's* sliding window, including cached K.
    Proxy/target clocks are aligned, with a fixed 0.5 role offset for proxies.
    """
    if mode not in {"bid", "tf", "rollout", "commit"}:
        raise ValueError(f"Unknown layout mode {mode}")
    if min(chunks, height, width) <= 0 or height % 2 or width % 2 or chunk_size != 5:
        raise ValueError("Positive dimensions, even latent H/W and five-latent chunks required")
    device = tags.device
    positions, modalities, times, runs = [], [], [], []
    video, audio, text = [], [], []
    cursor = 0

    def append(pos, modal, tid, role=None, chunk=-1, kind="video"):
        nonlocal cursor
        n = len(pos)
        positions.append(pos)
        modalities.extend([modal] * n if isinstance(modal, int) else modal)
        times.extend([tid] * n)
        if role:
            runs.append(Run(role, chunk, cursor, cursor + n))
        if kind == "video":
            video.extend(range(cursor, cursor + n))
        elif kind == "audio":
            audio.extend(range(cursor, cursor + n))
        elif kind == "text":
            text.extend(range(cursor, cursor + n))
        cursor += n

    text_pos = torch.zeros(tags.numel(), 3, dtype=torch.float64)
    text_pos[:, 0] = torch.arange(tags.numel())
    append(text_pos, tags.cpu().tolist(), 0, kind="text")
    for index, ref in enumerate(references):
        if ref.shape[2] != 1:
            raise ValueError("Static appearance references must each contain one latent frame")
        pos = native_video_position_grid(
            1, ref.shape[-2], ref.shape[-1], origin=float(tags.numel() + index)
        )
        append(torch.from_numpy(pos), 0, 0)
    ap = torch.zeros(audio_rows, 3, dtype=torch.float64)
    ap[:, 0] = float(tags.numel()) + torch.arange(audio_rows) % max(audio_rows // 2, 1)
    if audio_rows:
        append(ap, 2, 0, kind="audio")
    runs.append(Run("static", -1, 0, cursor))
    origin = float(tags.numel() + len(references) + 1)
    base = torch.from_numpy(native_video_position_grid(chunk_size, height, width, origin=origin))
    proxy_height = height if proxy_height is None else proxy_height
    proxy_width = width if proxy_width is None else proxy_width
    if min(proxy_height, proxy_width) < 2 or proxy_height % 2 or proxy_width % 2:
        raise ValueError("Proxy latent H/W must be positive and divisible by two")
    proxy_base = (
        base
        if (proxy_height, proxy_width) == (height, width)
        else torch.from_numpy(
            native_video_position_grid(chunk_size, proxy_height, proxy_width, origin=origin)
        )
    )
    active = range(chunks) if current_chunk is None else [current_chunk]
    for i in active:
        pos = proxy_base.clone()
        pos[:, 0] += 0.5
        append(pos, 0, 0, "proxy", i)
    if mode == "tf":
        for i in range(chunks):
            append(base.clone(), 0, 0, "clean", i)
    offset = len(video)
    role = "clean" if mode == "commit" else "noisy"
    for i in active:
        append(base.clone(), 0, 0 if role == "clean" else 1, role, i)
    return Layout(
        torch.cat(positions).to(device),
        torch.tensor(modalities, dtype=torch.long, device=device),
        torch.tensor(times, dtype=torch.long, device=device),
        torch.tensor(text, dtype=torch.long, device=device),
        torch.tensor(video, dtype=torch.long, device=device),
        torch.tensor(audio, dtype=torch.long, device=device),
        tuple(runs),
        slice(offset, len(video)),
        origin,
        height,
        width,
        chunk_size,
    )


def visible(query: Run, key: Run, *, mode: str, history: int):
    if mode == "bid":
        return True
    if query.role == "static":
        return key.role == "static"
    if key.role == "static":
        return True
    first = max(0, query.chunk - history)
    if not first <= key.chunk <= query.chunk:
        return False
    if query.role == "proxy":
        return key.role == "proxy"
    if key.role == "proxy":
        return True
    if query.role == "clean":
        return key.role == "clean"
    return (key.role == "clean" and key.chunk < query.chunk) or (
        key.role == "noisy" and key.chunk == query.chunk
    )


__all__ = ["Layout", "Run", "build_layout", "patchify_video", "unpatchify_video", "visible"]
