"""Prepare user-supplied semantic proxy video and appearance reference."""

import gc
from pathlib import Path

import torch

from .core import Condition
from .preencode import encode_reference, encode_text_audio_condition, fit_reference
from .tae_proxy import resize_proxy_pixels
from .v4_data import TAEH3VisualEncoder, validate_static


def proxy_chunks(path):
    """Read constant-rate video in 17-frame blocks, padding only the final block."""
    import av
    import numpy as np

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Proxy input must contain a video stream")
        stream = container.streams.video[0]
        stream.thread_count = 1
        if stream.average_rate is None or abs(float(stream.average_rate) - 24) > 0.02:
            raise ValueError("Proxy video must use constant 24 fps")
        frames, origin, previous = [], None, None
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Proxy video must have constant-rate timestamps")
            timestamp = float(frame.pts * frame.time_base)
            if origin is None:
                origin = timestamp
            tolerance = max(1e-4, min(float(frame.time_base), 1 / 96))
            if (abs(timestamp - origin - index / 24) > tolerance
                    or previous is not None and timestamp <= previous):
                raise ValueError("Proxy video must have constant-rate timestamps")
            previous = timestamp
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == 17:
                yield torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2)[None], 17
                frames = []
        if origin is None:
            raise ValueError("Proxy video is empty")
        if frames:
            visible = len(frames)
            frames.extend([frames[-1]] * (17 - visible))
            yield torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2)[None], visible


@torch.inference_mode()
def prepare_condition(cfg, proxy_video, reference_image, tae_assets, *, device):
    """Encode only the supplied reference and proxy; no target or dataset is needed."""
    from solarwm.backends.minimax_h3.optional import load_conditioners

    for path in (proxy_video, reference_image):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    image = fit_reference(reference_image)
    modules = load_conditioners(
        {"checkpoint_path": cfg["model"]["checkpoint"], "torch_dtype": "bfloat16"},
        device=torch.device(device),
    )
    modules.audio_vae.set_attention_backend("native")
    try:
        static = encode_text_audio_condition(modules, [image], device=device)
    finally:
        del modules
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    encoder = TAEH3VisualEncoder(tae_assets, device=device)
    chunks, visible_frames = [], 0
    try:
        static["reference_000"] = encode_reference(encoder, image)
        validate_static(static)
        for pixels, visible in proxy_chunks(proxy_video):
            pixels = resize_proxy_pixels(
                pixels, cfg["model"]["proxy_input_height"], cfg["model"]["proxy_input_width"],
            )
            latents = encoder.encode_chunk(pixels)
            expected = (1, 24, 5, 22, 38)
            if tuple(latents.shape) != expected or not bool(latents.isfinite().all()):
                raise ValueError("TAEH3 produced invalid proxy latents")
            chunks.append(latents.cpu().bfloat16().contiguous())
            visible_frames += visible
            del pixels, latents
    finally:
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    condition = Condition(
        prompt=static["prompt_embeds"][None],
        tags=static["text_token_tags"],
        references=[static["reference_000"][None]],
        proxy=torch.cat(chunks, dim=2),
        audio=static["silence_rows"][None],
        output_size=(34, 60),
    )
    return condition, visible_frames


def distributed_condition(cfg, args, *, device, rank):
    """Prepare once before model allocation, then share CPU conditions with all ranks."""
    import torch.distributed as dist

    if not dist.is_initialized():
        condition, visible = prepare_condition(
            cfg, args.proxy_video, args.reference_image, args.tae_assets, device=device,
        )
        return condition.to(device), visible
    payload = [None]
    if rank == 0:
        try:
            payload[0] = {
                "prepared": prepare_condition(
                    cfg, args.proxy_video, args.reference_image, args.tae_assets, device=device,
                ),
            }
        except Exception as error:
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(payload, src=0, device=device)
    if "error" in payload[0]:
        raise ValueError(f"Inference input preparation failed: {payload[0]['error']}")
    condition, visible = payload[0]["prepared"]
    return condition.to(device), visible
