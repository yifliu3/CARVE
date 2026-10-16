"""Bidirectional or causal v4 inference with matching TAEH3 decode."""

import json
from pathlib import Path

import torch

from . import checkpoint
from .core import Core
from .data import EncodedDataset
from .evaluation import generate_test_sample, inference_steps
from .model import build, initialize_distributed


def run(cfg, args):
    if not getattr(args, "tae_assets", None):
        raise ValueError("--tae-assets is required to encode inputs and decode generated video")
    proxy_video = getattr(args, "proxy_video", None)
    reference_image = getattr(args, "reference_image", None)
    if bool(proxy_video) != bool(reference_image):
        raise ValueError("--proxy-video and --reference-image must be provided together")
    if proxy_video and not getattr(args, "tae_assets", None):
        raise ValueError("--tae-assets is required for raw inference inputs")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty inference output directory")
    device, rank, _ = initialize_distributed(cfg)
    root, meta = checkpoint.read(args.weights, cfg)
    stage = meta["stage"]
    if stage != cfg["stage"]:
        raise ValueError("Inference config.stage must match the checkpoint stage")
    steps = inference_steps(cfg, args.steps)
    visible_frames = None
    if proxy_video:
        from .inference_inputs import distributed_condition

        condition, visible_frames = distributed_condition(cfg, args, device=device, rank=rank)
    else:
        dataset = EncodedDataset(
            cfg["data"]["manifest"],
            split=cfg["data"]["split"],
            checkpoint_id=cfg["model"]["checkpoint_id"],
            qwen_dynamic_proxy=False,
            qwen_proxy_scope="reference_only",
            stage=stage,
            camera_conditioning="none",
            proxy_conditioning="tae_tokens",
            visual_encoder=cfg["data"]["visual_encoder"],
            proxy_variant=cfg["data"]["proxy_variant"],
        )
        for axis in ("height", "width"):
            if dataset.codec["proxy_" + axis] != cfg["model"]["proxy_input_" + axis]:
                raise ValueError("Model and dataset proxy input dimensions differ")
        if not 0 <= args.sample_index < len(dataset):
            raise ValueError("sample-index is outside the selected split")
        condition = dataset[args.sample_index].to(device)
        condition.target = None
    model = build(cfg, device, anyflow=meta["anyflow"], frozen=True)
    checkpoint.initialize(model, root, cfg, allowed_stages={stage}, source=args.weight_source)
    core = Core(
        model.model,
        history=cfg["model"]["history_chunks"],
        shift=cfg["train"]["video_shift"],
        anyflow=meta["anyflow"],
    )
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        generated, metrics = generate_test_sample(
            core,
            condition,
            stage=stage,
            steps=steps,
            seed=cfg["train"]["seed"],
            progress=lambda item: print(json.dumps(item), flush=True) if rank == 0 else None,
        )
    if rank == 0:
        if visible_frames is not None:
            metrics.update(
                input_frames=visible_frames,
                padded_frames=generated.shape[2] // 5 * 17 - visible_frames,
            )
        from .tae_decode import TAEH3VideoDecoder

        decoder = TAEH3VideoDecoder(args.tae_assets, device=device)
        decode_causal_video(generated, args.tae_assets, out / "generated.mp4",
                            device=device, decoder=decoder, visible_frames=visible_frames)
        metrics.update(decoder.identity)
        (out / "timings.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def decode_causal_video(latents, tae_assets, output, *, device, visible_frames=None,
                        skip_frames=0, decoder=None):
    """Publish five-latent blocks as 17 RGB frames using the matching visual codec."""
    if (latents.ndim != 5 or latents.shape[:2] != (1, 24)
            or latents.shape[2] < 5 or latents.shape[2] % 5):
        raise ValueError("Causal decode requires complete five-latent blocks")
    if type(skip_frames) is not int or skip_frames < 0:
        raise ValueError("skip_frames must be a nonnegative integer")
    available = latents.shape[2] // 5 * 17 - skip_frames
    visible = available if visible_frames is None else visible_frames
    if type(visible) is not int or not 1 <= visible <= available:
        raise ValueError("visible_frames must fit the generated blocks after skip_frames")
    if not latents.is_floating_point() or not bool(latents.isfinite().all()):
        raise ValueError("Generated latents must be finite floating-point tensors")
    return decode_video(latents, tae_assets, output, device=device, decoder=decoder,
                        visible_frames=visible, skip_frames=skip_frames)


@torch.inference_mode()
def decode_video(latents, tae_assets, output, *, device, visible_frames=None, skip_frames=0,
                 decoder=None):
    import imageio.v2 as imageio
    from .tae_decode import TAEH3VideoDecoder

    if type(skip_frames) is not int or skip_frames < 0:
        raise ValueError("skip_frames must be a nonnegative integer")
    if visible_frames is not None and (type(visible_frames) is not int or visible_frames < 1):
        raise ValueError("visible_frames must be a positive integer")
    decoder = decoder if decoder is not None else TAEH3VideoDecoder(tae_assets, device=device)
    seen = written = 0
    with imageio.get_writer(str(output), fps=24, codec="libx264", macro_block_size=1) as writer:
        for rgb in decoder.decode_chunks(latents):
            for frame in rgb[0].unbind(1):
                seen += 1
                if seen <= skip_frames:
                    continue
                if visible_frames is not None and written == visible_frames:
                    return written
                pixels = (frame.cpu().float().clamp(0, 1) * 255).round().byte()
                writer.append_data(pixels.permute(1, 2, 0).numpy())
                written += 1
    if visible_frames is not None and written != visible_frames:
        raise ValueError(f"Decoder produced {written} frames; expected {visible_frames}")
    return written
