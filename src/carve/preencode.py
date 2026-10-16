"""Encode colored trajectory pairs with the same TAEH3 for all visual roles."""

import gc
import json
from pathlib import Path

import numpy as np
import torch

from .data import asset, digest, read_video, validate_raw
from .palette import palette_bytes, palette_identity
from .prompt import STATIC_PROMPT, static_prompt_hash
from .tae_proxy import resize_proxy_pixels
from .v4_data import MODALITIES, SCHEMA, TAEH3VisualEncoder, V4EncodedDataset, codec


@torch.inference_mode()
def encode_qwen_condition(modules, images, prompt_text, *, device, fps=24):
    """Only the reference image and released static prompt enter Qwen."""
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3Ref2VATextEncoderStep,
        get_qwen3vl_prompt_embeds,
    )
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference

    refs = [MiniMaxH3ImageReference(image=image) for image in images]
    step = MiniMaxH3Ref2VATextEncoderStep(video_sample_fps=2.0)
    vision, image_counts, video_counts, timestamps = step._gather_vision_features(
        modules.processor, refs, fps,
    )
    tokens, tags = step._build_presentation(
        modules.tokenizer, prompt_text, refs, image_counts, video_counts, timestamps,
    )
    prompt = get_qwen3vl_prompt_embeds(
        modules.text_encoder, modules.processor, tokens, vision,
        text_encoder_layer=50, device=torch.device(device), dtype=torch.bfloat16,
    )
    return {
        "prompt_embeds": prompt[0].cpu().contiguous(),
        "text_token_tags": torch.tensor(tags, dtype=torch.long),
    }


@torch.inference_mode()
def encode_text_audio_condition(modules, images, *, device):
    if len(images) != 1:
        raise ValueError("The released static prompt requires one reference image")
    result = encode_qwen_condition(modules, images, STATIC_PROMPT, device=device)
    waveform = torch.zeros(2, 1, 32000, device=device)
    audio = modules.audio_vae.encode(waveform, return_dict=False)[0].mode().float().cpu()
    mean = torch.tensor(modules.audio_vae.config.latents_mean).view(1, -1, 1)
    std = torch.tensor(modules.audio_vae.config.latents_std).view(1, -1, 1)
    result["silence_rows"] = (
        ((audio - mean) / std).transpose(1, 2).reshape(-1, 32).bfloat16().contiguous()
    )
    return result


def fit_reference(path, width=960, height=544):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    factor = min(1.0, width / image.width, height / image.height)
    size = (max(32, int(image.width * factor) // 32 * 32),
            max(32, int(image.height * factor) // 32 * 32))
    return image.resize(size, Image.Resampling.LANCZOS)


@torch.inference_mode()
def encode_reference(encoder, image):
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None]
    return encoder.encode_image(pixels)[0].cpu().bfloat16().contiguous()


def group_raw_pairs(rows):
    """One coarse/fine pair must share the original reference and target."""
    pairs = {}
    shared = ("scene_id", "source_scene_id", "split", "references", "prompt",
              "target", "fps", "num_frames", "width", "height")
    for row in rows:
        variant, sid = row.get("proxy_variant"), row.get("source_sample_id")
        if variant not in MODALITIES or not isinstance(sid, str) or not sid.strip():
            raise ValueError("v4 raw inputs require source_sample_id and coarse/fine proxy_variant")
        if (row["width"], row["height"], row["fps"]) != (960, 544, 24):
            raise ValueError("v4 requires 960x544 target video at 24 fps")
        pair = pairs.setdefault(sid, {})
        if variant in pair:
            raise ValueError("Repeated proxy variant in one trajectory")
        pair[variant] = row
    for pair in pairs.values():
        if set(pair) != set(MODALITIES):
            raise ValueError("Missing colored coarse/fine proxy variant")
        a, b = [pair[v] for v in MODALITIES]
        if any(a.get(k) != b.get(k) for k in shared):
            raise ValueError("Coarse/fine trajectories or target/reference differ")
    return [(sid, pairs[sid]) for sid in sorted(pairs)]


@torch.inference_mode()
def run(
    manifest, output, checkpoint, checkpoint_id, *, tae_assets, device="cuda:0",
    seed=42, proxy_height=352, proxy_width=608, palette=None,
):
    from safetensors.torch import load_file, save_file
    from solarwm.backends.minimax_h3.optional import load_conditioners

    pairs = group_raw_pairs(validate_raw(manifest))
    if (proxy_height, proxy_width) != (352, 608):
        raise ValueError("v4 uniform TAEH3 requires 608x352 proxy pixels")
    colors = palette_bytes(palette)
    root, out = Path(manifest).resolve().parent, Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preencoding requires an empty output directory")
    identity = codec({
        "checkpoint_id": checkpoint_id, "qwen_proxy_scope": "reference_only",
        "qwen_layer": 50, "static_prompt_sha256": static_prompt_hash(),
        "silence_seconds": 1, **palette_identity(),
    })
    out.mkdir(parents=True, exist_ok=True)
    modules = load_conditioners(
        {"checkpoint_path": str(checkpoint), "torch_dtype": "bfloat16"},
        device=torch.device(device),
    )
    modules.audio_vae.set_attention_backend("native")
    references, records = {}, []
    for sid, pair in pairs:
        row = pair["coarse"]
        reference_path = asset(root, row["references"][0])
        key = digest(reference_path)
        if key not in references:
            image = fit_reference(reference_path)
            name = f"condition-{len(references):06d}.safetensors"
            save_file(
                encode_text_audio_condition(modules, [image], device=device),
                str(out / name), metadata={"codec": json.dumps(identity)},
            )
            references[key] = (name, reference_path, image.size)
        name, _, (ref_width, ref_height) = references[key]
        records.append({
            "schema": SCHEMA, "dataset_version": "v4", "sample_id": sid,
            "source_sample_id": sid, "scene_id": row["scene_id"],
            "source_scene_id": row.get("source_scene_id", row["scene_id"]),
            "split": row["split"], "fps": row["fps"], "num_frames": row["num_frames"],
            "height": row["height"], "width": row["width"], "codec": identity,
            "condition": name, "latents": f"target-{len(records):07d}.safetensors",
            "proxy_latents": f"proxy-{len(records):07d}.safetensors",
            "num_references": 1, "reference_height": ref_height, "reference_width": ref_width,
            "proxy_height": 352, "proxy_width": 608, "proxy_variants": list(MODALITIES),
            "source_hashes": {
                "references": [key], "target": digest(asset(root, row["target"])),
                "proxies": {v: digest(asset(root, pair[v]["proxy"])) for v in MODALITIES},
                "prompt": static_prompt_hash(),
            },
            **palette_identity(),
        })
    # Release the native conditioners before loading the uniform visual encoder.
    del modules
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    encoder = TAEH3VisualEncoder(tae_assets, device=device)
    for name, reference_path, _ in references.values():
        static = load_file(str(out / name))
        static["reference_000"] = encode_reference(encoder, fit_reference(reference_path))
        save_file(static, str(out / name), metadata={"codec": json.dumps(identity)})
    for (_, pair), record in zip(pairs, records, strict=True):
        row = pair["coarse"]
        pixels = torch.from_numpy(read_video(asset(root, row["target"]), row)).permute(
            3, 0, 1, 2,
        )[None]
        target = encoder.encode_video(pixels)[0].cpu().bfloat16().contiguous()
        save_file({"target_latents": target}, str(out / record["latents"]))
        del pixels, target
        proxies = {}
        for variant in MODALITIES:
            source = pair[variant]
            pixels = torch.from_numpy(read_video(asset(root, source["proxy"]), source)).permute(
                3, 0, 1, 2,
            )[None]
            pixels = resize_proxy_pixels(pixels, 352, 608)
            proxies[f"{variant}_latents"] = encoder.encode_video(pixels)[0].cpu().bfloat16().contiguous()
        save_file(proxies, str(out / record["proxy_latents"]))
        print(json.dumps({"encoded": record["sample_id"], "visual_encoder": "taeh3"}), flush=True)
    (out / "prompt.txt").write_text(STATIC_PROMPT + "\n")
    (out / "palette.json").write_bytes(colors)
    temporary = out / "manifest.jsonl.tmp"
    temporary.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in records))
    dataset = V4EncodedDataset(temporary)
    for index in range(len(dataset)):
        dataset[index]
    temporary.replace(out / "manifest.jsonl")
    (out / "COMPLETE.json").write_text(json.dumps({
        "schema": SCHEMA, "trajectories": len(records),
        "manifest_sha256": digest(out / "manifest.jsonl"),
        "codec": identity, "seed": seed,
    }, indent=2) + "\n")
