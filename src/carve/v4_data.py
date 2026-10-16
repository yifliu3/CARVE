"""Independent storage contract for colored proxy pairs and uniform TAEH3."""

import json
from pathlib import Path

import torch

from .palette import palette_identity, validate_palette_identity
from .tae_proxy import (
    TAE_COMMIT,
    TAE_SOURCE_SHA256,
    TAE_WEIGHTS_SHA256,
    TAEH3ProxyEncoder,
)

MODALITIES = ("coarse", "fine")
SCHEMA = "cwm-h3.encoded.v4-uniform-taeh3.v1"


def codec(parent, palette_sha256=None):
    from .prompt import static_prompt_hash

    if (parent.get("qwen_proxy_scope") != "reference_only"
            or parent.get("static_prompt_sha256") != static_prompt_hash()):
        raise ValueError("v4 requires the unchanged reference-only structured prompt")
    if not (palette_sha256 or parent.get("proxy_palette_sha256")):
        raise ValueError("Missing v4 semantic palette hash")
    validate_palette_identity({
        **palette_identity(), **parent,
        "proxy_palette_sha256": palette_sha256 or parent["proxy_palette_sha256"],
    })
    result = {
        "schema": "cwm-h3.codec.v4-uniform-taeh3-colored-pair.v1",
        "checkpoint_id": parent["checkpoint_id"],
        "visual_encoder": "taeh3",
        "visual_encoder_commit": TAE_COMMIT,
        "visual_encoder_source_sha256": TAE_SOURCE_SHA256,
        "visual_encoder_weights_sha256": TAE_WEIGHTS_SHA256,
        "visual_encoder_dtype": "float16",
        "visual_storage_dtype": "bfloat16",
        "pixel_normalization": "rgb_0_1",
        "visual_posterior": "deterministic",
        "latent_normalization": "taeh3_output_no_extra_rescale",
        "temporal_policy": "zero_prefix3_per17_reset_memory_global_drop3",
        "reference_policy": "repeat_image17_same_video_adapter_first_latent",
        "reference_resize": "fit-no-upscale-multiple32",
        "proxy_resize": "nearest_before_tae",
        "proxy_height": 352,
        "proxy_width": 608,
        "proxy_modalities": list(MODALITIES),
        "proxy_conditioning": "tae_tokens",
        "proxy_selection": "one_variant_per_condition_not_concatenated",
        "proxy_history_tokens": True,
        "proxy_projection": "shared_native_proj_in",
        "proxy_latent_patch": [1, 2, 2],
        "proxy_color_mode": "semantic_class",
        "proxy_palette_version": "semantic_palette_fixed_v1",
        "proxy_palette_sha256": palette_sha256 or parent["proxy_palette_sha256"],
        "qwen_dynamic_proxy": False,
        "qwen_proxy_scope": "reference_only",
        "qwen_layer": parent["qwen_layer"],
        "static_prompt_sha256": parent["static_prompt_sha256"],
        "silence_seconds": parent["silence_seconds"],
    }
    validate_palette_identity(result)
    return result


def latent_length(num_frames):
    if type(num_frames) is not int or num_frames < 39 or (num_frames - 5) % 17:
        raise ValueError("v4 video requires 17*n+5 frames, n>=2")
    return (num_frames - 5) // 17 * 5 + 2


class TAEH3VisualEncoder(TAEH3ProxyEncoder):
    """Encode references using the identical repeated-image video adapter."""

    @torch.no_grad()
    def encode_image(self, pixels):
        if (pixels.ndim != 5 or pixels.shape[1:3] != (3, 1)
                or pixels.dtype != torch.uint8 or pixels.shape[-2] % 32
                or pixels.shape[-1] % 32):
            raise ValueError("TAEH3 image must be uint8 [B,3,1,H,W], multiple32")
        z = self.encode_chunk(
            pixels.expand(-1, -1, 17, -1, -1).contiguous()
        )[:, :, :1].contiguous()
        expected = (pixels.shape[0], 24, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
        if z.shape != expected or not bool(torch.isfinite(z).all()):
            raise ValueError("Invalid TAEH3 reference latent")
        return z


class V4EncodedDataset:
    """One stored trajectory with two selectable proxies and shared appearance."""

    def __init__(self, manifest, split=None, proxy_variant=None):
        from .data import read_rows, validate_geometry, validate_scene_groups

        self.root = Path(manifest).resolve().parent
        all_rows = read_rows(manifest)
        validate_scene_groups(all_rows)
        if len({r["sample_id"] for r in all_rows}) != len(all_rows):
            raise ValueError("Duplicate v4 sample IDs")
        if proxy_variant not in (None, *MODALITIES):
            raise ValueError("Select coarse or fine proxy")
        self.proxy_variant = proxy_variant
        identities = {json.dumps(r["codec"], sort_keys=True) for r in all_rows}
        if len(identities) != 1:
            raise ValueError("Mixed v4 codec identities")
        self.codec = all_rows[0]["codec"]
        if self.codec != codec(self.codec):
            raise ValueError("Invalid uniform TAEH3 v4 contract")
        for row in all_rows:
            if row["schema"] != SCHEMA or row["dataset_version"] != "v4":
                raise ValueError("v4 requires its own encoded schema and dataset version")
            if row["split"] not in {"train", "val", "test"}:
                raise ValueError("Invalid v4 split")
            if row.get("proxy_variants") != list(MODALITIES) or row.get("num_references", 1) != 1:
                raise ValueError("v4 requires coarse/fine proxies and one reference")
            validate_geometry(row)
            if (row["height"], row["width"], row["fps"]) != (544, 960, 24):
                raise ValueError("Invalid v4 video geometry")
            if (row["proxy_height"], row["proxy_width"]) != (352, 608):
                raise ValueError("Invalid v4 proxy geometry")
            for key in ("reference_height", "reference_width"):
                if type(row[key]) is not int or row[key] < 32 or row[key] % 32:
                    raise ValueError("Invalid v4 reference dimensions")
            if "proxy_color_mode" in row or "proxy_palette_version" in row:
                validate_palette_identity({**self.codec, **row}, require_hash=False)
        self.rows = [r for r in all_rows if split is None or r["split"] == split]
        if not self.rows:
            raise ValueError("Empty v4 dataset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from safetensors.torch import load_file
        from .data import asset

        row = self.rows[index]
        static = load_file(str(asset(self.root, row["condition"])))
        target = load_file(str(asset(self.root, row["latents"])))
        proxies = load_file(str(asset(self.root, row["proxy_latents"])))
        if set(target) != {"target_latents"} or set(proxies) != {f"{m}_latents" for m in MODALITIES}:
            raise ValueError("Unexpected v4 latent payload")
        if set(static) != {"prompt_embeds", "text_token_tags", "silence_rows", "reference_000"}:
            raise ValueError("v4 must retain reference-only static conditioning")
        length = latent_length(row["num_frames"])
        tensors = [target["target_latents"], *proxies.values(), static["reference_000"]]
        if any(z.dtype != torch.bfloat16 or not bool(z.isfinite().all()) for z in tensors):
            raise ValueError("v4 paired proxy/reference/target latents must be finite BF16")
        if tuple(target["target_latents"].shape) != (24, length, 34, 60):
            raise ValueError("v4 target shape differs")
        if any(tuple(z.shape) != (24, length, 22, 38) for z in proxies.values()):
            raise ValueError("Invalid v4 paired proxy shape")
        if static["reference_000"].shape != (
            24, 1, row["reference_height"] // 16, row["reference_width"] // 16,
        ):
            raise ValueError("v4 reference shape differs")
        validate_static(static)
        result = {
            "static": static,
            "target": target["target_latents"],
            "proxies": {m: proxies[f"{m}_latents"] for m in MODALITIES},
        }
        if self.proxy_variant is not None:
            result["proxy"] = result["proxies"][self.proxy_variant]
            result["proxy_variant"] = self.proxy_variant
        return result


def validate_static(static):
    """Shared checks for dataset and interactive reference-only conditions."""
    if set(static) != {"prompt_embeds", "text_token_tags", "silence_rows", "reference_000"}:
        raise ValueError("Static condition requires prompt/tags/silence/reference only")
    prompt, tags, audio, ref = (
        static[k] for k in ("prompt_embeds", "text_token_tags", "silence_rows", "reference_000")
    )
    if (prompt.ndim != 2 or prompt.shape[1] != 5120 or not len(prompt)
            or prompt.dtype != torch.bfloat16 or tags.shape != prompt.shape[:1]
            or tags.dtype != torch.long or not set(tags.tolist()) <= {0, 1}):
        raise ValueError("Invalid static Qwen shape/dtype/tags")
    if (audio.ndim != 2 or audio.shape[1] != 32 or not len(audio)
            or audio.dtype != torch.bfloat16):
        raise ValueError("Invalid encoded silence")
    if (ref.ndim != 4 or tuple(ref.shape[:2]) != (24, 1)
            or any(n < 2 or n % 2 for n in ref.shape[-2:]) or ref.dtype != torch.bfloat16):
        raise ValueError("Invalid static TAEH3 reference latents")
    if any(not bool(t.isfinite().all()) for t in (prompt, audio, ref)):
        raise ValueError("Static condition contains nonfinite tensors")
