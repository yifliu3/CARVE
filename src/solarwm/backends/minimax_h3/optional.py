"""Lazy Qwen and audio-silence loaders for v4 reference conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solarwm.errors import BackendContractError


@dataclass
class H3RuntimeModules:
    text_encoder: Any = None
    processor: Any = None
    tokenizer: Any = None
    audio_vae: Any = None


def require_h3_runtime() -> tuple[Any, Any, Any]:
    """Resolve heavy packages before allocating the reference conditioners."""

    try:
        import diffusers
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise BackendContractError(
            "V4 reference conditioning requires torch, H3-compatible Diffusers "
            "and Transformers"
        ) from exc
    if not torch.cuda.is_available():
        raise BackendContractError("V4 reference conditioning requires CUDA")
    return torch, diffusers, transformers


def load_conditioners(
    model_cfg: Mapping[str, Any],
    *,
    device: Any,
) -> H3RuntimeModules:
    """Load reference-only Qwen and the audio VAE used for encoded silence."""

    torch, _diffusers, _transformers = require_h3_runtime()
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from transformers import (
        Qwen2TokenizerFast,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )

    root = Path(str(model_cfg["checkpoint_path"]))
    if model_cfg.get("torch_dtype", "bfloat16") != "bfloat16":
        raise BackendContractError("V4 reference conditioners require bfloat16")
    return H3RuntimeModules(
        text_encoder=(
            Qwen3VLForConditionalGeneration.from_pretrained(
                str(root), subfolder="text_encoder", torch_dtype=torch.bfloat16
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        ),
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(root / "tokenizer")),
        processor=Qwen3VLProcessor.from_pretrained(str(root / "processor")),
        audio_vae=(
            AutoencoderKLMiniMaxH3Audio.from_pretrained(
                str(root), subfolder="audio_vae", torch_dtype=torch.bfloat16
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        ),
    )


__all__ = [
    "H3RuntimeModules",
    "load_conditioners",
    "require_h3_runtime",
]
