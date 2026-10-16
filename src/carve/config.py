"""Configuration and checkpoint identity for the released v4 uniform TAEH3 model."""

import copy
import json
import math
from pathlib import Path

import yaml

DEFAULT_STAGES = ("bidirectional_fm", "causal_tf_anyflow", "causal_dmd_sgf")
STAGES = set(DEFAULT_STAGES)
DEFAULTS = {
    "schema": "cwm-h3.run.v1",
    "stage": "bidirectional_fm",
    "model": {
        "checkpoint": "models/Ref2VA",
        "checkpoint_id": "minimax-h3-ref2va-hymm-ef520db195ef3286",
        "subfolder": "transformer_ref",
        "lora_rank": 512,
        "lora_alpha": 512,
        "history_chunks": 5,
        "activation_checkpointing": True,
        "activation_checkpointing_exempt_layers": 0,
        "attention_checkpointing": True,
        "tiny": False,
        "camera_conditioning": "none",
        "proxy_conditioning": "tae_tokens",
        "proxy_projection_trainable": False,
        "proxy_input_height": 352,
        "proxy_input_width": 608,
    },
    "data": {
        "manifest": "data/encoded/manifest.jsonl",
        "score_manifest": "",
        "split": "train",
        "qwen_dynamic_proxy": False,
        "qwen_proxy_scope": "reference_only",
        "visual_encoder": "taeh3",
        "proxy_variant": "mixed",
    },
    "distributed": {"sp_size": 8, "fsdp_sharding": "full_shard"},
    "train": {
        "max_steps": 2000,
        "grad_accum": 1,
        "learning_rate": 1e-4,
        "critic_learning_rate": 4e-7,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "warmup_steps": 500,
        "seed": 42,
        "save_every": 200,
        "ema_decay": 0.9999,
        "critic_updates_per_student": 5,
        "denoise_steps": 4,
        "video_shift": 12.0,
        "anyflow_gate": 0.25,
    },
    "checkpoint": {
        "initialize": "",
        "teacher": "",
        "resume": "",
        "initialize_weights": "live",
        "teacher_weights": "live",
        "allow_conditioning_transition": False,
        "shared_optimizer": True,
        "async_save": False,
        "local_dir": "",
    },
    "runtime": {
        "output_dir": "outputs/stage0p5",
        "device": "cuda",
        "tensorboard_dir": "",
        "optimizer_state_offload": False,
        "experiment_version": "v4-colored-uniform-taeh3-tokens608-frozen-proj",
        "experiment_id": "",
    },
}


def resolve(path, overrides=()):
    path = Path(path)
    raw = (
        json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    )

    def merge(base, extra):
        if not isinstance(extra, dict):
            raise ValueError("Configuration sections must be mappings")
        for key, value in extra.items():
            if key not in base:
                raise ValueError(f"Unknown configuration field: {key}")
            if isinstance(base[key], dict):
                merge(base[key], value)
            else:
                base[key] = value

    cfg = copy.deepcopy(DEFAULTS)
    merge(cfg, raw)
    for override in overrides:
        key, value = override.split("=", 1)
        target = cfg
        fields = key.split(".")
        for part in fields[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown override: {key}")
            target = target[part]
        if fields[-1] not in target:
            raise ValueError(f"Unknown override: {key}")
        target[fields[-1]] = yaml.safe_load(value)
    if cfg["schema"] != DEFAULTS["schema"] or cfg["stage"] not in STAGES:
        raise ValueError("Unsupported CARVE schema/stage")
    for section, keys in (
        ("model", ("lora_rank", "lora_alpha", "history_chunks")),
        (
            "train",
            (
                "max_steps",
                "grad_accum",
                "save_every",
                "critic_updates_per_student",
                "denoise_steps",
            ),
        ),
        ("distributed", ("sp_size",)),
    ):
        for key in keys:
            if type(cfg[section][key]) is not int or cfg[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for key in ("activation_checkpointing_exempt_layers",):
        if type(cfg["model"][key]) is not int or cfg["model"][key] < 0:
            raise ValueError(f"model.{key} must be nonnegative")
    if type(cfg["train"]["warmup_steps"]) is not int or cfg["train"]["warmup_steps"] < 0:
        raise ValueError("train.warmup_steps must be nonnegative")
    for key in ("learning_rate", "critic_learning_rate", "grad_clip", "video_shift"):
        value = cfg["train"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"train.{key} must be finite and positive")
    if not 0 <= cfg["train"]["ema_decay"] < 1 or not 0 <= cfg["train"]["anyflow_gate"] <= 1:
        raise ValueError("Invalid EMA decay or AnyFlow gate")
    if cfg["runtime"]["device"] not in {"cpu", "cuda"}:
        raise ValueError("runtime.device must be cpu or cuda")
    if type(cfg["runtime"]["optimizer_state_offload"]) is not bool:
        raise ValueError("runtime.optimizer_state_offload must be a boolean")
    if cfg["data"]["split"] not in {"train", "val", "test"}:
        raise ValueError("data.split must be train, val or test")
    if cfg["distributed"]["fsdp_sharding"] not in {"full_shard", "hybrid_shard"}:
        raise ValueError("Unsupported FSDP sharding")
    for key in ("initialize_weights", "teacher_weights"):
        if cfg["checkpoint"][key] not in {"live", "ema"}:
            raise ValueError(f"checkpoint.{key} must be live or ema")
    for key in ("tiny", "activation_checkpointing", "attention_checkpointing"):
        if type(cfg["model"][key]) is not bool:
            raise ValueError(f"model.{key} must be a boolean")
    for key in ("shared_optimizer", "async_save", "allow_conditioning_transition"):
        if type(cfg["checkpoint"][key]) is not bool:
            raise ValueError(f"checkpoint.{key} must be a boolean")
    if cfg["checkpoint"]["allow_conditioning_transition"]:
        raise ValueError("v4 requires the same conditioning contract across stages")
    if cfg["checkpoint"]["async_save"]:
        local = cfg["checkpoint"]["local_dir"]
        if not cfg["checkpoint"]["shared_optimizer"] or not local:
            raise ValueError("Async checkpoints require shared_optimizer and local_dir")
        if Path(local).resolve() == Path(cfg["runtime"]["output_dir"]).resolve():
            raise ValueError("Checkpoint staging must differ from output_dir")
    if not cfg["model"]["tiny"] and not cfg["model"]["checkpoint_id"]:
        raise ValueError("Set a stable model.checkpoint_id")
    validate_conditioning(cfg)
    return cfg


def validate_conditioning(cfg):
    mc, dc = cfg["model"], cfg["data"]
    if (
        mc["proxy_conditioning"] != "tae_tokens"
        or mc["proxy_projection_trainable"] is not False
        or mc["camera_conditioning"] != "none"
        or dc["qwen_dynamic_proxy"] is not False
        or dc["qwen_proxy_scope"] != "reference_only"
        or dc["score_manifest"]
        or dc["visual_encoder"] != "taeh3"
    ):
        raise ValueError("v4 requires uniform TAEH3, frozen proj_in and reference-only Qwen")
    if dc["proxy_variant"] not in {"mixed", "coarse", "fine"}:
        raise ValueError("data.proxy_variant must be mixed, coarse or fine")
    if (mc["proxy_input_height"], mc["proxy_input_width"]) != (352, 608):
        raise ValueError("v4 uniform TAEH3 requires 608x352 proxy pixels")


def score_config(cfg):
    if cfg["stage"] != "causal_dmd_sgf":
        raise ValueError("Score configuration requires causal_dmd_sgf")
    result = copy.deepcopy(cfg)
    result["stage"] = "bidirectional_fm"
    validate_conditioning(result)
    return result


def model_contract(cfg):
    """Keep the upstream v4 checkpoint identity byte-for-byte compatible."""
    from .prompt import static_prompt_hash
    from .tae_proxy import TAE_COMMIT, TAE_SOURCE_SHA256, TAE_TOKENS_CONTRACT, TAE_WEIGHTS_SHA256

    validate_conditioning(cfg)
    mc = cfg["model"]
    return {
        "schema": "cwm-h3.model.v1",
        "checkpoint_id": mc["checkpoint_id"],
        "subfolder": mc["subfolder"],
        "rank": mc["lora_rank"],
        "alpha": mc["lora_alpha"],
        "history_chunks": mc["history_chunks"],
        "chunk_latents": 5,
        "proxy_role_offset": 0.5,
        "static_condition_time": 1.0,
        "qwen_dynamic_proxy": False,
        "camera_prope": False,
        "anyflow_gate": cfg["train"]["anyflow_gate"],
        "tiny": mc["tiny"],
        "proxy_conditioning": "tae_tokens",
        "proxy_contract": TAE_TOKENS_CONTRACT,
        "proxy_latent_input": True,
        "proxy_history_tokens": True,
        "proxy_patch": [1, 2, 2],
        "proxy_injection": "independent_sequence_tokens",
        "proxy_resize": "nearest_before_tae",
        "proxy_input_height": mc["proxy_input_height"],
        "proxy_input_width": mc["proxy_input_width"],
        "proxy_encoder": "taeh3",
        "proxy_encoder_commit": TAE_COMMIT,
        "proxy_encoder_source_sha256": TAE_SOURCE_SHA256,
        "proxy_encoder_weights_sha256": TAE_WEIGHTS_SHA256,
        "proxy_temporal_policy": "zero_prefix3_per17_reset_memory_global_drop3",
        "proxy_encoder_dtype": "float16",
        "proxy_projection_trainable": False,
        "qwen_proxy_scope": "reference_only",
        "static_prompt_sha256": static_prompt_hash(),
        "visual_encoder": "taeh3",
        "visual_encoder_commit": TAE_COMMIT,
        "visual_encoder_source_sha256": TAE_SOURCE_SHA256,
        "visual_encoder_weights_sha256": TAE_WEIGHTS_SHA256,
        "visual_roles": ["reference", "proxy", "target"],
        "pixel_normalization": "rgb_0_1",
        "latent_normalization": "taeh3_output_no_extra_rescale",
        "reference_policy": "repeat_image17_same_video_adapter_first_latent",
    }
