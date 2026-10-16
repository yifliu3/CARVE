"""Demo configuration. Relative paths are resolved against the config file."""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "v4-colored-uniform-taeh3-tokens608-frozen-proj"
ENV_PATHS = {
    "data_dir": "CWM_DEMO_DATA_DIR",
    "run_config": "CWM_RUN_CONFIG",
    "checkpoint": "CWM_CHECKPOINT",
    "taeh3_assets": "CWM_TAEH3_ASSETS",
    "static_condition": "CWM_STATIC_CONDITION",
    "static_manifest": "CWM_STATIC_MANIFEST",
    "chromium_executable": "CWM_CHROMIUM_EXECUTABLE",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config(path=None, *, overrides=None):
    path = Path(path or ROOT / "configs/demo.json").resolve()
    cfg = read(path)
    for key, env in ENV_PATHS.items():
        if os.environ.get(env):
            cfg[key] = str(Path(os.environ[env]).expanduser().resolve())
    if os.environ.get("CWM_DEMO_PORT"):
        cfg["port"] = int(os.environ["CWM_DEMO_PORT"])
    if os.environ.get("CWM_STATIC_SAMPLE_ID"):
        cfg["static_sample_id"] = os.environ["CWM_STATIC_SAMPLE_ID"]
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    for key in ENV_PATHS:
        if cfg.get(key):
            value = Path(cfg[key]).expanduser()
            cfg[key] = str((path.parent / value).resolve()) if not value.is_absolute() else str(value)
    if cfg.get("version") != VERSION:
        raise ValueError("Only the released v4 uniform TAEH3 version is supported")
    if (cfg.get("width"), cfg.get("height"), cfg.get("fps"), cfg.get("chunk_frames")) != (960, 544, 24, 17):
        raise ValueError("Demo geometry must be 960x544, 24 fps, 17-frame chunks")
    if (cfg.get("proxy_width"), cfg.get("proxy_height")) != (608, 352):
        raise ValueError("v4 requires a 608x352 RGB proxy before TAE encoding")
    if type(cfg.get("port")) is not int or not 1 <= cfg["port"] <= 65535:
        raise ValueError("Invalid server port")
    if cfg.get("weights") not in {"live", "ema"}:
        raise ValueError("weights must be live or ema")
    if type(cfg.get("denoise_steps")) is not int or cfg["denoise_steps"] != 4:
        raise ValueError("v4 demo requires denoise_steps=4")
    if type(cfg.get("seed")) is not int or not 0 <= cfg["seed"] < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    for key in ("sp_size", "worker_timeout_seconds"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg.get("decode_video")) is not bool:
        raise ValueError("decode_video must be a boolean")
    for key in ("encoder_device", "decoder_device"):
        if not isinstance(cfg.get(key), str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", cfg[key]):
            raise ValueError(f"{key} must be cpu, cuda or cuda:index")
    return cfg


# Fixed recording limits are shared with the browser; no model/device paths here.
CFG = {
    "translation_m_per_s": 0.6,
    "rotation_deg_per_s": 30,
    "fps": 24,
    "max_triangles": 2_000_000,
}
