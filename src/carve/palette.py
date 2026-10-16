"""Pinned semantic RGB palette for v4 colored proxy pairs."""

import hashlib
from pathlib import Path

PALETTE_VERSION = "semantic_palette_fixed_v1"
PALETTE_SHA256 = "c1331659f7d40a6e95ea98a7dc6f30521ebf90dfd5b1e17c9ff1177422c1368f"
PALETTE_PATH = Path(__file__).parent / "assets" / f"{PALETTE_VERSION}.json"


def palette_identity():
    return {
        "proxy_color_mode": "semantic_class",
        "proxy_palette_version": PALETTE_VERSION,
        "proxy_palette_sha256": PALETTE_SHA256,
    }


def validate_palette_identity(value, *, require_hash=True):
    for key, expected in palette_identity().items():
        if key == "proxy_palette_sha256" and not require_hash and key not in value:
            continue
        if value.get(key) != expected:
            raise ValueError(f"Semantic palette identity differs: {key}")


def palette_bytes(path=None):
    payload = Path(path or PALETTE_PATH).read_bytes()
    if hashlib.sha256(payload).hexdigest() != PALETTE_SHA256:
        raise ValueError("Semantic palette file differs from the pinned training palette")
    return payload
