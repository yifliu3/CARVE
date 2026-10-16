"""Portable raw manifests and v4 uniform TAEH3 training sample views."""

import hashlib
import json
from pathlib import Path

from .palette import validate_palette_identity

RAW_SCHEMA = "cwm-h3.raw.v1"
ENCODED_SCHEMA = "cwm-h3.encoded.v4-uniform-taeh3.v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def asset(root, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("Asset paths must be relative to the manifest directory")
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or out-of-root asset: {name}")
    return path


def read_rows(manifest):
    rows = []
    with Path(manifest).open() as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Row must be an object")
                    rows.append(row)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{manifest}:{number}: {error}") from error
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def validate_scene_groups(rows):
    groups, variants = {}, {}
    for row in rows:
        scene = row["scene_id"]
        source = row.get("source_scene_id", scene)
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source_scene_id must be a nonempty string")
        if variants.setdefault(scene, source) != source:
            raise ValueError("One scene variant must retain its source_scene_id")
        if groups.setdefault(source, row["split"]) != row["split"]:
            raise ValueError("Source scene leaks across dataset splits")


def validate_geometry(row):
    for key in ("width", "height", "num_frames"):
        if type(row[key]) is not int or row[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if row["width"] % 32 or row["height"] % 32:
        raise ValueError("Video width/height must be multiples of 32")
    if row["num_frames"] < 39 or (row["num_frames"] - 5) % 17:
        raise ValueError("Use 17*n+5 frames, n>=2 (for example 39 or 107)")
    if float(row["fps"]) != 24.0:
        raise ValueError("Proxy and target must use constant 24 fps")


def read_video(path, row):
    import av
    import numpy as np

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        if (
            (stream.width, stream.height) != (row["width"], row["height"])
            or stream.average_rate is None
            or abs(float(stream.average_rate) - 24) > 0.02
        ):
            raise ValueError(f"Video dimensions/fps differ: {path}")
        frames = []
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - index / 24) > 1e-4:
                raise ValueError(f"Video must have aligned constant-rate timestamps: {path}")
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != row["num_frames"]:
        raise ValueError(f"Video frame count differs: {path}")
    return np.stack(frames)


def validate_raw(manifest, *, inspect_media=False):
    from .prompt import static_prompt_hash

    root = Path(manifest).resolve().parent
    rows = read_rows(manifest)
    ids, scenes = set(), {}
    required = {
        "schema",
        "sample_id",
        "scene_id",
        "split",
        "references",
        "prompt",
        "proxy",
        "target",
        "fps",
        "num_frames",
        "width",
        "height",
    }
    for row in rows:
        if required - row.keys():
            raise ValueError(f"Missing raw fields: {sorted(required - row.keys())}")
        if row["schema"] != RAW_SCHEMA or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unsupported raw schema or split")
        for key in ("sample_id", "scene_id"):
            value = row[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty string")
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample_id")
        ids.add(row["sample_id"])
        if not isinstance(row["references"], list) or len(row["references"]) != 1:
            raise ValueError("The v4 prompt requires exactly one appearance reference")
        prompt = asset(root, row["prompt"]).read_text().strip()
        if hashlib.sha256(prompt.encode()).hexdigest() != static_prompt_hash():
            raise ValueError("Use the released reference_only_ref2va_v4.txt prompt")
        refs = [asset(root, name) for name in row["references"]]
        identity = (row["split"], tuple(digest(p) for p in refs))
        if scenes.setdefault(row["scene_id"], identity) != identity:
            raise ValueError("One scene must keep its split and reference image")
        validate_geometry(row)
        validate_palette_identity(row, require_hash=False)
        for key in ("proxy", "target"):
            path = asset(root, row[key])
            if inspect_media:
                read_video(path, row)
    validate_scene_groups(rows)
    return rows


class EncodedDataset:
    def __init__(
        self,
        manifest,
        *,
        split="train",
        checkpoint_id=None,
        qwen_dynamic_proxy=None,
        stage=None,
        qwen_proxy_scope=None,
        camera_conditioning=None,
        proxy_conditioning=None,
        visual_encoder=None,
        proxy_variant="mixed",
    ):
        from .v4_data import MODALITIES, V4EncodedDataset

        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        if visual_encoder not in {None, "taeh3"}:
            raise ValueError("Dataset and configured visual encoder differ")
        if proxy_variant not in {"mixed", *MODALITIES}:
            raise ValueError("Select mixed, coarse or fine proxy")
        self.v4_dataset = V4EncodedDataset(manifest, split=split)
        self.v4 = True
        self.visual_encoder = "taeh3"
        variants = MODALITIES if proxy_variant == "mixed" else (proxy_variant,)
        self.rows = [
            {**row, "sample_id": f"{row['sample_id']}-{variant}",
             "source_sample_id": row["sample_id"], "proxy_variant": variant}
            for row in self.v4_dataset.rows for variant in variants
        ]
        self._storage_indices = [
            i for i in range(len(self.v4_dataset)) for _ in variants
        ]
        if len({(r["num_frames"], r["height"], r["width"], r["fps"]) for r in self.rows}) != 1:
            raise ValueError("One split must have uniform frame count, dimensions and fps")
        self.codec = self.v4_dataset.codec
        self.proxy_conditioning = "tae_tokens"
        codec = self.codec
        if proxy_conditioning not in {None, "tae_tokens"} or camera_conditioning not in {
            None,
            "none",
        }:
            raise ValueError("Dataset and model conditioning differ")
        if checkpoint_id and codec["checkpoint_id"] != checkpoint_id:
            raise ValueError("Dataset encoder and base checkpoint identities differ")
        self.validate_conditioning(
            qwen_dynamic_proxy=qwen_dynamic_proxy,
            qwen_proxy_scope=qwen_proxy_scope,
            stage=stage,
        )

    def validate_conditioning(self, *, qwen_dynamic_proxy=None, stage=None, qwen_proxy_scope=None):
        if qwen_dynamic_proxy not in {None, False} or qwen_proxy_scope not in {
            None,
            "reference_only",
        }:
            raise ValueError("TAE tokens require reference-only Qwen")
        if stage not in {None, "bidirectional_fm", "causal_tf_anyflow", "causal_dmd_sgf"}:
            raise ValueError("Unsupported stage")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from .core import Condition

        row = self.rows[index]
        stored = self.v4_dataset[self._storage_indices[index]]
        static = stored["static"]
        target, proxy = stored["target"], stored["proxies"][row["proxy_variant"]]
        length = (row["num_frames"] - 5) // 17 * 5
        prompt, tags, audio = (
            static["prompt_embeds"],
            static["text_token_tags"],
            static["silence_rows"],
        )
        refs = [static["reference_000"][None]]
        return Condition(
            prompt[None],
            tags,
            refs,
            proxy[None, :, :length],
            audio[None],
            target[None, :, :length],
            output_size=(row["height"] // 16, row["width"] // 16),
        )
