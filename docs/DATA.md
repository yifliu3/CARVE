# Data Preparation

CARVE uses an appearance reference and semantic proxy video to condition
generation. Target video provides supervision during training. A matching
preencoded dataset can be used directly.

## Raw inputs

Each raw trajectory is represented by two rows sharing `source_sample_id`.
One row selects `proxy_variant: coarse`, the other `fine`. Each pair contains:

- One RGB appearance reference.
- Colored coarse/fine proxy and target MP4s aligned at 24 fps.
- The released [reference-only prompt](../src/carve/prompts/reference_only_ref2va_v4.txt).
- A manifest row identifying the scene, dataset split and semantic palette.

Frame count must be `17*n+5`, with `n>=2`, such as 39 or 107 frames.
Width and height must be multiples of 32. The default training recipe uses
960 × 544 video. Keep all variants of an original scene in the same split;
use `source_scene_id` to group variants.

Example layout:

```text
data/raw/
├── manifest.jsonl
├── prompt.txt
└── scene01/
    ├── reference.png
    ├── coarse.mp4
    ├── fine.mp4
    └── target.mp4
```

Copy the released prompt:

```bash
mkdir -p data/raw
cp src/carve/prompts/reference_only_ref2va_v4.txt data/raw/prompt.txt
```

Each line of `manifest.jsonl` is one JSON object. This expanded example should
be written on a single line:

```json
{
  "schema": "cwm-h3.raw.v1",
  "sample_id": "scene01-path01-coarse",
  "source_sample_id": "scene01-path01",
  "proxy_variant": "coarse",
  "scene_id": "scene01",
  "source_scene_id": "scene01",
  "split": "train",
  "references": ["scene01/reference.png"],
  "prompt": "prompt.txt",
  "proxy": "scene01/coarse.mp4",
  "target": "scene01/target.mp4",
  "fps": 24,
  "num_frames": 107,
  "width": 960,
  "height": 544,
  "proxy_color_mode": "semantic_class",
  "proxy_palette_version": "semantic_palette_fixed_v1"
}
```

Asset paths are relative to the manifest directory. Training and inference
config paths are relative to the invoking directory.
Add the matching fine row with `sample_id: scene01-path01-fine`,
`proxy_variant: fine` and `proxy: scene01/fine.mp4`. Keep reference, target,
scene, split, frame count and geometry identical. The v4 release requires
960 x 544 target/proxy source video, and stores proxy pixels at 608 x 352.

## Semantic palette

Render proxies with the exact **25-class**
[semantic palette](../src/carve/assets/semantic_palette_fixed_v1.json).
Its SHA256 is:

```text
c1331659f7d40a6e95ea98a7dc6f30521ebf90dfd5b1e17c9ff1177422c1368f
```

Raw rows require `proxy_color_mode: semantic_class` and
`proxy_palette_version: semantic_palette_fixed_v1`.
`proxy_palette_sha256` is optional for raw rows; when provided, it must match
the hash above. Arbitrary texture or vertex colors do not reproduce this
semantic input.

Preencoding writes all three palette identity fields and a byte-identical
`palette.json`. Encoded codec metadata requires the hash, mode and version;
encoded rows require mode and version.

## TAEH3 assets

All visual encoding and RGB decoding require `taehv.py` and `taeh3.safetensors` from revision
`011dfc2112197741c540e0bdd5b7b67bcc930771`. The encoder verifies their hashes
before loading:

| Asset | SHA256 |
| --- | --- |
| `taehv.py` | `99545f093b36c16297ad182ce18a5dce2ebc98bb611f5e8dfa2dce431ba39ce2` |
| `taeh3.safetensors` | `4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13` |

Use the same native MiniMax-H3 base snapshot for encoding and training.
Its default identity is `minimax-h3-ref2va-hymm-ef520db195ef3286`; update
`checkpoint_id` consistently if changing the base model.

## Validate and preencode

Run from the repository root:

```bash
python -m carve validate-data \
  --manifest data/raw/manifest.jsonl --inspect-media

python -m carve preencode \
  --manifest data/raw/manifest.jsonl --output data/encoded \
  --checkpoint models/Ref2VA \
  --checkpoint-id minimax-h3-ref2va-hymm-ef520db195ef3286 \
  --tae-assets models/taeh3 --device cuda:0
```

Use an empty encoding output directory. The bundled palette is the default;
`--palette` accepts an external JSON file only if its hash matches.

Preencoding writes **one record per trajectory**, with shared reference/target
and both colored proxies. Qwen receives only the reference image and static
prompt; audio silence retains the H3 audio codec. The native visual VAE is not
loaded. Qwen/audio conditioners unload before the pinned TAEH3 visual encoder.
Reference, proxy and target use that same encoder with RGB in `[0,1]`,
deterministic output and no extra latent normalization or posterior sampling.
Proxy RGB is nearest-resized to 608 x 352 before encoding. Target stays
960 x 544; reference uses fit-without-upscale and multiples of 32.

Each 17-frame chunk gets three zero prefix frames and freshly reset encoder
memory. Incomplete chunks repeat their final input frame. Concatenate the
five-latent chunks and remove three final latents once: 107 RGB frames produce
32 stored latents; training uses the first 30. A reference image repeats to
17 frames and selects the first latent from the identical chunk adapter.

| Payload | Stored Tensor for 107 Frames |
| --- | --- |
| `condition.safetensors:reference_000` | BF16 `[24,1,H_ref/16,W_ref/16]` |
| `latents.safetensors:target_latents` | BF16 `[24,32,34,60]` |
| `proxy.safetensors:coarse_latents` | BF16 `[24,32,22,38]` |
| `proxy.safetensors:fine_latents` | BF16 `[24,32,22,38]` |

The encoded schema is `cwm-h3.encoded.v4-uniform-taeh3.v1`; codec schema is
`cwm-h3.codec.v4-uniform-taeh3-colored-pair.v1`. The manifest is published after
all payloads pass the production reader, followed by a dataset `COMPLETE.json`.
Use that completed output rather than a partially populated output directory.

For the default geometry, each complete block has:

| Tensor | Shape |
| --- | --- |
| Proxy tokens | `[1,24,5,22,38]` |
| Target latents | `[1,24,5,34,60]` |

Set `data.manifest=data/encoded/manifest.jsonl` for training or inference.
All configurations declare `data.visual_encoder: taeh3`. The default
`data.proxy_variant: mixed` exposes coarse then fine views for each trajectory;
`coarse` and `fine` expose one view instead. Views share the stored reference and
target, and do not alter the manifest or concatenate proxy variants.

Checkpoint identity includes the pinned encoder, all visual roles, normalization,
reference policy and frozen shared projection. Initialization and resume require
the same v4 model contract. Exact resume also requires the original split and
proxy variant selection.
