# V4 Release Validation

Recorded on **October 15, 2026** for
`v4-colored-uniform-taeh3-tokens608-frozen-proj`. The release uses the same
pinned TAEH3 for reference, proxy and target, frozen shared `proj_in`, rank512
LoRA and reference-only static Qwen. External models, datasets and checkpoints
are not bundled.

## Executed Release Checks

The regression suite and reference probes were run before packaging. Test
sources are not included in this release; the results below are historical
validation records.

| Check | Observed Result | Scope |
| --- | --- | --- |
| Static checks | Ruff passed before packaging | Standalone sources, tests and demo |
| CPU regression suite | **104 passed, 4 CUDA cases skipped in 45.91 seconds** | Three-stage LIVE handoffs, exact resume, data, reference preparation, decoder, optimizer offload, worker and browser |
| Reference model comparison | **1,585 tensors exactly equal**, including dtype; report JSON equal | CPU forward outputs, gradients, model state, AnyFlow/DMD, replay/cache/session checks |
| Published v4 dataset | **312 mixed test views**, or **156 coarse/fine views** each | Production reader; shared reference/target and independent proxy selection |
| Real demo static condition | Matching v4 dataset condition loaded successfully | Manifest binding, one TAEH3 reference and no target input |
| Real visual shapes | Target `[1,24,30,34,60]`; proxy `[1,24,30,22,38]`; reference `[1,24,1,34,60]` | Actual paired v4 payloads at full stored/training geometry |
| Completed checkpoint identity | Stage1 step800 and Stage2 step1000 contracts equal the release contract | Completed metadata; no full-size model allocation |
| Pinned TAEH3 assets | Both source and weight SHA256 checks passed | Actual supplied assets, not a mock |
| Real TAEH3 decoder | **Zero maximum difference** from upstream padded-lookahead decode; 34 frames at 32 x 32 | Two complete chunks on CPU; exact session reset and future isolation |
| Real TAEH3 reference encoder | Reference equals first latent of the identical repeated-image video chunk | Pinned FP16 encoder on CPU, 32 x 32 reference |
| RGB preparation orchestration | Mock conditioners/encoder produced a validated v4 trajectory pair | Reference/target sharing, nearest RGB proxy resize, palette and final publication |
| Browser acceptance | Both desktop/mobile recording and server/live-preview matching cases passed; reference preparation enables generation | Existing Chromium runtime with SwiftShader; reference preparation uses mocked heavy conditioners |
| Hosted CI | Pending | Workflow provided; no hosted execution claimed |
| Renamed wheel installation | `carve-4.0.0` built and installed into an isolated temporary directory | Package import, installed `carve` help/resolve and bundled prompt/palette checked outside the source tree |

The validation environment uses Python 3.12, Diffusers 0.40.0,
Transformers 5.12.1, PEFT 0.20.0 and Hugging Face Hub 1.33.0. It inherits the host's
PyTorch 2.6.0 development build. A temporary environment excludes unrelated
image-specific optional package hooks and disables unrelated pytest plugins.
This is not a clean official-wheel CUDA environment validation.

The project, Python package and command are named **CARVE** / `carve`, with
the title **Proxy-Guided Causal Diffusion Rendering with Coding Agents**.
CARVE expands to **Causal Agent-assisted Rendering of Virtual Environments**.
README, demo branding, imports and CLI examples use the new name. The `cwm-h3.*` serialized schemas,
v4 experiment identity and `CWM_*` environment variables are retained for
existing checkpoints, datasets and local configuration. The full suite,
browser checks and 1,585-tensor reference comparison passed after the rename.

The suite and reference comparison were rerun after removing unused additive
codecs, native video-VAE decoding/loading, historical packed layouts and
geometry profiles, unused framework helpers and obsolete checkpoint transition
handling. The prompt is now `reference_only_ref2va_v4.txt`; its content hash and
the fixed semantic palette hash are unchanged. Palette codec construction also
rejects conflicting supplied color modes and palette versions. All four release
configs and the published v4 dataset still validate.

Demo synchronization added the bundled v4 SGF inference config, disabled
inference checkpointing, and made CLI devices/SP size and worker seed follow
demo configuration. Uploaded reference images retain their aspect ratio and
use the offline training preparation path. Static conditions validate before
atomic publication, and tagged conditions can be used alongside a configured
dataset manifest. HTTP reference preparation, rejected publication, CLI
defaults, worker seed forwarding and the browser reference-ready flow passed.
The model and visual assets in these new orchestration tests are mocked; the
previous real codec and reference parity checks remain the model evidence.

## Reference Training And Selection

These are completed reference v4 runs, read from their saved selection and
completion records on October 15, 2026. They are separate from executions of the
standalone release:

| Stage | Selected LIVE | Steps for Evaluation | Mean LPIPS | Mean PSNR | Mean SSIM |
| --- | --- | --- | --- | --- | --- |
| 0.5, bidirectional FM | **2000** | 30 | 0.137189641 | 19.10178465 | 0.560415707 |
| 1, causal AnyFlow | **800** | 4 | 0.175777446 | 17.34000862 | 0.481606512 |
| 2, causal SGF | **1000** | 4 | 0.160664163 | 17.88717020 | 0.503986231 |

Each stage selected the minimum mean LPIPS on the fixed historical 20-view set:
ten trajectories, each with coarse/fine proxies, across five test scenes.
Coarse/fine are equally weighted. This is checkpoint selection data, not an
independent quality estimate, and stage solvers differ.

Stage2 completed 1000 outer steps on October 15, 2026 at 15:46:58 CST.
It initialized the student from Stage1 step800 LIVE and teacher/critic from
Stage0.5 step2000 LIVE. CPU optimizer-state offload and the frozen projection
were retained. All configurations use SP8; the reference training allocation
used 128 GPUs with DP16.

Selected LIVE payload SHA256:

| Stage | SHA256 |
| --- | --- |
| 0.5 | `0e9ea758dc44add4762c13a80e3a63da4af536ed0f1fe5a81eaa2a8a89b65251` |
| 1 | `a1ed7248b8942488e60c1cd1e1fa891157759272313cdbcac50df318e5b0bb49` |
| 2 | `5c07fcbeb12d1c48c0b918bcd039cbe2fa422086054f0ebd69b98a7c9e98b1b8` |

Default relative config paths are installation templates; users must supply
the matching completed external v4 checkpoint directories. Initialization and
resume require an identical model contract.

## Release Checks

Use the [release environment](../README.md#installation) for static and CLI
checks:

```bash
python -m pip install -e '.[dev]'
ruff check src demo
python -m carve --help
python -m carve resolve --config configs/inference.yaml
```

## Full-Size Inference Recipe

The following is a reproduction recipe, not an executed v4 release smoke:

```bash
export CWM_MODEL_DIR="${CWM_MODEL_DIR:?Set the complete Ref2VA model directory}"
export CWM_PROXY_VIDEO="${CWM_PROXY_VIDEO:?Set the semantic proxy video at 24 fps}"
export CWM_REFERENCE_IMAGE="${CWM_REFERENCE_IMAGE:?Set the scene appearance reference}"
export CWM_CHECKPOINT="${CWM_CHECKPOINT:?Set a matching completed Stage2 checkpoint}"
export CWM_TAEH3_ASSETS="${CWM_TAEH3_ASSETS:?Set pinned TAEH3 assets}"
export CWM_SMOKE_DIR="$(mktemp -d /tmp/cwm-smoke.XXXXXX)"
torchrun --standalone --nproc_per_node=8 -m carve infer \
  --config configs/inference.yaml \
  --model-dir "$CWM_MODEL_DIR" \
  --weights "$CWM_CHECKPOINT" --weight-source live \
  --proxy-video "$CWM_PROXY_VIDEO" --reference-image "$CWM_REFERENCE_IMAGE" \
  --steps 4 --output "$CWM_SMOKE_DIR" --tae-assets "$CWM_TAEH3_ASSETS"
```

For a 107-frame proxy video, require 107 MP4 frames at 960x544 / 24 fps.
Inference writes `generated.mp4` and `timings.json`, without saving generated
latents. `timings.json` records decoder identity and final-block padding.
Changing solver steps changes the comparison.

## Decode And Demo Behavior

TAEH3 carries memory through complete five-latent chunks. Each chunk expands
to 20 RGB frames, removes the three prefix frames and retains 17 generated frames.
No native VAE RGB/latent normalization or terminal latent repetition is used.

The HTTP demo replays all generated latent chunks from the session to restore
decoder memory before publishing the current request. This keeps complete
decoder history but increases decode work with session length. Model history
remains bounded to five chunks. Short requests pad their final proxy chunk;
the padded generated latents enter history although only visible RGB frames
are published. Later continuation includes that padding in model time.

## Remaining Validation

- Full-size v4 standalone SP8 training/inference and resident worker execution.
- Complete real raw RGB/Qwen/audio/TAEH3 preparation and HTTP generation.
- CUDA SP2 parity, CPU-slot optimizer arithmetic on CUDA and FSDP capacity.
- Full-size injected-failure worker cleanup and long-session latency/memory.
- Independent multi-scene quality evaluation and hosted Python 3.10/3.12 CI.
