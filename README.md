<div align="center">

<h1>CARVE</h1>
<p><strong>Proxy-Guided Causal Diffusion Rendering with Coding Agents</strong></p>

<p>
  <a href="#quick-start">Quick Start</a> &nbsp; / &nbsp;
  <a href="#demo">Demo</a> &nbsp; / &nbsp;
  <a href="#training">Training</a>
</p>

</div>

CARVE is a causal diffusion renderer that turns depth, semantic, coarse-geometry or fine-geometry proxy videos and a reference image into video with consistent appearance and temporal continuity.

![Depth, semantic segmentation, coarse-geometry and fine-geometry proxies paired with their generated videos](assets/proxy_video_teaser.gif)

> CARVE is an open-source project and is still under development.

## Key Features

- **Multiple proxy representations.** Condition generation on depth maps, semantic segmentation maps, coarse-geometry renders or fine-geometry renders.
- **Reference-conditioned appearance.** Use a reference image to supply the scene's visual appearance.
- **Causal video continuation.** Generate successive video blocks with cached temporal context.
- **Two-step rendering.** The final Stage2 model uses two SGF denoising steps per video block.
- **Interactive proxy rendering.** Import scene meshes, render proxy views, and record or replay trajectories in the browser.

## Quick Start

### Installation

Run from the repository root in a separate Linux environment.

| Python | PyTorch / CUDA | Diffusers | Transformers | PEFT |
| :---: | :---: | :---: | :---: | :---: |
| 3.12 | 2.6.0 / 12.4 | 0.40.0 | 5.12.1 | 0.20.0 |

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

### Models

Place the **2-step Stage2 CARVE DiT checkpoint**, **MiniMax-H3 Ref2VA** model in Diffusers format and pinned **TAEH3** source and weights under `models/`. The CARVE checkpoint will be available on Hugging Face soon. Use the same model versions for data encoding and generation.

| Model | Download | Local directory | Version |
| --- | --- | --- | --- |
| CARVE DiT | Hugging Face — coming soon | `models/CARVE/` | Stage2 · 2-step SGF |
| MiniMax-H3 | [Official Hugging Face repository](https://huggingface.co/MiniMaxAI/MiniMax-H3) · [Ref2VA files](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/Ref2VA) | `models/Ref2VA/` | Ref2VA / Diffusers |
| TAEH3 | [Pinned upstream repository](https://github.com/madebyollin/taehv/tree/011dfc2112197741c540e0bdd5b7b67bcc930771) · [taehv.py](https://raw.githubusercontent.com/madebyollin/taehv/011dfc2112197741c540e0bdd5b7b67bcc930771/taehv.py) · [taeh3.safetensors](https://raw.githubusercontent.com/madebyollin/taehv/011dfc2112197741c540e0bdd5b7b67bcc930771/safetensors/taeh3.safetensors) | `models/taeh3/` | Commit `011dfc2112197741c540e0bdd5b7b67bcc930771` |

<details>
<summary><strong>Model directory layout</strong></summary>

```text
models/
├── CARVE/                   # 2-step Stage2 CARVE checkpoint
├── Ref2VA/
│   ├── transformer_ref/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── processor/
│   ├── vae/
│   └── audio_vae/
└── taeh3/                   # Required for visual encoding and RGB decode
    ├── taehv.py
    └── taeh3.safetensors
```

</details>

### Demo

Preview scene meshes, proxy renders and camera trajectories:

```bash
python -m pip install --no-deps -r demo/requirements-optional.txt
python -m demo.server --preview --port 8080
```

Open **http://127.0.0.1:8080/** and focus the scene. Use **W / A / S / D** to move and **arrow keys** to turn, then record, replay or export your camera path.

For video generation, use the downloaded models and a 2-step Stage2 CARVE checkpoint, then start the demo in generation mode:

```bash
python -m playwright install chromium
export CWM_CHECKPOINT="$PWD/models/CARVE"
export CWM_TAEH3_ASSETS="$PWD/models/taeh3"
python -m demo.server --generate --port 8080
```

Replace the paths with your local assets. Import a scene mesh and its appearance reference, click **Prepare reference**, then record a trajectory and generate video. The bundled demo uses two SGF denoising steps per 17-frame block and checks the checkpoint's two-step training recipe before loading H3.

### Inference

Provide a **proxy video** and an **appearance reference image** of the same scene, together with a **2-step Stage2** CARVE checkpoint. Supported proxy representations include depth maps, semantic segmentation maps, coarse-geometry renders and fine-geometry renders. The released coarse/fine semantic data convention and palette are documented in the [data guide](docs/DATA.md). Use a constant **24 fps** proxy video; generated video is **960 × 544** at **24 fps**.

```bash
# Set these paths to your own inputs and downloaded model assets.
PROXY_VIDEO=/path/to/proxy.mp4                 # Depth, semantic, coarse or fine proxy, 24 fps
REFERENCE_IMAGE=/path/to/reference.png         # Appearance reference of the same scene
CARVE_CHECKPOINT=models/CARVE                  # 2-step Stage2 CARVE checkpoint directory
H3_MODEL_DIR=models/Ref2VA                     # H3 Ref2VA model in Diffusers format
TAEH3_ASSETS=models/taeh3                      # taehv.py and taeh3.safetensors
OUTPUT_DIR=outputs/inference/sample-000        # Empty output directory

torchrun --standalone --nproc_per_node=8 -m carve infer \
  --config configs/inference.yaml \
  --model-dir "$H3_MODEL_DIR" \
  --weights "$CARVE_CHECKPOINT" \
  --proxy-video "$PROXY_VIDEO" \
  --reference-image "$REFERENCE_IMAGE" \
  --tae-assets "$TAEH3_ASSETS" \
  --steps 2 \
  --output "$OUTPUT_DIR"
```

Replace the paths with your own video, image and model assets, and choose an empty output directory. CARVE encodes the inputs automatically; no target video or preencoded dataset is required. The output preserves the proxy video's frame count, including a partial final block.

The bundled inference config defaults to **2 denoising steps per block**, so `--steps 2` may be omitted. Use a checkpoint trained with the matching two-step SGF recipe, identified by `v4-2step-colored-uniform-taeh3-tokens608-frozen-proj`. Checkpoint metadata must match the inference config's stage, denoising step count and experiment version.

| Output | Contents |
| --- | --- |
| `generated.mp4` | Decoded video at 24 fps |
| `timings.json` | Denoising step count, sampling mode and chunk timings |

## Training

Prepare data with the **[data guide](docs/DATA.md)** and update the model, dataset and checkpoint paths in the configs, then train the three stages in order:

| Stage | Objective | Config |
| --- | --- | --- |
| **0.5** | Bidirectional flow matching | [stage0p5.yaml](configs/stage0p5.yaml) |
| **1** | Causal teacher forcing / AnyFlow | [stage1.yaml](configs/stage1.yaml) |
| **2** | Causal distillation / **2-step SGF** | [stage2.yaml](configs/stage2.yaml) |

```bash
torchrun --standalone --nproc_per_node=8 -m carve train --config configs/stage0p5.yaml
torchrun --standalone --nproc_per_node=8 -m carve train --config configs/stage1.yaml
torchrun --standalone --nproc_per_node=8 -m carve train --config configs/stage2.yaml
```

Stage1 initializes from Stage0.5. Stage2 initializes from Stage1 and uses Stage0.5 as its teacher. Set `checkpoint.initialize` and `checkpoint.teacher` to the corresponding completed checkpoints.

Stage2 sets `train.denoise_steps: 2` for both student and critic rollouts. The default recipe runs 1,000 outer training iterations, updates the critic on each iteration and updates the student every five iterations after warm-up. The two denoising steps are the sampling budget per block, not the number of training iterations. Stage0.5 and Stage1 keep their existing recipes; their reference evaluation budgets are 30 and 4 steps, respectively.

Start the two-step Stage2 with LIVE weights from Stage1 and LIVE teacher/critic weights from Stage0.5. For exact resume, use a checkpoint from the same two-step recipe. See the [data guide](docs/DATA.md) for the shared visual contract and the [validation record](docs/VALIDATION.md) for the scope of reported results.

## Acknowledgements and License

This code builds on [SolarWM](https://github.com/Junchao-cs/SolarWM), [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and [TAEH3](https://github.com/madebyollin/taehv). Code is licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights and external datasets retain their publishers' terms.
