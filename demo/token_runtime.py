"""Current-block generation and reference preparation for the v4 TAEH3 model."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "demo"

from .settings import DENOISE_STEPS, ROOT, VERSION, load_config, read

PROXY_SHAPE = (24, 5, 22, 38)
OUTPUT_SIZE = (34, 60)


def import_core():
    # Only the sibling release package is considered, never a training checkout.
    source = ROOT.parent / "src"
    if source.is_dir():
        sys.path.insert(0, str(source))


def validate_release_config(cfg):
    import_core()
    from carve.config import validate_conditioning

    validate_conditioning(cfg)
    model, data = cfg["model"], cfg["data"]
    expected = {
        "lora_rank": 512, "lora_alpha": 512,
        "proxy_conditioning": "tae_tokens",
        "proxy_projection_trainable": False,
        "proxy_input_height": 352, "proxy_input_width": 608,
        "history_chunks": 5,
    }
    if any(model.get(k) != v for k, v in expected.items()):
        raise ValueError("Require rank/alpha512 frozen-projection TAE tokens at 608x352")
    if model.get("tiny", False):
        raise ValueError("A randomly initialized tiny model is not a release checkpoint")
    if data.get("qwen_dynamic_proxy", False) or data.get("qwen_proxy_scope") != "reference_only":
        raise ValueError("v4 requires reference-only static Qwen conditioning")
    if data.get("visual_encoder") != "taeh3":
        raise ValueError("v4 requires uniform TAEH3 for reference/proxy/target")
    if cfg["runtime"].get("experiment_version") != VERSION:
        raise ValueError("Run config must identify the released v4 2-step version")
    if cfg["stage"] != "causal_dmd_sgf":
        raise ValueError("The released demo requires a Stage2 causal SGF run config")
    if cfg["train"]["denoise_steps"] != DENOISE_STEPS:
        raise ValueError("The released Stage2 demo requires two denoising steps")


def inspect_checkpoint(path, cfg):
    """Validate the complete causal v4 identity before allocating H3."""
    root = Path(path)
    marker = root / "COMPLETE.json"
    if not marker.is_file():
        marker = root / "LOCAL_COMPLETE.json"
    meta = read(marker)
    if meta.get("schema") != "cwm-h3.checkpoint.v1":
        raise ValueError("Not a completed CARVE checkpoint")
    if meta.get("stage") != "causal_dmd_sgf":
        raise ValueError("Supply a matching Stage2 2-step causal SGF checkpoint")
    if meta.get("config", {}).get("runtime", {}).get("experiment_version") != VERSION:
        raise ValueError("Checkpoint provenance does not identify the released v4 2-step version")
    validate_release_config(cfg)
    import_core()
    from carve.config import model_contract
    from carve.evaluation import validate_inference_checkpoint

    validate_inference_checkpoint(cfg, meta)
    if meta.get("contract") != model_contract(cfg):
        raise ValueError("Checkpoint base/conditioning/projection/LoRA contract differs")
    if meta.get("anyflow") is not True:
        raise ValueError("Checkpoint stage and AnyFlow architecture disagree")
    return meta


def resolve_run_config(path, *, model_dir=None, sp_size=None):
    if not path:
        raise ValueError("Set run_config / CWM_RUN_CONFIG to the v4 release configuration")
    import_core()
    from carve.config import resolve

    cfg = resolve(path)
    if model_dir:
        cfg["model"]["checkpoint"] = str(Path(model_dir).expanduser().resolve())
    if sp_size is not None:
        if type(sp_size) is not int or sp_size < 1:
            raise ValueError("sp_size must be a positive integer")
        cfg["distributed"]["sp_size"] = sp_size
    cfg["model"]["activation_checkpointing"] = False
    cfg["model"]["attention_checkpointing"] = False
    validate_release_config(cfg)
    return cfg


def validate_current(latents):
    import torch

    if latents.ndim != 5 or latents.shape[0] != 1 or tuple(latents.shape[1:]) != PROXY_SHAPE:
        raise ValueError("encoded_current must be [1,24,5,22,38]")
    if not latents.is_floating_point() or not bool(torch.isfinite(latents).all()):
        raise ValueError("encoded_current must contain finite floating-point TAE latents")
    return latents


class CurrentTokenEncoder:
    def __init__(self, assets, device):
        if not assets:
            raise ValueError("Set taeh3_assets / CWM_TAEH3_ASSETS to pinned real TAEH3 assets")
        import_core()
        from carve.tae_proxy import TAEH3ProxyEncoder

        self.encoder = TAEH3ProxyEncoder(assets, device=device)

    def encode_current(self, pixels):
        import torch
        from carve.tae_proxy import resize_proxy_pixels

        if pixels.dtype != torch.uint8 or tuple(pixels.shape) != (1, 3, 17, 544, 960):
            raise ValueError("Current proxy pixels must be uint8 [1,3,17,544,960]")
        # Resize RGB, not latents; retain independent native proxy/target grids.
        resized = resize_proxy_pixels(pixels, 352, 608)
        return validate_current(self.encoder.encode_chunk(resized)).bfloat16().contiguous()


def load_static(path, cfg, *, manifest=None, sample_id=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from carve.prompt import static_prompt_hash
    from carve.v4_data import codec as v4_codec, validate_static

    with safe_open(str(path), framework="pt", device="cpu") as payload:
        metadata = payload.metadata() or {}
    codec = json.loads(metadata.get("codec", "{}"))
    if manifest and metadata.get("version") != VERSION:
        from carve.data import asset, read_rows
        from carve.v4_data import SCHEMA

        rows = [row for row in read_rows(manifest)
                if (not sample_id or row.get("sample_id") == sample_id)
                and asset(Path(manifest).resolve().parent, row["condition"]) == Path(path).resolve()]
        if not rows:
            raise ValueError("Static condition does not belong to the selected dataset sample")
        codec = rows[0]["codec"]
        if rows[0].get("schema") != SCHEMA or codec != v4_codec(codec):
            raise ValueError("Dataset static requires the exact v4 uniform TAEH3 codec")
        if codec.get("checkpoint_id") != cfg["model"]["checkpoint_id"] or codec.get("qwen_dynamic_proxy") is not False:
            raise ValueError("Dataset static base or Qwen conditioning differs")
    elif metadata.get("version") != VERSION:
        raise ValueError("Untagged dataset static requires its matching v4 manifest")
    elif metadata.get("checkpoint_id") != cfg["model"]["checkpoint_id"] or metadata.get("static_prompt_sha256") != static_prompt_hash():
        raise ValueError("Static condition base or prompt differs")
    if codec != v4_codec(codec) or codec.get("checkpoint_id") != cfg["model"]["checkpoint_id"]:
        raise ValueError("Static condition uniform TAEH3 codec differs")
    tensors = load_file(str(path))
    validate_static(tensors)
    prompt, tags, audio = (tensors[k] for k in ("prompt_embeds", "text_token_tags", "silence_rows"))
    from carve.core import Condition

    return Condition(
        prompt=prompt[None], tags=tags, references=[tensors["reference_000"][None]],
        proxy=torch.empty(1, 24, 0, 22, 38), audio=audio[None], output_size=OUTPUT_SIZE,
    )


class TokenRuntime:
    """Real H3 session; callers provide only the current encoded proxy block."""

    def __init__(self, cfg, checkpoint_path, *, source="live"):
        import_core()
        from carve import checkpoint
        from carve.core import Core
        from carve.model import build, initialize_distributed

        self.cfg = cfg
        self.meta = inspect_checkpoint(checkpoint_path, cfg)
        self.cfg["stage"] = self.meta["stage"]
        self.device, self.rank, self.world = initialize_distributed(cfg)
        handle = build(cfg, self.device, anyflow=self.meta["anyflow"])
        checkpoint.initialize(handle, checkpoint_path, cfg, allowed_stages={self.meta["stage"]}, source=source)
        self.core = Core(
            handle.model, history=cfg["model"]["history_chunks"],
            shift=cfg["train"]["video_shift"], anyflow=self.meta["anyflow"],
        )
        self.session = None

    def start(self, static_path, *, manifest=None, sample_id=None):
        from carve.streaming import CausalSession

        condition = load_static(static_path, self.cfg, manifest=manifest, sample_id=sample_id).to(self.device)
        self.session = CausalSession(self.core, condition, seed=self.cfg["train"]["seed"])

    def generate(self, encoded_current, *, steps=None, cancelled=None):
        if self.session is None:
            raise ValueError("Start a session with its matching static condition first")
        steps = self.cfg["train"]["denoise_steps"] if steps is None else steps
        if type(steps) is not int or steps != DENOISE_STEPS:
            raise ValueError("The released Stage2 demo requires steps=2")
        self.session.propose_token_chunk(
            validate_current(encoded_current).to(self.device).float(),
            steps=steps, sgf=self.meta["stage"] == "causal_dmd_sgf", cancelled=cancelled,
        )
        if self.session.pending is None:
            return None
        return self.session.commit()

    def reset(self):
        if self.session is not None:
            self.session.reset()
        self.session = None


def save_current(path, latents):
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    save_file({"encoded_current": validate_current(latents).cpu().contiguous()}, str(path),
              metadata={"version": VERSION, "output_size": json.dumps(OUTPUT_SIZE)})


def prepare_static(args, cfg):
    import torch
    from safetensors.torch import save_file
    from carve.prompt import static_prompt_hash
    from carve.preencode import encode_text_audio_condition, encode_reference, fit_reference
    from carve.v4_data import TAEH3VisualEncoder, codec
    from carve.palette import palette_identity
    from solarwm.backends.minimax_h3.optional import load_conditioners

    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    if len(args.reference) != 1:
        raise ValueError("The v4 static prompt supports exactly one reference image")
    images = [fit_reference(args.reference[0])]
    modules = load_conditioners(
        {"checkpoint_path": cfg["model"]["checkpoint"], "torch_dtype": "bfloat16"},
        device=torch.device(args.device),
    )
    modules.audio_vae.set_attention_backend("native")
    static = encode_text_audio_condition(modules, images, device=args.device)
    del modules
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    encoder = TAEH3VisualEncoder(args.tae_assets, device=args.device)
    static["reference_000"] = encode_reference(encoder, images[0])
    identity = codec({
        "checkpoint_id": cfg["model"]["checkpoint_id"],
        "qwen_proxy_scope": "reference_only", "qwen_layer": 50,
        "static_prompt_sha256": static_prompt_hash(), "silence_seconds": 1,
        **palette_identity(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        save_file(static, str(temporary), metadata={
            "version": VERSION, "checkpoint_id": cfg["model"]["checkpoint_id"],
            "static_prompt_sha256": static_prompt_hash(),
            "codec": json.dumps(identity),
        })
        load_static(temporary, cfg)
        if path.exists():
            raise FileExistsError(path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/demo.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    encode = sub.add_parser("encode", help="Encode one real 17-frame proxy RGB block")
    encode.add_argument("--pixels", required=True, help="uint8 .npy [17,544,960,3]")
    encode.add_argument("--assets")
    encode.add_argument("--device")
    encode.add_argument("--output", required=True)
    prepare = sub.add_parser("prepare-static", help="Encode uniform TAEH3 reference-only conditions")
    prepare.add_argument("--reference", action="append", required=True)
    prepare.add_argument("--run-config")
    prepare.add_argument("--model-dir")
    prepare.add_argument("--tae-assets")
    prepare.add_argument("--device")
    prepare.add_argument("--output", required=True)
    rollout = sub.add_parser("rollout", help="Generate from one or more current token blocks")
    rollout.add_argument("--run-config")
    rollout.add_argument("--checkpoint")
    rollout.add_argument("--model-dir")
    rollout.add_argument("--static")
    rollout.add_argument("--static-manifest")
    rollout.add_argument("--sample-id")
    rollout.add_argument("--tokens", action="append", required=True)
    rollout.add_argument("--sp-size", type=int)
    rollout.add_argument("--seed", type=int)
    rollout.add_argument("--steps", type=int)
    rollout.add_argument("--output", required=True)
    decode = sub.add_parser("decode", help="Decode consecutive generated chunks to an MP4")
    decode.add_argument("--run-config")
    decode.add_argument("--model-dir")
    decode.add_argument("--tae-assets")
    decode.add_argument("--latents", action="append", required=True)
    decode.add_argument("--visible-frames", type=int)
    decode.add_argument("--device")
    decode.add_argument("--output", required=True)
    args = parser.parse_args()
    demo = load_config(args.config)
    import_core()
    if args.command == "encode":
        import numpy as np
        import torch

        frames = np.load(args.pixels, allow_pickle=False)
        if frames.dtype != np.uint8 or frames.shape != (17, 544, 960, 3):
            raise ValueError("--pixels must be uint8 [17,544,960,3]")
        pixels = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].contiguous()
        encoder = CurrentTokenEncoder(
            args.assets or demo["taeh3_assets"], args.device or demo["encoder_device"],
        )
        save_current(args.output, encoder.encode_current(pixels))
    else:
        size = getattr(args, "sp_size", None)
        cfg = resolve_run_config(
            args.run_config or demo["run_config"], model_dir=args.model_dir,
            sp_size=demo["sp_size"] if size is None else size,
        )
        if args.command == "prepare-static":
            args.device = args.device or demo["encoder_device"]
            args.tae_assets = args.tae_assets or demo["taeh3_assets"]
            if not args.tae_assets:
                raise ValueError("Set --tae-assets / CWM_TAEH3_ASSETS")
            prepare_static(args, cfg)
        elif args.command == "decode":
            import torch
            from safetensors import safe_open
            from safetensors.torch import load_file
            from carve.inference import decode_causal_video

            chunks = []
            for name in args.latents:
                with safe_open(name, framework="pt", device="cpu") as payload:
                    if (payload.metadata() or {}).get("version") != VERSION:
                        raise ValueError("Generated latent artifact version differs")
                tensors = load_file(name)
                if set(tensors) != {"latents"}:
                    raise ValueError("Expected generated latents only")
                latent = tensors["latents"]
                if tuple(latent.shape) != (1, 24, 5, *OUTPUT_SIZE):
                    raise ValueError("Expected one generated [1,24,5,34,60] block")
                if not latent.is_floating_point() or not bool(latent.isfinite().all()):
                    raise ValueError("Generated latents must be finite")
                chunks.append(latent)
            target = Path(args.output)
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            assets = args.tae_assets or demo["taeh3_assets"]
            if not assets:
                raise ValueError("Set --tae-assets / CWM_TAEH3_ASSETS")
            decode_causal_video(torch.cat(chunks, dim=2), assets, target,
                                device=torch.device(args.device or demo["decoder_device"]),
                                visible_frames=args.visible_frames)
        else:
            from safetensors import safe_open
            from safetensors.torch import load_file, save_file
            import torch.distributed as dist

            checkpoint_path = args.checkpoint or demo["checkpoint"]
            if not checkpoint_path:
                raise ValueError("Supply a matching causal checkpoint, not Stage0.5 weights")
            runtime = None
            try:
                if args.steps is not None and args.steps != DENOISE_STEPS:
                    raise ValueError("The released Stage2 demo requires --steps 2")
                seed = demo["seed"] if args.seed is None else args.seed
                if not 0 <= seed < 2**63:
                    raise ValueError("--seed must be in [0, 2**63)")
                cfg["train"]["seed"] = seed
                out = Path(args.output)
                if out.exists() and any(out.iterdir()):
                    raise FileExistsError("Use an empty rollout output directory")
                out.mkdir(parents=True, exist_ok=True)
                # Every SP rank validates every immutable input before entering collectives.
                for path in args.tokens:
                    with safe_open(path, framework="pt", device="cpu") as f:
                        if (f.metadata() or {}).get("version") != VERSION or set(f.keys()) != {"encoded_current"}:
                            raise ValueError("Token artifact version or payload differs")
                        validate_current(f.get_tensor("encoded_current"))
                static_path = args.static or demo["static_condition"]
                manifest = args.static_manifest or demo.get("static_manifest")
                sample_id = args.sample_id or demo.get("static_sample_id")
                load_static(static_path, cfg, manifest=manifest, sample_id=sample_id)
                runtime = TokenRuntime(cfg, checkpoint_path, source=demo["weights"])
                runtime.start(static_path, manifest=manifest, sample_id=sample_id)
                for index, path in enumerate(args.tokens):
                    with safe_open(path, framework="pt", device="cpu") as f:
                        if (f.metadata() or {}).get("version") != VERSION:
                            raise ValueError("Token artifact version differs")
                    payload = load_file(path)
                    if set(payload) != {"encoded_current"}:
                        raise ValueError("Only encoded_current may enter the token adapter")
                    clean = runtime.generate(payload["encoded_current"], steps=args.steps or demo["denoise_steps"])
                    if runtime.rank == 0:
                        target = out / f"chunk-{index:05d}.safetensors"
                        if target.exists():
                            raise FileExistsError(target)
                        save_file({"latents": clean.cpu().contiguous()}, str(target),
                                  metadata={"version": VERSION, "stage": runtime.meta["stage"]})
            finally:
                if runtime is not None:
                    runtime.reset()
                if dist.is_initialized():
                    dist.destroy_process_group()
    print(json.dumps({"complete": True, "version": VERSION, "output": args.output}))


if __name__ == "__main__":
    main()
