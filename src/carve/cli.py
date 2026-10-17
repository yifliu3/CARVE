"""Small release entry point; CUDA dependencies are loaded only when needed."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        prog="carve", description="CARVE: Proxy-Guided Causal Diffusion Rendering with Coding Agents",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--inspect-media", action="store_true")
    pre = commands.add_parser("preencode")
    pre.add_argument("--manifest", required=True)
    pre.add_argument("--output", required=True)
    pre.add_argument("--checkpoint", required=True)
    pre.add_argument("--checkpoint-id", required=True)
    pre.add_argument("--tae-assets", required=True)
    pre.add_argument("--device", default="cuda:0")
    pre.add_argument("--seed", type=int, default=42)
    pre.add_argument("--proxy-height", type=int, default=352)
    pre.add_argument("--proxy-width", type=int, default=608)
    pre.add_argument("--palette", help="Exact pinned palette JSON; defaults to the bundled file")
    for name in ("resolve", "train", "infer"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--set", action="append", default=[])
        if name == "infer":
            sub.add_argument("--weights", required=True, help="CARVE checkpoint directory (2-step Stage2)")
            sub.add_argument(
                "--model-dir", help="Downloaded H3 Ref2VA directory; overrides model.checkpoint",
            )
            sub.add_argument(
                "--proxy-video", required=True,
                help="User-provided coarse or fine semantic proxy video at constant 24 fps",
            )
            sub.add_argument(
                "--reference-image", required=True,
                help="User-provided appearance reference image of the same scene",
            )
            sub.add_argument("--sample-index", type=int, default=0)
            sub.add_argument("--weight-source", choices=("live", "ema"), default="live")
            sub.add_argument(
                "--steps", type=int, default=None,
                help="Denoising steps per chunk; defaults to the config (2 for released Stage2)",
            )
            sub.add_argument("--output", required=True, help="Empty directory for generated.mp4")
            sub.add_argument(
                "--decode", action="store_true", default=True,
                help="Decode video (enabled by default)",
            )
            sub.add_argument(
                "--tae-assets", required=True,
                help="Directory containing pinned taehv.py and taeh3.safetensors",
            )
    args = parser.parse_args()
    if args.command == "infer":
        raw_inputs = bool(args.proxy_video or args.reference_image)
        if raw_inputs and not (args.proxy_video and args.reference_image):
            parser.error("--proxy-video and --reference-image must be provided together")
        if (raw_inputs or args.decode) and not args.tae_assets:
            parser.error("--tae-assets is required for raw inputs or --decode")
    if args.command == "validate-data":
        from .data import validate_raw

        rows = validate_raw(args.manifest, inspect_media=args.inspect_media)
        print(json.dumps({"samples": len(rows), "scenes": len({r["scene_id"] for r in rows})}))
    elif args.command == "preencode":
        from .preencode import run

        run(
            args.manifest,
            args.output,
            args.checkpoint,
            args.checkpoint_id,
            tae_assets=args.tae_assets,
            device=args.device,
            seed=args.seed,
            proxy_height=args.proxy_height,
            proxy_width=args.proxy_width,
            palette=args.palette,
        )
    else:
        from .config import resolve

        cfg = resolve(args.config, args.set)
        if args.command == "infer" and args.model_dir:
            cfg["model"]["checkpoint"] = args.model_dir
        if args.command == "resolve":
            print(json.dumps(cfg, indent=2))
        elif args.command == "train":
            from .train import run

            run(cfg)
        else:
            import torch.distributed as dist
            from .inference import run

            try:
                run(cfg, args)
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()
