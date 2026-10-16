"""Persistent torchrun model owner controlled by a local JSON command stream."""

import argparse
import json
import sys
import traceback


def main():
    import torch.distributed as dist
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from .token_runtime import VERSION, TokenRuntime, resolve_run_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sp-size", type=int, required=True)
    parser.add_argument("--source", choices=("live", "ema"), default="live")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    runtime = None
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def respond(value):
        protocol.write(json.dumps(value, allow_nan=False) + "\n")
        protocol.flush()

    try:
        cfg = resolve_run_config(args.run_config, sp_size=args.sp_size)
        if args.seed is not None:
            if not 0 <= args.seed < 2**63:
                raise ValueError("--seed must be in [0, 2**63)")
            cfg["train"]["seed"] = args.seed
        runtime = TokenRuntime(cfg, args.checkpoint, source=args.source)
        if runtime.rank == 0:
            respond({"ready": True})
        while True:
            message = [None]
            if runtime.rank == 0:
                line = sys.stdin.readline()
                message[0] = json.loads(line) if line else {"operation": "close"}
            if dist.is_initialized():
                dist.broadcast_object_list(message, src=0)
            request = message[0]
            operation = request["operation"]
            if operation == "close":
                break
            try:
                if operation == "start":
                    runtime.start(request["static"], manifest=request.get("manifest"),
                                  sample_id=request.get("sample_id"))
                elif operation == "generate":
                    with safe_open(request["tokens"], framework="pt", device="cpu") as payload:
                        if (payload.metadata() or {}).get("version") != VERSION:
                            raise ValueError("Token artifact version differs")
                    payload = load_file(request["tokens"])
                    if set(payload) != {"encoded_current"}:
                        raise ValueError("Expected encoded_current only")
                    clean = runtime.generate(payload["encoded_current"], steps=request["steps"])
                    if runtime.rank == 0:
                        save_file(
                            {"latents": clean.cpu().contiguous()}, request["output"],
                            metadata={"version": cfg["runtime"]["experiment_version"],
                                      "stage": runtime.meta["stage"]},
                        )
                elif operation == "reset":
                    runtime.reset()
                else:
                    raise ValueError("Unknown model operation")
                if dist.is_initialized():
                    dist.barrier()
                if runtime.rank == 0:
                    respond({"complete": True})
            except Exception as error:
                traceback.print_exc()
                if runtime.rank == 0:
                    respond({"error": f"{type(error).__name__}: {error}"})
                # A rank failure may leave peers inside a collective. Exit the
                # entire torchrun job; the server starts a fresh session next time.
                raise
    finally:
        if runtime is not None:
            runtime.reset()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
