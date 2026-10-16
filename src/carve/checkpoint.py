"""Atomic per-rank training state and explicit, weights-only stage transitions."""

import json
import random
from pathlib import Path

import torch

from .config import model_contract


def values(handle):
    return {k: v.detach().cpu().clone() for k, v in handle.trainables.items()}


def copy_values(handle, state, *, allow_new_anyflow=False):
    missing, extra = set(handle.trainables) - state.keys(), state.keys() - set(handle.trainables)
    allowed = (
        {k for k in missing if k.startswith("delta_embedding.")} if allow_new_anyflow else set()
    )
    if missing - allowed or extra:
        raise ValueError(
            f"Checkpoint parameter mismatch: missing={missing - allowed}, extra={extra}"
        )
    with torch.no_grad():
        for name, value in state.items():
            param = handle.trainables[name]
            if param.shape != value.shape:
                raise ValueError(f"Checkpoint shape mismatch: {name}")
            param.copy_(value.to(param.device, param.dtype))


def read(path, cfg):
    root = Path(path)
    marker = root / "COMPLETE.json"
    if not marker.is_file():
        marker = root / "LOCAL_COMPLETE.json"
    meta = json.loads(marker.read_text())
    if meta.get("schema") != "cwm-h3.checkpoint.v1":
        raise ValueError("Not a CARVE checkpoint")
    expected = model_contract(cfg)
    if meta["contract"] != expected:
        raise ValueError("Checkpoint base/conditioning/LoRA contract differs")
    return root, meta


def initialize(handle, path, cfg, *, allowed_stages, source="live"):
    root, meta = read(path, cfg)
    if meta["stage"] not in allowed_stages:
        raise ValueError(f"Cannot initialize this role from {meta['stage']}")
    validate_files(root, meta, [f"{source}.pt"] if meta.get("split_weights") else ["weights.pt"])
    selected = load_weights(root, meta, source)
    if selected is None:
        raise ValueError(f"No {source} weights in checkpoint")
    copy_values(handle, selected, allow_new_anyflow=handle.anyflow and not meta["anyflow"])
    return meta


def load_weights(root, meta, source):
    if meta.get("split_weights"):
        path = root / f"{source}.pt"
        return torch.load(path, map_location="cpu", weights_only=True) if path.is_file() else None
    return torch.load(root / "weights.pt", map_location="cpu", weights_only=True)[source]


def validate_files(root, meta, names):
    """Check the exact requested payloads before changing any live state.

    Each node leader hashes its local replica once, then all training ranks
    agree on success. Synchronous checkpoints require nonempty files; async
    split checkpoints require hashes for every payload.
    """
    import os

    import torch.distributed as dist

    from .async_checkpoint import digest

    distributed = dist.is_initialized()
    leader = not distributed or int(os.environ.get("LOCAL_RANK", dist.get_rank())) == 0
    failure = None
    if leader:
        try:
            files = meta.get("files", {})
            for name in names:
                path = root / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Checkpoint missing required file: {path}")
                item = files.get(name)
                if meta.get("split_weights") and item is None:
                    raise ValueError(f"Checkpoint integrity metadata missing for {name}")
                if item is not None and (
                    path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]
                ):
                    raise ValueError(f"Checkpoint integrity mismatch: {path}")
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
    if distributed:
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, failure)
        failure = next((x for x in failures if x is not None), None)
    if failure:
        raise ValueError(failure)


def metadata(runtime):
    return dict(
        schema="cwm-h3.checkpoint.v1",
        contract=model_contract(runtime.cfg),
        stage=runtime.cfg["stage"],
        anyflow=runtime.student.anyflow,
        world_size=runtime.world,
        sp_size=runtime.cfg["distributed"]["sp_size"],
        step=runtime.step,
        student_step=runtime.student_step,
        config=runtime.cfg,
        dataset_digest=runtime.dataset_digest,
        shared_optimizer=runtime.cfg["checkpoint"].get("shared_optimizer", False),
    )


def rank_state(runtime):
    return dict(
        step=runtime.step,
        student_step=runtime.student_step,
        cursor=runtime.cursor,
        optimizer=None,
        critic_optimizer=None,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state() if runtime.device.type == "cuda" else None,
        python_rng=random.getstate(),
        dataset_digest=runtime.dataset_digest,
    )


def save_async(runtime, writer, path):
    """Gather small RNG/cursor records; snapshot replicated state only on rank0."""
    import torch.distributed as dist

    from .async_checkpoint import AsyncCheckpointWriter

    if not runtime.cfg["checkpoint"].get("shared_optimizer"):
        raise ValueError("Async save currently requires replicated/shared optimizer")
    local = rank_state(runtime)
    records = [None] * runtime.world if runtime.rank == 0 else None
    if dist.is_initialized():
        dist.gather_object(local, records, dst=0, group=runtime.checkpoint_group)
    else:
        records = [local]
    info, failure = None, None
    try:
        if runtime.rank == 0:
            if writer is None:
                writer = AsyncCheckpointWriter(
                    Path(runtime.cfg["checkpoint"]["local_dir"]) / "writer-status.json"
                )
            files = {
                "live.pt": runtime.student.trainables,
                "ema.pt": runtime.ema,
                "optimizer.pt": {
                    "optimizer": runtime.optimizer.state_dict(),
                    "critic_optimizer": None
                    if runtime.critic_optimizer is None
                    else runtime.critic_optimizer.state_dict(),
                },
                **{f"rank-{rank}.pt": state for rank, state in enumerate(records)},
            }
            if runtime.critic is not None:
                files["critic.pt"] = runtime.critic.trainables
            meta = {**metadata(runtime), "split_weights": True}
            local_path = Path(runtime.cfg["checkpoint"]["local_dir"]) / Path(path).name
            info = writer.submit(files, meta, local_path, path)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        if runtime.rank == 0 and writer is not None:
            writer.abort()
    # No rank starts another optimizer update until the immutable snapshot is
    # complete. This also propagates staging failures to every rank.
    if dist.is_initialized():
        status = [failure]
        dist.broadcast_object_list(status, src=0, group=runtime.checkpoint_group)
        failure = status[0]
    if failure:
        raise RuntimeError(f"Checkpoint staging failed: {failure}")
    return writer, info


def save(runtime, path):
    import torch.distributed as dist

    root = Path(path)
    if (root / "COMPLETE.json").exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    shared_optimizer = runtime.cfg["checkpoint"].get("shared_optimizer", False)
    payload = dict(
        step=runtime.step,
        student_step=runtime.student_step,
        cursor=runtime.cursor,
        optimizer=None if shared_optimizer else runtime.optimizer.state_dict(),
        critic_optimizer=None
        if shared_optimizer or runtime.critic_optimizer is None
        else runtime.critic_optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state() if runtime.device.type == "cuda" else None,
        python_rng=random.getstate(),
        dataset_digest=runtime.dataset_digest,
    )
    temporary = root / f"rank-{runtime.rank}.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(root / f"rank-{runtime.rank}.pt")
    if runtime.rank == 0:
        if shared_optimizer:
            # LoRA/target-time parameters and gradients are replicated and explicitly
            # synchronized across SP/DP. Adam slots are identical on every rank.
            torch.save(
                {
                    "optimizer": runtime.optimizer.state_dict(),
                    "critic_optimizer": None
                    if runtime.critic_optimizer is None
                    else runtime.critic_optimizer.state_dict(),
                },
                root / "optimizer.pt.tmp",
            )
            (root / "optimizer.pt.tmp").replace(root / "optimizer.pt")
        torch.save(
            dict(
                live=values(runtime.student),
                ema=runtime.ema,
                critic=None if runtime.critic is None else values(runtime.critic),
            ),
            root / "weights.pt.tmp",
        )
        (root / "weights.pt.tmp").replace(root / "weights.pt")
    if dist.is_initialized():
        dist.barrier()
    if runtime.rank == 0:
        meta = dict(
            schema="cwm-h3.checkpoint.v1",
            contract=model_contract(runtime.cfg),
            stage=runtime.cfg["stage"],
            anyflow=runtime.student.anyflow,
            world_size=runtime.world,
            sp_size=runtime.cfg["distributed"]["sp_size"],
            step=runtime.step,
            student_step=runtime.student_step,
            config=runtime.cfg,
            dataset_digest=runtime.dataset_digest,
            shared_optimizer=shared_optimizer,
        )
        (root / "COMPLETE.json.tmp").write_text(json.dumps(meta, indent=2) + "\n")
        (root / "COMPLETE.json.tmp").replace(root / "COMPLETE.json")
    if dist.is_initialized():
        dist.barrier()


def resume(runtime, path):
    root, meta = read(path, runtime.cfg)
    if (
        meta["stage"] != runtime.cfg["stage"]
        or meta["world_size"] != runtime.world
        or meta["sp_size"] != runtime.cfg["distributed"]["sp_size"]
    ):
        raise ValueError("Resume requires the same stage and distributed topology")
    if meta["dataset_digest"] != runtime.dataset_digest:
        raise ValueError("Resume dataset manifest changed")
    for key in ("visual_encoder", "proxy_variant", "split"):
        if meta["config"]["data"].get(key) != runtime.cfg["data"][key]:
            raise ValueError(f"Resume data.{key} changed")
    # Only scheduling-independent output/step-limit fields may change on resume.
    previous, current = dict(meta["config"]["train"]), dict(runtime.cfg["train"])
    for key in ["max_steps", "save_every"]:
        previous.pop(key)
        current.pop(key)
    if previous != current:
        raise ValueError(
            "Resume optimizer/RNG/training settings changed; use initialize for a new stage"
        )
    if meta["config"]["checkpoint"]["teacher"] != runtime.cfg["checkpoint"]["teacher"]:
        raise ValueError("Resume must use the original frozen teacher checkpoint")
    if runtime.cfg["stage"] == "causal_dmd_sgf" and (
        meta["config"]["checkpoint"].get("teacher_weights")
        != runtime.cfg["checkpoint"]["teacher_weights"]
    ):
        raise ValueError("Resume must use the original frozen teacher weight source")
    names = ["live.pt", "ema.pt"] if meta.get("split_weights") else ["weights.pt"]
    if meta.get("split_weights") and runtime.critic is not None:
        names.append("critic.pt")
    names += [f"rank-{rank}.pt" for rank in range(runtime.world)]
    if meta.get("shared_optimizer"):
        names.append("optimizer.pt")
    validate_files(root, meta, names)
    state = torch.load(root / f"rank-{runtime.rank}.pt", map_location="cpu", weights_only=True)
    if (
        state["step"] != meta["step"]
        or state["student_step"] != meta["student_step"]
        or state["dataset_digest"] != meta["dataset_digest"]
    ):
        raise ValueError("Checkpoint rank state does not match its completion metadata")
    copy_values(runtime.student, load_weights(root, meta, "live"))
    runtime.ema = load_weights(root, meta, "ema")
    if runtime.critic is not None:
        copy_values(runtime.critic, load_weights(root, meta, "critic"))
    optimizer_state = (
        torch.load(root / "optimizer.pt", map_location="cpu", weights_only=True)
        if meta.get("shared_optimizer", False)
        else state
    )
    runtime.optimizer.load_state_dict(optimizer_state["optimizer"])
    if runtime.critic_optimizer is not None:
        runtime.critic_optimizer.load_state_dict(optimizer_state["critic_optimizer"])
    runtime.step, runtime.student_step, runtime.cursor = (
        state["step"],
        state["student_step"],
        state["cursor"],
    )
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
