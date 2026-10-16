"""Three-stage v4 training with EMA and exact resume."""

import json
import random
import time
import traceback
from pathlib import Path

import torch

from solarwm.backends.minimax_h3.distributed import state, sync_lora_gradients
from solarwm.backends.minimax_h3.optimizer import FP32MasterAdamW
from solarwm.training.sgf import should_update_student

from . import checkpoint
from .config import score_config, validate_conditioning
from .core import Core, dmd_critic_loss, dmd_student_loss
from .data import EncodedDataset, digest
from .model import build, initialize_distributed


class Runtime:
    def __init__(self, cfg, *, dataset=None):
        validate_conditioning(cfg)
        self.cfg = cfg
        stage = cfg["stage"]
        self.device, self.rank, self.world = initialize_distributed(cfg)
        # Checkpoint metadata consists of CPU RNG/cursor records. Keep its
        # collectives independent of NCCL's training communicators and buffers.
        self.checkpoint_group = None
        if torch.distributed.is_initialized() and cfg["checkpoint"].get("async_save"):
            self.checkpoint_group = torch.distributed.new_group(backend="gloo")
        self.step, self.student_step, self.cursor = 0, 0, 0
        seed = cfg["train"]["seed"]
        torch.manual_seed(seed)
        random.seed(seed)
        self.dataset = (
            dataset
            if dataset is not None
            else EncodedDataset(
                cfg["data"]["manifest"],
                split=cfg["data"]["split"],
                checkpoint_id=cfg["model"]["checkpoint_id"],
                qwen_dynamic_proxy=cfg["data"].get("qwen_dynamic_proxy", False),
                qwen_proxy_scope="reference_only",
                stage=cfg["stage"],
                camera_conditioning=cfg["model"].get("camera_conditioning", "none"),
                proxy_conditioning="tae_tokens",
                visual_encoder=cfg["data"]["visual_encoder"],
                proxy_variant=cfg["data"]["proxy_variant"],
            )
        )
        if isinstance(self.dataset, EncodedDataset):
            if self.dataset.visual_encoder != cfg["data"]["visual_encoder"]:
                raise ValueError("Dataset and configured reference/target visual encoder differ")
            if cfg["model"].get("proxy_conditioning") == "tae_tokens":
                for axis in ("height", "width"):
                    if self.dataset.codec["proxy_" + axis] != cfg["model"]["proxy_input_" + axis]:
                        raise ValueError("Model and dataset proxy input dimensions differ")
            self.dataset.validate_conditioning(
                qwen_dynamic_proxy=cfg["data"].get("qwen_dynamic_proxy", False),
                qwen_proxy_scope="reference_only",
                stage=cfg["stage"],
            )
        self.score_cfg = score_config(cfg) if stage == "causal_dmd_sgf" else None
        self.dataset_digest = (
            "synthetic-test-only" if dataset is not None else digest(cfg["data"]["manifest"])
        )
        af = stage in {"causal_tf_anyflow", "causal_dmd_sgf"}
        self.student = build(cfg, self.device, anyflow=af)
        self.teacher = self.critic = self.critic_optimizer = None
        init = cfg["checkpoint"]["initialize"]
        resuming = bool(cfg["checkpoint"]["resume"])
        if init and not resuming:
            allowed = {
                "causal_tf_anyflow": {"bidirectional_fm"},
                "causal_dmd_sgf": {"causal_tf_anyflow"},
            }.get(stage, {stage})
            checkpoint.initialize(
                self.student,
                init,
                cfg,
                allowed_stages=allowed,
                source=cfg["checkpoint"]["initialize_weights"],
            )
        elif stage != "bidirectional_fm" and not resuming and dataset is None:
            raise ValueError("This stage requires checkpoint.initialize")
        self.core = Core(
            self.student.model,
            history=cfg["model"]["history_chunks"],
            shift=cfg["train"]["video_shift"],
            anyflow=af,
        )
        self.optimizer = self.make_optimizer(self.student, cfg["train"]["learning_rate"])
        if stage == "causal_dmd_sgf":
            # Three resident models must release gathered frozen blocks after
            # each forward, including the bidirectional score models.
            self.teacher = build(self.score_cfg, self.device, frozen=True, root_wrap=True)
            self.critic = build(self.score_cfg, self.device, root_wrap=True)
            teacher_path = cfg["checkpoint"]["teacher"]
            if teacher_path:
                for role in (self.teacher, self.critic):
                    checkpoint.initialize(
                        role,
                        teacher_path,
                        self.score_cfg,
                        allowed_stages={"bidirectional_fm"},
                        source=cfg["checkpoint"].get("teacher_weights", "live"),
                    )
            elif dataset is None:
                raise ValueError("DMD requires checkpoint.teacher from bidirectional adaptation")
            self.teacher_core = Core(
                self.teacher.model, history=self.core.history, shift=self.core.shift
            )
            self.critic_core = Core(
                self.critic.model, history=self.core.history, shift=self.core.shift
            )
            self.critic_optimizer = self.make_optimizer(
                self.critic, cfg["train"]["critic_learning_rate"]
            )
        self.ema = {k: v.float() for k, v in checkpoint.values(self.student).items()}
        # Different logical DP samples/noise, identical within each SP group.
        torch.manual_seed(seed + state().dp_rank)
        if resuming:
            checkpoint.resume(self, cfg["checkpoint"]["resume"])

    def make_optimizer(self, handle, lr):
        return FP32MasterAdamW(
            handle.trainables.values(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=self.cfg["train"]["weight_decay"],
            state_device="cpu" if self.cfg["runtime"]["optimizer_state_offload"] else None,
        )

    def next_batch(self):
        topology = state()
        index = (self.cursor * topology.dp_world_size + topology.dp_rank) % len(self.dataset)
        self.cursor += 1
        return self.dataset[index].to(self.device)

    def finish(self, handle, optimizer):
        params = tuple(handle.trainables.values())
        sync_lora_gradients(params)
        norm = torch.nn.utils.clip_grad_norm_(
            params, self.cfg["train"]["grad_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(norm)

    def update_ema(self):
        decay = self.cfg["train"]["ema_decay"]
        for key, param in self.student.trainables.items():
            self.ema[key].lerp_(param.detach().float().cpu(), 1 - decay)

    def train_step(self):
        cfg, stage = self.cfg["train"], self.cfg["stage"]
        accum, sp = cfg["grad_accum"], state().sp_size
        self.optimizer.zero_grad(set_to_none=True)
        self.student.model.train()
        if self.critic is not None:
            self.critic.model.train()
            self.critic_optimizer.zero_grad(set_to_none=True)
        student_update = stage != "causal_dmd_sgf" or should_update_student(
            self.step, cfg["critic_updates_per_student"]
        )
        loss_sum = critic_sum = 0.0
        for micro in range(accum):
            condition = self.next_batch()
            score_condition = None
            if stage == "bidirectional_fm":
                loss = self.core.fm_loss(condition, bidirectional=True)
            elif stage == "causal_tf_anyflow":
                kind = ["diffusion", "diffusion", "consistency", "map"][
                    (self.step * accum + micro + state().dp_rank) % 4
                ]
                loss = self.core.anyflow_loss(
                    condition,
                    sample_type=kind,
                    process_group=state().dp_group if state().dp_world_size > 1 else None,
                )
            elif student_update:
                loss = dmd_student_loss(
                    self.core,
                    self.teacher_core,
                    self.critic_core,
                    condition,
                    seed=cfg["seed"] + self.cursor,
                    steps=cfg["denoise_steps"],
                    score_condition=score_condition,
                )
            else:
                loss = None
            if loss is not None:
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite student loss")
                (loss / (accum * sp)).backward()
                loss_sum += float(loss.detach()) / accum
            if stage == "causal_dmd_sgf":
                closs = dmd_critic_loss(
                    self.core,
                    self.critic_core,
                    condition,
                    seed=cfg["seed"] + 100000 + self.cursor,
                    steps=cfg["denoise_steps"],
                    score_condition=score_condition,
                )
                (closs / (accum * sp)).backward()
                critic_sum += float(closs.detach()) / accum
        warmup = cfg["warmup_steps"]
        for group in self.optimizer.param_groups:
            group["lr"] = cfg["learning_rate"] * min(1.0, (self.student_step + 1) / max(1, warmup))
        norm = 0.0
        critic_norm = 0.0
        if student_update:
            norm = self.finish(self.student, self.optimizer)
            self.student_step += 1
            self.update_ema()
        if self.critic is not None:
            critic_norm = self.finish(self.critic, self.critic_optimizer)
        self.step += 1
        return dict(
            step=self.step,
            student_step=self.student_step,
            loss=loss_sum,
            critic_loss=critic_sum,
            grad_norm=norm,
            critic_grad_norm=critic_norm,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            critic_learning_rate=0.0
            if self.critic_optimizer is None
            else self.critic_optimizer.param_groups[0]["lr"],
            student_updated=student_update,
        )


def run(cfg):
    out = Path(cfg["runtime"]["output_dir"])
    if (out / "resolved-config.json").exists() and not cfg["checkpoint"]["resume"]:
        raise FileExistsError(
            "Output already has a training run; resume or choose a new output_dir"
        )
    runtime = Runtime(cfg)
    out.mkdir(parents=True, exist_ok=True)
    if runtime.rank == 0:
        (out / "resolved-config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    writer = None
    checkpoint_writer = None
    if runtime.rank == 0 and cfg["runtime"].get("tensorboard_dir"):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            log_dir=cfg["runtime"]["tensorboard_dir"],
            purge_step=runtime.step + 1 if cfg["checkpoint"]["resume"] else None,
            flush_secs=10,
        )
        writer.add_text("run/config", json.dumps(cfg, indent=2), runtime.step)
    failed = False
    try:
        while runtime.step < cfg["train"]["max_steps"]:
            # All ranks participate in error propagation. Successful writes do
            # not add a disk/network wait to the training step.
            _poll_checkpoint_writer(runtime, checkpoint_writer, writer)
            start = time.monotonic()
            metrics = runtime.train_step()
            metrics["seconds"] = time.monotonic() - start
            if torch.distributed.is_initialized():
                # SP peers compute the same loss; the world mean is the logical
                # DP mean. Keep TensorBoard representative of the full 128-card run.
                keys = ("loss", "critic_loss")
                reduced = torch.tensor([metrics[key] for key in keys], device=runtime.device)
                torch.distributed.all_reduce(reduced)
                reduced /= runtime.world
                for key, value in zip(keys, reduced.tolist(), strict=True):
                    metrics[key] = value
            if runtime.device.type == "cuda":
                metrics["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                metrics["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            if runtime.rank == 0:
                line = json.dumps(metrics)
                print(line, flush=True)
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(line + "\n")
                if writer is not None:
                    for key, value in metrics.items():
                        if key != "step":
                            writer.add_scalar(f"train/{key}", value, runtime.step)
            if (
                runtime.step % cfg["train"]["save_every"] == 0
                or runtime.step == cfg["train"]["max_steps"]
            ):
                path = out / f"checkpoint-{runtime.step:06d}"
                if cfg["checkpoint"].get("async_save"):
                    checkpoint_writer, info = checkpoint.save_async(
                        runtime, checkpoint_writer, path
                    )
                    if info is not None:
                        _record_checkpoint_events(runtime, [info], writer)
                else:
                    checkpoint.save(runtime, path)
                if writer is not None:
                    writer.flush()
    except BaseException:
        failed = True
        traceback.print_exc()
        if checkpoint_writer is not None:
            checkpoint_writer.abort()
        raise
    finally:
        try:
            if not failed:
                _poll_checkpoint_writer(runtime, checkpoint_writer, writer, close=True)
        finally:
            if writer is not None:
                writer.close()
    return runtime


def _record_checkpoint_events(runtime, events, tb_writer):
    for event in events:
        print(json.dumps({"checkpoint": event}), flush=True)
        if tb_writer is not None:
            for key in ["snapshot_seconds", "snapshot_bytes", "seconds"]:
                if key in event:
                    tb_writer.add_scalar(
                        f"checkpoint/{event['event']}/{key}", event[key], event["step"]
                    )


def _poll_checkpoint_writer(runtime, checkpoint_writer, tb_writer, *, close=False):
    if not runtime.cfg["checkpoint"].get("async_save"):
        return
    failure = None
    try:
        if checkpoint_writer is not None:
            events = checkpoint_writer.close() if close else checkpoint_writer.poll()
            _record_checkpoint_events(runtime, events, tb_writer)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    if torch.distributed.is_initialized():
        status = [failure]
        torch.distributed.broadcast_object_list(status, src=0, group=runtime.checkpoint_group)
        failure = status[0]
    if failure:
        raise RuntimeError(failure)
