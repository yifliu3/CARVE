"""Stable test selection shared by standalone inference and its supervisor."""


def inference_steps(cfg, steps=None):
    """Use the causal solver budget from the training recipe unless overridden."""
    if cfg["stage"] not in {"bidirectional_fm", "causal_tf_anyflow", "causal_dmd_sgf"}:
        raise ValueError(
            "Test evaluation supports bidirectional_fm, causal_tf_anyflow and causal_dmd_sgf"
        )
    from .config import validate_conditioning

    validate_conditioning(cfg)
    if steps is None:
        steps = 30 if cfg["stage"] == "bidirectional_fm" else cfg["train"]["denoise_steps"]
    if type(steps) is not int or steps < 1:
        raise ValueError("Inference steps must be a positive integer")
    return steps


def generate_test_sample(core, condition, *, stage, steps, seed, progress=None):
    """Generation never receives GT; causal calls receive only the current block."""
    import time

    import torch

    from solarwm.backends.minimax_h3.torch_flow import make_shifted_schedule

    if condition.target is not None:
        raise ValueError("Clear target latents before test generation")
    device = condition.device
    if stage == "bidirectional_fm":
        rng = torch.Generator(device=device).manual_seed(seed)
        current = torch.randn(condition.video_shape, generator=rng, device=device)
        schedule = make_shifted_schedule(steps + 1, shift=core.shift, device=device)
        for index, t in enumerate(schedule.timesteps):
            velocity = core.forward(condition, current, t, mode="bid")
            current = current + (schedule.sigmas[index] - schedule.sigmas[index + 1]) * velocity
        return current, {"sampling_mode": "bidirectional"}
    if stage not in {"causal_tf_anyflow", "causal_dmd_sgf"} or not core.anyflow:
        raise ValueError("Causal evaluation requires an AnyFlow model")
    from .streaming import CausalSession

    count = condition.video_shape[2] // 5
    session = CausalSession(core, condition, seed=seed)
    chunks, timings = [], []
    for index in range(count):
        started = time.monotonic()
        chunk = session.generate_token_chunk(
            condition.proxy[:, :, index * 5 : (index + 1) * 5],
            steps=steps,
            sgf=stage == "causal_dmd_sgf",
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        expected = list(range(max(0, index - core.history + 1), index + 1))
        if not session.cache.layers:
            raise ValueError("Causal generation failed to populate history")
        for entries in session.cache.layers.values():
            if sorted({run.chunk for run, _, _ in entries}) != expected:
                raise ValueError("Causal history cache has an invalid window")
        timing = {
            "chunk": index,
            "seed": seed + index,
            "seconds": time.monotonic() - started,
            "history_chunks": expected,
        }
        timings.append(timing)
        if progress is not None:
            progress(timing)
        chunks.append(chunk)
    return torch.cat(chunks, 2), {
        "sampling_mode": "causal_sgf" if stage == "causal_dmd_sgf" else "causal_anyflow",
        "conditioning_scope": "current_chunk",
        "chunk_count": count,
        "chunk_timings": timings,
        "target_video_input": False,
    }


def sample_indices(size, nsample=10, *, shard=0, shards=1):
    """Apply the global video limit before distributing samples across nodes."""
    if type(nsample) is not int or (nsample != -1 and nsample < 1):
        raise ValueError("nsample must be a positive integer or -1 for the full test set")
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Invalid shard or shard count")
    count = size if nsample == -1 else min(size, nsample)
    return list(range(count))[shard::shards]
