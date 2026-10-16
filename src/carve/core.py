"""Shared TAE-token H3 forwards for three-stage training."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from solarwm.backends.minimax_h3.anyflow_loss import h3_anyflow_v15_loss
from solarwm.backends.minimax_h3.distributed import is_sequence_parallel_enabled
from solarwm.backends.minimax_h3.torch_flow import (
    make_shifted_schedule,
    predict_clean_sample,
    sample_shifted_timestep,
    scale_noise,
)
from solarwm.training.sgf import compute_sgf_kl_gradient, sgf_student_loss

from .attention import CWMControl, KVCache
from .layout import build_layout, patchify_video, unpatchify_video


@dataclass
class Condition:
    prompt: torch.Tensor
    tags: torch.Tensor
    references: list[torch.Tensor]
    proxy: torch.Tensor | None
    audio: torch.Tensor
    target: torch.Tensor | None = None
    output_size: tuple[int, int] | None = None  # Target latent H/W for a smaller token proxy.

    @property
    def video_shape(self):
        if self.proxy is not None:
            if self.output_size is not None:
                if len(self.output_size) != 2 or any(
                    type(v) is not int or v < 2 or v % 2 for v in self.output_size
                ):
                    raise ValueError("Token-proxy output_size must contain even target latent H/W")
                return (*self.proxy.shape[:3], *self.output_size)
            return self.proxy.shape
        raise ValueError("Generation requires proxy tokens to define video geometry")

    @property
    def device(self):
        return self.prompt.device

    def to(self, device):
        return Condition(
            self.prompt.to(device),
            self.tags.to(device),
            [x.to(device).float() for x in self.references],
            None if self.proxy is None else self.proxy.to(device).float(),
            self.audio.to(device).float(),
            None if self.target is None else self.target.to(device).float(),
            self.output_size,
        )


class Core:
    def __init__(self, model, *, history=5, shift=12.0, anyflow=False):
        self.model = model
        self.history, self.shift, self.anyflow = history, shift, anyflow

    def forward(
        self, condition, noisy, t, *, clean=None, r=None, mode="tf", cache=None, chunk=None
    ):
        count = condition.video_shape[2] // 5
        streaming = chunk is not None and condition.video_shape[2] == 5
        if streaming:
            count = chunk + 1
        layout = build_layout(
            condition.tags,
            condition.references,
            condition.audio.shape[1],
            chunks=count,
            height=noisy.shape[-2],
            width=noisy.shape[-1],
            mode=mode,
            current_chunk=chunk,
            proxy_height=condition.proxy.shape[-2],
            proxy_width=condition.proxy.shape[-1],
        )
        parts = [patchify_video(x) for x in condition.references]
        proxy = (
            condition.proxy
            if chunk is None or streaming
            else condition.proxy[:, :, chunk * 5 : (chunk + 1) * 5]
        )
        parts.append(patchify_video(proxy))
        if mode == "tf":
            if clean is None or tuple(clean.shape) != tuple(condition.video_shape):
                raise ValueError(
                    "TF/replay requires a clean history with the complete target shape"
                )
            parts.append(patchify_video(clean.detach()))
        parts.append(patchify_video(noisy))
        # Static references/proxy/silence are clean and immutable. Native H3 t=1 is clean.
        time = torch.as_tensor(t, device=noisy.device, dtype=torch.float32).reshape(())
        times = torch.stack((torch.ones_like(time), time))
        rt = None
        if self.anyflow:
            endpoint = (
                torch.ones_like(time)
                if r is None
                else torch.as_tensor(r, device=noisy.device).reshape(())
            )
            rt = torch.stack((torch.ones_like(time), endpoint))
        control = CWMControl(layout, self.model.rope, mode, self.history, cache, chunk)
        control.checkpoint_segments = getattr(self.model, "cwm_checkpoint_segments", True)
        video, _ = self.model(
            hidden_states=torch.cat(parts, 1),
            audio_hidden_states=condition.audio,
            encoder_hidden_states=condition.prompt,
            timestep=times,
            r_timestep=rt,
            attention_mask=control,
            fused_prope=False,
            packed_sequence_parallel=is_sequence_parallel_enabled(),
            return_dict=False,
            **layout.kwargs(),
        )
        return unpatchify_video(
            video[:, layout.output_slice], noisy.shape[2], noisy.shape[-2], noisy.shape[-1]
        ).float()

    def fm_loss(self, condition, *, bidirectional=False):
        clean = condition.target
        if clean is None:
            raise ValueError("Training requires target latents")
        noise = torch.randn_like(clean)
        t = sample_shifted_timestep(1, self.shift, clean.device, torch.float32)[0]
        prediction = self.forward(
            condition,
            scale_noise(clean, noise, t),
            t,
            clean=clean,
            mode="bid" if bidirectional else "tf",
        )
        return F.mse_loss(prediction, clean - noise)

    def anyflow_loss(self, condition, *, sample_type, process_group=None):
        # Cycle all three types even with microbatch=1/DP1. Solar's rank-rounded
        # type allocation otherwise leaves some types absent on small jobs.
        clean = condition.target
        noise = torch.randn_like(clean)
        u = torch.rand(1, device=clean.device)
        sigma_t = self.shift * u / (1 + (self.shift - 1) * u)
        if sample_type == "diffusion":
            sigma_r = sigma_t
        elif sample_type == "consistency":
            sigma_r = torch.zeros_like(sigma_t)
        elif sample_type == "map":
            sigma_r = sigma_t * torch.rand_like(sigma_t)
        else:
            raise ValueError(sample_type)
        return h3_anyflow_v15_loss(
            lambda x, t, r: self.forward(condition, x, t, r=r, clean=clean),
            clean,
            noise,
            sigma_t,
            sigma_r,
            torch.tensor([sample_type == "diffusion"], device=clean.device),
            epsilon=5.0,
            shift=self.shift,
            process_group=process_group,
        )

    def iter_generate(self, condition, *, steps=4, seed=42, sgf=False, exit_index=None):
        """Yield clean chunks; same raw-KV commit semantics as training rollout."""
        device = condition.device
        rng = torch.Generator(device=device).manual_seed(seed)
        schedule = make_shifted_schedule(steps + 1, shift=self.shift, device=device)
        cache = KVCache(self.history)
        exit_index = steps - 1 if exit_index is None else exit_index
        if not 0 <= exit_index < steps:
            raise ValueError("Invalid rollout exit step")
        try:
            with torch.no_grad():
                for chunk in range(condition.video_shape[2] // 5):
                    shape = (*condition.video_shape[:2], 5, *condition.video_shape[3:])
                    current = torch.randn(shape, generator=rng, device=device)
                    saved = None
                    for step, t in enumerate(schedule.timesteps):
                        # SGF predicts a clean endpoint at each denoising step;
                        # ordinary AnyFlow sampling predicts the next endpoint.
                        r = (
                            (1.0 if sgf else 1 - schedule.sigmas[step + 1])
                            if self.anyflow
                            else None
                        )
                        velocity = self.forward(
                            condition, current, t, r=r, mode="rollout", cache=cache, chunk=chunk
                        )
                        x0 = predict_clean_sample(current, velocity, t)
                        if step == exit_index:
                            saved = (x0.detach(), current.detach().clone(), t.detach(), r)
                        if step + 1 < steps:
                            if sgf:
                                eps = torch.randn(shape, generator=rng, device=device)
                                current = scale_noise(x0, eps, schedule.timesteps[step + 1])
                            else:
                                current = (
                                    current
                                    + (schedule.sigmas[step] - schedule.sigmas[step + 1]) * velocity
                                )
                    committed = saved[0]
                    self.forward(
                        condition, committed, 1.0, r=1.0, mode="commit", cache=cache, chunk=chunk
                    )
                    yield saved
        finally:
            cache.layers.clear()

    def rollout(self, condition, *, steps=4, seed=42, sgf=False, exit_index=None):
        chunks = list(
            self.iter_generate(condition, steps=steps, seed=seed, sgf=sgf, exit_index=exit_index)
        )
        return (
            torch.cat([x[0] for x in chunks], 2),
            torch.cat([x[1] for x in chunks], 2),
            chunks[0][2],
            chunks[0][3],
        )

    def replay(self, condition, rollout):
        clean, noisy, t, r = rollout
        v = self.forward(condition, noisy, t, r=r, clean=clean)
        return predict_clean_sample(noisy, v, t)


def _dmd_score_condition(condition, score_condition):
    if score_condition is None:
        score_condition = condition
    if score_condition.video_shape != condition.video_shape:
        raise ValueError("Student and score proxy geometries differ")
    if (score_condition.output_size is None) != (condition.output_size is None):
        raise ValueError("Token-proxy DMD requires matching score-model conditioning")
    return score_condition


def dmd_student_loss(student, teacher, critic, condition, *, seed, steps=4, score_condition=None):
    score_condition = _dmd_score_condition(condition, score_condition)
    exit_index = int(torch.randint(steps, (), device=condition.device))
    rollout = student.rollout(condition, steps=steps, seed=seed, sgf=True, exit_index=exit_index)
    output = student.replay(condition, rollout)
    with torch.no_grad():
        t = sample_shifted_timestep(1, student.shift, output.device, torch.float32)[0].clamp(
            0.02, 0.98
        )
        noise = torch.randn_like(output)
        xt = scale_noise(output.detach(), noise, t)
        real = predict_clean_sample(xt, teacher.forward(score_condition, xt, t, mode="bid"), t)
        fake = predict_clean_sample(xt, critic.forward(score_condition, xt, t, mode="bid"), t)
        gradient = compute_sgf_kl_gradient(
            fake_x0=fake, real_x0=real, student_output=output.detach()
        )
    return sgf_student_loss(output, gradient)


def dmd_critic_loss(student, critic, condition, *, seed, steps=4, score_condition=None):
    score_condition = _dmd_score_condition(condition, score_condition)
    with torch.no_grad():
        rollout = student.rollout(condition, steps=steps, seed=seed, sgf=True)
        clean = student.replay(condition, rollout).detach()
        noise = torch.randn_like(clean)
        t = sample_shifted_timestep(1, student.shift, clean.device, torch.float32)[0].clamp(
            0.02, 0.98
        )
        xt = scale_noise(clean, noise, t)
    return F.mse_loss(critic.forward(score_condition, xt, t, mode="bid"), clean - noise)
