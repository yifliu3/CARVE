"""Interactive generation from only the current TAE proxy-token block."""

import torch

from solarwm.backends.minimax_h3.torch_flow import make_shifted_schedule, predict_clean_sample

from .attention import KVCache
from .core import Condition


class CausalSession:
    """Fixed reference, bounded past KV, and atomic commit after a complete block.

    A control action defines the next 17 frames (five latents). Future controls
    are not accepted. Discard unshown work before a new action; once a block is
    displayed/committed, it cannot be rewritten. Use propose_token_chunk then
    commit to discard an unshown block when controls change.
    """

    def __init__(self, core, condition, *, seed=42):
        self.core = core
        self.seed = seed
        if condition.proxy is None:
            raise ValueError("A token session requires proxy latents")
        self.static = Condition(
            condition.prompt,
            condition.tags,
            condition.references,
            condition.proxy[:, :, :0],
            condition.audio,
            output_size=condition.output_size,
        )
        self.reset()

    def reset(self):
        self.cache = KVCache(self.core.history)
        self.index = 0
        self.pending = None

    def discard_pending(self):
        """Keep committed history when an unshown proposed block becomes stale."""
        self.pending = None

    @torch.no_grad()
    def propose_token_chunk(self, proxy, *, steps=4, sgf=False, cancelled=None):
        """Supply current proxy tokens without changing history.

        Cancellation is checked between forwards. SP peers must receive the
        same cancellation decision because they share collectives.
        """
        if proxy.ndim != 5 or proxy.shape[1:3] != (24, 5):
            raise ValueError("TAE token session requires [B,24,5,H,W] current proxy latents")
        c = Condition(
            self.static.prompt,
            self.static.tags,
            self.static.references,
            proxy,
            self.static.audio,
            output_size=self.static.output_size,
        )
        return self._propose_condition(c, steps=steps, sgf=sgf, cancelled=cancelled)

    @torch.no_grad()
    def generate_token_chunk(self, proxy, *, steps=4, sgf=False):
        self.propose_token_chunk(proxy, steps=steps, sgf=sgf)
        return self.commit()

    def _propose_condition(self, c, *, steps, sgf, cancelled):
        if self.pending is not None:
            raise ValueError("Commit or discard the pending chunk before proposing another")
        if steps < 1:
            raise ValueError("steps must be positive")
        rng = torch.Generator(device=c.device).manual_seed(self.seed + self.index)
        current = torch.randn(c.video_shape, generator=rng, device=c.device)
        schedule = make_shifted_schedule(steps + 1, shift=self.core.shift, device=c.device)
        for i, t in enumerate(schedule.timesteps):
            if cancelled is not None and cancelled():
                return None
            r = (1.0 if sgf else 1 - schedule.sigmas[i + 1]) if self.core.anyflow else None
            velocity = self.core.forward(
                c, current, t, r=r, mode="rollout", cache=self.cache, chunk=self.index
            )
            clean = predict_clean_sample(current, velocity, t)
            if i + 1 < steps:
                if sgf:
                    eps = torch.randn(c.video_shape, generator=rng, device=c.device)
                    next_t = schedule.timesteps[i + 1]
                    current = next_t * clean + (1 - next_t) * eps
                else:
                    current = current + (schedule.sigmas[i] - schedule.sigmas[i + 1]) * velocity
        if not bool(torch.isfinite(clean).all()):
            raise FloatingPointError("Nonfinite generated chunk; history was not committed")
        if cancelled is not None and cancelled():
            return None
        self.pending = (c, clean)
        return clean

    @torch.no_grad()
    def commit(self):
        """Accept the proposed block, then make its proxy/clean-video KV visible."""
        if self.pending is None:
            raise ValueError("No proposed block to commit")
        c, clean = self.pending
        previous = dict(self.cache.layers)
        try:
            self.core.forward(
                c, clean, 1.0, r=1.0, mode="commit", cache=self.cache, chunk=self.index
            )
        except BaseException:
            self.cache.layers = previous
            raise
        self.index += 1
        self.pending = None
        return clean
