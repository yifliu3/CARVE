"""Full bidirectional and segmented causal attention without dense token masks."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from solarwm.backends.minimax_h3.distributed import (
    is_sequence_parallel_enabled,
    sequence_all_to_all,
)

from .layout import Layout, visible


@dataclass
class KVCache:
    history: int = 5
    layers: dict = field(default_factory=dict)

    def prior(self, layer, chunk):
        entries = self.layers.get(layer, [])
        ids = sorted({r.chunk for r, _, _ in entries})
        if ids != list(range(max(0, chunk - self.history), chunk)):
            raise RuntimeError(f"Cache discontinuity: chunk={chunk}, cached={ids}")
        return entries

    def commit(self, layer, chunk, entries):
        if torch.is_grad_enabled():
            raise RuntimeError("KV commit must run under no_grad")
        prior = self.prior(layer, chunk)
        self.layers[layer] = [
            (r, k.detach().clone(), v.detach().clone())
            for r, k, v in [*prior, *entries]
            if r.chunk >= chunk - self.history + 1
        ]


@dataclass
class CWMControl:
    layout: Layout
    rope: object
    mode: str
    history: int = 5
    cache: KVCache | None = None
    chunk: int | None = None
    checkpoint_segments: bool = True

    def validate_sequence_length(self, length):
        if length != self.layout.length:
            raise ValueError("CARVE attention layout does not match packed length")

    def _positions(self, run, first, device):
        if run.role == "static":
            return self.layout.positions[run.start : run.stop]
        # A token proxy may have a smaller spatial grid than the target. Reuse
        # only a prototype from the same modality, including for cached rows.
        roles = {"proxy"} if run.role == "proxy" else {"clean", "noisy"}
        prototype = next(r for r in self.layout.runs if r.role in roles)
        pos = self.layout.positions[prototype.start : prototype.stop].clone()
        if len(pos) != run.stop - run.start:
            raise ValueError("Cached attention run geometry differs from its modality prototype")
        if prototype.role == "proxy" and run.role != "proxy":
            pos[:, 0] -= 0.5
        elif prototype.role != "proxy" and run.role == "proxy":
            pos[:, 0] += 0.5
        # Five H3 latents cover 17 video frames; native rotary rescale is 5/3.
        pos[:, 0] += (run.chunk - first) * (17.0 * 5.0 / 3.0)
        return pos.to(device)

    def forward(self, attn, query, key, value):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        sp = is_sequence_parallel_enabled()
        if sp:
            query, key, value = [
                sequence_all_to_all(x, scatter_dim=2, gather_dim=1) for x in (query, key, value)
            ]
        if self.mode == "bid":
            # Preserve absolute chunk times and the proxy role offset. The
            # layout itself stores chunk-local positions for causal replay.
            positions = torch.cat(
                [self._positions(run, 0, query.device) for run in self.layout.runs]
            )
            rotary = self.rope(positions)
            query = _apply_rotary_emb(query, *rotary)
            key = _apply_rotary_emb(key, *rotary)
            # Conditions and noisy targets all attend to the entire document.
            # A single unmasked call retains the CUDA Flash Attention path.
            out = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0
            ).transpose(1, 2)
        else:
            out = self._causal_forward(attn, query, key, value)
        if sp:
            out = sequence_all_to_all(out, scatter_dim=1, gather_dim=2)
        return attn.to_out[1](attn.to_out[0](out.flatten(2, 3)))

    def _causal_forward(self, attn, query, key, value):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        layer = attn.h3_layer_index
        prior = [] if self.cache is None else self.cache.prior(layer, self.chunk)
        current = [
            (r, key[:, r.start : r.stop], value[:, r.start : r.stop]) for r in self.layout.runs
        ]
        results = []
        for run in self.layout.runs:
            selected = [
                (r, k, v)
                for r, k, v in [*prior, *current]
                if visible(run, r, mode=self.mode, history=self.history)
            ]
            first = max(0, run.chunk - self.history)
            selected_runs = tuple(r for r, _, _ in selected)

            def attend(q_slice, *kv, query_run=run, key_runs=selected_runs, origin=first):
                # Bind run metadata now: checkpoint recomputation happens after
                # this loop has advanced. Inputs are views of the shared Q/K/V,
                # not the repeatedly concatenated context of every query block.
                count = len(key_runs)
                qpos = self._positions(query_run, origin, q_slice.device)
                kpos = torch.cat([self._positions(r, origin, q_slice.device) for r in key_runs])
                q = _apply_rotary_emb(q_slice, *self.rope(qpos))
                k = _apply_rotary_emb(torch.cat(kv[:count], 1), *self.rope(kpos))
                v = torch.cat(kv[count:], 1)
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0
                ).transpose(1, 2)

            inputs = (
                query[:, run.start : run.stop],
                *[k for _, k, _ in selected],
                *[v for _, _, v in selected],
            )
            if self.checkpoint_segments and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                out = checkpoint(attend, *inputs, use_reentrant=False, preserve_rng_state=False)
            else:
                out = attend(*inputs)
            results.append(out)
        if self.mode == "commit":
            if self.cache is None:
                raise ValueError("Commit requires a cache")
            self.cache.commit(
                layer,
                self.chunk,
                [(r, k, v) for r, k, v in current if r.role in {"proxy", "clean"}],
            )
        return torch.cat(results, 1)
