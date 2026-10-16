"""Native H3 parameters with CARVE token attention and AnyFlow time conditioning."""

from dataclasses import dataclass
from typing import Any

import torch
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3AttnProcessor,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerOutput,
)
from diffusers.utils import apply_lora_scale

from .anyflow_conditioning import H3AnyFlowConditioningMixin
from .diffusers_compat import patch_minimax_h3_parameter_dtype
from .distributed import (
    get_sp_size,
    is_sequence_parallel_enabled,
    sequence_all_gather,
    shard_stage0p5_packed_sequence,
    state,
    validate_packed_attention_mask,
)

get_parameter_dtype = patch_minimax_h3_parameter_dtype()


@dataclass(frozen=True)
class H3AttentionControl:
    attention_mask: Any
    sequence_lengths: tuple[int, ...] = ()


class SolarMiniMaxH3AttnProcessor(MiniMaxH3AttnProcessor):
    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from carve.attention import CWMControl

        if isinstance(attention_mask, H3AttentionControl):
            if attention_mask.sequence_lengths:
                state().sequence_lengths = attention_mask.sequence_lengths
            attention_mask = attention_mask.attention_mask
        if not isinstance(attention_mask, CWMControl):
            return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = (
                attn.to_q(hidden_states),
                attn.to_k(hidden_states),
                attn.to_v(hidden_states),
            )
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        return attention_mask.forward(attn, query, key, value)


class SolarMiniMaxH3Transformer3DModel(H3AnyFlowConditioningMixin, MiniMaxH3Transformer3DModel):
    def _install_solar_processors(self):
        for index, block in enumerate(self.transformer_blocks):
            block.attn.set_processor(SolarMiniMaxH3AttnProcessor())
            block.attn.h3_layer_index = index

    @classmethod
    def from_config(cls, config=None, return_unused_kwargs=False, **kwargs):
        result = super().from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)
        if return_unused_kwargs:
            model, unused = result
            model._install_solar_processors()
            return model, unused
        result._install_solar_processors()
        return result

    @classmethod
    def strict_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("output_loading_info", None)
        model, info = cls.from_pretrained(
            pretrained_model_name_or_path,
            output_loading_info=True,
            **kwargs,
        )
        problems = {
            name: info[name]
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if info.get(name)
        }
        if problems:
            raise RuntimeError(f"H3 checkpoint did not strict-load: {problems}")
        return model

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        attention_kwargs=None,
        return_dict=True,
        *,
        r_timestep=None,
        attention_mask=None,
        fused_prope=False,
        packed_sequence_parallel=False,
    ):
        from carve.attention import CWMControl

        if fused_prope is not False or not isinstance(attention_mask, CWMControl):
            raise ValueError("The release requires CARVE token attention without camera PRoPE")
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError("position_ids must be [sequence,3]")
        length = position_ids.shape[0]
        if token_tags.shape != (length,) or timestep_indices.shape != (length,):
            raise ValueError("Packed tags/timesteps must match the sequence length")
        sp = is_sequence_parallel_enabled()
        if sp != bool(packed_sequence_parallel):
            raise ValueError(f"Packed sequence-parallel contract differs (SP={get_sp_size()})")
        if sp:
            validate_packed_attention_mask(attention_mask, total_tokens=length)
        video = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
        text = self.token_refiner(
            self.context_embedder(
                encoder_hidden_states.to(get_parameter_dtype(self.context_embedder))
            )
        )
        packed = text.new_zeros((text.shape[0], length, text.shape[-1]))
        packed = packed.index_copy(1, text_indices, text)
        packed = packed.index_copy(1, video_indices, video.to(text.dtype))
        packed = packed.index_copy(1, audio_indices, audio.to(text.dtype))
        temb = self._time_condition(timestep, r_timestep, parameter_dtype=get_parameter_dtype)
        if sp:
            shard = shard_stage0p5_packed_sequence(
                hidden_states=packed,
                position_ids=position_ids,
                token_tags=token_tags,
                timestep_indices=timestep_indices,
            )
            packed, position_ids = shard.hidden_states, shard.position_ids
            token_tags, timestep_indices = shard.token_tags, shard.timestep_indices
            control = H3AttentionControl(attention_mask, shard.sequence_lengths)
        else:
            control = attention_mask
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        rotary = self.rope(position_ids)
        for block in self.transformer_blocks:
            # A separate identity branch lets each FSDP block release its full
            # parameters when its local time-condition gradient is available.
            block_temb = temb.clone() if temb.requires_grad else temb
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def custom_forward(states, time_condition, current_block=block):
                    return current_block(states, time_condition, adaln_indices, rotary, control)

                packed = self._gradient_checkpointing_func(custom_forward, packed, block_temb)
            else:
                packed = block(packed, block_temb, adaln_indices, rotary, control)
        packed = self.norm_out(packed, temb, timestep_indices).to(
            get_parameter_dtype(self.proj_out)
        )
        video = self.proj_out(packed)
        audio = self.audio_proj_out(packed)
        if sp:
            video, audio = sequence_all_gather(video, dim=1), sequence_all_gather(audio, dim=1)
            if video.shape[1] != length or audio.shape[1] != length:
                raise RuntimeError("SP output gather did not restore the complete document")
        video, audio = video.index_select(1, video_indices), audio.index_select(1, audio_indices)
        if not return_dict:
            return video, audio
        return MiniMaxH3TransformerOutput(sample=video, audio_sample=audio)
