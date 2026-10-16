"""Decode complete five-latent H3 chunks with pinned TAEH3 temporal memory."""

import importlib.util
from pathlib import Path

import torch

from .data import digest
from .tae_proxy import TAE_COMMIT, TAE_SOURCE_SHA256, TAE_WEIGHTS_SHA256


class TAEH3VideoDecoder:
    def __init__(self, assets, device="cuda:0"):
        assets = Path(assets)
        for name, expected in (("taehv.py", TAE_SOURCE_SHA256),
                               ("taeh3.safetensors", TAE_WEIGHTS_SHA256)):
            if digest(assets / name) != expected:
                raise ValueError(f"Pinned TAEH3 decoder asset differs: {name}")
        spec = importlib.util.spec_from_file_location("cwm_pinned_tae_decode", assets / "taehv.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.device = torch.device(device)
        self.model = self.module.TAEHV(str(assets / "taeh3.safetensors"), arch_name="taeh3")
        del self.model.encoder
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model.requires_grad_(False).eval().to(self.device, self.dtype)
        self.identity = {
            "visual_decoder": "taeh3", "decoder_commit": TAE_COMMIT,
            "decoder_source_sha256": TAE_SOURCE_SHA256,
            "decoder_weights_sha256": TAE_WEIGHTS_SHA256, "decoder_dtype": str(self.dtype),
            "decoder_temporal_policy": "five_latents_to17_trim3_each_chunk_keep_all_generated_chunks",
            "decoder_rgb_normalization": "rgb_0_1_no_extra_rescale",
        }

    @torch.inference_mode()
    def decode_chunks(self, latents):
        if (latents.ndim != 5 or latents.shape[:2] != (1, 24)
                or latents.shape[2] < 5 or latents.shape[2] % 5):
            raise ValueError("TAEH3 generated video requires [1,24,5*K,H,W] latents")
        memory = {}
        try:
            for part in latents.split(5, dim=2):
                x = part.permute(0, 2, 1, 3, 4).to(self.device, self.dtype).flatten(0, 1)
                for index, block in enumerate(self.model.decoder):
                    if isinstance(block, (self.module.MemBlock, self.module.SuperMemBlock)):
                        sequence = x.unsqueeze(0)
                        previous = memory.get(index)
                        if previous is None:
                            previous = torch.zeros_like(sequence[:, :1])
                        past = torch.cat((previous, sequence[:, :-1]), dim=1).flatten(0, 1)
                        memory[index] = sequence[:, -1:].clone()
                        x = block(x, past)
                    else:
                        x = block(x)
                if x.shape[0] != 20:
                    raise ValueError("TAEH3 decoder temporal expansion changed")
                pixels = self.model.postprocess_output_frames(x[None, 3:])
                if not bool(torch.isfinite(pixels).all()):
                    raise FloatingPointError("Nonfinite TAEH3 decode")
                yield pixels.permute(0, 2, 1, 3, 4).contiguous().cpu()
        finally:
            memory.clear()
