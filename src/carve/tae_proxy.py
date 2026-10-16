"""Pinned TAEH3 chunk encoding for v4 uniform visual latents."""

import importlib.util
from pathlib import Path

import torch

TAE_COMMIT = "011dfc2112197741c540e0bdd5b7b67bcc930771"
TAE_SOURCE_SHA256 = "99545f093b36c16297ad182ce18a5dce2ebc98bb611f5e8dfa2dce431ba39ce2"
TAE_WEIGHTS_SHA256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"
TAE_TOKENS_CONTRACT = "cwm-h3.proxy.tae-tokens.v1"


def resize_proxy_pixels(pixels, height, width):
    """Same nearest-neighbor pixel resize online/offline; never resize latent features."""
    if pixels.ndim != 5 or pixels.shape[1] != 3 or pixels.dtype != torch.uint8:
        raise ValueError("Proxy resize expects uint8 [B,3,T,H,W]")
    if height % 32 or width % 32 or min(height, width) < 32:
        raise ValueError("Proxy output dimensions must be multiples of32")
    b, _, t, h, w = pixels.shape
    if (h, w) == (height, width):
        return pixels
    x = pixels.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w).float()
    x = torch.nn.functional.interpolate(x, size=(height, width), mode="nearest").to(torch.uint8)
    return x.reshape(b, t, 3, height, width).permute(0, 2, 1, 3, 4).contiguous()


class TAEH3ProxyEncoder:
    """Encode each current 17-frame chunk independently, without final-tail trimming.

    The offline adapter concatenates these chunks then removes the final three
    latents once. This matches H3's 107->32 storage / first30 training geometry.
    An online call needs only the current17 frames and returns all five latents.
    """

    def __init__(self, assets, device="cuda:0"):
        from .data import digest

        assets = Path(assets)
        for name, expected in (
            ("taehv.py", TAE_SOURCE_SHA256),
            ("taeh3.safetensors", TAE_WEIGHTS_SHA256),
        ):
            if digest(assets / name) != expected:
                raise ValueError(f"Pinned TAEH3 asset differs: {name}")
        spec = importlib.util.spec_from_file_location("cwm_pinned_taehv", assets / "taehv.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        self.device = torch.device(device)
        self.model = module.TAEHV(str(assets / "taeh3.safetensors"), arch_name="taeh3")
        self.model.requires_grad_(False).eval()
        del self.model.decoder
        self.model.to(self.device, dtype=torch.float16)

    @torch.no_grad()
    def encode_chunk(self, pixels):
        if (
            pixels.ndim != 5
            or pixels.shape[1:3] != (3, 17)
            or pixels.dtype != torch.uint8
            or pixels.shape[-2] % 32
            or pixels.shape[-1] % 32
        ):
            raise ValueError("TAEH3 expects uint8 [B,3,17,H,W], H/W divisible by32")
        x = pixels.permute(0, 2, 1, 3, 4).to(self.device, dtype=torch.float16) / 255
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, 3, 0))
        x = self.model.preprocess_input_frames(x)
        z = self.module.apply_model_with_memblocks(self.model.encoder, x, True, False)
        z = z.permute(0, 2, 1, 3, 4).contiguous()
        expected = (pixels.shape[0], 24, 5, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
        if z.shape != expected or not bool(torch.isfinite(z).all()):
            raise ValueError("TAEH3 produced invalid proxy latents")
        return z

    @torch.no_grad()
    def encode_video(self, pixels):
        if pixels.ndim != 5 or pixels.shape[2] < 1:
            raise ValueError("Expected nonempty [B,3,T,H,W] video")
        missing = -pixels.shape[2] % 17
        if missing:
            pixels = torch.cat((pixels, pixels[:, :, -1:].expand(-1, -1, missing, -1, -1)), 2)
        return torch.cat(
            [self.encode_chunk(pixels[:, :, i : i + 17]) for i in range(0, pixels.shape[2], 17)], 2
        )[:, :, :-3]
