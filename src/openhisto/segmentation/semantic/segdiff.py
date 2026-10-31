"""Embedding-conditioned mask diffusion, ported from the AIM Lab training code.

The original single-token cross-attention is intentionally preserved. Its
conditioning is spatially uniform; attention/Mamba options operate on the
bottleneck only, because the source's other branches never reached its output.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    frequencies = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000) / (half - 1)))
    angles = t.float()[:, None] * frequencies[None]
    result = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return F.pad(result, (0, dim % 2))


class GlobalCrossAttention(nn.Module):
    def __init__(self, channels, embed_dim, attn_dim=256):
        super().__init__()
        self.q_proj = nn.Linear(channels, attn_dim)
        self.k_proj = nn.Linear(embed_dim, attn_dim)
        self.v_proj = nn.Linear(embed_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, channels)

    def forward(self, x, embedding):
        # A single conditioning token makes the attention softmax identically 1.
        # Evaluate the same function directly rather than allocating H*W*attn_dim
        # query/value maps (gigabytes at the source's 1000px training default).
        conditioning = self.out_proj(self.v_proj(embedding))
        return x + conditioning[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x))) + self.time_proj(t)[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class SpatialBlock(nn.Module):
    def __init__(self, channels, style, options):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.style = style
        if style == "mamba":
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise ImportError(
                    "SegDiff network_style=mamba requires the optional mamba-ssm package and a supported CUDA environment"
                ) from exc
            self.network = Mamba(
                d_model=channels,
                d_state=options.get("d_state", 16),
                d_conv=options.get("d_conv", 4),
                expand=options.get("expand", 2),
            )
        else:
            heads = int(options.get("num_heads", 4))
            if channels % heads:
                raise ValueError("SegDiff bottleneck channels must be divisible by num_heads")
            self.network = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * options.get("mlp_ratio", 4.0))
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )

    def forward(self, x, t=None):
        b, c, h, w = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        normalized = self.norm1(flat)
        if self.style == "mamba":
            flat = flat + self.network(normalized)
        else:
            flat = flat + self.network(normalized, normalized, normalized, need_weights=False)[0]
        flat = flat + self.mlp(self.norm2(flat))
        return flat.reshape(b, h, w, c).permute(0, 3, 1, 2)


class DiTBlock(nn.Module):
    def __init__(self, dim, time_dim, heads, ratio):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.time1, self.time2 = nn.Linear(time_dim, 2 * dim), nn.Linear(time_dim, 2 * dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * ratio)), nn.GELU(), nn.Linear(int(dim * ratio), dim)
        )

    def forward(self, x, t):
        gamma, beta = self.time1(t).chunk(2, -1)
        h = gamma[:, None] * self.norm1(x) + beta[:, None]
        x = x + self.attention(h, h, h, need_weights=False)[0]
        gamma, beta = self.time2(t).chunk(2, -1)
        return x + self.mlp(gamma[:, None] * self.norm2(x) + beta[:, None])


class LowDimDiT(nn.Module):
    def __init__(self, channels, time_dim, options):
        super().__init__()
        dim, heads = int(options.get("low_dim", 256)), int(options.get("num_heads", 4))
        if dim % heads:
            raise ValueError("SegDiff low_dim must be divisible by num_heads")
        self.in_proj, self.out_proj = nn.Linear(channels, dim), nn.Linear(dim, channels)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(dim, time_dim, heads, options.get("mlp_ratio", 4.0))
                for _ in range(int(options.get("depth", 4)))
            ]
        )

    def forward(self, x, t):
        b, c, h, w = x.shape
        flat = self.in_proj(x.permute(0, 2, 3, 1).reshape(b, h * w, c))
        for block in self.blocks:
            flat = block(flat, t)
        return self.out_proj(flat).reshape(b, h, w, c).permute(0, 3, 1, 2)


class UNetDenoiser(nn.Module):
    def __init__(self, classes, base, mults, time_dim, style, options):
        super().__init__()
        channels = [base * m for m in mults]
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.in_conv = nn.Conv2d(classes, channels[0], 3, padding=1)
        self.downs = nn.ModuleList(
            [ResBlock(a, b, time_dim) for a, b in zip(channels[:-1], channels[1:])]
        )
        self.mid = ResBlock(channels[-1], channels[-1], time_dim)
        self.ups = nn.ModuleList(
            [
                ResBlock(2 * channels[-1 - i], channels[-2 - i], time_dim)
                for i in range(len(channels) - 1)
            ]
        )
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]), nn.SiLU(), nn.Conv2d(channels[0], classes, 3, padding=1)
        )
        self.inner = (
            LowDimDiT(channels[-1], time_dim, options)
            if style == "low_dim"
            else SpatialBlock(channels[-1], style, options)
            if style in {"attention", "mamba"}
            else None
        )

    def forward(self, x, t):
        t = self.time_mlp(timestep_embedding(t, self.time_dim))
        x, skips = self.in_conv(x), []
        for block in self.downs:
            x = block(x, t)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.mid(x, t)
        if self.inner is not None:
            x = self.inner(x, t)
        for block, skip in zip(self.ups, reversed(skips)):
            x = block(torch.cat((F.interpolate(x, scale_factor=2, mode="nearest"), skip), dim=1), t)
        return self.out_conv(x)


class SegDiffModel(nn.Module):
    def __init__(self, num_classes, options):
        super().__init__()
        self.num_classes = num_classes
        self.timesteps = int(options.get("timesteps", 1000))
        base = int(options.get("base_channels", 64))
        mults = tuple(int(x) for x in options.get("channel_mults", (1, 2, 4)))
        time_dim, embed_dim = (
            int(options.get("time_emb_dim", 256)),
            int(options.get("embed_dim", 256)),
        )
        feature_dim = int(options.get("feature_dim", 0))
        style = options.get("network_style") or None
        if (
            self.timesteps < 1
            or feature_dim < 1
            or not mults
            or any(m < 1 or base * m % 8 for m in mults)
            or time_dim < 4
        ):
            raise ValueError(
                "SegDiff requires positive timesteps/feature_dim, channel widths divisible by 8, and time_emb_dim >= 4"
            )
        if style not in {None, "none", "low_dim", "attention", "mamba"}:
            raise ValueError("SegDiff network_style must be none, low_dim, attention, or mamba")
        self.stride = 2 ** (len(mults) - 1)
        self.feature_projector = nn.Linear(feature_dim, embed_dim)
        self.conditioning = GlobalCrossAttention(
            num_classes, embed_dim, int(options.get("attn_dim", 256))
        )
        self.denoiser = UNetDenoiser(num_classes, base, mults, time_dim, style, options)
        schedule = options.get("beta_schedule", "linear")
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.timesteps)
        elif schedule == "sqrt_linear":
            betas = torch.linspace(1e-4, 0.02, self.timesteps).sqrt()
        elif schedule == "cosine":
            steps = torch.linspace(0, 1, self.timesteps + 1)
            bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2).square()
            bar = bar / bar[0]
            betas = (1 - bar[1:] / bar[:-1]).clamp(1e-8, 0.999)
        else:
            raise ValueError("Unknown SegDiff beta_schedule")
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0))
        self.ce_weight, self.dice_weight = (
            float(options.get("ce_weight", 0.1)),
            float(options.get("dice_weight", 0.1)),
        )
        if not all(math.isfinite(w) and w >= 0 for w in (self.ce_weight, self.dice_weight)):
            raise ValueError("SegDiff CE/Dice weights must be finite and nonnegative")

    def forward(self, x, t, features):
        return self.denoiser(self.conditioning(x, self.feature_projector(features)), t)

    def loss(self, labels, features, ignore_index):
        valid = labels.ne(ignore_index)
        if not valid.any():
            raise ValueError("SegDiff training mask contains no valid labeled pixels")
        if torch.any((labels[valid] < 0) | (labels[valid] >= self.num_classes)):
            raise ValueError("SegDiff mask contains a class outside num_classes")
        safe = labels.masked_fill(~valid, 0)
        target = F.one_hot(safe, self.num_classes).permute(0, 3, 1, 2).float()
        x0 = (target + 1e-8).log().masked_fill(~valid[:, None], 0)
        t = torch.randint(self.timesteps, (labels.shape[0],), device=labels.device)
        a = self.alpha_bar[t, None, None, None].sqrt()
        b = (1 - self.alpha_bar[t, None, None, None]).sqrt()
        noise = torch.randn_like(x0)
        noisy = a * x0 + b * noise
        prediction = self(noisy, t, features)
        valid_channels = valid[:, None].float()
        loss = ((prediction - noise).square() * valid_channels).sum() / (
            valid.sum() * self.num_classes
        )
        if self.ce_weight or self.dice_weight:
            reconstructed = (noisy - b * prediction) / (a + 1e-8)
            if self.ce_weight:
                loss = loss + self.ce_weight * F.cross_entropy(
                    reconstructed, labels, ignore_index=ignore_index
                )
            if self.dice_weight:
                probs, truth = reconstructed.softmax(1) * valid_channels, target * valid_channels
                dims = (0, 2, 3)
                dice = (2 * (probs * truth).sum(dims) + 1e-6) / (
                    probs.sum(dims) + truth.sum(dims) + 1e-6
                )
                loss = loss + self.dice_weight * (1 - dice.mean())
        return loss

    @torch.no_grad()
    def sample(self, features, size, generator=None):
        def random_normal(shape):
            # MPS has no Generator implementation; a CPU generator still gives
            # deterministic explicit-seed sampling without changing global RNG.
            device = features.device if generator is None else generator.device
            return torch.randn(shape, device=device, generator=generator).to(features.device)

        x = random_normal((len(features), self.num_classes, *size))
        for t in reversed(range(self.timesteps)):
            time = torch.full((len(features),), t, device=x.device, dtype=torch.long)
            noise = self(x, time, features)
            beta, alpha, bar = self.betas[t], self.alphas[t], self.alpha_bar[t]
            mean = (x - beta / (1 - bar).sqrt() * noise) / alpha.sqrt()
            if t:
                variance = beta * (1 - self.alpha_bar[t - 1]) / (1 - bar)
                x = mean + variance.sqrt() * random_normal(x.shape)
            else:
                x = mean
        return x
