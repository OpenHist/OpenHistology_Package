"""Lazy, deterministic generator inference for CycleGAN and Pix2Pix."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from openhisto import _common


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            'GAN methods require PyTorch; pip install "openhisto[stain-transfer-gan]"'
        ) from exc
    return torch


def resolve_device(requested="auto"):
    return _common.resolve_device(require_torch(), requested)


def grayscale_tensor(tensor):
    """Replicate luminance in three channels, including for normalized [-1,1] input."""
    weights = tensor.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (tensor * weights).sum(1, keepdim=True).expand(-1, 3, -1, -1)


def _load(path):
    return require_torch().load(str(path), map_location="cpu", weights_only=True)


def checkpoint_metadata(path):
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists():
        return {}
    with sidecar.open() as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {sidecar}")
    return metadata


class GANNormalizer:
    def __init__(self, domain, config):
        torch = require_torch()
        from .networks import generator

        if not domain.checkpoint:
            raise ValueError(f"{domain.method} requires a checkpoint")
        checkpoint = Path(domain.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        metadata = checkpoint_metadata(checkpoint)
        loaded = _load(checkpoint)
        full = isinstance(loaded, dict) and loaded.get("format") == "stain_transfer.training.v1"
        if full:
            embedded = loaded.get("metadata", {})
            if not isinstance(embedded, dict):
                raise ValueError("Full checkpoint metadata must be a dictionary")
            for field in (
                "method",
                "architecture",
                "input_mode",
                "ngf",
                "norm",
                "dropout",
                "direction",
            ):
                if field in metadata and field in embedded and metadata[field] != embedded[field]:
                    raise ValueError(f"Sidecar {field} conflicts with embedded checkpoint metadata")
            metadata = {**embedded, **metadata}
        if metadata.get("method", domain.method) != domain.method:
            raise ValueError("Checkpoint method does not match the selected GAN method")
        allowed = {"ngf", "norm", "dropout", "inference_size"}
        unknown = set(domain.options) - allowed
        if unknown:
            raise ValueError(f"Unknown GAN options: {', '.join(sorted(unknown))}")
        for field in ("architecture", "input_mode", "direction"):
            explicit, saved = getattr(domain, field), metadata.get(field)
            if explicit is not None and saved is not None and explicit != saved:
                if field != "direction" or not (full and domain.method == "cyclegan"):
                    raise ValueError(f"Explicit {field} conflicts with checkpoint metadata")
        self.method = domain.method
        self.architecture = domain.architecture or metadata.get("architecture")
        self.input_mode = domain.input_mode or metadata.get("input_mode")
        self.direction = domain.direction or metadata.get("direction")
        if not all((self.architecture, self.input_mode, self.direction)):
            raise ValueError(
                "Raw GAN checkpoints require explicit architecture, input_mode, and direction; "
                "exported generators provide these in an adjacent JSON sidecar"
            )
        if self.input_mode not in {"rgb", "grayscale"}:
            raise ValueError("GAN input_mode must be rgb or grayscale")
        if self.direction not in {"AtoB", "BtoA"}:
            raise ValueError("GAN direction must be AtoB or BtoA")
        if (
            self.method == "pix2pix"
            and full
            and self.direction != metadata.get("direction", "AtoB")
        ):
            raise ValueError("Pix2Pix has one trained direction; its generator cannot be reversed")
        options = {**metadata, **dict(domain.options)}
        for field in ("ngf", "norm", "dropout"):
            if (
                field in domain.options
                and field in metadata
                and domain.options[field] != metadata[field]
            ):
                raise ValueError(f"Explicit {field} conflicts with checkpoint metadata")
        if (
            isinstance(options.get("ngf", 64), bool)
            or not isinstance(options.get("ngf", 64), int)
            or options.get("ngf", 64) <= 0
        ):
            raise ValueError("GAN ngf must be a positive integer")
        if not isinstance(options.get("dropout", False), bool):
            raise ValueError("GAN dropout must be boolean")
        self.resize = options.get("inference_size")
        if self.resize is not None and (
            isinstance(self.resize, bool) or not isinstance(self.resize, int) or self.resize <= 0
        ):
            raise ValueError("GAN inference_size must be a positive integer")
        self.device = resolve_device(config.device)
        normalization = options.get("norm", "batch" if self.method == "pix2pix" else "instance")
        if self.method == "pix2pix" and normalization != "batch":
            raise ValueError("Pix2Pix uses batch normalization; norm must be batch")
        self.model = generator(
            self.method,
            self.architecture,
            ngf=int(options.get("ngf", 64)),
            norm=normalization,
            dropout=bool(options.get("dropout", False)),
        )
        if full:
            key = (
                ("G_A" if self.direction == "AtoB" else "G_B") if self.method == "cyclegan" else "G"
            )
            state = loaded["networks"][key]
        else:
            state = loaded
        if isinstance(state, dict) and state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.provenance = {
            "method": self.method,
            "checkpoint": str(checkpoint.resolve()),
            "architecture": self.architecture,
            "input_mode": self.input_mode,
            "direction": self.direction,
            "device": str(self.device),
            "ngf": int(options.get("ngf", 64)),
            "norm": normalization,
            "inference_size": self.resize,
            "checkpoint_sha256": digest.hexdigest(),
            "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
            "preprocessing": {
                "input_mode": self.input_mode,
                "normalization": "[-1,1]",
                "padding": "replicate",
                "preserve_dimensions": True,
                "inference_size": self.resize,
            },
            "source_name": metadata.get(
                "target_name" if full and self.direction == "BtoA" else "source_name"
            ),
            "target_name": metadata.get(
                "source_name" if full and self.direction == "BtoA" else "target_name"
            ),
        }

    def transform(self, rgb, valid_bounds=None, patch_id=None):
        torch = require_torch()
        from .networks import spatial_requirements

        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("GAN normalization expects an H×W×3 uint8 RGB array")
        height, width = rgb.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Cannot normalize an empty image")
        x = (
            torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
            / 127.5
            - 1
        )
        if self.resize is not None:
            size = int(self.resize)
            if size <= 0:
                raise ValueError("inference_size must be positive")
            x = torch.nn.functional.interpolate(
                x, size=(size, size), mode="bicubic", align_corners=False
            )
        input_h, input_w = x.shape[-2:]
        stride, minimum = spatial_requirements(self.method, self.architecture)
        padded_h = max(minimum, math.ceil(input_h / stride) * stride)
        padded_w = max(minimum, math.ceil(input_w / stride) * stride)
        x = torch.nn.functional.pad(
            x, (0, padded_w - input_w, 0, padded_h - input_h), mode="replicate"
        )
        if self.input_mode == "grayscale":
            x = grayscale_tensor(x)
        with torch.inference_mode():
            output = self.model(x.to(self.device))[..., :input_h, :input_w]
            if self.resize is not None:
                output = torch.nn.functional.interpolate(
                    output, size=(height, width), mode="bicubic", align_corners=False
                )
            result = ((output[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return result.permute(1, 2, 0).cpu().numpy()


def build(domain, config):
    return GANNormalizer(domain, config)
