"""Small trainable-adapter contract and aligned RGB/label preprocessing."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import numpy as np
import torch
from torch import nn

from ..config import ModelConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationAdapter(ABC):
    """Adapters own their trainable module; the engine owns optimization/I/O."""

    supports_logits = True
    supports_training = True
    score_semantics = "class_logits"

    def __init__(self, config: ModelConfig, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model: nn.Module

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @abstractmethod
    def loss(self, samples):
        """Return a differentiable scalar for loaded, aligned labeled samples."""

    @abstractmethod
    def infer(self, samples):
        """Return per-sample CHW NumPy probabilities/scores/native logits."""

    @abstractmethod
    def save(self, directory):
        """Save backend state beneath the caller-owned bundle directory."""


BaseSegAdapter = SegmentationAdapter


def preprocess_batch(
    samples,
    *,
    device="cpu",
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    ignore_index=-100,
    divisor=32,
    require_labels=False,
):
    """Normalize once and pad aligned labels with ignore_index, without resizing.

    Input images are uint8 HWC RGB. Padded image pixels are zero in normalized
    space. Image/label geometry and class IDs are never changed by a processor.
    """
    if not samples:
        raise ValueError("At least one sample is required")
    mean = torch.as_tensor(mean, dtype=torch.float32).reshape(3, 1, 1)
    std = torch.as_tensor(std, dtype=torch.float32).reshape(3, 1, 1)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Normalization requires finite mean and positive std")
    images, labels, sizes = [], [], []
    labeled = [sample.mask is not None for sample in samples]
    if require_labels and not all(labeled):
        raise ValueError("Training samples require masks")
    for sample in samples:
        image = np.asarray(sample.image)
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[-1] != 3
            or min(image.shape[:2]) < 1
        ):
            raise ValueError("Adapters require nonempty uint8 HWC RGB images")
        size = tuple(image.shape[:2])
        sizes.append(size)
        value = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255)
        images.append((value - mean) / std)
        if all(labeled):
            mask = np.asarray(sample.mask)
            if mask.shape != size or mask.dtype.kind not in "iub":
                raise ValueError("Masks must be integer HW maps aligned to their image")
            labels.append(torch.from_numpy(np.array(mask, dtype=np.int64, copy=True)))
    height = math.ceil(max(h for h, _ in sizes) / divisor) * divisor
    width = math.ceil(max(w for _, w in sizes) / divisor) * divisor
    pixels = torch.zeros((len(samples), 3, height, width), dtype=torch.float32)
    pixel_mask = torch.zeros((len(samples), height, width), dtype=torch.bool)
    targets = (
        torch.full((len(samples), height, width), ignore_index, dtype=torch.long)
        if labels
        else None
    )
    for i, ((h, w), image) in enumerate(zip(sizes, images)):
        pixels[i, :, :h, :w] = image
        pixel_mask[i, :h, :w] = True
        if targets is not None:
            targets[i, :h, :w] = labels[i]
    return (
        pixels.to(device),
        None if targets is None else targets.to(device),
        sizes,
        pixel_mask.to(device),
    )


def validate_targets(labels, config):
    valid = labels != config.ignore_index
    if not valid.any():
        raise ValueError("Batch contains no labeled pixels")
    if ((labels[valid] < 0) | (labels[valid] >= config.num_classes)).any():
        raise ValueError("Mask contains class IDs outside ModelConfig.num_classes")
    return valid
