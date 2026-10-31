"""Shared, gradient-preserving MIL adapter contract."""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class MILOutput:
    logits: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    attention: Any = None
    aux: dict = field(default_factory=dict)
    native_output: Any = None


class MILAdapter(nn.Module):
    """One bag per call; subclasses retain their native trainable components."""

    def __init__(self):
        super().__init__()
        self.metadata = {}


def as_bag(inputs, input_dim=None, *, batched=False):
    value = inputs.get("features") if isinstance(inputs, dict) else inputs
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a feature tensor or {'features': tensor}")
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("Expected one nonempty bag [N,D] or [1,N,D]")
    if input_dim is not None and value.shape[-1] != input_dim:
        raise ValueError(f"Expected feature dimension {input_dim}, received {value.shape[-1]}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("Features must be finite floating-point tensors")
    return value.unsqueeze(0) if batched else value


def output_from_logits(logits, *, embedding=None, attention=None, aux=None, native=None):
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    probabilities = logits.sigmoid() if logits.shape[-1] == 1 else logits.softmax(-1)
    return MILOutput(logits, probabilities, embedding, attention, aux or {}, native)
