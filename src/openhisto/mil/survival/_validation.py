"""Small CPU-only validators shared by survival heads and numerical helpers."""

import torch


def vector(value, name, *, floating=True):
    result = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if result.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError(f"{name} must be a nonempty vector [N] or [N,1]")
    if result.is_complex() or not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real values")
    return result.to(dtype=torch.float64) if floating else result


def targets(durations, events):
    times = vector(durations, "durations")
    flags = vector(events, "events", floating=False)
    if len(times) != len(flags):
        raise ValueError("durations and events must have the same length")
    if (times < 0).any():
        raise ValueError("durations must be nonnegative in the supplied time unit")
    if not ((flags == 0) | (flags == 1)).all():
        raise ValueError("events must be 1 for observed events and 0 for right censoring")
    return times, flags.bool()


def training_split(split):
    if split != "train":
        raise ValueError('Fit survival calibration/time grids using split="train" only')


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value
