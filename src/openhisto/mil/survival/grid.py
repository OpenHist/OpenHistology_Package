"""Training-fitted, positive-width time intervals for discrete hazards.

K logits describe K intervals with K+1 boundaries starting at zero. Intervals
are [0,b1], (b1,b2], ... . Events use their enclosing interval; censoring
contributes only intervals whose right boundary has actually been reached.
"""

from dataclasses import dataclass

import torch

from ._validation import positive_integer, targets, training_split, vector


@dataclass(frozen=True)
class DiscreteTargets:
    """Transformed labels plus untouched copies of the supplied observations.

    These subject-level tensors are returned to the caller, never placed in a
    head's config or calibration buffers.
    """

    durations: torch.Tensor
    events: torch.Tensor
    event_bin: torch.Tensor
    completed_intervals: torch.Tensor
    original_durations: torch.Tensor
    original_events: torch.Tensor
    boundaries: torch.Tensor
    truncated_count: int = 0
    truncated_event_count: int = 0

    @property
    def metadata(self):
        return {
            "event_convention": "1=observed_event,0=right_censored",
            "num_bins": len(self.boundaries) - 1,
            "horizon": float(self.boundaries[-1]),
            "administratively_truncated": self.truncated_count,
            "observed_events_truncated": self.truncated_event_count,
        }


@dataclass(frozen=True)
class TimeGrid:
    boundaries: torch.Tensor
    fitted_count: int = 0
    fit_split: str = "train"

    def __post_init__(self):
        bounds = vector(self.boundaries, "boundaries").detach().clone()
        if len(bounds) < 2 or bounds[0] != 0 or not (bounds[1:] > bounds[:-1]).all():
            raise ValueError(
                "boundaries must start at 0 and be strictly increasing with positive interval widths"
            )
        if self.fitted_count < 0:
            raise ValueError("fitted_count cannot be negative")
        training_split(self.fit_split)
        object.__setattr__(self, "boundaries", bounds)

    @property
    def num_bins(self):
        return len(self.boundaries) - 1

    @property
    def horizon(self):
        return float(self.boundaries[-1])

    @property
    def metadata(self):
        return {
            "boundaries": self.boundaries.tolist(),
            "num_bins": self.num_bins,
            "fit_split": self.fit_split,
            "fitted_count": self.fitted_count,
            "interval_convention": "[0,b1],(b1,b2],...",
            "time_unit": "unchanged_from_supplied_durations",
        }

    def transform(self, durations, events, *, administrative_truncation=False):
        if not isinstance(administrative_truncation, bool):
            raise ValueError("administrative_truncation must be an explicit boolean")
        times, flags = targets(durations, events)
        original_times, original_flags = times.detach().clone(), flags.detach().clone()
        beyond = times > self.boundaries[-1]
        if beyond.any() and not administrative_truncation:
            raise ValueError(
                "durations exceed the fitted horizon; opt into administrative_truncation explicitly"
            )
        effective_times = times.clamp(max=self.horizon)
        effective_events = flags & ~beyond
        right = self.boundaries[1:]
        event_bin = torch.searchsorted(right, effective_times.contiguous(), right=False)
        completed = torch.searchsorted(right, effective_times.contiguous(), right=True)
        return DiscreteTargets(
            effective_times,
            effective_events,
            event_bin,
            completed,
            original_times,
            original_flags,
            self.boundaries.clone(),
            int(beyond.sum()),
            int((beyond & flags).sum()),
        )


def fit_time_grid(
    durations,
    events=None,
    *,
    num_bins=20,
    boundaries=None,
    administrative_truncation=False,
    split="train",
):
    """Fit equal-width intervals using training follow-up only.

    Explicit boundaries override equal-width fitting, but supplied training
    observations beyond their horizon still require explicit truncation.
    No validation/test durations are used to expand a fitted horizon.
    """
    training_split(split)
    times = vector(durations, "durations")
    if (times < 0).any():
        raise ValueError("durations must be nonnegative")
    flags = (
        torch.zeros(len(times), dtype=torch.bool) if events is None else targets(times, events)[1]
    )
    if boundaries is None:
        count = positive_integer(num_bins, "num_bins")
        maximum = float(times.max())
        if maximum <= 0:
            raise ValueError(
                "A positive training follow-up horizon is required for positive-width intervals"
            )
        boundaries = torch.linspace(0.0, maximum, count + 1, dtype=torch.float64)
    grid = TimeGrid(boundaries, len(times), split)
    grid.transform(times, flags, administrative_truncation=administrative_truncation)
    return grid
