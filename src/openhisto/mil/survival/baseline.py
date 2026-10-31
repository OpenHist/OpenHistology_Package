"""Train-only Breslow calibration for absolute Cox survival probabilities."""

from dataclasses import dataclass
import math

import torch

from openhisto.mil.heads.types import SurvivalOutput
from ._validation import targets, training_split, vector


def finite_exp(value):
    """Exponentiate in float64 with finite saturation at representable extremes."""
    value = value.to(torch.float64)
    # Margin avoids rounding exp(log(max_float)) up to infinity.
    return torch.exp(value.clamp(max=math.log(torch.finfo(torch.float64).max) - 2.0))


@dataclass(frozen=True)
class CoxBaseline:
    """Aggregate baseline only: no individual training risks/labels are stored.

    log_cumulative_hazard is on the centered-risk scale. Keeping this quantity
    in log space avoids underflow from large positive training risk offsets.
    """

    times: torch.Tensor
    log_cumulative_hazard: torch.Tensor
    risk_center: torch.Tensor
    training_horizon: torch.Tensor
    event_count: int
    fit_split: str = "train"

    def __post_init__(self):
        training_split(self.fit_split)
        times = vector(self.times, "baseline times").detach().clone()
        log_h = (
            vector(self.log_cumulative_hazard, "log cumulative baseline hazard").detach().clone()
        )
        center = torch.as_tensor(self.risk_center, dtype=torch.float64).detach().clone()
        horizon = torch.as_tensor(self.training_horizon, dtype=torch.float64).detach().clone()
        if len(times) != len(log_h) or (times < 0).any() or not (times[1:] > times[:-1]).all():
            raise ValueError(
                "Baseline event times must be nonnegative, unique, sorted and aligned with hazards"
            )
        if not (log_h[1:] >= log_h[:-1]).all():
            raise ValueError("Cumulative baseline hazard must be nondecreasing")
        if (
            center.ndim != 0
            or horizon.ndim != 0
            or not torch.isfinite(center)
            or not torch.isfinite(horizon)
        ):
            raise ValueError("Baseline risk center and training horizon must be finite scalars")
        if (
            center.device.type != "cpu"
            or horizon.device.type != "cpu"
            or horizon < times[-1]
            or self.event_count < 1
        ):
            raise ValueError("Invalid CPU baseline calibration horizon or event count")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "log_cumulative_hazard", log_h)
        object.__setattr__(self, "risk_center", center)
        object.__setattr__(self, "training_horizon", horizon)

    @property
    def metadata(self):
        return {
            "method": "breslow",
            "fit_split": self.fit_split,
            "event_count": self.event_count,
            "horizon": float(self.training_horizon),
            "risk_center": float(self.risk_center),
            "storage": "log_cumulative_centered_baseline",
            "time_unit": "unchanged_from_supplied_durations",
        }

    @property
    def cumulative_hazard(self):
        """Uncentered H0 at event times; inference uses log storage directly."""
        return finite_exp(self.log_cumulative_hazard - self.risk_center)

    def predict(self, log_risk, times=None, *, allow_extrapolation=False):
        if not isinstance(allow_extrapolation, bool):
            raise ValueError("allow_extrapolation must be an explicit boolean")
        risk = vector(log_risk, "log_risk")
        if times is None:
            times = self.times
            if times[0] > 0:
                times = torch.cat((times.new_zeros(1), times))
        query = vector(times, "prediction times")
        if (query < 0).any() or not (query[1:] > query[:-1]).all():
            raise ValueError("Prediction times must be nonnegative and strictly increasing")
        if query.max() > self.training_horizon and not allow_extrapolation:
            raise ValueError(
                "Prediction times exceed the training horizon; no extrapolation is enabled"
            )
        index = torch.searchsorted(self.times, query.contiguous(), right=True) - 1
        log_h0 = self.log_cumulative_hazard[index.clamp_min(0)]
        log_h0 = log_h0.masked_fill(index < 0, -torch.inf)
        log_h = log_h0[None, :] + (risk - self.risk_center)[:, None]
        cumulative = finite_exp(log_h)
        survival = torch.exp(-cumulative)
        return SurvivalOutput(
            risk=risk,
            logits=risk[:, None],
            survival=survival,
            time_grid=query,
            cumulative_hazard=cumulative,
            aux={
                **self.metadata,
                "risk_definition": "log_relative_hazard",
                "extrapolation": "carry_last_baseline"
                if query.max() > self.training_horizon
                else None,
            },
        )


def fit_cox_baseline(log_risk, durations, events, *, split="train"):
    """Fit grouped Breslow increments d(t)/sum(exp(risk[R(t)])) after training.

    Same-time censored subjects remain in R(t). Refit after updating the model;
    absolute curves from one set of weights must not calibrate another model.
    """
    training_split(split)
    risk = vector(log_risk, "log_risk").detach()
    times, flags = targets(durations, events)
    if len(risk) != len(times) or len(risk) < 2:
        raise ValueError("Baseline fitting requires at least two aligned training subjects")
    if not flags.any():
        raise ValueError("Baseline fitting requires at least one observed training event")
    center = risk.max()
    centered = risk - center
    event_times = torch.unique(times[flags], sorted=True)
    log_increments = []
    for time in event_times:
        count = (flags & (times == time)).sum().to(torch.float64)
        denominator = torch.logsumexp(centered[times >= time], dim=0)
        log_increments.append(count.log() - denominator)
    cumulative = torch.logcumsumexp(torch.stack(log_increments), dim=0)
    return CoxBaseline(event_times, cumulative, center, times.max(), int(flags.sum()), split)
