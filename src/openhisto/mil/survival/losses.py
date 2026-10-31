"""Censor-aware CPU losses; event=1 is observed, event=0 is censored."""

import torch
from torch.nn import functional as F

from ._validation import targets, vector
from .grid import DiscreteTargets, TimeGrid


def cox_ph_loss(
    log_risk,
    durations,
    events,
    *,
    ties="efron",
    reduction="mean",
    allow_no_events=False,
    allow_singleton=False,
):
    """True Cox partial likelihood with grouped risk sets and tied events.

    R(t) contains every subject with duration >=t, including subjects censored
    at t. Mean reduction divides by observed event count for both tie methods.
    The caller must supply an actual cohort/risk set: separate one-slide losses
    are not a substitute for a cohort Cox objective.
    """
    if ties not in {"efron", "breslow"}:
        raise ValueError("ties must be efron or breslow")
    if not isinstance(allow_no_events, bool) or not isinstance(allow_singleton, bool):
        raise ValueError("allow_no_events and allow_singleton must be explicit booleans")
    if reduction not in {"mean", "sum"}:
        raise ValueError("Cox reduction must be mean or sum")
    risk = vector(log_risk, "log_risk")
    times, flags = targets(durations, events)
    if len(risk) != len(times):
        raise ValueError("log_risk and targets must have the same length")
    if len(risk) < 2 and not allow_singleton:
        raise ValueError("Cox loss requires at least two subjects in the supplied risk set")
    event_count = int(flags.sum())
    if event_count == 0:
        if allow_no_events:
            return risk.sum() * 0.0
        raise ValueError("Cox loss requires at least one observed event")
    contributions = []
    for time in torch.unique(times[flags], sorted=True):
        deaths = flags & (times == time)
        count = int(deaths.sum())
        log_denominator = torch.logsumexp(risk[times >= time], dim=0)
        numerator = risk[deaths].sum()
        if ties == "breslow" or count == 1:
            contributions.append(count * log_denominator - numerator)
        else:
            # D_r = R - r/d * E, in log space. E/R<=1 so log1p is
            # stable even when raw relative risks would overflow exp().
            log_event_sum = torch.logsumexp(risk[deaths], dim=0)
            event_fraction = torch.exp((log_event_sum - log_denominator).clamp(max=0.0))
            fractions = torch.arange(count, dtype=risk.dtype) / count
            denominators = log_denominator + torch.log1p(-fractions * event_fraction)
            contributions.append(denominators.sum() - numerator)
    result = torch.stack(contributions).sum()
    return result / event_count if reduction == "mean" else result


def discrete_hazard_nll(logits, labels: DiscreteTargets, *, reduction="mean"):
    """Event: preceding survival + event-bin hazard; censor: completed intervals.

    This is an interval likelihood, not cross entropy over mutually exclusive
    classes. Censoring before the first positive right boundary has zero loss.
    """
    if not isinstance(labels, DiscreteTargets):
        raise TypeError("labels must be a DiscreteTargets object from TimeGrid.transform")
    if not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or logits.ndim != 2:
        raise ValueError("logits must be a CPU tensor [N,K]")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    if logits.shape != (len(labels.events), len(labels.boundaries) - 1):
        raise ValueError("logits must match the target count and fitted interval count")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none, mean or sum")
    work = logits.to(torch.float64)
    preceding = torch.where(labels.events, labels.event_bin, labels.completed_intervals)
    mask = torch.arange(logits.shape[1])[None, :] < preceding[:, None]
    loss = F.softplus(work).masked_fill(~mask, 0.0).sum(dim=1)
    event_terms = F.softplus(-work).gather(1, labels.event_bin[:, None]).squeeze(1)
    loss = loss + event_terms * labels.events.to(work.dtype)
    return loss if reduction == "none" else loss.mean() if reduction == "mean" else loss.sum()


def discrete_hazard_loss(
    logits,
    durations,
    events,
    grid,
    *,
    administrative_truncation=False,
    reduction="mean",
    return_targets=False,
):
    """Transform raw observations and evaluate their interval likelihood."""
    grid = grid if isinstance(grid, TimeGrid) else TimeGrid(grid)
    labels = grid.transform(durations, events, administrative_truncation=administrative_truncation)
    result = discrete_hazard_nll(logits, labels, reduction=reduction)
    return (result, labels) if return_targets else result
