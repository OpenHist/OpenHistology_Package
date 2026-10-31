"""Cox and discrete-hazard heads with explicit censoring and calibration."""

from .baseline import CoxBaseline, fit_cox_baseline
from .grid import DiscreteTargets, TimeGrid, fit_time_grid
from .heads import CoxHead, DiscreteHazardHead
from .losses import cox_ph_loss, discrete_hazard_loss, discrete_hazard_nll


def head_info(name):
    if name not in {"cox", "discrete_hazard"}:
        raise ValueError("Survival head must be cox or discrete_hazard")
    return {
        "family": "survival",
        "name": name,
        "networks": ["linear", "mlp"],
        "supported_devices": ["cpu"],
        "event_convention": "1=observed_event,0=right_censored",
        "calibration": "training-only Breslow baseline"
        if name == "cox"
        else "training-only positive-width time grid",
        "outputs": "log relative risk; absolute survival after baseline fit"
        if name == "cox"
        else "conditional hazards; time-indexed survival after grid fit",
    }


def list_heads():
    return [head_info(name) for name in ("cox", "discrete_hazard")]


def create_survival_head(name, input_dim, **options):
    head_info(name)
    cls = CoxHead if name == "cox" else DiscreteHazardHead
    return cls(input_dim, **options).cpu()


__all__ = [
    "CoxBaseline",
    "CoxHead",
    "DiscreteHazardHead",
    "DiscreteTargets",
    "TimeGrid",
    "create_survival_head",
    "list_heads",
    "head_info",
    "fit_time_grid",
    "fit_cox_baseline",
    "cox_ph_loss",
    "discrete_hazard_loss",
    "discrete_hazard_nll",
]
