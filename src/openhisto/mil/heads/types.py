"""Outputs shared by classification, survival and MIL attachment."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClassificationOutput:
    logits: Any
    probabilities: Any
    kl: Any = None
    uncertainty: dict = field(default_factory=dict)


@dataclass
class SurvivalOutput:
    risk: Any
    logits: Any = None
    hazards: Any = None
    survival: Any = None
    time_grid: Any = None
    cumulative_hazard: Any = None
    aux: dict = field(default_factory=dict)


@dataclass
class TaskOutput:
    head: Any
    representation: Any
    mil: Any
    representation_spec: Any = None
