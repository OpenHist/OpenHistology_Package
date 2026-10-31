"""Serializable public configuration and data contracts (no model imports)."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math

from openhisto._common import json_safe


class Serializable:
    def to_dict(self):
        return json_safe(asdict(self))

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class MaskSpec(Serializable):
    kind: str = "auto"  # auto, binary, indexed, rgb, channels, instance, geojson
    value_map: dict = field(default_factory=dict)
    palette: dict = field(default_factory=dict)  # "r,g,b" -> class ID
    channel_axis: int | None = None
    channel_map: dict = field(default_factory=dict)
    class_property: str = "classification.name"
    class_map: dict = field(default_factory=dict)
    class_priority: list[int] = field(default_factory=list)
    origin: tuple[float, float] | None = None
    downsample: tuple[float, float] | None = None
    ignore_values: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.kind not in {"auto", "binary", "indexed", "rgb", "channels", "instance", "geojson"}:
            raise ValueError("Unknown mask kind")
        for key in ("origin", "downsample"):
            value = getattr(self, key)
            if value is not None and (
                len(value) != 2 or not all(math.isfinite(float(x)) for x in value)
            ):
                raise ValueError(f"{key} must contain two finite numbers")
        if self.downsample is not None and any(float(x) <= 0 for x in self.downsample):
            raise ValueError("downsample must be positive")


@dataclass
class Sample:
    image: Any
    mask: Any = None
    sample_id: str = ""
    slide_id: str | None = None
    mask_spec: MaskSpec | None = None
    xy_level0: tuple[float, float] | None = None
    footprint_level0: tuple[float, float] | None = None
    valid_bounds: tuple[int, int, int, int] | None = None
    prompts: dict | None = None
    embedding: Any = None


@dataclass
class LoadedSample:
    image: Any
    mask: Any = None
    sample_id: str = ""
    slide_id: str | None = None
    xy_level0: tuple[float, float] | None = None
    footprint_level0: tuple[float, float] | None = None
    valid_bounds: tuple[int, int, int, int] | None = None
    prompts: dict | None = None
    embedding: Any = None


@dataclass
class ModelConfig(Serializable):
    model: str = "mask2former"
    num_classes: int = 2
    model_name_or_path: str | None = None
    pretrained: bool = True
    class_names: list[str] = field(default_factory=list)
    ignore_index: int = -100
    image_size: tuple[int, int] = (1000, 1000)
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            isinstance(self.num_classes, bool)
            or not isinstance(self.num_classes, int)
            or self.num_classes < 2
        ):
            raise ValueError("num_classes must be an integer >= 2")
        if len(self.image_size) != 2 or any(
            isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in self.image_size
        ):
            raise ValueError("image_size must be positive (height, width)")
        self.image_size = tuple(self.image_size)
        if 0 <= self.ignore_index < self.num_classes:
            raise ValueError("ignore_index must be outside the class range")
        if self.class_names and len(self.class_names) != self.num_classes:
            raise ValueError("class_names must contain one name per class")


@dataclass
class TrainConfig(Serializable):
    num_epochs: int = 30
    batch_size: int = 5
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    lr_gamma: float = 0.95
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    save_every: int = 1
    augment: bool = True

    def __post_init__(self):
        for name in ("num_epochs", "batch_size", "save_every"):
            if (
                isinstance(getattr(self, name), bool)
                or not isinstance(getattr(self, name), int)
                or getattr(self, name) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not 0 < self.lr_gamma <= 1
        ):
            raise ValueError("Invalid optimizer/scheduler settings")


@dataclass
class PredictConfig(Serializable):
    batch_size: int = 1
    device: str = "auto"
    probabilities: bool = True
    scores: bool = False
    logits: bool = False

    def __post_init__(self):
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be positive")


@dataclass
class Prediction:
    labels: Any
    sample_id: str
    probabilities: Any = None
    scores: Any = None
    logits: Any = None
    image: Any = None
    slide_id: str | None = None
    xy_level0: tuple[float, float] | None = None
    footprint_level0: tuple[float, float] | None = None
    valid_bounds: tuple[int, int, int, int] | None = None
    class_names: list[str] = field(default_factory=list)
    score_semantics: str = ""


@dataclass
class TrainingResult(Serializable):
    output: str
    history: list[dict]
    completed_epochs: int
    best_val_loss: float


class CapabilityError(ValueError):
    """The selected model cannot provide the requested behavior."""
