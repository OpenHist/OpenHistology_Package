"""Serializable execution settings. Credentials deliberately live elsewhere."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from openhisto import _common
from openhisto._common import fingerprint, json_safe


def reject_secrets(value):
    _common.reject_secrets(
        value,
        "Credentials must be supplied through AuthConfig or environment variables, not execution settings",
    )


def mag_key(value) -> str:
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("Magnification must be positive and finite")
    from decimal import Decimal

    number = format(Decimal(str(float(value))).normalize(), "f")
    return number + "x"


@dataclass(frozen=True)
class EncoderSpec:
    model: str
    revision: str | None = None
    dimension: int | None = None
    normalize: bool = False
    checkpoint: str | None = None
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a registry profile ID")
        if self.dimension is not None and (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension < 1
        ):
            raise ValueError("dimension must be a positive integer")
        if not isinstance(self.normalize, bool) or not isinstance(self.options, dict):
            raise ValueError("normalize must be boolean and options must be a dictionary")
        reject_secrets(self.options)
        json.dumps(json_safe(self.options), allow_nan=False)

    def to_dict(self):
        return json_safe(asdict(self))


@dataclass(frozen=True)
class EncodeConfig:
    slide_id: str | None = None
    batch_size: int = 32
    device: str = "auto"
    precision: str = "auto"
    workers: int = 0
    compression_level: int = 4
    offline: bool = False
    cache_dir: str | None = None
    prompt_credentials: bool = False
    timeout: float = 60.0
    max_retries: int = 3
    cloud_concurrency: int = 4
    metadata_policy: str = "preserve"

    def __post_init__(self):
        for name in ("batch_size", "cloud_concurrency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("workers", "max_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            not isinstance(self.compression_level, int)
            or isinstance(self.compression_level, bool)
            or not 0 <= self.compression_level <= 9
        ):
            raise ValueError("compression_level must be an integer from 0 to 9")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("precision must be auto, float32, float16, or bfloat16")
        if self.device not in {"auto", "cpu", "mps", "cuda"} and not self.device.startswith(
            "cuda:"
        ):
            raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<index>")
        if self.metadata_policy not in {"preserve", "update", "error"}:
            raise ValueError("metadata_policy must be preserve, update, or error")

    def identity(self):
        return {"device": self.device, "precision": self.precision, "output_dtype": "float32"}


@dataclass(frozen=True)
class SourceSpec:
    path: str | Path
    source_id: str | None = None
    mag: float | None = None


@dataclass(frozen=True)
class EncodingReport:
    output: Path
    patches: int
    completed: int
    reused: int
    pending: int
    failed: int
    runs: tuple[str, ...] = ()
    metadata_updated: int = 0
    metadata_preserved: int = 0

    def to_dict(self):
        return {**asdict(self), "output": str(self.output)}
