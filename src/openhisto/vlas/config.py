"""Serializable jobs; credentials belong in the runtime environment."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from openhisto import _common
from openhisto._common import fingerprint, json_safe


def reject_secrets(value):
    _common.reject_secrets(
        value,
        "Supply credentials through environment variables or an existing Hugging Face login, never job settings",
    )


def mag_key(value):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("Magnification must be positive and finite")
    return format(float(value), ".12g") + "x"


@dataclass(frozen=True)
class SourceSpec:
    path: str | Path
    source_id: str | None = None
    mag: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    model: str
    revision: str | None = None
    checkpoint: str | None = None
    python_executable: str | None = None
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a nonempty catalogue ID")
        for name in ("revision", "checkpoint", "python_executable"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (str, Path)) or not str(value)):
                raise ValueError(f"{name} must be a nonempty string when supplied")
        if not isinstance(self.options, dict):
            raise ValueError("options must be an object")
        reject_secrets(self.options)
        json.dumps(json_safe(self.options), allow_nan=False)

    def to_dict(self):
        return json_safe(asdict(self))


@dataclass(frozen=True)
class DescribeConfig:
    slide_id: str | None = None
    batch_size: int = 1
    workers: int = 0
    device: str = "cuda"
    precision: str = "auto"
    max_new_tokens: int = 2048
    do_sample: bool = False
    temperature: float = 1.0
    seed: int = 0
    offline: bool = False
    cache_dir: str | None = None
    timeout: float = 600.0
    startup_timeout: float = 600.0
    worker_diagnostics: bool = False
    patch_keys: tuple[str, ...] | None = None

    def __post_init__(self):
        for name in ("batch_size", "max_new_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("workers", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("timeout", "startup_timeout", "temperature"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.worker_diagnostics, bool):
            raise ValueError("worker_diagnostics must be boolean")
        if self.precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("precision must be auto, float32, float16, or bfloat16")
        if self.device not in {"auto", "cpu", "mps", "cuda"} and not self.device.startswith(
            "cuda:"
        ):
            raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<index>")
        if not isinstance(self.do_sample, bool) or not isinstance(self.offline, bool):
            raise ValueError("do_sample and offline must be booleans")
        if self.slide_id is not None and (not isinstance(self.slide_id, str) or not self.slide_id):
            raise ValueError("slide_id must be a nonempty string")
        if self.patch_keys is not None:
            if (
                not isinstance(self.patch_keys, (list, tuple))
                or any(not isinstance(key, str) or not key for key in self.patch_keys)
                or len(set(self.patch_keys)) != len(self.patch_keys)
            ):
                raise ValueError("patch_keys must be a sequence of unique patch keys")

    def identity(self):
        return {
            key: getattr(self, key)
            for key in ("device", "precision", "max_new_tokens", "do_sample", "temperature", "seed")
        }

    def to_dict(self):
        return json_safe(asdict(self))


@dataclass(frozen=True)
class DescriptionReport:
    output: str
    patches: int
    completed: int = 0
    reused: int = 0
    truncated: int = 0
    failed: int = 0
    pending: int = 0
    runs: tuple[str, ...] = ()

    def to_dict(self):
        return json_safe(asdict(self))
