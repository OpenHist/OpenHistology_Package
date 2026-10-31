"""Serializable settings for patch stain normalization and GAN training."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from openhisto._common import json_safe

METHODS = ("vahadane", "macenko", "reinhard", "histogram", "cyclegan", "pix2pix")
ALIASES = {
    "default": "vahadane",
    "hist": "histogram",
    "cycleGAN": "cyclegan",
    "cycle_gan": "cyclegan",
}


class _Unspecified:
    def __repr__(self):
        return "<unspecified>"


_UNSPECIFIED = _Unspecified()
_RESUME_DEFAULTS = {
    "output": "stain_training",
    "device": "auto",
    "workers": 0,
    "sample_every": 1,
    "checkpoint_every": 1,
}


def canonical_method(value):
    if not isinstance(value, str):
        raise ValueError("method must be a string")
    value = ALIASES.get(value, value.lower())
    value = ALIASES.get(value, value)
    if value not in METHODS:
        raise ValueError(f"Unknown method {value!r}; choose from {', '.join(METHODS)}")
    return value


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _device(value):
    if not isinstance(value, str) or (
        value not in {"auto", "cpu", "mps", "cuda"}
        and not (value.startswith("cuda:") and value[5:].isdigit())
    ):
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<index>")


@dataclass(frozen=True)
class DomainSpec:
    """Destination appearance. Mixtures explicitly enumerate destination candidates."""

    method: str = "vahadane"
    references: tuple[str, ...] = ()
    checkpoint: str | None = None
    architecture: str | None = None
    input_mode: str | None = None
    direction: str | None = None
    options: dict = field(default_factory=dict)
    candidates: tuple["DomainSpec", ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "method", canonical_method(self.method))
        refs = (
            (self.references,)
            if isinstance(self.references, (str, Path))
            else tuple(self.references)
        )
        if any(not isinstance(v, (str, Path)) or not str(v) for v in refs):
            raise ValueError("references must contain image paths")
        object.__setattr__(self, "references", tuple(str(v) for v in refs))
        if self.checkpoint is not None:
            object.__setattr__(self, "checkpoint", str(self.checkpoint))
        if not isinstance(self.options, dict):
            raise ValueError("options must be a dictionary")
        json.dumps(json_safe(self.options), allow_nan=False)
        candidates = tuple(DomainSpec(**v) if isinstance(v, dict) else v for v in self.candidates)
        if any(not isinstance(v, DomainSpec) or v.candidates for v in candidates):
            raise ValueError("candidates must be DomainSpec objects without nested mixtures")
        object.__setattr__(self, "candidates", candidates)
        if self.input_mode not in {None, "rgb", "grayscale"}:
            raise ValueError("input_mode must be rgb or grayscale")
        if self.direction not in {None, "AtoB", "BtoA"}:
            raise ValueError("direction must be AtoB or BtoA")
        if candidates and (refs or self.checkpoint is not None):
            raise ValueError("A mixture uses candidates rather than parent references/checkpoint")
        if candidates and (
            self.options
            or any(
                value is not None for value in (self.architecture, self.input_mode, self.direction)
            )
        ):
            raise ValueError("Mixture backend options must be supplied on each candidate")
        if candidates and self.method != "vahadane":
            raise ValueError("A mixture specifies methods on its candidates, not on the parent")

    def to_dict(self):
        return json_safe(asdict(self))


@dataclass(frozen=True)
class TransferConfig:
    seed: int = 256
    device: str = "auto"
    standardize: bool = False
    histogram_clip: tuple[float, float] | None = (0.01, 0.99)
    mix: bool = False
    overwrite: bool = False
    workers: int = 0
    batch_size: int = 1
    png_compression: int = 6

    def __post_init__(self):
        _integer(self.seed, "seed", 0)
        _integer(self.workers, "workers", 0)
        _integer(self.batch_size, "batch_size", 1)
        _integer(self.png_compression, "png_compression", 0)
        if self.png_compression > 9:
            raise ValueError("png_compression must be <= 9")
        for name in ("standardize", "mix", "overwrite"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        _device(self.device)
        if self.histogram_clip is not None:
            clip = tuple(self.histogram_clip)
            if (
                len(clip) != 2
                or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in clip
                )
                or not 0 <= clip[0] < clip[1] <= 1
            ):
                raise ValueError(
                    "histogram_clip must be null or [low, high] with 0 <= low < high <= 1"
                )
            object.__setattr__(self, "histogram_clip", clip)

    def to_dict(self):
        return json_safe(asdict(self))


@dataclass(frozen=True)
class TrainConfig:
    method: str = "cyclegan"
    source: str | None = None
    target: str | None = None
    output: str = _UNSPECIFIED
    pairs: str | None = None
    paired_ab: str | None = None
    grayscale_source: bool = False
    source_name: str = "source"
    target_name: str = "target"
    device: str = _UNSPECIFIED
    seed: int = 256
    batch_size: int | None = None
    epochs: int | None = None
    decay_epochs: int | None = None
    load_size: int | None = None
    crop_size: int = 256
    workers: int = _UNSPECIFIED
    ngf: int = 64
    ndf: int | None = None
    architecture: str | None = None
    input_mode: str | None = None
    lr: float | None = None
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    lambda_l1: float = 100.0
    lambda_mask: float = 0.0
    lambda_real_seg: float = 1.0
    seg_channels: int = 32
    seg_depth: int = 4
    resume: str | None = None
    warm_start: str | None = None
    sample_every: int = _UNSPECIFIED
    checkpoint_every: int = _UNSPECIFIED
    max_steps: int | None = None
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        # Retain ordinary public defaults and checkpoint dictionaries, while
        # distinguishing omitted resume settings from explicit default values.
        supplied = set()
        for name, default in _RESUME_DEFAULTS.items():
            if getattr(self, name) is _UNSPECIFIED:
                object.__setattr__(self, name, default)
            else:
                supplied.add(name)
        object.__setattr__(self, "_specified_resume_fields", frozenset(supplied))
        method = canonical_method(self.method)
        if method not in {"cyclegan", "pix2pix"}:
            raise ValueError("Only cyclegan and pix2pix have trainable models")
        object.__setattr__(self, "method", method)
        for name in ("source", "target", "output", "pairs", "paired_ab", "resume", "warm_start"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, str(value))
        for name in ("seed", "workers", "sample_every", "checkpoint_every"):
            _integer(
                getattr(self, name), name, 0 if name in {"seed", "workers", "sample_every"} else 1
            )
        for name in ("crop_size", "ngf", "seg_channels", "seg_depth"):
            _integer(getattr(self, name), name, 1)
        for name in ("batch_size", "load_size", "ndf", "max_steps"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name, 1)
        for name in ("epochs", "decay_epochs"):
            if getattr(self, name) is not None:
                _integer(getattr(self, name), name, 0)
        for name in (
            "lambda_cycle",
            "lambda_identity",
            "lambda_l1",
            "lambda_mask",
            "lambda_real_seg",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.lr is not None and (
            isinstance(self.lr, bool)
            or not isinstance(self.lr, (int, float))
            or not math.isfinite(self.lr)
            or self.lr <= 0
        ):
            raise ValueError("lr must be positive and finite")
        _device(self.device)
        if self.input_mode not in {None, "rgb", "grayscale"}:
            raise ValueError("input_mode must be rgb or grayscale")
        if not isinstance(self.grayscale_source, bool) or not isinstance(self.options, dict):
            raise ValueError("grayscale_source must be boolean and options a dictionary")
        if self.resume and self.warm_start:
            raise ValueError("resume and warm_start are mutually exclusive")
        if not self.output or not self.source_name or not self.target_name:
            raise ValueError("output and domain names must be nonempty")
        json.dumps(json_safe(self.options), allow_nan=False)

    def to_dict(self):
        return json_safe(asdict(self))


@dataclass(frozen=True)
class TransferReport:
    output: Path
    processed: int
    blank: int
    failed: int
    method: str

    def to_dict(self):
        return json_safe(asdict(self))


@dataclass(frozen=True)
class TrainReport:
    output: Path
    epochs_completed: int
    global_step: int
    checkpoint: Path
    exports: tuple[Path, ...]

    def to_dict(self):
        return json_safe(asdict(self))
