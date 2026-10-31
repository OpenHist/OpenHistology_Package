"""Configuration and results shared by the extractors and command line."""

from dataclasses import asdict, dataclass
from pathlib import Path
import math
from numbers import Real


@dataclass(frozen=True)
class ExtractionConfig:
    """Output dimensions are pixels; stride is measured at target magnification.

    ``source_mag`` overrides objective-power metadata. No scanner magnification
    is assumed when both are absent. ``min_box_size`` is in input mask pixels.
    """

    patch_size: int = 512
    stride: int | None = None
    target_mag: float = 20.0
    source_mag: float | None = None
    foreground_method: str = "HSV"
    sat_thresh: float = 0.05
    tissue_fraction: float = 0.05
    lum_thresh: float = 220.0
    sl_tissue_fraction: float = 0.2
    edge: str = "drop"
    codec: str = "png"
    png_compression: int = 6
    jpeg_quality: int = 95
    min_box_size: float = 0.0
    filter_mask_patches: bool = False
    overwrite: bool = False

    def __post_init__(self):
        for name in ("patch_size", "stride"):
            value = getattr(self, name)
            if name == "stride" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("target_mag", "source_mag"):
            value = getattr(self, name)
            if value is None and name == "source_mag":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if self.edge not in {"drop", "pad"}:
            raise ValueError("edge must be 'drop' or 'pad'")
        if self.codec not in {"png", "jpeg"}:
            raise ValueError("codec must be 'png' or 'jpeg'")
        for name, low, high in (("png_compression", 0, 9), ("jpeg_quality", 1, 95)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if not isinstance(self.foreground_method, str) or self.foreground_method.lower() not in {
            "hsv",
            "sl",
            "luminesence",
            "luminance",
            "otsu",
            "none",
        }:
            raise ValueError("foreground_method must be HSV, SL, luminance, or none")
        for name in ("sat_thresh", "tissue_fraction", "sl_tissue_fraction"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            not isinstance(self.lum_thresh, Real)
            or not math.isfinite(self.lum_thresh)
            or not 0 <= self.lum_thresh <= 255
        ):
            raise ValueError("lum_thresh must be in [0, 255]")
        if (
            not isinstance(self.min_box_size, Real)
            or not math.isfinite(self.min_box_size)
            or self.min_box_size < 0
        ):
            raise ValueError("min_box_size must be nonnegative and finite")
        for name in ("filter_mask_patches", "overwrite"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

    @property
    def effective_stride(self) -> int:
        return self.patch_size if self.stride is None else self.stride

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionSummary:
    archive_path: Path
    candidate_count: int
    patch_count: int
    skipped_background: int = 0
    skipped_edges: int = 0
    skipped_size: int = 0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["archive_path"] = str(self.archive_path)
        return result
