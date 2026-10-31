"""Lossless label, score, visualization and polygon prediction exports."""

from __future__ import annotations
import json
import os
from pathlib import Path
from numbers import Integral
import tempfile
import numpy as np
from PIL import Image
from .config import CapabilityError, json_safe
from .annotations import labels_to_geojson


def _color_image(labels, palette=None):
    # RGB throughout, with sparse class IDs supported without a huge color table.
    explicit = {}
    for key, color in (palette or {}).items():
        index = int(key)
        if index < 0 or len(color) != 3 or any(not 0 <= int(c) <= 255 for c in color):
            raise ValueError("Palette must map nonnegative class IDs to RGB values in 0..255")
        explicit[index] = tuple(int(c) for c in color)
    result = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for index in np.unique(labels):
        i = int(index)
        color = explicit.get(
            i,
            (0, 0, 0)
            if i == 0
            else ((i * 67) % 224 + 31, (i * 113) % 224 + 31, (i * 173) % 224 + 31),
        )
        result[labels == index] = color
    return result


def export_prediction(
    prediction, output_dir, *, formats=("labels",), overwrite=False, palette=None
):
    """Write requested artifacts and metadata; refuse existing files by default.

    Labels are lossless PNG (uint8/uint16) or TIFF. Probabilities, scores and logits
    are separate NumPy arrays; requesting unavailable fields raises CapabilityError.
    Polygon coordinates use level-0 geometry when supplied, native pixels otherwise.
    """
    labels = np.asarray(prediction.labels)
    if (
        labels.ndim != 2
        or not np.issubdtype(labels.dtype, np.integer)
        or labels.size == 0
        or np.any(labels < 0)
    ):
        raise ValueError("Prediction labels must be a nonempty 2D nonnegative integer array")
    foreground_labels = labels
    if prediction.valid_bounds is not None:
        bounds = prediction.valid_bounds
        if len(bounds) != 4 or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in bounds
        ):
            raise ValueError("valid_bounds must contain four integer output-pixel bounds")
        x0, y0, x1, y1 = (int(v) for v in bounds)
        h, w = labels.shape
        if not (0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h):
            raise ValueError("valid_bounds must lie inside prediction labels")
        foreground_labels = np.zeros_like(labels)
        foreground_labels[y0:y1, x0:x1] = labels[y0:y1, x0:x1]
    name = str(prediction.sample_id)
    if not name or name in {".", ".."} or any(c in name for c in ("/", "\\", "\0")):
        raise ValueError("sample_id must be a nonempty filename component")
    if isinstance(formats, str):
        formats = (formats,)
    formats = tuple(dict.fromkeys(formats))
    supported = {
        "labels",
        "labels_png",
        "labels_tiff",
        "tiff",
        "binary",
        "binary_masks",
        "probabilities",
        "scores",
        "logits",
        "palette",
        "overlay",
        "geojson",
        "qupath",
    }
    unknown = set(formats) - supported
    if unknown:
        raise ValueError(f"Unknown export formats: {sorted(unknown)}")
    if not formats:
        raise ValueError("At least one export format is required")
    if (prediction.xy_level0 is None) != (prediction.footprint_level0 is None):
        raise ValueError("xy_level0 and footprint_level0 must be specified together")
    if prediction.xy_level0 is not None:
        if len(prediction.xy_level0) != 2 or not np.isfinite(prediction.xy_level0).all():
            raise ValueError("xy_level0 must contain finite x and y")
        if (
            len(prediction.footprint_level0) != 2
            or not np.isfinite(prediction.footprint_level0).all()
            or min(prediction.footprint_level0) <= 0
        ):
            raise ValueError("footprint_level0 must contain positive finite width and height")
    folder = Path(output_dir)
    max_label = int(labels.max())
    label_dtype = np.uint8 if max_label <= 255 else np.uint16 if max_label <= 65535 else np.uint32
    if max_label > np.iinfo(np.uint32).max:
        raise ValueError("Label IDs exceed TIFF uint32 capacity")
    encoded_labels = labels.astype(label_dtype)
    jobs = []
    metadata = {
        "sample_id": name,
        "slide_id": prediction.slide_id,
        "shape": list(labels.shape),
        "class_names": prediction.class_names,
        "xy_level0": prediction.xy_level0,
        "footprint_level0": prediction.footprint_level0,
        "valid_bounds": prediction.valid_bounds,
        "score_semantics": prediction.score_semantics,
        "coordinate_space": "level0" if prediction.xy_level0 is not None else "image_pixels",
        "formats": list(formats),
    }

    def add(suffix, kind, value):
        jobs.append((folder / f"{name}{suffix}", kind, value))

    for fmt in formats:
        if fmt in {"labels", "labels_png", "labels_tiff", "tiff"}:
            use_tiff = fmt in {"labels_tiff", "tiff"} or (
                fmt == "labels" and label_dtype == np.uint32
            )
            if fmt == "labels_png" and label_dtype == np.uint32:
                raise ValueError("PNG labels require IDs <=65535; request labels_tiff")
            add(
                "_labels.tiff" if use_tiff else "_labels.png",
                "tiff" if use_tiff else "png",
                encoded_labels,
            )
        elif fmt in {"probabilities", "scores", "logits"}:
            value = getattr(prediction, fmt)
            if value is None:
                raise CapabilityError(f"The prediction does not provide {fmt}")
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                raise ValueError(f"{fmt} must contain finite numeric values")
            if array.ndim < 2 or tuple(array.shape[-2:]) != labels.shape:
                raise ValueError(f"{fmt} must end with native image (height,width) dimensions")
            add(f"_{fmt}.npy", "npy", array)
        elif fmt in {"binary", "binary_masks"}:
            classes = set(int(v) for v in np.unique(foreground_labels) if v > 0) | set(
                range(1, len(prediction.class_names))
            )
            for class_id in sorted(classes):
                add(
                    f"_class_{class_id}.png",
                    "png",
                    (foreground_labels == class_id).astype(np.uint8) * 255,
                )
        elif fmt in {"palette", "overlay"}:
            rgb = _color_image(foreground_labels, palette)
            if fmt == "overlay":
                if prediction.image is None:
                    raise CapabilityError("Overlay export requires the native RGB image")
                image = np.asarray(prediction.image)
                if image.shape != labels.shape + (3,) or image.dtype != np.uint8:
                    raise ValueError("Overlay image must be native uint8 RGB aligned with labels")
                blended = np.rint(0.55 * image + 0.45 * rgb).astype(np.uint8)
                rgb = np.where((foreground_labels != 0)[..., None], blended, image)
            add(f"_{fmt}.png", "png", rgb)
        else:
            geojson = labels_to_geojson(
                foreground_labels,
                xy_level0=prediction.xy_level0,
                footprint_level0=prediction.footprint_level0,
                class_names=prediction.class_names,
                qupath=fmt == "qupath",
            )
            add(f"_{fmt}.geojson", "json", geojson)
    # Alias combinations should not produce duplicate writes.
    unique = {}
    for job in jobs:
        unique[job[0]] = job
    jobs = list(unique.values())
    add("_metadata.json", "json", metadata)
    if not overwrite:
        existing = [str(path) for path, _, _ in jobs if path.exists()]
        if existing:
            raise FileExistsError(f"Export files already exist: {existing}")
    folder.mkdir(parents=True, exist_ok=True)
    staged = []
    published = []
    try:
        for path, kind, value in jobs:
            fd, temp_path = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=folder)
            os.close(fd)
            staged.append((Path(temp_path), path))
            if kind == "png":
                Image.fromarray(value).save(temp_path, format="PNG")
            elif kind == "tiff":
                import tifffile

                tifffile.imwrite(temp_path, value, photometric="minisblack")
            elif kind == "npy":
                with open(temp_path, "wb") as handle:
                    np.save(handle, value, allow_pickle=False)
            else:
                Path(temp_path).write_text(
                    json.dumps(json_safe(value), allow_nan=False, indent=2) + "\n"
                )
        for temporary, destination in staged:
            if overwrite:
                os.replace(temporary, destination)
            else:
                # Hard-link publication is atomic and cannot clobber a racing writer.
                os.link(temporary, destination)
                temporary.unlink()
            published.append(destination)
    except BaseException:
        if not overwrite:
            for path in published:
                path.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return [path for path, _, _ in jobs]
