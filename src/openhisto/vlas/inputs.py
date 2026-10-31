"""Lightweight patch inputs compatible with CPath feature observation identities."""

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SourceSpec, fingerprint, json_safe, reject_secrets


def _number(value, name, *, required=False, positive=False):
    if value is None or value == "":
        if required:
            raise ValueError(f"Patch metadata requires {name}")
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if not math.isfinite(result) or positive and result <= 0:
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _image(path):
    with Image.open(path) as image:
        if image.mode not in {"RGB", "RGBA", "L", "LA", "P", "1"}:
            raise ValueError(
                "Patch manifests require 8-bit RGB/grayscale images; convert high-bit-depth images explicitly"
            )
        if image.mode in {"RGBA", "LA", "P"}:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            return np.array(white.convert("RGB"), dtype=np.uint8)
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _content_hash(image):
    """Match feature_encoder's decoded RGB hash, including image dimensions."""
    digest = hashlib.sha256()
    digest.update(np.asarray(image.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(stream)
            if (
                not reader.fieldnames
                or any(not name for name in reader.fieldnames)
                or len(reader.fieldnames) != len(set(reader.fieldnames))
            ):
                raise ValueError("Patch CSV needs unique, nonempty column names")
            for row in reader:
                if None in row:
                    raise ValueError("Patch CSV row is wider than its header")
                yield row
        else:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Patch JSONL must contain row objects")
                    yield row


def _source(value):
    if isinstance(value, SourceSpec):
        return value
    if isinstance(value, dict):
        return SourceSpec(**value)
    if isinstance(value, (str, Path)):
        return SourceSpec(value)
    # Accept feature_encoder.SourceSpec without importing that heavier package.
    if hasattr(value, "path"):
        return SourceSpec(
            value.path, source_id=getattr(value, "source_id", None), mag=getattr(value, "mag", None)
        )
    raise TypeError("Patch sources must be paths, source dictionaries or SourceSpec objects")


def _get_patch_reader():
    from openhisto.patch_extraction import PatchReader

    return PatchReader


class PatchCollection:
    """Canonical metadata with images decoded only for the requested patch keys."""

    def __init__(self):
        self.rows = []
        self.slide_id = None
        self.sources = []
        self._loaders = {}
        self._rows_by_key = {}
        self._readers = []

    def __len__(self):
        return len(self.rows)

    def read_images(self, patch_keys, workers=0):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        if isinstance(patch_keys, str):
            patch_keys = [patch_keys]
        keys = list(patch_keys)

        def checked(key):
            if key not in self._loaders:
                raise ValueError(f"Unknown patch key: {key}")
            image = self._loaders[key]()
            if _content_hash(image) != self._rows_by_key[key]["content_hash"]:
                raise ValueError(
                    "Patch content changed during description; input sources must remain immutable during a run"
                )
            return image

        if workers and keys:
            with ThreadPoolExecutor(max_workers=min(workers, len(keys))) as executor:
                return list(executor.map(checked, keys))
        return [checked(key) for key in keys]

    def close(self):
        for reader in self._readers:
            reader.close()
        self._readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def prepare_inputs(inputs, config):
    """Index patch archives/manifests using feature_encoder-compatible keys.

    No slide images are extracted here. Dataset acquisition manifests are not
    patch manifests: callers must first extract them into patch archives.
    """
    if isinstance(inputs, (str, Path, SourceSpec, dict)) or hasattr(inputs, "path"):
        inputs = [inputs]
    specifications = [_source(item) for item in inputs]
    if not specifications:
        raise ValueError("At least one patch source is required")
    if (
        len(specifications) > 1
        and not config.slide_id
        and any(Path(item.path).suffix.lower() not in {".csv", ".jsonl"} for item in specifications)
    ):
        raise ValueError("Combining patch archives requires config.slide_id")
    collection = PatchCollection()

    def add(meta, image, loader):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Patch sources must supply uint8 HWC RGB images")
        slide = str(meta["slide_id"])
        if not slide:
            raise ValueError("slide_id cannot be empty")
        if collection.slide_id is not None and collection.slide_id != slide:
            raise ValueError("One description job can contain only one slide")
        collection.slide_id = slide
        key = "p_" + fingerprint([slide, meta["source_id"], meta["patch_id"]])
        if key in collection._loaders:
            raise ValueError(
                "Duplicate patch identity; assign distinct source IDs or fix duplicate manifest patch IDs"
            )
        meta["patch_key"] = key
        meta["content_hash"] = _content_hash(image)
        meta["patch_height"], meta["patch_width"] = image.shape[:2]
        x0, y0, x1, y1 = (meta[name] for name in ("valid_x0", "valid_y0", "valid_x1", "valid_y1"))
        if not (0 <= x0 <= x1 <= image.shape[1] and 0 <= y0 <= y1 <= image.shape[0]):
            raise ValueError("Patch valid bounds must lie within its image")
        collection.rows.append(meta)
        collection._rows_by_key[key] = meta
        collection._loaders[key] = loader

    try:
        for specification in specifications:
            path = Path(specification.path).expanduser().resolve(strict=True)
            source = specification.source_id or path.name
            if not isinstance(source, str) or not source:
                raise ValueError("source_id must be a nonempty string")
            explicit_mag = _number(specification.mag, "mag", positive=True)
            if path.suffix.lower() in {".csv", ".jsonl"}:
                observed_mags = {explicit_mag} if explicit_mag is not None else set()
                for item in _manifest(path):
                    reject_secrets(item)
                    if not str(item.get("patch_id", "")) or not item.get("image_path"):
                        raise ValueError(
                            "Patch manifests require patch_id and image_path; whole-slide dataset manifests must first be extracted into patch archives"
                        )
                    slide = config.slide_id or item.get("slide_id")
                    if not slide:
                        raise ValueError("Patch manifests require slide_id or config.slide_id")
                    if (
                        config.slide_id
                        and item.get("slide_id")
                        and str(item["slide_id"]) != config.slide_id
                    ):
                        raise ValueError("Manifest slide_id disagrees with config.slide_id")
                    mag = _number(
                        item.get("mag", item.get("magnification", explicit_mag)),
                        "mag",
                        required=True,
                        positive=True,
                    )
                    if explicit_mag is not None and mag != explicit_mag:
                        raise ValueError("Manifest magnification disagrees with SourceSpec.mag")
                    observed_mags.add(mag)
                    image_path = Path(item["image_path"]).expanduser()
                    image_path = (
                        image_path if image_path.is_absolute() else path.parent / image_path
                    ).resolve(strict=True)
                    image = _image(image_path)
                    valid = item.get("valid_bounds")
                    if valid is not None:
                        if not isinstance(valid, list) or len(valid) != 4:
                            raise ValueError(
                                "JSONL valid_bounds must contain four output-pixel bounds"
                            )
                    else:
                        valid = [
                            item.get(name, default)
                            for name, default in zip(
                                ("valid_x0", "valid_y0", "valid_x1", "valid_y1"),
                                (0, 0, image.shape[1], image.shape[0]),
                            )
                        ]
                    bounds = [_number(value, "valid_bounds", required=True) for value in valid]
                    if any(value != int(value) for value in bounds):
                        raise ValueError("valid_bounds must contain integer pixel bounds")
                    explicit_index = item.get("patch_index", item.get("source_index"))
                    index = (
                        _number(explicit_index, "patch_index")
                        if explicit_index not in {None, ""}
                        else -1
                    )
                    if index != int(index) or index < -1:
                        raise ValueError("patch_index must be a nonnegative integer when supplied")
                    meta = {
                        "patch_id": str(item["patch_id"]),
                        "slide_id": str(slide),
                        "source_id": source,
                        "source_type": "manifest",
                        "source_index": int(index),
                        "mag": mag,
                        "source_mag": _number(item.get("source_mag"), "source_mag", positive=True),
                        "metadata_json": json.dumps(
                            json_safe(item), sort_keys=True, allow_nan=False
                        ),
                        **{
                            name: _number(
                                item.get(name),
                                name,
                                positive=name in {"width_level0", "height_level0"},
                            )
                            for name in ("x_level0", "y_level0", "width_level0", "height_level0")
                        },
                        **dict(
                            zip(("valid_x0", "valid_y0", "valid_x1", "valid_y1"), map(int, bounds))
                        ),
                    }
                    add(meta, image, lambda current=image_path: _image(current))
                collection.sources.append(
                    {
                        "source_id": source,
                        "source_type": "manifest",
                        "path": str(path),
                        "magnifications": sorted(observed_mags),
                    }
                )
            else:
                if path.suffix.lower() not in {".h5", ".hdf5"}:
                    raise ValueError(
                        "Patch sources must be HDF5 patch archives or CSV/JSONL patch manifests"
                    )
                reader = _get_patch_reader()(path)
                collection._readers.append(reader)
                metadata = reader.metadata
                reject_secrets(metadata)
                native_mag = metadata.get(
                    "target_mag", metadata.get("config", {}).get("target_mag")
                )
                mag = _number(
                    native_mag if native_mag is not None else explicit_mag,
                    "archive target_mag",
                    required=True,
                    positive=True,
                )
                if explicit_mag is not None and mag != explicit_mag:
                    raise ValueError("Archive magnification disagrees with SourceSpec.mag")
                slide = config.slide_id or path.stem
                collection.slide_id = collection.slide_id or str(slide)
                if collection.slide_id != str(slide):
                    raise ValueError("One description job can contain only one slide")
                for index in range(len(reader)):
                    record = reader[index]
                    meta = {
                        "patch_id": str(index),
                        "slide_id": str(slide),
                        "source_id": source,
                        "source_type": "archive",
                        "source_index": index,
                        "mag": mag,
                        "x_level0": float(record.xy_level0[0]),
                        "y_level0": float(record.xy_level0[1]),
                        "width_level0": float(record.footprint_level0[0]),
                        "height_level0": float(record.footprint_level0[1]),
                        "source_mag": _number(
                            metadata.get("source_mag"), "source_mag", positive=True
                        ),
                        "metadata_json": "{}",
                        **dict(
                            zip(
                                ("valid_x0", "valid_y0", "valid_x1", "valid_y1"),
                                record.valid_bounds,
                            )
                        ),
                    }
                    add(
                        meta,
                        record.image,
                        lambda current=index, source_reader=reader: source_reader[current].image,
                    )
                collection.sources.append(
                    {
                        "source_id": source,
                        "source_type": "archive",
                        "path": str(path),
                        "magnifications": [mag],
                        "extraction": metadata,
                    }
                )
        collection.slide_id = (
            collection.slide_id or config.slide_id or Path(specifications[0].path).stem
        )
        return collection
    except BaseException:
        collection.close()
        raise
