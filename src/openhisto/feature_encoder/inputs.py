"""Ordered patch sources, stable identities, and bounded image loading."""

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import SourceSpec, fingerprint, json_safe, mag_key, reject_secrets


OBS_COLUMNS = [
    "patch_id",
    "slide_id",
    "source_id",
    "source_type",
    "source_index",
    "mag",
    "content_hash",
    "x_level0",
    "y_level0",
    "width_level0",
    "height_level0",
    "patch_width",
    "patch_height",
    "valid_x0",
    "valid_y0",
    "valid_x1",
    "valid_y1",
    "source_mag",
    "metadata_json",
]
_FLOAT_COLUMNS = {"mag", "x_level0", "y_level0", "width_level0", "height_level0", "source_mag"}
_INT_COLUMNS = {
    "source_index",
    "patch_width",
    "patch_height",
    "valid_x0",
    "valid_y0",
    "valid_x1",
    "valid_y1",
}


def _number(value, name, *, required=False, positive=False):
    if value is None or value == "":
        if required:
            raise ValueError(f"Patch metadata requires {name}")
        return float("nan")
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (ValueError, TypeError):
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
    digest = hashlib.sha256()
    digest.update(np.asarray(image.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
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


class PatchCollection:
    """All metadata is indexed; image buffers are loaded only for requested rows."""

    def __init__(self):
        self.obs = pd.DataFrame(columns=OBS_COLUMNS)
        self.slide_id = None
        self.sources = []
        self._loaders = {}
        self._readers = []

    def read_images(self, patch_ids, workers=0):
        def checked(identifier):
            image = self._loaders[str(identifier)]()
            if _content_hash(image) != self.obs.at[identifier, "content_hash"]:
                raise ValueError(
                    "Patch content changed during encoding; input sources must remain immutable during a run"
                )
            return image

        if workers:
            with ThreadPoolExecutor(max_workers=min(workers, max(1, len(patch_ids)))) as executor:
                return list(executor.map(checked, patch_ids))
        return [checked(identifier) for identifier in patch_ids]

    def close(self):
        for reader in self._readers:
            reader.close()
        self._readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def prepare_inputs(inputs, config):
    """Read hashes/geometry without retaining the slide's decoded images.

    Archive IDs default to the input basename; explicit SourceSpec.source_id
    values keep identities stable across renamed files. Manifest patch IDs must
    remain stable when its rows are reordered or appended.
    """
    if isinstance(inputs, (str, Path, SourceSpec, dict)):
        inputs = [inputs]
    if not inputs:
        raise ValueError("At least one patch source is required")
    specifications = [
        item
        if isinstance(item, SourceSpec)
        else SourceSpec(**item)
        if isinstance(item, dict)
        else SourceSpec(item)
        for item in inputs
    ]
    if (
        len(specifications) > 1
        and not config.slide_id
        and any(Path(item.path).suffix.lower() not in {".csv", ".jsonl"} for item in specifications)
    ):
        raise ValueError("Combining patch archives requires config.slide_id")
    collection = PatchCollection()
    rows, ids = [], []

    def add(meta, image, loader):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Patch sources must supply uint8 HWC RGB images")
        slide = str(meta["slide_id"])
        if not slide:
            raise ValueError("slide_id cannot be empty")
        if collection.slide_id is not None and collection.slide_id != slide:
            raise ValueError("One feature file can contain only one slide")
        collection.slide_id = slide
        # Magnification is immutable metadata, not identity: changing it for an
        # existing source/index must fail reconciliation rather than append.
        identifier = "p_" + fingerprint([slide, meta["source_id"], meta["patch_id"]])
        if identifier in collection._loaders:
            raise ValueError(
                "Duplicate patch identity; assign distinct source IDs or fix duplicate manifest patch IDs"
            )
        meta["content_hash"] = _content_hash(image)
        meta["patch_height"], meta["patch_width"] = image.shape[:2]
        x0, y0, x1, y1 = (meta[key] for key in ("valid_x0", "valid_y0", "valid_x1", "valid_y1"))
        if not (0 <= x0 <= x1 <= image.shape[1] and 0 <= y0 <= y1 <= image.shape[0]):
            raise ValueError("Patch valid bounds must lie within its image")
        rows.append(meta)
        ids.append(identifier)
        collection._loaders[identifier] = loader

    try:
        for specification in specifications:
            path = Path(specification.path).expanduser().resolve(strict=True)
            source = specification.source_id or path.name
            if not isinstance(source, str) or not source:
                raise ValueError("source_id must be a nonempty string")
            if specification.mag is not None:
                mag_key(specification.mag)
            if path.suffix.lower() in {".csv", ".jsonl"}:
                observed_mags = (
                    {float(specification.mag)} if specification.mag is not None else set()
                )
                for item in _manifest(path):
                    reject_secrets(item)
                    if not str(item.get("patch_id", "")) or not item.get("image_path"):
                        raise ValueError("Patch manifests require patch_id and image_path")
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
                        item.get("mag", item.get("magnification", specification.mag)),
                        "mag",
                        required=True,
                        positive=True,
                    )
                    if specification.mag is not None and mag != float(specification.mag):
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
                            item.get(key, default)
                            for key, default in zip(
                                ("valid_x0", "valid_y0", "valid_x1", "valid_y1"),
                                (0, 0, image.shape[1], image.shape[0]),
                            )
                        ]
                    valid_numbers = [
                        _number(value, "valid_bounds", required=True) for value in valid
                    ]
                    if any(value != int(value) for value in valid_numbers):
                        raise ValueError("valid_bounds must contain integer pixel bounds")
                    explicit_index = item.get("patch_index", item.get("source_index"))
                    index = (
                        _number(explicit_index, "patch_index")
                        if explicit_index not in {None, ""}
                        else -1
                    )
                    if not math.isfinite(index) or index != int(index) or index < -1:
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
                            key: _number(
                                item.get(key),
                                key,
                                positive=key in {"width_level0", "height_level0"},
                            )
                            for key in ("x_level0", "y_level0", "width_level0", "height_level0")
                        },
                        **dict(
                            zip(
                                ("valid_x0", "valid_y0", "valid_x1", "valid_y1"),
                                map(int, valid_numbers),
                            )
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
                from openhisto.patch_extraction import PatchReader

                reader = PatchReader(path)
                collection._readers.append(reader)
                metadata = reader.metadata
                reject_secrets(metadata)
                native_mag = metadata.get(
                    "target_mag", metadata.get("config", {}).get("target_mag")
                )
                mag = _number(
                    native_mag if native_mag is not None else specification.mag,
                    "archive target_mag",
                    required=True,
                    positive=True,
                )
                if specification.mag is not None and mag != float(specification.mag):
                    raise ValueError("Archive magnification disagrees with SourceSpec.mag")
                slide = config.slide_id or path.stem
                collection.slide_id = collection.slide_id or str(slide)
                if collection.slide_id != str(slide):
                    raise ValueError("One feature file can contain only one slide")
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
        collection.obs = pd.DataFrame(
            rows, columns=OBS_COLUMNS, index=pd.Index(ids, name="patch_key")
        )
        for column in OBS_COLUMNS:
            dtype = (
                "float64"
                if column in _FLOAT_COLUMNS
                else "int64"
                if column in _INT_COLUMNS
                else "object"
            )
            collection.obs[column] = collection.obs[column].astype(dtype)
        return collection
    except BaseException:
        collection.close()
        raise
