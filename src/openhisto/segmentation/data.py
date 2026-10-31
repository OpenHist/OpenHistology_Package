"""Native-resolution samples, explicit mask decoding and patch-archive adapters."""

from __future__ import annotations
import csv
from dataclasses import fields, replace
import json
from pathlib import Path
from numbers import Integral
import numpy as np
from PIL import Image
from .config import MaskSpec, Sample, LoadedSample
from .annotations import rasterize_geojson

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _spec(value):
    if value is None or isinstance(value, MaskSpec):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    return MaskSpec(**value)


def _resolve(value, base):
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        return path if path.is_absolute() else base / path
    return value


def _json_cell(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
    return value


def _descriptor(value, base):
    value = _json_cell(value)
    if isinstance(value, dict):
        value = dict(value)
        for key in ("path", "archive_path", "feature_store"):
            if key in value:
                value[key] = _resolve(value[key], base)
        return value
    return _resolve(value, base)


def _record(row, base, default_spec):
    values = dict(row)
    values = {
        k: v
        for k, v in values.items()
        if v is not None and not (isinstance(v, str) and not v.strip())
    }
    image = values.pop("image_path", values.pop("image", None))
    mask = values.pop(
        "mask_path",
        values.pop("annotation_path", values.pop("annotation", values.pop("mask", None))),
    )
    if image is None:
        raise ValueError("Each sample requires image or image_path")
    for name in ("xy_level0", "footprint_level0", "valid_bounds", "prompts"):
        if name in values:
            values[name] = _json_cell(values[name])
    if "x" in values and "y" in values:
        values["xy_level0"] = (float(values.pop("x")), float(values.pop("y")))
    if "footprint_width" in values and "footprint_height" in values:
        values["footprint_level0"] = (
            float(values.pop("footprint_width")),
            float(values.pop("footprint_height")),
        )
    values["mask_spec"] = _spec(values.get("mask_spec", default_spec))
    image = _descriptor(image, base)
    if isinstance(image, (str, Path)):
        image = _resolve(image, base)
        values.setdefault("sample_id", image.stem)
    mask = _resolve(mask, base)
    if "embedding" in values:
        values["embedding"] = _descriptor(values["embedding"], base)
    valid = {f.name for f in fields(Sample)} - {"image", "mask"}
    unexpected = set(values) - valid
    if unexpected:
        raise ValueError(f"Unknown sample fields: {sorted(unexpected)}")
    return Sample(image=image, mask=mask, **values)


def _get_patch_reader():
    from openhisto.patch_extraction import PatchReader

    return PatchReader


def prepare_dataset(source, *, mask_spec=None, num_classes=2, ignore_index=-100):
    """Normalize arrays, image paths, manifests or archive specs to lazy Samples.

    CSV paths resolve relative to the CSV. Legacy slide_dir/mask_dir manifests pair
    image stems to '<stem>_mask' labels. Directories without a mask column are
    valid unlabeled prediction inputs. Archive masks use explicit origin/downsample.
    """
    if num_classes < 2 or 0 <= ignore_index < num_classes:
        raise ValueError("Invalid class count or ignore_index")
    default_spec = _spec(mask_spec)
    base = Path.cwd()

    def expand(item, root):
        if isinstance(item, Sample):
            return [replace(item, mask_spec=item.mask_spec or default_spec)]
        if isinstance(item, np.ndarray) or isinstance(item, Image.Image):
            return [Sample(item, mask_spec=default_spec)]
        if isinstance(item, dict):
            if "samples" in item:
                if not isinstance(item["samples"], list):
                    raise ValueError(
                        "Dataset samples must be a list; use manifest.csv for prepared datasets"
                    )
                return [sample for row in item["samples"] for sample in expand(row, root)]
            item = {
                k: v
                for k, v in item.items()
                if v is not None and not (isinstance(v, str) and not v.strip())
            }
            if "archive_path" in item:
                # Reader access remains local to each process and each read.
                archive_path = _resolve(item["archive_path"], root)
                with _get_patch_reader()(archive_path) as reader:
                    count = len(reader)
                mask = _resolve(
                    item.get(
                        "annotation_path",
                        item.get("annotation", item.get("mask_path", item.get("mask"))),
                    ),
                    root,
                )
                selected = item.get("patch_index", item.get("index"))
                if selected is None or selected == "":
                    selected = None
                    indices = range(count)
                else:
                    if isinstance(selected, (bool, np.bool_)) or float(selected) != int(selected):
                        raise ValueError("Archive index must be an integer")
                    selected = int(selected)
                    if selected < 0 or selected >= count:
                        raise ValueError("Archive index is out of bounds")
                    indices = [selected]
                name = str(item.get("sample_id", Path(archive_path).stem))
                return [
                    Sample(
                        image={"archive_path": str(archive_path), "index": i},
                        mask=mask,
                        sample_id=name
                        if selected is not None and item.get("sample_id")
                        else f"{name}_{i:06d}",
                        slide_id=item.get("slide_id", Path(archive_path).stem),
                        mask_spec=_spec(item.get("mask_spec") or default_spec),
                        prompts=_json_cell(item.get("prompts")),
                        embedding=_descriptor(item.get("embedding"), root),
                    )
                    for i in indices
                ]
            if "slide_dir" in item:
                image_dir = _resolve(item["slide_dir"], root)
                mask_dir = _resolve(item.get("mask_dir"), root)
                if not image_dir.is_dir() or (mask_dir is not None and not mask_dir.is_dir()):
                    raise FileNotFoundError("Manifest image or mask directory does not exist")
                masks = {}
                if mask_dir is not None:
                    for path in sorted(mask_dir.iterdir()):
                        if path.suffix.lower() in _IMAGE_SUFFIXES | {".npy", ".json", ".geojson"}:
                            key = path.stem.removesuffix("_mask")
                            if key in masks:
                                raise ValueError(f"Ambiguous masks for {key}")
                            masks[key] = path
                result = []
                for path in sorted(image_dir.iterdir()):
                    if path.suffix.lower() not in _IMAGE_SUFFIXES:
                        continue
                    if mask_dir is not None and path.stem not in masks:
                        raise ValueError(f"Missing mask for {path.name}")
                    slide_id = item.get("slide_id", item.get("slide_name", image_dir.name))
                    result.append(
                        Sample(
                            path,
                            masks.get(path.stem),
                            sample_id=f"{slide_id}_{path.stem}",
                            slide_id=str(slide_id),
                            mask_spec=_spec(item.get("mask_spec") or default_spec),
                        )
                    )
                if mask_dir is not None:
                    image_stems = {
                        path.stem
                        for path in image_dir.iterdir()
                        if path.suffix.lower() in _IMAGE_SUFFIXES
                    }
                    orphaned = set(masks) - image_stems
                    if orphaned:
                        raise ValueError(f"Masks without matching images: {sorted(orphaned)}")
                return result
            return [_record(item, root, default_spec)]
        if isinstance(item, (str, Path)):
            path = _resolve(item, root)
            if path.suffix.lower() == ".json":
                content = json.loads(path.read_text())
                if isinstance(content, dict) and "samples" in content:
                    content = content["samples"]
                if isinstance(content, list):
                    return [sample for row in content for sample in expand(row, path.parent)]
                return expand(content, path.parent)
            if path.suffix.lower() == ".csv":
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                result = []
                for row in rows:
                    result.extend(expand(row, path.parent))
                return result
            if path.is_dir():
                return expand({"slide_dir": path}, root)
            if path.suffix.lower() in {".h5", ".hdf5"}:
                return expand({"archive_path": path}, root)
            return [Sample(path, sample_id=path.stem, mask_spec=default_spec)]
        raise TypeError(f"Unsupported dataset source: {type(item).__name__}")

    if isinstance(source, (list, tuple)):
        samples = [sample for item in source for sample in expand(item, base)]
    else:
        samples = expand(source, base)
    if not samples:
        raise ValueError("Dataset contains no samples")
    ids = set()
    for i, sample in enumerate(samples):
        if not sample.sample_id:
            sample.sample_id = f"sample_{i:06d}"
        if sample.sample_id in ids:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        ids.add(sample.sample_id)
    return samples


def _pillow_array(image, decode_colors=False):
    if decode_colors and image.mode in {"P", "LA"}:
        # Keep alpha for the image reader's white-background compositing.
        image = image.convert("RGBA")
    elif decode_colors and image.mode == "1":
        image = image.convert("L")
    return np.array(image)


def _array(source, *, decode_colors=False):
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, Image.Image):
        return _pillow_array(source, decode_colors)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                if len(arrays.files) != 1:
                    raise ValueError("NPZ must contain exactly one array")
                return arrays[arrays.files[0]]
        if path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile

            try:
                return tifffile.memmap(path)
            except (ValueError, TypeError):
                return tifffile.imread(path)
        with Image.open(path) as image:
            return _pillow_array(image, decode_colors)
    raise TypeError("Expected an array or image/NumPy path")


def _class_validate(array, num_classes, ignore_index):
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Decoded mask must be a two-dimensional integer label map")
    invalid = (array != ignore_index) & ((array < 0) | (array >= num_classes))
    if np.any(invalid):
        raise ValueError(
            f"Mask contains class IDs outside 0..{num_classes - 1}: {np.unique(array[invalid]).tolist()}"
        )
    return array.astype(np.int64, copy=False)


def decode_mask(source, spec=None, *, num_classes=2, ignore_index=-100):
    """Decode masks without grayscale conversion or loss of integer precision."""
    spec = _spec(spec) or MaskSpec()
    array = _array(source, decode_colors=spec.kind == "rgb")
    kind = spec.kind
    if kind == "auto":
        if array.ndim != 2:
            raise ValueError("Multichannel masks require explicit rgb/channels kind and mapping")
        values = set(np.unique(array).tolist()) - set(spec.ignore_values) - {ignore_index}
        kind = (
            "binary" if num_classes == 2 and (values <= {0, 1} or values <= {0, 255}) else "indexed"
        )
    if kind == "rgb":
        if array.ndim != 3 or array.shape[-1] not in (3, 4) or not spec.palette:
            raise ValueError("RGB masks require HWC colors and an explicit palette")
        colors = array[..., :3]
        result = np.full(colors.shape[:2], ignore_index, dtype=np.int64)
        known = np.zeros(result.shape, dtype=bool)
        for color, class_id in spec.palette.items():
            rgb = tuple(int(v) for v in (color.split(",") if isinstance(color, str) else color))
            if len(rgb) != 3:
                raise ValueError("Palette colors must contain r,g,b")
            where = np.all(colors == rgb, axis=-1)
            result[where] = int(class_id)
            known |= where
        if not known.all():
            raise ValueError(
                f"Unknown RGB colors: {np.unique(colors[~known], axis=0)[:8].tolist()}"
            )
    elif kind == "channels":
        if array.ndim != 3 or spec.channel_axis is None or not spec.channel_map:
            raise ValueError("Channel masks require channel_axis and channel_map")
        channels = np.moveaxis(array, spec.channel_axis, 0)
        result = np.zeros(channels.shape[1:], dtype=np.int64)
        active = []
        for channel, class_id in spec.channel_map.items():
            index = int(channel)
            if index < 0 or index >= len(channels):
                raise ValueError("channel_map index is out of bounds")
            values = channels[index]
            if not np.isfinite(values).all() or not np.isin(values, [0, 1, 255]).all():
                raise ValueError("Channel masks must contain binary 0/1 or 0/255 values")
            active.append((int(class_id), values != 0))
        if spec.class_priority:
            rank = {int(v): i for i, v in enumerate(spec.class_priority)}
            if {cid for cid, _ in active} - set(rank):
                raise ValueError("class_priority must include all mapped channel classes")
            active.sort(key=lambda pair: rank[pair[0]], reverse=True)
        else:
            used = np.zeros(result.shape, dtype=bool)
            for class_id, where in active:
                if np.any(used & where & (result != class_id)):
                    raise ValueError("Overlapping channel classes require class_priority")
                result[where] = class_id
                used |= where
        for class_id, where in active:
            result[where] = class_id
    elif kind in {"binary", "indexed", "instance"}:
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.number) and array.dtype != bool:
            raise ValueError("Binary/indexed masks must be two-dimensional numeric arrays")
        if not np.isfinite(array).all() or np.any(array != np.floor(array)):
            raise ValueError("Mask values must be finite integers")
        ignored = np.isin(array, list(spec.ignore_values) + [ignore_index])
        if spec.value_map:
            result = np.full(array.shape, ignore_index, dtype=np.int64)
            known = ignored.copy()
            for raw, class_id in spec.value_map.items():
                where = array == int(raw)
                result[where] = int(class_id)
                known |= where
            if not known.all():
                raise ValueError(f"Unmapped mask values: {np.unique(array[~known]).tolist()}")
        elif kind == "instance":
            raise ValueError(
                "Instance masks require an explicit instance ID to semantic class value_map"
            )
        elif kind == "binary":
            known = array[~ignored]
            if not (np.isin(known, [0, 1]).all() or np.isin(known, [0, 255]).all()):
                raise ValueError("Binary masks must use 0/1 or 0/255")
            result = (array != 0).astype(np.int64)
        else:
            result = array.astype(np.int64)
        result[ignored] = ignore_index
    else:
        raise ValueError("GeoJSON masks must be rasterized with image geometry")
    return _class_validate(result, num_classes, ignore_index)


def _sample_mask(mask, spec, image_shape, xy, footprint, num_classes, ignore_index):
    is_json = (
        isinstance(mask, dict)
        or isinstance(mask, list)
        or (isinstance(mask, (str, Path)) and Path(mask).suffix.lower() in {".json", ".geojson"})
    )
    if spec.kind == "geojson" or is_json:
        return rasterize_geojson(
            mask,
            image_shape,
            spec,
            xy_level0=xy,
            footprint_level0=footprint,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
    if spec.origin is None and spec.downsample is None:
        array = decode_mask(mask, spec, num_classes=num_classes, ignore_index=ignore_index)
        if array.shape != image_shape:
            raise ValueError(
                "Mask/image dimensions differ; specify mask origin and downsample for slide-level masks"
            )
        return array
    if spec.origin is None or spec.downsample is None or xy is None or footprint is None:
        raise ValueError(
            "Slide mask sampling requires origin, downsample, xy_level0 and footprint_level0"
        )
    raw = _array(mask, decode_colors=spec.kind == "rgb")
    decode_spec = spec
    if spec.kind == "channels":
        if raw.ndim != 3 or spec.channel_axis is None:
            raise ValueError("Channel masks require explicit channel_axis")
        raw = np.moveaxis(raw, spec.channel_axis, 0)
        mh, mw = raw.shape[1:]
        decode_spec = replace(spec, channel_axis=0)
    elif spec.kind == "rgb":
        if raw.ndim != 3:
            raise ValueError("RGB masks require HWC colors")
        mh, mw = raw.shape[:2]
    else:
        if raw.ndim != 2:
            raise ValueError("Multichannel masks require explicit rgb/channels kind and mapping")
        mh, mw = raw.shape
    h, w = image_shape
    xs = xy[0] + (np.arange(w) + 0.5) * footprint[0] / w
    ys = xy[1] + (np.arange(h) + 0.5) * footprint[1] / h
    ix = np.floor((xs - spec.origin[0]) / spec.downsample[0]).astype(np.int64)
    iy = np.floor((ys - spec.origin[1]) / spec.downsample[1]).astype(np.int64)
    vx, vy = (ix >= 0) & (ix < mw), (iy >= 0) & (iy < mh)
    result = np.full((h, w), ignore_index, dtype=np.int64)
    if not vx.any() or not vy.any():
        return result
    if spec.kind == "channels":
        tile = raw[:, iy[vy][:, None], ix[vx][None, :]]
    elif spec.kind == "rgb":
        tile = raw[iy[vy][:, None], ix[vx][None, :], :]
    else:
        tile = raw[np.ix_(iy[vy], ix[vx])]
    result[np.ix_(vy, vx)] = decode_mask(
        tile, decode_spec, num_classes=num_classes, ignore_index=ignore_index
    )
    return result


def read_sample(sample, *, num_classes=2, ignore_index=-100):
    if not isinstance(sample, Sample):
        sample = prepare_dataset(sample, num_classes=num_classes, ignore_index=ignore_index)[0]
    xy, footprint, bounds = sample.xy_level0, sample.footprint_level0, sample.valid_bounds
    if isinstance(sample.image, dict) and "archive_path" in sample.image:
        with _get_patch_reader()(sample.image["archive_path"]) as reader:
            record = reader[int(sample.image["index"])]
        image = record.image
        xy, footprint, bounds = record.xy_level0, record.footprint_level0, record.valid_bounds
    else:
        image = _array(sample.image, decode_colors=True)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3 or image.shape[-1] not in (3, 4) or image.dtype != np.uint8:
            raise ValueError("Images must be uint8 HWC RGB/RGBA or grayscale")
        if image.shape[-1] == 4:
            alpha = image[..., 3:4].astype(np.float32) / 255
            image = np.rint(image[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    if not image.shape[0] or not image.shape[1]:
        raise ValueError("Image dimensions must be nonempty")
    if (xy is None) != (footprint is None):
        raise ValueError("xy_level0 and footprint_level0 must be specified together")
    if footprint is not None and (
        len(footprint) != 2 or not np.isfinite(footprint).all() or min(footprint) <= 0
    ):
        raise ValueError("footprint_level0 must contain positive finite width and height")
    if xy is not None and (len(xy) != 2 or not np.isfinite(xy).all()):
        raise ValueError("xy_level0 must contain finite x and y")
    mask = (
        None
        if sample.mask is None
        else _sample_mask(
            sample.mask,
            _spec(sample.mask_spec) or MaskSpec(),
            image.shape[:2],
            xy,
            footprint,
            num_classes,
            ignore_index,
        )
    )
    if bounds is not None:
        if len(bounds) != 4 or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in bounds
        ):
            raise ValueError("valid_bounds must contain four integer output-pixel bounds")
        bounds = tuple(int(v) for v in bounds)
        x0, y0, x1, y1 = bounds
        h, w = image.shape[:2]
        if not (0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h):
            raise ValueError("valid_bounds must lie inside the native image")
        if mask is not None:
            valid = np.zeros(mask.shape, dtype=bool)
            valid[y0:y1, x0:x1] = True
            mask[~valid] = ignore_index
    embedding = sample.embedding
    if isinstance(embedding, (str, Path)):
        embedding = _array(embedding)
    return LoadedSample(
        np.ascontiguousarray(image),
        mask,
        sample.sample_id,
        sample.slide_id,
        xy,
        footprint,
        bounds,
        sample.prompts,
        embedding,
    )
