"""Streaming whole-slide extraction with explicit level-0 coordinates."""

from pathlib import Path
import math
import numbers

import numpy as np
from PIL import Image

from .archive import PatchArchiveWriter
from .config import ExtractionConfig, ExtractionSummary


def _open_slide(path):
    # Importing the package must not require loading the OpenSlide native library.
    try:
        from openslide import OpenSlide
    except (ImportError, OSError) as exc:
        raise ImportError("Extraction requires openslide-python and the OpenSlide library") from exc
    return OpenSlide(str(path))


def _ensure_distinct_output(output_path, *input_paths):
    output = Path(output_path)
    for input_path in input_paths:
        source = Path(input_path)
        if output.resolve() == source.resolve():
            raise ValueError("output_path must differ from all input files")
        try:
            same_file = output.samefile(source)
        except OSError:
            same_file = False
        if same_file:
            raise ValueError("output_path must differ from all input files")


def _resolution(slide, config):
    source_mag = config.source_mag
    if source_mag is None:
        value = slide.properties.get("openslide.objective-power")
        try:
            source_mag = float(value)
        except (TypeError, ValueError):
            raise ValueError("Source magnification is missing; set source_mag explicitly") from None
    if not math.isfinite(source_mag) or source_mag <= 0:
        raise ValueError("Source magnification must be positive and finite")
    if config.target_mag > source_mag:
        raise ValueError("target_mag cannot be finer than source_mag")
    ratio = source_mag / config.target_mag
    if not math.isfinite(ratio):
        raise ValueError("Requested magnification ratio is too large")
    levels = []
    for index, value in enumerate(slide.level_downsamples):
        downsample = float(value)
        if math.isfinite(downsample) and 0 < downsample <= ratio:
            levels.append((downsample, index))
    if not levels:
        raise ValueError("No slide pyramid level is fine enough for the target magnification")
    downsample, level = max(levels)
    return float(source_mag), ratio, level, downsample


def _grid(width, height, stride_level0):
    row = 0
    while True:
        y = round(row * stride_level0)
        if y >= height:
            break
        column = 0
        while True:
            x = round(column * stride_level0)
            if x >= width:
                break
            yield x, y, True
            column += 1
        row += 1


def _valid_bounds(x, y, footprint, width, height, patch_size):
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + footprint), min(height, y + footprint)
    if right <= left or bottom <= top:
        return None
    # Only fully valid output pixels enter tissue statistics. Integer division
    # avoids floating-point off-by-one errors at exact pixel boundaries.
    x0 = ((left - x) * patch_size + footprint - 1) // footprint
    y0 = ((top - y) * patch_size + footprint - 1) // footprint
    x1 = ((right - x) * patch_size) // footprint
    y1 = ((bottom - y) * patch_size) // footprint
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def _read_patch(slide, x, y, footprint, level, downsample, patch_size):
    extent = footprint / downsample
    read_size = math.ceil(extent)
    region = slide.read_region((int(x), int(y)), level, (read_size, read_size))
    # OpenSlide pads outside the slide with transparent pixels. Composite onto
    # white before resampling so transparency cannot become black tissue.
    rgba = region.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    rgb = background.convert("RGB")
    return rgb.resize(
        (patch_size, patch_size),
        resample=Image.Resampling.LANCZOS,
        box=(0.0, 0.0, extent, extent),
    )


def _is_foreground(image, valid_bounds, config):
    method = config.foreground_method.lower()
    if method == "none":
        return True
    x0, y0, x1, y1 = valid_bounds
    pixels = np.asarray(image, dtype=np.float32)[y0:y1, x0:x1]
    if pixels.size == 0:
        return False
    maximum = pixels.max(axis=-1)
    minimum = pixels.min(axis=-1)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    if method == "hsv":
        return bool(np.mean(saturation > config.sat_thresh) > config.tissue_fraction)
    luminance = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if method == "sl":
        return bool(
            luminance.mean() < config.lum_thresh
            and np.mean(saturation > config.sat_thresh) > config.sl_tissue_fraction
        )
    from skimage.filters import threshold_otsu

    return bool(luminance.mean() < threshold_otsu(luminance))


def _pair(value, name, *, positive):
    if isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)):
        values = (value, value)
    else:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be a scalar or a pair of numbers")
        try:
            values = tuple(value)
        except TypeError:
            raise ValueError(f"{name} must be a scalar or a pair of numbers") from None
        if len(values) != 2:
            raise ValueError(f"{name} must contain two numbers")
    result = []
    for item in values:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real):
            raise ValueError(f"{name} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number) or (positive and number <= 0):
            raise ValueError(f"{name} must contain {'positive ' if positive else ''}finite numbers")
        result.append(number)
    return tuple(result)


def _load_binary_mask(mask):
    if isinstance(mask, np.ndarray):
        array = mask
    elif isinstance(mask, (str, Path)) or hasattr(mask, "__fspath__"):
        path = Path(mask)
        if path.suffix.lower() in {".tif", ".tiff"}:
            import tifffile

            array = tifffile.imread(path)
        else:
            with Image.open(path) as image:
                array = np.array(image)
    else:
        raise TypeError("mask must be a 2D NumPy array or an image path")
    if array.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if not all(array.shape):
        raise ValueError("mask dimensions must be nonzero")
    if array.dtype.kind not in "buif":
        raise ValueError("mask must contain binary numeric values")
    seen_one = seen_255 = False
    # Validate in bounded blocks instead of creating several whole-mask boolean
    # temporaries. Values must use one binary convention, not mixed 1 and 255.
    rows = max(1, 1_048_576 // array.shape[1])
    for start in range(0, array.shape[0], rows):
        values = np.unique(array[start : start + rows])
        if not np.isin(values, (0, 1, 255)).all():
            raise ValueError("mask values must be 0/1 or 0/255")
        seen_one = seen_one or bool(np.any(values == 1))
        seen_255 = seen_255 or bool(np.any(values == 255))
        if seen_one and seen_255:
            raise ValueError("mask cannot mix 0/1 and 0/255 encodings")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _mask_boxes(mask):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Mask-centered extraction requires cv2; install openhisto[patch-extraction-opencv] "
            "or provide an existing compatible OpenCV distribution"
        ) from exc

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(
        (cv2.boundingRect(contour) for contour in contours),
        key=lambda box: (box[1], box[0], box[2], box[3]),
    )


def _mask_candidates(boxes, footprint, downsample, origin, min_box_size):
    for x, y, width, height in boxes:
        center_x = origin[0] + (x + width / 2.0) * downsample[0]
        center_y = origin[1] + (y + height / 2.0) * downsample[1]
        yield (
            round(center_x - footprint / 2.0),
            round(center_y - footprint / 2.0),
            width >= min_box_size and height >= min_box_size,
        )


def _extract(slide_path, output_path, config, *, mode, boxes=None, mask_transform=None):
    if config is None:
        config = ExtractionConfig()
    if not isinstance(config, ExtractionConfig):
        raise TypeError("config must be an ExtractionConfig")
    slide = _open_slide(slide_path)
    try:
        width, height = (int(value) for value in slide.dimensions)
        if width <= 0 or height <= 0:
            raise ValueError("Slide dimensions must be positive")
        source_mag, ratio, level, downsample = _resolution(slide, config)
        requested_footprint = config.patch_size * ratio
        requested_stride = config.effective_stride * ratio
        coordinate_limit = np.iinfo(np.int64).max
        if (
            not math.isfinite(requested_footprint)
            or not math.isfinite(requested_stride)
            or requested_footprint >= coordinate_limit
            or requested_stride >= coordinate_limit
        ):
            raise ValueError("Requested footprint or stride exceeds supported level-0 coordinates")
        footprint = round(requested_footprint)
        metadata = {
            "mode": mode,
            "slide_dimensions_level0": [width, height],
            "source_mag": source_mag,
            "target_mag": config.target_mag,
            "scale": ratio,
            "selected_level": level,
            "selected_level_downsample": downsample,
            "effective_scale": footprint / config.patch_size,
            "config": config.to_dict(),
        }
        if mask_transform is not None:
            metadata["mask_transform"] = mask_transform
        if mode == "grid":
            candidates = _grid(width, height, requested_stride)
        else:
            candidates = _mask_candidates(
                boxes,
                footprint,
                mask_transform["downsample"],
                mask_transform["origin_level0"],
                config.min_box_size,
            )
        counts = dict(
            candidate_count=0, patch_count=0, skipped_background=0, skipped_edges=0, skipped_size=0
        )
        with PatchArchiveWriter(
            output_path,
            patch_size=config.patch_size,
            metadata=metadata,
            codec=config.codec,
            png_compression=config.png_compression,
            jpeg_quality=config.jpeg_quality,
            overwrite=config.overwrite,
        ) as writer:
            for x, y, size_ok in candidates:
                counts["candidate_count"] += 1
                if not size_ok:
                    counts["skipped_size"] += 1
                    continue
                inside = x >= 0 and y >= 0 and x + footprint <= width and y + footprint <= height
                bounds = _valid_bounds(x, y, footprint, width, height, config.patch_size)
                if bounds is None or (config.edge == "drop" and not inside):
                    counts["skipped_edges"] += 1
                    continue
                patch = _read_patch(slide, x, y, footprint, level, downsample, config.patch_size)
                if (mode == "grid" or config.filter_mask_patches) and not _is_foreground(
                    patch, bounds, config
                ):
                    counts["skipped_background"] += 1
                    continue
                writer.append(patch, (x, y), (footprint, footprint), bounds)
                counts["patch_count"] += 1
            writer.set_counts(counts)
        return ExtractionSummary(archive_path=Path(output_path), **counts)
    finally:
        slide.close()


def extract_grid(slide_path, output_path, config=None) -> ExtractionSummary:
    """Extract a row-major grid into one compressed archive.

    Magnification comes from ``config.source_mag`` or the slide objective-power
    property. Grid stride is measured in output pixels; stored coordinates and
    footprints are always level-0 pixels. No slide name or path is archived.
    """
    _ensure_distinct_output(output_path, slide_path)
    return _extract(slide_path, output_path, config, mode="grid")


def extract_mask_centered(
    slide_path,
    mask,
    output_path,
    config=None,
    *,
    mask_downsample,
    mask_origin=(0, 0),
) -> ExtractionSummary:
    """Extract one square patch per external binary-mask bounding box.

    ``mask_downsample`` is level-0 pixels per mask pixel, supplied as a scalar
    or ``(x, y)`` pair. ``mask_origin`` is the level-0 origin of the mask. Boxes
    are ordered by mask row then column. ``min_box_size`` is an inclusive
    minimum for both box dimensions, measured in input mask pixels. This mode
    bypasses tissue filtering unless ``filter_mask_patches`` is enabled.
    """
    input_paths = [slide_path]
    if isinstance(mask, (str, Path)) or hasattr(mask, "__fspath__"):
        input_paths.append(mask)
    _ensure_distinct_output(output_path, *input_paths)
    downsample = _pair(mask_downsample, "mask_downsample", positive=True)
    origin = _pair(mask_origin, "mask_origin", positive=False)
    binary_mask = _load_binary_mask(mask)
    boxes = _mask_boxes(binary_mask)
    transform = {"downsample": list(downsample), "origin_level0": list(origin)}
    return _extract(
        slide_path, output_path, config, mode="mask_centered", boxes=boxes, mask_transform=transform
    )
