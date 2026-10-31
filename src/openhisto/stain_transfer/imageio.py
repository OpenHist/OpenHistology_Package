"""Strict 8-bit RGB loading and atomic, lossless PNG publication."""

import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


def validate_rgb(image):
    """Return an array after checking the public uint8 HWC RGB contract."""
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Images must be uint8 HWC RGB arrays with three channels")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError("Images must have positive width and height")
    return array


def load_rgb(path):
    """Load RGB/grayscale 8-bit data; composite alpha onto a white background.

    High-bit-depth and floating-point inputs need an explicit caller conversion;
    silently converting them through Pillow would discard their dynamic range.
    """
    with Image.open(path) as image:
        if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
            raise ValueError(
                f"Image {path} is not 8-bit RGB/grayscale; convert high-bit-depth "
                "or other color-space data explicitly"
            )
        if image.mode in {"LA", "P", "RGBA"}:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            array = np.array(white.convert("RGB"), dtype=np.uint8)
        else:
            array = np.array(image.convert("RGB"), dtype=np.uint8)
    return validate_rgb(array)


def _publish(temporary, destination, overwrite):
    if overwrite:
        os.replace(temporary, destination)
    else:
        # Unlike a check followed by rename, linking cannot replace a file
        # another writer published after our initial existence check.
        os.link(temporary, destination)
        temporary.unlink()


def save_rgb(image, path, *, overwrite=False, png_compression=6):
    """Atomically save an RGB array as PNG, refusing existing files by default."""
    array = validate_rgb(image)
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Normalized image output must have a .png extension")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be boolean")
    if (
        isinstance(png_compression, bool)
        or not isinstance(png_compression, int)
        or not 0 <= png_compression <= 9
    ):
        raise ValueError("png_compression must be an integer from 0 to 9")
    if path.is_dir():
        raise IsADirectoryError(path)
    if not overwrite and os.path.lexists(path):
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        Image.fromarray(array).save(temporary, format="PNG", compress_level=png_compression)
        _publish(temporary, path, overwrite)
    finally:
        temporary.unlink(missing_ok=True)
    return path
