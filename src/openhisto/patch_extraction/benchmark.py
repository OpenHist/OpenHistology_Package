"""Compare PNG files with a patch archive using deterministic synthetic images.

Run with ``python -m openhisto.patch_extraction.benchmark``.
Images are generated one at a time; all output files are temporary. Timings
include encoding and API writes/close, exclude image generation, and do not
force durable writes or clear the operating system's page cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import h5py
import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from .archive import PatchArchiveWriter, PatchReader


PNG_COMPRESSION = 6


def synthetic_patch(index: int, size: int, seed: int) -> Image.Image:
    """Create a reproducible RGB texture with pale tissue and purple dots."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    y, x = np.ogrid[:size, :size]
    phase = rng.uniform(0, 2 * np.pi, size=3)
    scale = max(size, 1)
    texture = (
        8 * np.sin(x * 28 / scale + phase[0])
        + 6 * np.cos(y * 23 / scale + phase[1])
        + 4 * np.sin((x + y) * 49 / scale + phase[2])
    )
    pixels = np.empty((size, size, 3), dtype=np.uint8)
    tissue = np.array([229, 184, 207], dtype=np.float32)
    for channel in range(3):
        values = tissue[channel] + texture + rng.normal(0, 3, (size, size))
        pixels[:, :, channel] = np.clip(values, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    radius_max = max(2, round(size / 100))
    for _ in range(max(1, size * size // 700)):
        cx, cy = rng.integers(0, size, size=2)
        rx, ry = rng.integers(1, radius_max + 1, size=2)
        color = tuple(int(v) for v in rng.integers([91, 52, 113], [150, 104, 165]))
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
    return image


def _digest(image) -> str:
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise AssertionError("Decoded patch must be uint8 HWC RGB")
    return hashlib.sha256(memoryview(np.ascontiguousarray(pixels))).hexdigest()


def _storage_stats(paths: list[Path]) -> dict:
    stats = [path.stat() for path in paths]
    return {
        "file_count": len(paths),
        "logical_bytes": sum(stat.st_size for stat in stats),
        "allocated_bytes": (
            sum(stat.st_blocks * 512 for stat in stats)
            if all(hasattr(stat, "st_blocks") for stat in stats)
            else None
        ),
    }


def _read_pass(read, indices: list[int], expected: list[str], size: int) -> dict:
    elapsed = 0.0
    for index in indices:
        started = perf_counter()
        image = read(index)
        elapsed += perf_counter() - started
        if np.asarray(image).shape != (size, size, 3):
            raise AssertionError(f"Unexpected decoded shape at patch {index}")
        if _digest(image) != expected[index]:
            raise AssertionError(f"Pixel verification failed at patch {index}")
    decoded_bytes = len(indices) * size * size * 3
    return {
        "read_wall_seconds": elapsed,
        "patches_per_second": len(indices) / elapsed,
        "decoded_mib_per_second": decoded_bytes / (1024**2 * elapsed),
        "verified_patches": len(indices),
    }


def run_benchmark(count: int = 64, patch_size: int = 512, seed: int = 0) -> dict:
    """Benchmark one generated corpus, verify all reads, then remove its files."""
    for name, value in (("count", count), ("patch_size", patch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    ordered = list(range(count))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC0A7]))
    shuffled = [int(index) for index in rng.permutation(count)]
    expected = []

    with TemporaryDirectory(prefix="cpath-storage-benchmark-") as temporary:
        root = Path(temporary)
        png_dir = root / "png"
        png_dir.mkdir()
        png_paths = [png_dir / f"patch_{index:08d}.png" for index in ordered]
        png_write_seconds = 0.0
        for index, path in enumerate(png_paths):
            image = synthetic_patch(index, patch_size, seed)
            expected.append(_digest(image))
            started = perf_counter()
            image.save(path, format="PNG", compress_level=PNG_COMPRESSION)
            png_write_seconds += perf_counter() - started

        archive_path = root / "patches.h5"
        started = perf_counter()
        with PatchArchiveWriter(
            archive_path,
            patch_size=patch_size,
            metadata={"source": "synthetic-benchmark", "seed": seed},
            codec="png",
            png_compression=PNG_COMPRESSION,
        ) as writer:
            archive_write_seconds = perf_counter() - started
            for index in ordered:
                image = synthetic_patch(index, patch_size, seed)
                if _digest(image) != expected[index]:
                    raise AssertionError("Synthetic generator is not deterministic")
                started = perf_counter()
                writer.append(
                    image,
                    xy_level0=(index * patch_size, 0),
                    footprint_level0=(patch_size, patch_size),
                    valid_bounds=(0, 0, patch_size, patch_size),
                )
                archive_write_seconds += perf_counter() - started
            started = perf_counter()
        archive_write_seconds += perf_counter() - started

        png_result = _storage_stats(png_paths)
        archive_result = _storage_stats([archive_path])
        for result, elapsed in (
            (png_result, png_write_seconds),
            (archive_result, archive_write_seconds),
        ):
            result["encode_write_wall_seconds"] = elapsed
            result["write_patches_per_second"] = count / elapsed

        def read_png(index):
            with Image.open(png_paths[index]) as image:
                return np.array(image.convert("RGB"))

        png_result["ordered_read"] = _read_pass(read_png, ordered, expected, patch_size)
        png_result["shuffled_read"] = _read_pass(read_png, shuffled, expected, patch_size)
        with PatchReader(archive_path) as reader:
            if len(reader) != count:
                raise AssertionError("Archive patch count does not match input")

            def read_archive(index):
                return reader.read(index).image

            archive_result["ordered_read"] = _read_pass(read_archive, ordered, expected, patch_size)
            archive_result["shuffled_read"] = _read_pass(
                read_archive, shuffled, expected, patch_size
            )

    return {
        "benchmark_version": 1,
        "settings": {
            "count": count,
            "patch_size": patch_size,
            "seed": seed,
            "codec": "png",
            "png_compression": PNG_COMPRESSION,
            "synthetic_generator": "numpy-sinusoidal-texture-pillow-ellipses-v1",
            "pixel_verification": "sha256 of every decoded RGB array",
            "shuffled_read": "one seeded permutation of all patches",
            "output": "temporary files removed after measurement",
        },
        "measurement": {
            "write_timing": "encode, API write and close; generation/hash excluded; no fsync",
            "read_timing": "per-patch open/decode for PNG; indexed decode for open HDF5 reader; hash excluded",
            "cache": "OS page cache retained; ordered pass precedes shuffled pass",
            "allocated_bytes": "regular-file st_blocks * 512; directory/inode overhead excluded",
            "interpretation": "host-dependent measurements; no guaranteed size or speed ratio",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": pillow_version,
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "png_files": png_result,
        "hdf5_archive": archive_result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(args.count, args.patch_size, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
