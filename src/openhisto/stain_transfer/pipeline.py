"""Normalization adapters for images, directory trees, manifests and archives."""

import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .config import DomainSpec, TransferConfig, TransferReport
from .imageio import _publish, load_rgb, save_rgb, validate_rgb
from .normalizer import create_normalizer, is_blank


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_PROVENANCE = "stain_transfer.json"


@contextmanager
def _managed_normalizer(domain, config):
    normalizer = create_normalizer(domain, config=config)
    try:
        yield normalizer
    finally:
        close = getattr(normalizer, "close", None)
        if close is not None:
            close()


def _check_distinct(source, destination, *, directory=False):
    if source == destination:
        raise ValueError("Input and output must be different paths")
    if directory and (source in destination.parents or destination in source.parents):
        raise ValueError("Input and output directories must not contain one another")


def _preflight(destinations, *, overwrite, input_paths=()):
    inputs = {Path(path).resolve() for path in input_paths}
    seen = set()
    for destination in destinations:
        canonical = destination.resolve()
        if canonical in seen:
            raise ValueError(f"Multiple input images map to the same output: {destination}")
        seen.add(canonical)
        if canonical in inputs:
            raise ValueError(f"Output would replace an input: {destination}")
        if destination.is_dir():
            raise IsADirectoryError(destination)
        if not overwrite and os.path.lexists(destination):
            raise FileExistsError(f"Output already exists: {destination}")
        for parent in destination.parents:
            if parent.exists() and not parent.is_dir():
                raise NotADirectoryError(parent)


def _atomic_text(path, text, *, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        _publish(temporary, path, overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def _write_provenance(path, normalizer, report, config, *, input_info=None):
    payload = {"stain_transfer": normalizer.provenance, "report": report.to_dict()}
    if input_info is not None:
        payload["input"] = input_info
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        overwrite=config.overwrite,
    )


def _transform(normalizer, image, *, bounds=None, patch_id=None):
    blank = bool(is_blank(image, bounds))
    result = validate_rgb(normalizer.transform(image, valid_bounds=bounds, patch_id=patch_id))
    if result.shape != image.shape:
        raise ValueError("Normalization must preserve image dimensions")
    return result, int(blank)


def _report(output, count, blanks, domain):
    method = "mixture" if domain.candidates or len(domain.references) > 1 else domain.method
    return TransferReport(output=output, processed=count, blank=blanks, failed=0, method=method)


def _image(source, output, domain, config):
    _check_distinct(source, output)
    if output.suffix.lower() != ".png":
        raise ValueError("Normalized image output must have a .png extension")
    sidecar = output.with_name(output.name + ".json")
    _preflight([output, sidecar], overwrite=config.overwrite, input_paths=[source])
    with _managed_normalizer(domain, config) as normalizer:
        image = load_rgb(source)
        digest = hashlib.sha256()
        # Match the classical-reference content hash: dimensions followed by
        # contiguous, decoded RGB bytes. This is independent of PNG compression.
        digest.update(np.asarray(image.shape, dtype="<i8").tobytes())
        digest.update(image.tobytes(order="C"))
        rgb, blank = _transform(normalizer, image, patch_id=source.name)
        save_rgb(rgb, output, overwrite=config.overwrite, png_compression=config.png_compression)
        report = _report(output, 1, blank, domain)
        _write_provenance(
            sidecar,
            normalizer,
            report,
            config,
            input_info={
                "rgb_sha256": digest.hexdigest(),
                "hash_encoding": "shape_le_i8_then_rgb_uint8",
            },
        )
        return report


def _directory(source, output, domain, config):
    _check_distinct(source, output, directory=True)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    files = sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    pairs = [(path, output / path.relative_to(source).with_suffix(".png")) for path in files]
    _preflight(
        [destination for _, destination in pairs] + [output / _PROVENANCE],
        overwrite=config.overwrite,
        input_paths=files,
    )
    with _managed_normalizer(domain, config) as normalizer:
        blanks = 0
        for path, destination in pairs:
            rgb, blank = _transform(
                normalizer, load_rgb(path), patch_id=path.relative_to(source).as_posix()
            )
            save_rgb(
                rgb, destination, overwrite=config.overwrite, png_compression=config.png_compression
            )
            blanks += blank
        report = _report(output, len(pairs), blanks, domain)
        _write_provenance(output / _PROVENANCE, normalizer, report, config)
        return report


def _reject_constant(value):
    raise ValueError(f"Manifest contains a nonfinite JSON value: {value}")


def _read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields or any(not name for name in fields) or len(fields) != len(set(fields)):
                raise ValueError("Manifest CSV must have unique, nonempty column names")
            rows = list(reader)
            if any(None in row for row in rows):
                raise ValueError("Manifest CSV row is wider than its header")
            return rows, fields
        rows = []
        for line in stream:
            if line.strip():
                row = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(row, dict):
                    raise ValueError("Manifest JSONL must contain row objects")
                rows.append(row)
        return rows, None


def _bounds(row, shape):
    valid = row.get("valid_bounds")
    if valid not in (None, ""):
        if isinstance(valid, str):
            valid = json.loads(valid, parse_constant=_reject_constant)
    elif any(
        row.get(key) not in (None, "") for key in ("valid_x0", "valid_y0", "valid_x1", "valid_y1")
    ):
        valid = [
            row.get(key, default)
            for key, default in zip(
                ("valid_x0", "valid_y0", "valid_x1", "valid_y1"), (0, 0, shape[1], shape[0])
            )
        ]
    else:
        return None
    if not isinstance(valid, (tuple, list)) or len(valid) != 4:
        raise ValueError("Manifest valid_bounds must contain four integer pixel bounds")
    integers = []
    for value in valid:
        if isinstance(value, bool):
            raise ValueError("Manifest valid_bounds must contain integers")
        try:
            number = float(value)
            integer = int(number)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("Manifest valid_bounds must contain integers") from None
        if integer != number:
            raise ValueError("Manifest valid_bounds must contain integers")
        integers.append(integer)
    x0, y0, x1, y1 = integers
    if not 0 <= x0 <= x1 <= shape[1] or not 0 <= y0 <= y1 <= shape[0]:
        raise ValueError("Manifest valid_bounds must lie within its image")
    return tuple(integers)


def _manifest(source, output, domain, config):
    _check_distinct(source, output)
    if output in source.parents:
        raise ValueError("Manifest output directory must not contain the input manifest")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    rows, fields = _read_manifest(source)
    plans, seen = [], set()
    for row in rows:
        value = row.get("image_path") or row.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError("Manifest rows require image_path (or path)")
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else source.parent / path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Manifest image is not a file: {path}")
        identifier = row.get("patch_id")
        if identifier not in (None, ""):
            identifier = "id:" + str(identifier)
        else:
            identifier = "path:" + value
        if identifier in seen:
            raise ValueError(
                "Manifest patch identities must be unique; supply unique patch_id values"
            )
        seen.add(identifier)
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
        relative = Path("images") / f"{digest}.png"
        updated = dict(row)
        updated["image_path"] = relative.as_posix()
        if "path" in updated:
            updated["path"] = relative.as_posix()
        plans.append((path, output / relative, identifier, updated))
    manifest_output = output / source.name
    _preflight(
        [destination for _, destination, _, _ in plans] + [manifest_output, output / _PROVENANCE],
        overwrite=config.overwrite,
        input_paths=[source] + [path for path, _, _, _ in plans],
    )
    with _managed_normalizer(domain, config) as normalizer:
        blanks = 0
        for (path, destination, identifier, _), row in zip(plans, rows):
            image = load_rgb(path)
            rgb, blank = _transform(
                normalizer, image, bounds=_bounds(row, image.shape), patch_id=identifier
            )
            save_rgb(
                rgb, destination, overwrite=config.overwrite, png_compression=config.png_compression
            )
            blanks += blank
        updated_rows = [row for _, _, _, row in plans]
        if fields is not None:
            from io import StringIO

            stream = StringIO(newline="")
            writer = csv.DictWriter(
                stream, fieldnames=fields if "image_path" in fields else fields + ["image_path"]
            )
            writer.writeheader()
            writer.writerows(updated_rows)
            manifest_text = stream.getvalue()
        else:
            manifest_text = "".join(
                json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n" for row in updated_rows
            )
        _atomic_text(manifest_output, manifest_text, overwrite=config.overwrite)
        report = _report(output, len(plans), blanks, domain)
        _write_provenance(output / _PROVENANCE, normalizer, report, config)
        return report


def _archive(source, output, domain, config):
    # Storage code is independent of OpenSlide; no native slide library is
    # initialized by this adapter.
    from openhisto.patch_extraction import PatchArchiveWriter, PatchReader

    _check_distinct(source, output)
    if output.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("Normalized archive output must have a .h5 or .hdf5 extension")
    _preflight([output], overwrite=config.overwrite, input_paths=[source])
    with _managed_normalizer(domain, config) as normalizer:
        with PatchReader(source) as reader:
            metadata = reader.metadata
            patch_size = metadata["patch_size"]
            previous = metadata.get("stain_transfer")
            metadata["stain_transfer"] = dict(normalizer.provenance)
            if previous is not None:
                metadata["stain_transfer"]["previous"] = previous
            # The writer sets these attributes itself; avoid contradictory values
            # when the source codec was JPEG and output is lossless PNG.
            for key in ("schema_version", "complete", "codec", "patch_size"):
                metadata.pop(key, None)
            original_counts = metadata.pop("counts", {})
            blanks = 0
            with PatchArchiveWriter(
                output,
                patch_size=patch_size,
                metadata=metadata,
                codec="png",
                png_compression=config.png_compression,
                overwrite=config.overwrite,
            ) as writer:
                for record in reader:
                    rgb, blank = _transform(
                        normalizer,
                        record.image,
                        bounds=record.valid_bounds,
                        patch_id=str(record.index),
                    )
                    writer.append(
                        rgb, record.xy_level0, record.footprint_level0, record.valid_bounds
                    )
                    blanks += blank
                writer.set_counts(original_counts)
            return _report(output, len(reader), blanks, domain)


def normalize(inputs, output, domain, config=None):
    """Normalize an image, recursive tree, patch manifest or native archive.

    Image outputs are PNG files. Trees/manifests write to a directory; archive
    outputs retain the native patch archive schema. Files publish atomically,
    while a failed multi-file job can leave already completed image files.
    """
    if isinstance(domain, dict):
        domain = DomainSpec(**domain)
    if not isinstance(domain, DomainSpec):
        raise TypeError("domain must be a DomainSpec or a domain dictionary")
    if config is None:
        config = TransferConfig()
    if not isinstance(config, TransferConfig):
        raise TypeError("config must be a TransferConfig")
    if config.workers != 0 or config.batch_size != 1:
        raise ValueError(
            "Normalization currently supports workers=0 and batch_size=1; images stream sequentially"
        )
    source = Path(inputs).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if source.is_dir():
        return _directory(source, output, domain, config)
    if not source.is_file():
        raise ValueError("Input must be an image, directory, manifest or patch archive")
    if source.suffix.lower() in {".csv", ".jsonl"}:
        return _manifest(source, output, domain, config)
    if source.suffix.lower() in {".h5", ".hdf5"}:
        return _archive(source, output, domain, config)
    return _image(source, output, domain, config)
