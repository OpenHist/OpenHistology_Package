"""Streaming, atomic image archives with process-local lazy readers.

Images are independently encoded as PNG or JPEG. Their concatenated bytes are
stored without HDF5 compression; offsets permit reading one image at a time.
Coordinates and footprints use level-zero slide pixels. ``valid_bounds`` uses
output-image pixels, with exclusive right and bottom bounds.
"""

from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
import json
from numbers import Integral
import operator
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
_BYTE_CHUNK = 64 * 1024
_ROW_CHUNK = 256


@dataclass(frozen=True)
class PatchRecord:
    image: np.ndarray
    index: int
    xy_level0: tuple[int, int]
    footprint_level0: tuple[int, int]
    valid_bounds: tuple[int, int, int, int]


def _integer(value, name, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _integer_tuple(value, length, name, dtype):
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain {length} integers") from exc
    if len(items) != length:
        raise ValueError(f"{name} must contain {length} integers")
    limits = np.iinfo(dtype)
    return tuple(_integer(item, name, minimum=limits.min, maximum=limits.max) for item in items)


def _json_dump(metadata):
    try:
        return json.dumps(metadata, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "metadata must be a JSON-serializable dictionary of finite values"
        ) from exc


def _text(value, name):
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Archive {name} must be UTF-8 text") from exc
    if not isinstance(value, str):
        raise ValueError(f"Archive {name} must be text")
    return value


def _reject_json_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _validate_archive(handle, *, require_complete=True):
    """Validate structure in bounded chunks, without decoding image payloads."""
    attrs = handle.attrs
    required_attrs = ("schema_version", "complete", "patch_size", "codec", "metadata_json")
    missing = [name for name in required_attrs if name not in attrs]
    if missing:
        raise ValueError(f"Archive is missing required attributes: {', '.join(missing)}")
    version = _integer(attrs["schema_version"], "Archive schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported archive schema_version {version}; expected {SCHEMA_VERSION}")
    complete = attrs["complete"]
    if not isinstance(complete, (bool, np.bool_, Integral)) or complete not in (0, 1):
        raise ValueError("Archive complete flag must be true or false")
    if require_complete and not complete:
        raise ValueError("Archive is incomplete and cannot be read")
    patch_size = _integer(attrs["patch_size"], "Archive patch_size", minimum=1)
    codec = _text(attrs["codec"], "codec")
    if codec not in {"png", "jpeg"}:
        raise ValueError(f"Unsupported archive codec: {codec!r}")
    try:
        metadata = json.loads(
            _text(attrs["metadata_json"], "metadata_json"),
            parse_constant=_reject_json_constant,
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("Archive metadata_json is not valid finite JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Archive metadata_json must contain an object")

    datasets = {}
    for name, width, dtype in (
        ("images/bytes", None, np.uint8),
        ("images/offsets", None, np.int64),
        ("coords", 2, np.int64),
        ("footprints", 2, np.int64),
        ("valid_bounds", 4, np.int32),
    ):
        if name not in handle or not isinstance(handle[name], h5py.Dataset):
            raise ValueError(f"Archive is missing dataset {name!r}")
        dataset = handle[name]
        if dataset.dtype != np.dtype(dtype):
            raise ValueError(f"Archive dataset {name!r} must have dtype {np.dtype(dtype)}")
        if width is None:
            valid_shape = dataset.ndim == 1
        else:
            valid_shape = dataset.ndim == 2 and dataset.shape[1] == width
        if not valid_shape:
            expected = "one dimension" if width is None else f"shape (N, {width})"
            raise ValueError(f"Archive dataset {name!r} must have {expected}")
        datasets[name] = dataset

    count = datasets["coords"].shape[0]
    for name in ("footprints", "valid_bounds"):
        if datasets[name].shape[0] != count:
            raise ValueError(f"Archive dataset {name!r} has inconsistent row count")
    offsets = datasets["images/offsets"]
    if offsets.shape[0] != count + 1:
        raise ValueError("Archive offsets must contain exactly N + 1 entries")
    if int(offsets[0]) != 0:
        raise ValueError("Archive offsets must start at zero")
    byte_count = datasets["images/bytes"].shape[0]
    if int(offsets[-1]) != byte_count:
        raise ValueError("Archive final offset must equal the image byte count")
    previous = 0
    for start in range(1, count + 1, _BYTE_CHUNK // 8):
        block = offsets[start : min(count + 1, start + _BYTE_CHUNK // 8)]
        if block[0] <= previous or np.any(block[1:] <= block[:-1]) or np.any(block > byte_count):
            raise ValueError("Archive offsets must increase strictly within the image byte count")
        previous = int(block[-1])

    for start in range(0, count, _ROW_CHUNK):
        stop = min(count, start + _ROW_CHUNK)
        if np.any(datasets["footprints"][start:stop] <= 0):
            raise ValueError("Archive footprints must contain positive level-zero dimensions")
        bounds = datasets["valid_bounds"][start:stop]
        if (
            np.any(bounds < 0)
            or np.any(bounds > patch_size)
            or np.any(bounds[:, 0] > bounds[:, 2])
            or np.any(bounds[:, 1] > bounds[:, 3])
        ):
            raise ValueError("Archive valid_bounds must lie within the output image")
    if "counts" in metadata:
        if not isinstance(metadata["counts"], dict):
            raise ValueError("Archive metadata counts must be an object")
        for name, value in metadata["counts"].items():
            _integer(value, f"Archive count {name!r}", minimum=0)
        if "patch_count" in metadata["counts"] and metadata["counts"]["patch_count"] != count:
            raise ValueError("Archive patch_count does not match the number of image rows")
    metadata.update(
        schema_version=version, complete=bool(complete), patch_size=patch_size, codec=codec
    )
    return count, metadata


class PatchArchiveWriter:
    """Write a single archive and publish it atomically on successful exit.

    ``metadata`` must already have been restricted to extraction information by
    the caller. Source paths and specimen identifiers are not added here.
    Existing output is refused unless ``overwrite=True``, including a file that
    appears after writing starts. A failed context removes its temporary file.
    """

    def __init__(
        self,
        output_path,
        *,
        patch_size: int,
        metadata: dict,
        codec="png",
        png_compression=6,
        jpeg_quality=95,
        overwrite=False,
    ):
        self.output_path = Path(output_path)
        self.patch_size = _integer(
            patch_size, "patch_size", minimum=1, maximum=np.iinfo(np.int32).max
        )
        if codec not in {"png", "jpeg"}:
            raise ValueError("codec must be 'png' or 'jpeg'")
        self.codec = codec
        self.png_compression = _integer(png_compression, "png_compression", minimum=0, maximum=9)
        self.jpeg_quality = _integer(jpeg_quality, "jpeg_quality", minimum=1, maximum=95)
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        self._metadata = json.loads(_json_dump(metadata))
        self._counts = {}
        if not isinstance(overwrite, bool):
            raise ValueError("overwrite must be a boolean")
        self.overwrite = overwrite
        self._file = None
        self._temporary_path = None
        self._state = "new"
        self._broken = False
        self._count = 0
        self._byte_count = 0

    def __enter__(self):
        if self._state != "new":
            raise RuntimeError("PatchArchiveWriter can only be entered once")
        if not self.overwrite and os.path.lexists(self.output_path):
            raise FileExistsError(f"Archive already exists: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{self.output_path.name}.tmp-", suffix=".h5", dir=self.output_path.parent
        )
        os.close(descriptor)
        self._temporary_path = Path(temporary_path)
        try:
            self._file = h5py.File(self._temporary_path, "w")
            self._file.attrs.update(
                schema_version=SCHEMA_VERSION,
                complete=False,
                patch_size=self.patch_size,
                codec=self.codec,
                metadata_json=_json_dump({**self._metadata, "counts": {}}),
            )
            images = self._file.create_group("images")
            images.create_dataset(
                "bytes", shape=(0,), maxshape=(None,), dtype=np.uint8, chunks=(_BYTE_CHUNK,)
            )
            offsets = images.create_dataset(
                "offsets", shape=(1,), maxshape=(None,), dtype=np.int64, chunks=(_ROW_CHUNK,)
            )
            offsets[0] = 0
            for name, width, dtype in (
                ("coords", 2, np.int64),
                ("footprints", 2, np.int64),
                ("valid_bounds", 4, np.int32),
            ):
                self._file.create_dataset(
                    name,
                    shape=(0, width),
                    maxshape=(None, width),
                    dtype=dtype,
                    chunks=(_ROW_CHUNK, width),
                )
            self._state = "open"
        except BaseException:
            self._abort()
            raise
        return self

    def _require_open(self):
        if self._state != "open" or self._file is None:
            raise RuntimeError("PatchArchiveWriter must be used inside its context")
        if self._broken:
            raise RuntimeError("Archive writing failed; this writer cannot be reused")

    def __len__(self):
        return self._count

    def append(self, image, xy_level0, footprint_level0, valid_bounds):
        self._require_open()
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.shape != (self.patch_size, self.patch_size, 3):
            raise ValueError("image must be uint8 RGB with shape (patch_size, patch_size, 3)")
        xy = _integer_tuple(xy_level0, 2, "xy_level0", np.int64)
        footprint = _integer_tuple(footprint_level0, 2, "footprint_level0", np.int64)
        if min(footprint) <= 0:
            raise ValueError("footprint_level0 dimensions must be positive")
        bounds = _integer_tuple(valid_bounds, 4, "valid_bounds", np.int32)
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 <= x1 <= self.patch_size and 0 <= y0 <= y1 <= self.patch_size):
            raise ValueError("valid_bounds must lie within the output image")
        stream = BytesIO()
        encoded_image = Image.fromarray(image)
        if self.codec == "png":
            encoded_image.save(stream, format="PNG", compress_level=self.png_compression)
        else:
            encoded_image.save(stream, format="JPEG", quality=self.jpeg_quality, subsampling=0)
        payload = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        index = self._count
        try:
            bytes_dataset = self._file["images/bytes"]
            bytes_dataset.resize((self._byte_count + len(payload),))
            bytes_dataset[self._byte_count :] = payload
            for name, value in (
                ("coords", xy),
                ("footprints", footprint),
                ("valid_bounds", bounds),
            ):
                dataset = self._file[name]
                dataset.resize((index + 1, dataset.shape[1]))
                dataset[index] = value
            offsets = self._file["images/offsets"]
            offsets.resize((index + 2,))
            offsets[index + 1] = self._byte_count + len(payload)
        except BaseException:
            self._broken = True
            raise
        self._count += 1
        self._byte_count += len(payload)
        return index

    def set_counts(self, counts: dict):
        self._require_open()
        if not isinstance(counts, dict) or any(not isinstance(name, str) for name in counts):
            raise ValueError("counts must be a dictionary with string keys")
        self._counts = {
            name: _integer(value, f"count {name!r}", minimum=0) for name, value in counts.items()
        }

    def close(self):
        """Validate, close, and atomically publish a successfully written archive."""
        if self._state == "closed":
            return
        self._require_open()
        try:
            counts = {"patch_count": self._count, **self._counts}
            self._file.attrs["metadata_json"] = _json_dump({**self._metadata, "counts": counts})
            _validate_archive(self._file, require_complete=False)
            self._file.attrs["complete"] = True
            self._file.flush()
            self._file.close()
            self._file = None
            if self.overwrite:
                os.replace(self._temporary_path, self.output_path)
            else:
                # A same-directory hard link publishes atomically without a
                # check-then-rename race that could replace another writer.
                os.link(self._temporary_path, self.output_path)
                self._temporary_path.unlink()
            self._temporary_path = None
            self._state = "closed"
        except BaseException:
            self._abort()
            raise

    def _abort(self):
        try:
            if self._file is not None:
                self._file.close()
        finally:
            self._file = None
            if self._temporary_path is not None:
                self._temporary_path.unlink(missing_ok=True)
                self._temporary_path = None
            self._state = "aborted"

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self._abort()
        else:
            try:
                self.close()
            except BaseException:
                self._abort()
                raise
        return False


class PatchReader:
    """Lazy random-access reader safe to pickle into worker processes.

    Each process opens and validates its own HDF5 handle. Validation scans small
    metadata arrays in bounded chunks; image payloads are read only on request.
    ``close`` releases the handle; a later operation may reopen it.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._file = None
        self._pid = None
        self._count = None
        self._metadata = None

    def _ensure_open(self):
        current_pid = os.getpid()
        if self._file is not None and self._pid == current_pid:
            return
        self.close()
        try:
            handle = h5py.File(self.path, "r")
        except FileNotFoundError:
            raise
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot open patch archive: {self.path}") from exc
        try:
            count, metadata = _validate_archive(handle)
        except BaseException:
            handle.close()
            raise
        self._file = handle
        self._pid = current_pid
        self._count = count
        self._metadata = metadata

    def __len__(self):
        self._ensure_open()
        return self._count

    @property
    def metadata(self):
        self._ensure_open()
        return deepcopy(self._metadata)

    def read(self, index):
        self._ensure_open()
        try:
            index = operator.index(index)
        except TypeError as exc:
            raise TypeError("Patch index must be an integer") from exc
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError("Patch index out of range")
        start, stop = (int(value) for value in self._file["images/offsets"][index : index + 2])
        payload = self._file["images/bytes"][start:stop].tobytes()
        try:
            with Image.open(BytesIO(payload)) as encoded_image:
                if encoded_image.size != (self._metadata["patch_size"],) * 2:
                    raise ValueError("Encoded image dimensions do not match patch_size")
                expected_format = "PNG" if self._metadata["codec"] == "png" else "JPEG"
                if encoded_image.format != expected_format:
                    raise ValueError("Encoded image format does not match archive codec")
                image = np.array(encoded_image.convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError, SyntaxError) as exc:
            raise ValueError(f"Cannot decode archived image at index {index}: {exc}") from exc
        return PatchRecord(
            image=image,
            index=index,
            xy_level0=tuple(int(value) for value in self._file["coords"][index]),
            footprint_level0=tuple(int(value) for value in self._file["footprints"][index]),
            valid_bounds=tuple(int(value) for value in self._file["valid_bounds"][index]),
        )

    __getitem__ = read

    def __iter__(self):
        for index in range(len(self)):
            yield self.read(index)

    def get_coordinates(self, start=0, stop=None):
        self._ensure_open()
        start, stop, _ = slice(start, stop).indices(self._count)
        if start >= stop:
            return np.empty((0, 2), dtype=np.int64)
        return self._file["coords"][start:stop]

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file = None
        self._pid = None
        self._count = None
        self._metadata = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_file=None, _pid=None, _count=None, _metadata=None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
