"""Atomic, resumable AnnData feature stores with selective feature reads.

AnnData's public writers own all encoding metadata. Numerical row updates use
HDF5 slices so adding an encoder never materializes older feature matrices.
"""

import json
import numbers
import os
from pathlib import Path
import re
import shutil
import uuid

import anndata as ad
from anndata.io import read_elem, write_elem
import h5py
import numpy as np
import pandas as pd

from .config import fingerprint, json_safe, mag_key, reject_secrets


_SCHEMA = 1
_STATE = "feature_encoder"
_CORE = (
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
)


def _json(value):
    value = json_safe(value)
    reject_secrets(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _key(value, name):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\0" in value
    ):
        raise ValueError(f"{name} must be a nonempty storage-safe string")
    return value


def _encoded_column(run_key):
    return "encoded_" + run_key


def _all_runs(uns):
    result = []
    for magnification, models in uns.get("feats", {}).items():
        for model, group in models.items():
            for key, run in group.get("runs", {}).items():
                result.append((magnification, model, key, run))
    return result


def _run_info(model, key, run, *, axes=False):
    value = {
        "run_key": key,
        "model": model,
        "mag": float(run["mag"]),
        "request_fingerprint": str(run["request_fingerprint"]),
        "dimension": int(run["dimension"]),
        "metadata": json.loads(run["metadata_json"]),
        "usage": json.loads(run.get("usage_json", "{}")),
    }
    if axes:
        value["feature_index"] = np.asarray(run["feature_index"]).astype(int).tolist()
        value["feature_names"] = np.asarray(run["feature_names"]).astype(str).tolist()
    return value


def _load(handle, *, published):
    if handle.attrs.get("encoding-type") != "anndata":
        raise ValueError("File is not a standard AnnData feature store")
    if "obs" not in handle or "var" not in handle or "uns" not in handle:
        raise ValueError("Feature store is missing AnnData metadata")
    obs, var, uns = read_elem(handle["obs"]), read_elem(handle["var"]), read_elem(handle["uns"])
    state = uns.get(_STATE, {})
    if state.get("schema_version") != _SCHEMA or not state.get("generation"):
        raise ValueError("Existing h5ad is not a compatible CPath feature store")
    if published and not state.get("complete", False):
        raise ValueError("Feature store is an unpublished partial")
    if len(var) or ("X" in handle and handle["X"].shape is not None):
        raise ValueError("Feature stores require X=None and an empty variable axis")
    if not obs.index.is_unique:
        raise ValueError("Feature store has duplicate observation IDs")
    found = set()
    for magnification, model, key, run in _all_runs(uns):
        if key in found or mag_key(run["mag"]) != magnification or run["model"] != model:
            raise ValueError("Feature run identities are inconsistent")
        found.add(key)
        if "obsm" not in handle or key not in handle["obsm"]:
            raise ValueError("Feature run is missing its embedding matrix")
        dataset = handle["obsm"][key]
        dimension = int(run["dimension"])
        if dataset.shape != (len(obs), dimension) or dataset.dtype != np.dtype("float32"):
            raise ValueError("Feature dimensions or dtype do not match metadata")
        if dataset.chunks is None or dataset.maxshape[0] is not None:
            raise ValueError("Feature matrices must be chunked with an unlimited row axis")
        column = _encoded_column(key)
        if column not in obs or obs[column].dtype != np.dtype(bool):
            raise ValueError("Feature run is missing its boolean encoded mask")
        if np.asarray(run["feature_index"]).shape != (dimension,) or np.asarray(
            run["feature_names"]
        ).shape != (dimension,):
            raise ValueError("Feature-axis metadata has incorrect dimensions")
        json.loads(run["metadata_json"])
    if set(handle.get("obsm", {})) != found:
        raise ValueError("Feature store contains unregistered observation matrices")
    return obs, uns


def _generation(path):
    if not path.exists():
        return ""
    with h5py.File(path, "r") as handle:
        _, uns = _load(handle, published=True)
        return str(uns[_STATE]["generation"])


def _write_frame(handle, frame):
    # Missing strings are encoded as standard categoricals, avoiding object
    # columns that mix NaN with UTF-8 strings in HDF5.
    frame = frame.copy()
    frame.index = pd.Index(frame.index.astype(str), name=None)
    for column in frame:
        if frame[column].dtype == object:
            nonmissing = frame[column].dropna()
            if all(isinstance(item, str) for item in nonmissing):
                if frame[column].isna().any():
                    frame[column] = pd.Categorical(frame[column])
            elif len(nonmissing):
                raise ValueError(
                    "Observation fields must be typed scalars; place nested metadata in metadata_json"
                )
    write_elem(handle, "obs", frame)


def _same(left, right):
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right) if not (pd.isna(left) or pd.isna(right)) else False


def _positions(frame, rows):
    n = len(frame)
    if rows is None:
        return np.arange(n, dtype=np.int64)
    if isinstance(rows, slice):
        return np.arange(n, dtype=np.int64)[rows]
    if isinstance(rows, str):
        rows = [rows]
    elif isinstance(rows, numbers.Integral) and not isinstance(rows, (bool, np.bool_)):
        rows = [rows]
    values = np.asarray(rows)
    if values.ndim != 1:
        raise ValueError("rows must be a one-dimensional selection")
    if not len(values):
        return np.empty(0, dtype=np.int64)
    if values.dtype.kind == "b":
        if len(values) != n:
            raise ValueError("Boolean row masks must match the observation count")
        return np.flatnonzero(values)
    if all(isinstance(value, str) for value in values):
        positions = frame.index.get_indexer(values)
        if np.any(positions < 0):
            raise ValueError("Requested observation IDs are absent")
        return positions.astype(np.int64)
    if values.dtype.kind not in "iu":
        raise ValueError("rows must contain integer positions or observation IDs")
    if values.dtype.kind == "u" and np.any(values >= n):
        raise IndexError("Observation position out of range")
    positions = values.astype(np.int64)
    positions = np.where(positions < 0, positions + n, positions)
    if np.any(positions < 0) or np.any(positions >= n):
        raise IndexError("Observation position out of range")
    return positions


class FeatureStore:
    """Update one published file through an exclusively owned resumable copy.

    Locks are never removed as stale automatically. A compatible partial keeps
    committed batches after exceptions. Successful no-op calls preserve the
    published file's bytes and modification time.
    """

    def __init__(self, output, compression_level=4):
        if (
            isinstance(compression_level, bool)
            or not isinstance(compression_level, int)
            or not 0 <= compression_level <= 9
        ):
            raise ValueError("compression_level must be an integer from 0 to 9")
        self.output = Path(output)
        self.partial_path = self.output.with_name(self.output.stem + ".partial.h5ad")
        self.lock_path = self.output.with_name(self.output.name + ".lock")
        if self.output == self.partial_path:
            raise ValueError("Output name conflicts with its partial file")
        self.compression_level = compression_level
        self._file = None
        self._obs = None
        self._uns = None
        self._lock_inode = None
        self._dirty = False
        self._entered = False
        self.metadata_changes = {"updated": 0, "preserved": 0}

    def __enter__(self):
        if self._entered:
            raise RuntimeError("FeatureStore can only be entered once")
        self._entered = True
        self.output.parent.mkdir(parents=True, exist_ok=True)
        for path in (self.output, self.partial_path):
            if path.is_symlink():
                raise ValueError("Feature output and partial files cannot be symlinks")
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise FileExistsError(
                "Feature output is locked; inspect the owner before explicitly removing a stale lock"
            ) from None
        try:
            with os.fdopen(fd, "w") as stream:
                stat = os.fstat(stream.fileno())
                self._lock_inode = (stat.st_dev, stat.st_ino)
                stream.write(json.dumps({"pid": os.getpid()}))
            self._baseline = _generation(self.output)
            if self.partial_path.exists():
                self._file = h5py.File(self.partial_path, "r+")
                self._obs, self._uns = _load(self._file, published=False)
                state = self._uns[_STATE]
                pristine_copy = (
                    state.get("complete", False) and state["generation"] == self._baseline
                )
                if not pristine_copy and state.get("baseline_generation", "") != self._baseline:
                    raise ValueError(
                        "Partial baseline differs from the published generation; preserve or remove it explicitly"
                    )
                if pristine_copy:
                    self._prepare_transaction()
                else:
                    # A completed stage may remain if atomic publication failed
                    # after final validation. It still needs to be published.
                    self._dirty = bool(state.get("transaction_dirty", False)) or bool(
                        state.get("complete", False) and state["generation"] != self._baseline
                    )
            elif self.output.exists():
                shutil.copyfile(self.output, self.partial_path)
                self._file = h5py.File(self.partial_path, "r+")
                self._obs, self._uns = _load(self._file, published=True)
                self._prepare_transaction()
            else:
                empty = pd.DataFrame(index=pd.Index([], dtype=str))
                state = dict(
                    schema_version=_SCHEMA,
                    generation=uuid.uuid4().hex,
                    baseline_generation="",
                    complete=False,
                    transaction_dirty=False,
                    slide_id="",
                    sources_json="[]",
                )
                ad.AnnData(
                    X=None, obs=empty, var=empty.copy(), uns={_STATE: state, "feats": {}}
                ).write_h5ad(self.partial_path)
                self._file = h5py.File(self.partial_path, "r+")
                self._obs, self._uns = _load(self._file, published=False)
            return self
        except BaseException:
            self._close_handle()
            self._release_lock()
            raise

    def _prepare_transaction(self):
        self._uns[_STATE].update(
            generation=uuid.uuid4().hex,
            baseline_generation=self._baseline,
            complete=False,
            transaction_dirty=False,
        )
        self._persist_uns()

    def _require_open(self):
        if self._file is None:
            raise RuntimeError("FeatureStore must be used as a context manager")

    def _persist_uns(self):
        write_elem(self._file, "uns", self._uns)
        self._file.flush()

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            self._uns[_STATE]["transaction_dirty"] = True
            self._persist_uns()

    @property
    def obs(self):
        self._require_open()
        return self._obs.copy()

    def reconcile(self, obs, *, slide_id, sources, metadata_policy="preserve"):
        """Reconcile rows, applying the selected policy to existing optional metadata.

        Content hashes and geometry always remain immutable. Counts of existing
        rows whose optional metadata was updated or preserved are available in
        ``metadata_changes`` after reconciliation.
        """
        self._require_open()
        if metadata_policy not in {"preserve", "update", "error"}:
            raise ValueError("metadata_policy must be preserve, update, or error")
        if not isinstance(obs, pd.DataFrame) or not obs.index.is_unique:
            raise ValueError("Observations must be a DataFrame with unique stable IDs")
        if any(not isinstance(value, str) or not value for value in obs.index):
            raise ValueError("Observation IDs must be nonempty strings")
        if not isinstance(slide_id, str) or not slide_id:
            raise ValueError("slide_id must be a nonempty string")
        if not isinstance(sources, list) or any(not isinstance(value, dict) for value in sources):
            raise ValueError("sources must be a list of dictionaries")
        _json(sources)
        incoming = obs.copy()
        if set(_CORE) - set(incoming):
            raise ValueError("Observations are missing canonical patch metadata columns")
        for column in incoming:
            _key(column, "Observation column")
            if column == "_index" or column.startswith("encoded_"):
                raise ValueError("Input observations cannot contain reserved columns")
        if len(incoming) and not (incoming["slide_id"] == slide_id).all():
            raise ValueError("All observations must belong to the specified slide")
        for value in incoming["mag"]:
            mag_key(value)
        current_slide = self._uns[_STATE]["slide_id"]
        if current_slide and current_slide != slide_id:
            raise ValueError("A feature store can contain only one logical slide")
        existing = incoming.index.intersection(self._obs.index)
        for observation in existing:
            if any(
                not _same(self._obs.at[observation, column], incoming.at[observation, column])
                for column in _CORE
            ):
                raise ValueError(
                    "Existing patch content or immutable geometry/source identity changed"
                )
        optional_columns = [
            column for column in incoming if column not in _CORE and column in self._obs
        ]
        metadata_rows = [
            observation
            for observation in existing
            if any(
                not _same(self._obs.at[observation, column], incoming.at[observation, column])
                for column in optional_columns
            )
        ]
        if metadata_rows and metadata_policy == "error":
            raise ValueError(
                "Optional patch metadata changed; choose metadata_policy='preserve' or 'update'"
            )
        self.metadata_changes = {
            "updated": len(metadata_rows) if metadata_policy == "update" else 0,
            "preserved": len(metadata_rows) if metadata_policy == "preserve" else 0,
        }
        added = incoming.loc[~incoming.index.isin(self._obs.index)].copy()
        changed = bool(len(added)) or not current_slide or bool(self.metadata_changes["updated"])
        unified = self._obs.copy()
        if metadata_policy == "update" and metadata_rows:
            for column in optional_columns:
                # Categorical columns read from h5ad may need new values.
                if isinstance(unified[column].dtype, pd.CategoricalDtype):
                    unified[column] = unified[column].astype(object)
                unified.loc[metadata_rows, column] = incoming.loc[metadata_rows, column]
        for column in incoming:
            if column not in unified:
                unified[column] = incoming[column].reindex(unified.index)
                changed = True
        if len(added):
            for column in unified:
                if column.startswith("encoded_"):
                    added[column] = False
            unified = (
                pd.concat([unified, added], axis=0)
                if len(unified)
                else added.reindex(columns=unified.columns)
            )
        unified.index = pd.Index(unified.index.astype(str), name=None)
        for _, _, key, _ in _all_runs(self._uns):
            column = _encoded_column(key)
            unified[column] = unified[column].fillna(False).astype(bool)
        stored_sources = json.loads(self._uns[_STATE]["sources_json"])
        signatures = {_json(value) for value in stored_sources}
        for source in sources:
            signature = _json(source)
            if signature not in signatures:
                stored_sources.append(json.loads(signature))
                signatures.add(signature)
                changed = True
        if changed:
            self._mark_dirty()
            self._obs = unified
            for _, _, key, run in _all_runs(self._uns):
                self._file["obsm"][key].resize((len(unified), int(run["dimension"])))
            _write_frame(self._file, unified)
            self._uns[_STATE].update(slide_id=slide_id, sources_json=_json(stored_sources))
            self._persist_uns()
        return self._obs.copy()

    def runs(self):
        self._require_open()
        return [_run_info(model, key, run) for _, model, key, run in sorted(_all_runs(self._uns))]

    def find_runs(self, model, mag, request_fingerprint):
        return [
            run
            for run in self.runs()
            if run["model"] == model
            and mag_key(run["mag"]) == mag_key(mag)
            and run["request_fingerprint"] == request_fingerprint
        ]

    def _run(self, run_key):
        for _, _, key, run in _all_runs(self._uns):
            if key == run_key:
                return run
        raise ValueError("Requested feature run is absent")

    def ensure_run(self, model, mag, request_fingerprint, encoder_metadata, dimension):
        self._require_open()
        if not self._uns[_STATE]["slide_id"]:
            raise ValueError("Reconcile patch observations before creating feature runs")
        _key(model, "model")
        _key(request_fingerprint, "request_fingerprint")
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, numbers.Integral)
            or dimension < 1
        ):
            raise ValueError("Feature dimension must be a positive integer")
        metadata_json = _json(encoder_metadata)
        if not isinstance(encoder_metadata, dict):
            raise ValueError("encoder_metadata must be a dictionary")
        existing = self.find_runs(model, mag, request_fingerprint)
        if existing:
            if len(existing) != 1 or existing[0]["dimension"] != dimension:
                raise ValueError("Existing run dimensions conflict with the requested encoder")
            key = existing[0]["run_key"]
            self.set_run_metadata(key, encoder_metadata)
            return key
        magnification = mag_key(mag)
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
        key = f"mag{magnification.replace('.', 'p')}__{slug}__{fingerprint([model, magnification, request_fingerprint])[:24]}"
        if any(item[2] == key for item in _all_runs(self._uns)):
            raise ValueError("Feature run key collision")
        self._mark_dirty()
        dimension = int(dimension)
        row_chunk = max(1, min(256, 1048576 // (4 * dimension)))
        write_elem(
            self._file["obsm"],
            key,
            np.empty((0, dimension), dtype=np.float32),
            dataset_kwargs=dict(
                maxshape=(None, dimension),
                chunks=(row_chunk, dimension),
                compression="gzip",
                compression_opts=self.compression_level,
                fillvalue=np.nan,
            ),
        )
        self._file["obsm"][key].resize((len(self._obs), dimension))
        self._obs[_encoded_column(key)] = np.zeros(len(self._obs), dtype=bool)
        _write_frame(self._file, self._obs)
        self._uns.setdefault("feats", {}).setdefault(magnification, {}).setdefault(
            model, {"runs": {}}
        )["runs"][key] = {
            "model": model,
            "mag": float(mag),
            "request_fingerprint": request_fingerprint,
            "dimension": dimension,
            "metadata_json": metadata_json,
            "metadata_fingerprint": fingerprint(encoder_metadata),
            "feature_index": np.arange(dimension, dtype=np.int64),
            "feature_names": np.asarray(
                [f"feature_{index:06d}" for index in range(dimension)], dtype=str
            ),
            "usage_json": "{}",
        }
        self._persist_uns()
        return key

    def encoded(self, run_key):
        self._require_open()
        self._run(run_key)
        return self._obs[_encoded_column(run_key)].to_numpy(dtype=bool, copy=True)

    def pending_rows(self, run_key):
        run = self._run(run_key)
        return np.flatnonzero(
            (self._obs["mag"].to_numpy(dtype=float) == float(run["mag"])) & ~self.encoded(run_key)
        )

    def write_batch(self, run_key, row_positions, features):
        self._require_open()
        run = self._run(run_key)
        values = np.asarray(row_positions)
        if values.ndim != 1 or (len(values) and values.dtype.kind not in "iu"):
            raise ValueError("Batch rows must be one-dimensional integer positions")
        positions = values.astype(np.int64)
        if (
            len(np.unique(positions)) != len(positions)
            or np.any(positions < 0)
            or np.any(positions >= len(self._obs))
        ):
            raise ValueError("Batch rows must be unique valid observation positions")
        array = np.asarray(features)
        if array.shape != (len(positions), int(run["dimension"])) or array.dtype.kind not in "fiu":
            raise ValueError("Feature batch shape or dtype does not match the run")
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            raise ValueError("Encoded features must be finite float32 values")
        if np.any(self._obs["mag"].to_numpy(dtype=float)[positions] != float(run["mag"])):
            raise ValueError("Batch rows do not belong to the run's magnification")
        if self.encoded(run_key)[positions].any():
            raise ValueError("Completed feature rows cannot be silently overwritten")
        if not len(positions):
            return
        order = np.argsort(positions)
        positions, array = positions[order], array[order]
        self._mark_dirty()
        self._file["obsm"][run_key][positions, :] = array
        self._file.flush()
        self._file["obs"][_encoded_column(run_key)][positions] = True
        self._file.flush()
        self._obs.iloc[positions, self._obs.columns.get_loc(_encoded_column(run_key))] = True

    def set_usage(self, run_key, usage):
        self._require_open()
        if not isinstance(usage, dict):
            raise ValueError("usage must be a dictionary")
        run = self._run(run_key)
        value = _json(usage)
        if value != run.get("usage_json", "{}"):
            self._mark_dirty()
            run["usage_json"] = value
            self._persist_uns()

    def set_run_metadata(self, run_key, metadata):
        self._require_open()
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        run = self._run(run_key)
        value = _json(metadata)
        if value == run["metadata_json"]:
            return
        if self.encoded(run_key).any():
            raise ValueError("Encoder metadata identity cannot change after rows have been encoded")
        self._mark_dirty()
        run.update(metadata_json=value, metadata_fingerprint=fingerprint(metadata))
        self._persist_uns()

    def _close_handle(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def _release_lock(self):
        if self._lock_inode is not None:
            try:
                stat = self.lock_path.lstat()
                if (stat.st_dev, stat.st_ino) == self._lock_inode:
                    self.lock_path.unlink()
            except FileNotFoundError:
                pass
            self._lock_inode = None

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                if self._dirty:
                    if _generation(self.output) != self._baseline:
                        raise ValueError("Published feature generation changed during the update")
                    _load(self._file, published=False)
                    self._uns[_STATE].update(complete=True, transaction_dirty=False)
                    self._persist_uns()
                    self._close_handle()
                    self.partial_path.replace(self.output)
                else:
                    self._close_handle()
                    self.partial_path.unlink(missing_ok=True)
        finally:
            self._close_handle()
            self._release_lock()
        return False


class FeatureReader:
    """Process-local reader that never eagerly loads unrelated embeddings."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = self._obs = self._uns = self._pid = None

    def _ensure_open(self):
        pid = os.getpid()
        if self._file is not None and self._pid == pid:
            return
        self.close()
        handle = h5py.File(self.path, "r")
        try:
            self._obs, self._uns = _load(handle, published=True)
        except BaseException:
            handle.close()
            raise
        self._file, self._pid = handle, pid

    def list_runs(self):
        self._ensure_open()
        return [_run_info(model, key, run) for _, model, key, run in sorted(_all_runs(self._uns))]

    def _select_run(self, model, mag, run_key):
        self._ensure_open()
        matches = [
            (key, run)
            for magnification, name, key, run in _all_runs(self._uns)
            if name == model
            and magnification == mag_key(mag)
            and (run_key is None or key == run_key)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Requested feature run is missing or ambiguous; select an explicit run_key"
            )
        return matches[0]

    def _rows(self, mag, rows):
        positions = _positions(self._obs, rows)
        if mag is not None:
            positions = positions[self._obs["mag"].to_numpy(dtype=float)[positions] == float(mag)]
        return positions

    def patch_metadata(self, mag=None, rows=None):
        self._ensure_open()
        if mag is not None:
            mag_key(mag)
        return self._obs.iloc[self._rows(mag, rows)].copy()

    def model_metadata(self, model, mag, run_key=None):
        key, run = self._select_run(model, mag, run_key)
        return _run_info(model, key, run, axes=True)

    def _read_features(self, key, run, positions, require_complete):
        if (
            require_complete
            and not self._obs[_encoded_column(key)].to_numpy(dtype=bool)[positions].all()
        ):
            raise ValueError("Requested feature rows are pending")
        if not len(positions):
            return np.empty((0, int(run["dimension"])), dtype=np.float32)
        unique, inverse = np.unique(positions, return_inverse=True)
        dataset = self._file["obsm"][key]
        if unique[-1] - unique[0] + 1 == len(unique):
            values = dataset[int(unique[0]) : int(unique[-1]) + 1, :]
        else:
            values = dataset[unique, :]
        result = values[inverse]
        if require_complete and not np.isfinite(result).all():
            raise ValueError("Completed feature rows contain nonfinite values")
        return result

    def features(self, model, mag, run_key=None, rows=None, require_complete=True):
        key, run = self._select_run(model, mag, run_key)
        return self._read_features(key, run, self._rows(mag, rows), require_complete)

    def iter_batches(
        self, model, mag, batch_size=32, *, run_key=None, rows=None, require_complete=True
    ):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        key, run = self._select_run(model, mag, run_key)
        positions = self._rows(mag, rows)
        if (
            require_complete
            and not self._obs[_encoded_column(key)].to_numpy(dtype=bool)[positions].all()
        ):
            raise ValueError("Requested feature rows are pending")
        for start in range(0, len(positions), batch_size):
            batch = positions[start : start + batch_size]
            yield (
                self._read_features(key, run, batch, require_complete),
                self._obs.iloc[batch].copy(),
            )

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file = self._obs = self._uns = self._pid = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __getstate__(self):
        return {"path": self.path, "_file": None, "_obs": None, "_uns": None, "_pid": None}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
