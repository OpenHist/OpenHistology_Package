"""Incremental patch-to-feature execution without implicit model upgrades."""

from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import NamedTuple

import numpy as np

from .base import EncoderError, ModelDescriptor, validate_features
from .config import (
    EncodeConfig,
    EncoderSpec,
    EncodingReport,
    fingerprint,
    json_safe,
    reject_secrets,
)
from .inputs import prepare_inputs
from .registry import get_descriptor, prepare_model
from .storage import FeatureStore


_CHECKPOINT_FILES = {
    "phikon": ("config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin"),
    "phikon_v2": (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ),
    "uni_v1": ("config.json", "pytorch_model.bin", "model.safetensors"),
    "conch15_titan": ("config.json", "conch_v1_5.py", "conch_v1_5_pytorch_model.bin"),
}


class EncodingFailure(EncoderError):
    """A failed run with counts and resumable partial work, without secret text."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _checkpoint_hash(path, model):
    path = Path(path).expanduser()
    directory = path if path.is_dir() else path.parent
    names = _CHECKPOINT_FILES.get(model, ())
    if path.is_dir():
        candidates = [directory / name for name in names if (directory / name).is_file()]
        if not candidates:
            raise ValueError("Checkpoint directory has no files for the selected encoder profile")
    else:
        # A single weight file is hashed together with its sibling configuration/code.
        candidates = [path] + [
            directory / name
            for name in names
            if name.endswith((".json", ".py")) and (directory / name).is_file()
        ]
    digest = hashlib.sha256()
    for candidate in sorted(set(candidates)):
        digest.update(candidate.name.encode())
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _request_identity(spec, config, descriptor):
    request = spec.to_dict()
    if spec.checkpoint:
        request["checkpoint"] = {"sha256": _checkpoint_hash(spec.checkpoint, spec.model)}
    return {
        "schema_version": 1,
        "spec": request,
        "profile": {
            "model_name": descriptor.model_name,
            "dimension": descriptor.dimension,
            "preprocessing": descriptor.preprocessing,
        },
        "inference": config.identity(),
    }


def _dimension(spec, descriptor):
    dimension = spec.dimension or descriptor.dimension
    value_range = descriptor.preprocessing.get("dimension_range")
    if value_range:
        valid = value_range[0] <= dimension <= value_range[1]
    else:
        valid = dimension in (descriptor.dimensions or (descriptor.dimension,))
    if not valid:
        raise ValueError("Requested output dimension is unsupported; inspect model_info()")
    return dimension


def _metadata(encoder, request, dimension):
    metadata = json_safe(dict(encoder.metadata))
    reject_secrets(metadata)
    json.dumps(metadata, allow_nan=False)
    return {
        "prepared": True,
        "request": request,
        "encoder": metadata,
        "normalization": "l2" if request["spec"]["normalize"] else "native",
        "output_dtype": "float32",
        "dimension": dimension,
        "execution": {
            "first_prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "core_versions": {
                name: version(name) for name in ("anndata", "numpy", "pandas", "h5py", "Pillow")
            },
        },
    }


def _check_frozen(saved, current):
    """A repeated default request keeps the originally resolved checkpoint."""
    old = saved.get("encoder", {})
    new = current.get("encoder", {})
    for name in (
        "resolved_revision",
        "checkpoint_sha256",
        "weights_sha256",
        "weights_hash",
        "checkpoint_config_sha256",
        "code_sha256",
        "pooling",
        "preprocessing",
        "profile_version",
        "adapter_version",
        "device",
        "precision",
        "library_versions",
        "sdk",
        "endpoint",
        "region",
        "task",
        "provider_precision",
    ):
        if name in old and old[name] != new.get(name):
            raise EncoderError(
                "Prepared model/runtime identity differs from saved features; use an explicit new configuration or options.cache_tag"
            )


def _usage_delta(before, after):
    return {
        key: value - before.get(key, 0)
        for key, value in after.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _add_usage(before, delta):
    result = dict(before)
    for key, value in delta.items():
        result[key] = result.get(key, 0) + value
    return result


def _is_oom(exc):
    # Matched by name so this module never imports torch.
    return type(exc).__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}


def _l2_normalized(features):
    norms = np.linalg.norm(features.astype(np.float64), axis=1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)


def _failure_message(exc):
    if isinstance(exc, (EncoderError, ValueError, OSError)):
        return str(exc)
    return f"Encoding failed ({type(exc).__name__}); compatible partial work was retained"


class _Request(NamedTuple):
    spec: EncoderSpec
    descriptor: ModelDescriptor
    dimension: int
    identity: dict
    fingerprint: str

    def unprepared_metadata(self):
        """Metadata for a run created before any model was prepared for it."""
        return {
            "prepared": False,
            "request": self.identity,
            "encoder": {"model_id": self.spec.model, "resolved_revision": None},
            "normalization": "l2" if self.spec.normalize else "native",
            "output_dtype": "float32",
            "dimension": self.dimension,
        }

    def frozen_spec(self, saved):
        """A repeated default HF request keeps the revision its saved run resolved."""
        revision = saved["metadata"].get("encoder", {}).get("resolved_revision") if saved else None
        if revision and self.spec.revision is None and self.descriptor.provider == "hf":
            return replace(self.spec, revision=revision)
        return self.spec


def _as_spec(item):
    if isinstance(item, EncoderSpec):
        return item
    if isinstance(item, dict):
        return EncoderSpec(**item)
    return EncoderSpec(item)


def _specs(models):
    if isinstance(models, (str, EncoderSpec, dict)):
        models = [models]
    specs = [_as_spec(item) for item in models]
    if not specs:
        raise ValueError("Select at least one encoder model")
    return specs


def _requests(specs, config):
    requests, seen = [], set()
    for spec in specs:
        descriptor = get_descriptor(spec)
        identity = _request_identity(spec, config, descriptor)
        request = _Request(
            spec, descriptor, _dimension(spec, descriptor), identity, fingerprint(identity)
        )
        if request.fingerprint in seen:
            raise ValueError("Duplicate encoder configuration in one request")
        seen.add(request.fingerprint)
        requests.append(request)
    return requests


def _stored_pending(store):
    return sum(len(store.pending_rows(run["run_key"])) for run in store.runs())


@contextmanager
def _usage_recorded(store, key, encoder, saved):
    """Add the encoder's usage during the block to the run's saved totals, even on failure."""
    before = dict(getattr(encoder, "usage", {}))
    try:
        yield
    finally:
        delta = _usage_delta(before, getattr(encoder, "usage", {}))
        store.set_usage(key, _add_usage(saved.get("usage", {}) if saved else {}, delta))


@dataclass
class _Progress:
    """Counts that are reported both on success and on failure."""

    total: int
    completed: int = 0
    reused: int = 0
    pending: int = 0
    failed: int = 0
    runs: list = field(default_factory=list)
    metadata_updated: int = 0
    metadata_preserved: int = 0

    def report(self, output, failed=0):
        return EncodingReport(
            output,
            self.total,
            self.completed,
            self.reused,
            self.pending,
            failed,
            tuple(self.runs),
            self.metadata_updated,
            self.metadata_preserved,
        )


class _Encoders:
    """Prepare each requested model at most once and close them all afterwards."""

    def __init__(self, config, auth):
        self._config, self._auth = config, auth
        self._prepared = {}

    def get(self, request, spec):
        if request.fingerprint not in self._prepared:
            self._prepared[request.fingerprint] = prepare_model(
                spec, config=self._config, auth=self._auth
            )
        return self._prepared[request.fingerprint]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        for encoder in self._prepared.values():
            # A failing close must not replace the encoding outcome.
            with suppress(Exception):
                encoder.close()


class _SlideRuns:
    """Bring every requested (model, magnification) run of one slide up to date."""

    def __init__(self, store, patches, encoders, config, progress):
        self.store, self.patches, self.encoders = store, patches, encoders
        self.config, self.progress = config, progress
        self.frame = store.reconcile(
            patches.obs,
            slide_id=patches.slide_id,
            sources=patches.sources,
            metadata_policy=config.metadata_policy,
        )
        progress.metadata_updated = store.metadata_changes["updated"]
        progress.metadata_preserved = store.metadata_changes["preserved"]
        self.mags = self.frame["mag"].to_numpy()
        progress.total = len(self.frame)

    def update(self, requests, magnifications):
        self.progress.pending = self._pending_rows(requests, magnifications)
        for request in requests:
            for mag in magnifications:
                self.progress.runs.append(self._update_run(request, mag))
        self.progress.pending = _stored_pending(self.store)
        self.progress.total = len(self.store.obs)

    def _pending_rows(self, requests, magnifications):
        """Gaps in saved runs plus every row of runs that do not exist yet."""
        count = _stored_pending(self.store)
        for request in requests:
            for mag in magnifications:
                if not self.store.find_runs(request.spec.model, mag, request.fingerprint):
                    count += int(np.count_nonzero(self.mags == mag))
        return count

    def _saved_run(self, request, mag):
        matching = self.store.find_runs(request.spec.model, mag, request.fingerprint)
        if len(matching) > 1:
            raise ValueError(
                "Several saved runs match one request; select an explicit revision or cache_tag"
            )
        if not matching:
            return None
        saved = matching[0]
        if saved["dimension"] != request.dimension:
            raise ValueError("Saved output dimension does not match the requested model")
        return saved

    def _update_run(self, request, mag):
        saved = self._saved_run(request, mag)
        if saved is None:
            key, pending = None, np.flatnonzero(self.mags == mag)
        else:
            key = saved["run_key"]
            self.progress.reused += int(
                np.count_nonzero(self.store.encoded(key) & (self.mags == mag))
            )
            pending = self.store.pending_rows(key)
        if len(pending):
            return self._encode_run(request, mag, saved, key, pending)
        if key is None:
            return self.store.ensure_run(
                request.spec.model,
                mag,
                request.fingerprint,
                request.unprepared_metadata(),
                request.dimension,
            )
        return key

    def _encode_run(self, request, mag, saved, key, pending):
        if any(
            identifier not in self.patches.obs.index for identifier in self.frame.index[pending]
        ):
            raise ValueError(
                "Pending feature rows need their original patch sources; include those archives/manifests in this request"
            )
        encoder = self.encoders.get(request, request.frozen_spec(saved))
        metadata = _metadata(encoder, request.identity, request.dimension)
        if saved and saved["metadata"].get("prepared"):
            _check_frozen(saved["metadata"], metadata)
        if key is None:
            key = self.store.ensure_run(
                request.spec.model, mag, request.fingerprint, metadata, request.dimension
            )
        elif not self.store.encoded(key).any():
            self.store.set_run_metadata(key, metadata)
        with _usage_recorded(self.store, key, encoder, saved):
            self._encode_rows(encoder, request, key, pending)
        return key

    def _encode_rows(self, encoder, request, key, positions):
        """Encode in batches, halving the batch size after running out of memory."""
        cursor, batch_size = 0, self.config.batch_size
        while cursor < len(positions):
            batch = positions[cursor : cursor + batch_size]
            images = self.patches.read_images(self.frame.index[batch].tolist(), self.config.workers)
            try:
                features = validate_features(encoder.encode(images), len(images), request.dimension)
            except Exception as exc:
                if _is_oom(exc) and len(images) > 1:
                    batch_size = max(1, len(images) // 2)
                    continue
                self.progress.failed = len(images)
                raise
            if request.spec.normalize:
                features = _l2_normalized(features)
            self.store.write_batch(key, batch, features)
            self.progress.completed += len(images)
            self.progress.pending -= len(images)
            cursor += len(images)


def encode_slide(inputs, output, models, config=None, auth=None) -> EncodingReport:
    """Add requested model/magnification results to one slide's standard h5ad.

    Existing rows and complete features are preserved. Repeated default model
    requests freeze their first resolved checkpoint. Revision/dimension/options
    changes create distinct runs. Other models' newly appended rows stay pending.
    No credentials are needed when all requested features are already complete.
    """
    config = config or EncodeConfig()
    if not isinstance(config, EncodeConfig):
        raise TypeError("config must be EncodeConfig")
    specs = _specs(models)
    output = Path(output).expanduser().absolute()
    if output.suffix.lower() != ".h5ad":
        raise ValueError("Feature output must end in .h5ad")
    with prepare_inputs(inputs, config) as patches, _Encoders(config, auth) as encoders:
        if any(Path(source["path"]).resolve() == output.resolve() for source in patches.sources):
            raise ValueError("Feature output cannot replace a patch input")
        requests = _requests(specs, config)
        magnifications = sorted(
            {float(mag) for source in patches.sources for mag in source["magnifications"]}
        )
        progress = _Progress(total=len(patches.obs))
        try:
            with FeatureStore(output, compression_level=config.compression_level) as store:
                _SlideRuns(store, patches, encoders, config, progress).update(
                    requests, magnifications
                )
        except Exception as exc:
            raise EncodingFailure(
                _failure_message(exc), progress.report(output, failed=max(1, progress.failed))
            ) from None
    return progress.report(output)
