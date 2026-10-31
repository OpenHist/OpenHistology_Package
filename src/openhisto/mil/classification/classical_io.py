"""Atomic data-only artifacts: restricted skops or native boosting formats.

Never uses pickle/joblib. The trust set is fixed in code, not taken from the
artifact or copied wholesale from get_untrusted_types(). Load artifacts from a
trusted source: checksums detect corruption, not the identity of their author.
"""

from hashlib import sha256
from importlib import metadata as package_metadata
import base64
import binascii
import json
import os
from pathlib import Path
import tempfile
import warnings
from zipfile import ZipFile, ZIP_STORED, BadZipFile

import numpy as np

from .classical import FrozenVectorClassifier, _PROFILES, _class_metadata, _json_safe


# Reviewed native sklearn implementation types needed by the supported models.
# In particular, do not trust arbitrary sklearn.*, scipy.*, or user classes.
_SKOPS_ALLOWED = frozenset(
    {
        "sklearn.tree._tree.Tree",
        "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
        "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
        "sklearn._loss.loss.HalfBinomialLoss",
        "sklearn._loss.loss.HalfMultinomialLoss",
        "sklearn._loss.link.LogitLink",
        "sklearn._loss.link.MultinomialLogit",
        "sklearn._loss._loss.CyHalfBinomialLoss",
        "sklearn._loss._loss.CyHalfMultinomialLoss",
        "sklearn.neighbors._kd_tree.KDTree",
        "sklearn.neighbors._ball_tree.BallTree",
        "sklearn.metrics._dist_metrics.EuclideanDistance64",
        "sklearn.metrics._dist_metrics.ManhattanDistance64",
        "sklearn.metrics._dist_metrics.MinkowskiDistance64",
        "sklearn.calibration._CalibratedClassifier",
        "sklearn.calibration._SigmoidCalibration",
        "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
        "sklearn.neural_network._stochastic_optimizers.SGDOptimizer",
    }
)
_PAYLOADS = {
    "sklearn": "estimator.skops",
    "xgboost": "model.json",
    "lightgbm": "model.txt",
    "catboost": "model.cbm",
}
_FORMAT = "cpath_frozen_vector_classifier"
_SCHEMA = 1


def _skops():
    try:
        import skops.io as sio
    except ImportError as exc:
        raise ImportError(
            'Install the mil-classical extra (pip install "openhisto[mil-classical]") to save/load sklearn estimators safely'
        ) from exc
    version = package_metadata.version("skops")
    numbers = tuple(int(part) for part in version.split(".")[:2])
    if numbers < (0, 16):
        raise ImportError("skops>=0.16 is required for current persistence security fixes")
    return sio


def _scaler_dict(scaler):
    if scaler is None:
        return None
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_samples_seen": int(scaler.n_samples_seen_),
    }


def _scaler_load(value, dimension):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"mean", "scale", "var", "n_samples_seen"}:
        raise ValueError("Malformed scaler metadata")
    arrays = {key: np.asarray(value[key], dtype=np.float64) for key in ("mean", "scale", "var")}
    if any(
        array.shape != (dimension,) or not np.isfinite(array).all() for array in arrays.values()
    ):
        raise ValueError("Scaler arrays must match feature_dim and be finite")
    count = value["n_samples_seen"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or (arrays["scale"] <= 0).any()
        or (arrays["var"] < 0).any()
    ):
        raise ValueError("Invalid scaler state")
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = arrays["mean"], arrays["scale"], arrays["var"]
    scaler.n_features_in_, scaler.n_samples_seen_ = dimension, np.int64(count)
    return scaler


def _payload(model, backend):
    if backend == "sklearn":
        return _skops().dumps(model.estimator_)
    with tempfile.TemporaryDirectory(prefix="cpath-native-classifier-") as temporary:
        path = Path(temporary) / _PAYLOADS[backend]
        if backend == "xgboost":
            model.estimator_.save_model(path)
        elif backend == "lightgbm":
            booster = model.estimator_ if model._native_loaded else model.estimator_.booster_
            booster.save_model(path)
        else:
            model.estimator_.save_model(str(path), format="cbm")
        return path.read_bytes()


def save_classifier(model, path, *, overwrite=False):
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be bool")
    if not isinstance(model, FrozenVectorClassifier) or not hasattr(model, "estimator_"):
        raise ValueError("Only fitted frozen-vector classifiers can be saved")
    path = Path(path).expanduser().absolute()
    if path.is_symlink():
        raise ValueError("Classifier artifact destination cannot be a symlink")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    backend = _PROFILES[model.name][0]
    payload = _payload(model, backend)
    metadata = dict(model.metadata)
    packages = ["numpy", "scipy", "scikit-learn"] + ([] if backend == "sklearn" else [backend])
    versions = {package: package_metadata.version(package) for package in packages}
    if metadata["versions"] != versions:
        # A native model may have been loaded with an explicit relaxed policy.
        # Its next export is written by the current libraries, while its
        # original fit/export environment remains available as provenance.
        metadata.setdefault("training_versions", metadata["versions"])
        metadata["versions"] = dict(versions)
    if backend == "sklearn":
        versions["skops"] = package_metadata.version("skops")
    manifest = {
        "artifact_type": _FORMAT,
        "schema_version": _SCHEMA,
        "complete": True,
        "name": model.name,
        "backend": backend,
        "scale": model.scale,
        "calibration": model.calibration,
        "options": model.options,
        "metadata": metadata,
        "versions": versions,
        "scaler": _scaler_dict(model.scaler_),
        "payload": _PAYLOADS[backend],
        "payload_sha256": sha256(payload).hexdigest(),
    }
    manifest_bytes = json.dumps(
        _json_safe(manifest), ensure_ascii=False, allow_nan=False, sort_keys=True
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + "-", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        with ZipFile(temporary, "w", compression=ZIP_STORED) as archive:
            archive.writestr("metadata.json", manifest_bytes)
            archive.writestr(_PAYLOADS[backend], payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Hard-link publication is atomic and cannot clobber a racing writer.
            os.link(temporary, path)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def _read(path, *, max_uncompressed_bytes=2_147_483_648):
    if (
        isinstance(max_uncompressed_bytes, bool)
        or not isinstance(max_uncompressed_bytes, int)
        or max_uncompressed_bytes < 1
    ):
        raise ValueError("max_uncompressed_bytes must be a positive integer")
    try:
        with ZipFile(Path(path).expanduser()) as archive:
            infos = archive.infolist()
            names = [entry.filename for entry in infos]
            if len(names) != 2 or len(set(names)) != 2 or "metadata.json" not in names:
                raise ValueError(
                    "Classifier artifact must contain exactly metadata and its model payload"
                )
            if sum(entry.file_size for entry in infos) > max_uncompressed_bytes:
                raise ValueError("Classifier artifact exceeds its uncompressed size limit")
            info = archive.getinfo("metadata.json")
            if info.file_size > 16 * 1024 * 1024:
                raise ValueError("Classifier metadata is unexpectedly large")
            manifest = json.loads(archive.read("metadata.json"))
            if not isinstance(manifest, dict):
                raise ValueError("Classifier metadata must be an object")
            _json_safe(manifest)
            if (
                manifest.get("artifact_type") != _FORMAT
                or manifest.get("schema_version") != _SCHEMA
                or manifest.get("complete") is not True
            ):
                raise ValueError("Unsupported or incomplete classifier artifact")
            backend = manifest.get("backend")
            if (
                backend not in _PAYLOADS
                or manifest.get("payload") != _PAYLOADS[backend]
                or set(names) != {"metadata.json", _PAYLOADS[backend]}
            ):
                raise ValueError("Invalid classifier payload format")
            payload = archive.read(_PAYLOADS[backend])
            if sha256(payload).hexdigest() != manifest.get("payload_sha256"):
                raise ValueError("Classifier payload checksum mismatch")
            return manifest, payload
    except (BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed classifier artifact") from exc


def _load_skops(payload, name, calibrated):
    sio = _skops()
    unknown = set(sio.get_untrusted_types(data=payload))
    forbidden = unknown - _SKOPS_ALLOWED
    if forbidden:
        raise ValueError("Unsupported serialized estimator types: " + ", ".join(sorted(forbidden)))
    estimator = sio.loads(payload, trusted=sorted(unknown & _SKOPS_ALLOWED))
    if calibrated:
        from sklearn.calibration import CalibratedClassifierCV

        expected = CalibratedClassifierCV
    else:
        from importlib import import_module

        _, module, cls, _ = _PROFILES[name]
        expected = getattr(import_module(module), cls)
    if type(estimator) is not expected:
        raise ValueError("Serialized estimator does not match its declared classifier")
    return estimator


def load_classifier(path, *, max_uncompressed_bytes=2_147_483_648, version_policy="strict"):
    """Load with exact dependency versions by default.

    ``version_policy='native'`` allows version differences only for native
    XGBoost/LightGBM/CatBoost payloads, with a warning. Their native readers and
    the usual feature/class/structure checks still validate the artifact.
    Sklearn/skops payloads always require the recorded environment.
    """
    if version_policy not in {"strict", "native"}:
        raise ValueError("version_policy must be 'strict' or 'native'")
    manifest, payload = _read(path, max_uncompressed_bytes=max_uncompressed_bytes)
    try:
        model = FrozenVectorClassifier(
            manifest["name"],
            scale=manifest["scale"],
            calibration=manifest["calibration"],
            **manifest["options"],
        )
        if _PROFILES[model.name][0] != manifest["backend"]:
            raise ValueError("Classifier backend does not match its name")
        if version_policy == "native" and manifest["backend"] == "sklearn":
            raise ValueError(
                "version_policy='native' applies only to native boosted-tree payloads; sklearn requires strict versions"
            )
        metadata = manifest["metadata"]
        if (
            not isinstance(metadata, dict)
            or metadata.get("name") != model.name
            or metadata.get("backend") != manifest["backend"]
            or metadata.get("scale") != model.scale
        ):
            raise ValueError("Classifier metadata does not match its configuration")
        dimension = metadata["feature_dim"]
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("Invalid classifier feature_dim")
        dtype = np.dtype(metadata["class_dtype"])
        if dtype.kind not in "biufUS" or dtype.itemsize > 1024 * 1024:
            raise ValueError("Invalid class mapping dtype")
        values = metadata["classes"]
        if dtype.kind == "S":
            if metadata.get("class_encoding") != "base64" or not isinstance(values, list):
                raise ValueError("Byte class labels require a base64 class mapping")
            try:
                values = [base64.b64decode(value, validate=True) for value in values]
            except (binascii.Error, TypeError, ValueError) as exc:
                raise ValueError("Invalid base64 class mapping") from exc
        elif metadata.get("class_encoding") is not None:
            raise ValueError("Unexpected class mapping encoding")
        classes = np.asarray(values, dtype=dtype)
        if (
            classes.ndim != 1
            or len(classes) < 2
            or not np.array_equal(np.unique(classes), classes)
            or _class_metadata(classes)["classes"] != metadata["classes"]
        ):
            raise ValueError("Class mapping must contain ordered unique classes")
        if classes.dtype.kind == "f" and not np.isfinite(classes).all():
            raise ValueError("Class mapping must be finite")
        versions = manifest["versions"]
        required = {"numpy", "scipy", "scikit-learn"}
        required.add("skops" if manifest["backend"] == "sklearn" else manifest["backend"])
        if not isinstance(versions, dict) or not required <= set(versions):
            raise ValueError("Classifier dependency versions are missing")
        differences = []
        for package in sorted(required):
            current = package_metadata.version(package)
            if current != versions[package]:
                if version_policy == "strict":
                    raise ValueError(
                        f"Classifier requires {package}=={versions[package]}; use the recorded isolated environment"
                    )
                differences.append(f"{package}: saved {versions[package]}, installed {current}")
            if package != "skops" and metadata["versions"].get(package) != versions[package]:
                raise ValueError("Classifier dependency version metadata is inconsistent")
        if differences:
            warnings.warn(
                "Loading a native classifier with different dependency versions ("
                + "; ".join(differences)
                + "); verify predictions in this environment",
                RuntimeWarning,
                stacklevel=2,
            )
        scaler = _scaler_load(manifest["scaler"], dimension)
        calibration = metadata.get("calibration")
        if calibration is not None:
            if (
                model.name != "svm"
                or not isinstance(calibration, dict)
                or calibration.get("method") not in {"sigmoid", "isotonic"}
                or calibration.get("mode") not in {"explicit_cv", "explicit_holdout"}
            ):
                raise ValueError("Invalid SVM calibration metadata")
            if manifest["calibration"] not in {None, calibration["method"]}:
                raise ValueError("SVM calibration configuration is inconsistent")
            if scaler is not None:
                raise ValueError("Calibrated SVM scaling must remain inside its fitted pipeline")
        if manifest["backend"] == "sklearn":
            estimator = _load_skops(payload, model.name, calibration is not None)
            if getattr(estimator, "n_features_in_", None) != dimension or not np.array_equal(
                estimator.classes_, np.arange(len(classes))
            ):
                raise ValueError(
                    "Serialized estimator feature/class dimensions do not match metadata"
                )
        else:
            with tempfile.TemporaryDirectory(prefix="cpath-load-classifier-") as temporary:
                native = Path(temporary) / manifest["payload"]
                native.write_bytes(payload)
                if manifest["backend"] == "lightgbm":
                    from lightgbm import Booster

                    estimator = Booster(model_file=str(native))
                    if (
                        estimator.num_feature() != dimension
                        or estimator.num_model_per_iteration()
                        != (1 if len(classes) == 2 else len(classes))
                    ):
                        raise ValueError("Native LightGBM dimensions do not match metadata")
                else:
                    estimator = model._estimator(len(classes))
                    if manifest["backend"] == "catboost":
                        estimator.load_model(str(native), format="cbm")
                    else:
                        estimator.load_model(native)
                    if len(estimator.classes_) != len(classes):
                        raise ValueError("Native classifier class count does not match metadata")
                    native_dimension = (
                        len(estimator.feature_names_)
                        if manifest["backend"] == "catboost"
                        else estimator.n_features_in_
                    )
                    if native_dimension != dimension:
                        raise ValueError(
                            "Native classifier feature dimension does not match metadata"
                        )
        model.estimator_, model.scaler_, model.classes_, model.n_features_in_ = (
            estimator,
            scaler,
            classes,
            dimension,
        )
        model.metadata = metadata
        model._native_loaded = manifest["backend"] != "sklearn"
        return model
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Malformed classifier metadata") from exc
