"""CPU estimators for frozen, fixed-width slide vectors.

The input rows are already pooled representations. These estimators do not
backpropagate through a MIL model. Scaling is fitted on training rows only;
calibrated SVMs fit a separate scaler inside each explicitly supplied fold.
"""

from copy import deepcopy
import base64
from importlib import import_module, metadata as package_metadata
import math

import numpy as np


_PROFILES = {
    "logistic_regression": ("sklearn", "sklearn.linear_model", "LogisticRegression", True),
    "random_forest": ("sklearn", "sklearn.ensemble", "RandomForestClassifier", False),
    "extra_trees": ("sklearn", "sklearn.ensemble", "ExtraTreesClassifier", False),
    "hist_gradient_boosting": (
        "sklearn",
        "sklearn.ensemble",
        "HistGradientBoostingClassifier",
        False,
    ),
    "svm": ("sklearn", "sklearn.svm", "SVC", True),
    "knn": ("sklearn", "sklearn.neighbors", "KNeighborsClassifier", True),
    "gaussian_nb": ("sklearn", "sklearn.naive_bayes", "GaussianNB", False),
    "sklearn_mlp": ("sklearn", "sklearn.neural_network", "MLPClassifier", True),
    "xgboost": ("xgboost", "xgboost", "XGBClassifier", False),
    "lightgbm": ("lightgbm", "lightgbm", "LGBMClassifier", False),
    "catboost": ("catboost", "catboost", "CatBoostClassifier", False),
}
_ALIASES = {
    "logreg": "logistic_regression",
    "lr": "logistic_regression",
    "rf": "random_forest",
    "extratrees": "extra_trees",
    "histgb": "hist_gradient_boosting",
    "svc": "svm",
    "gnb": "gaussian_nb",
    "gaussiannb": "gaussian_nb",
    "mlp": "sklearn_mlp",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
}
_TREE_PARAMS = {
    "n_estimators",
    "criterion",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "bootstrap",
    "max_samples",
    "class_weight",
    "n_jobs",
    "random_state",
    "ccp_alpha",
}
_PARAMS = {
    "logistic_regression": {
        "C",
        "penalty",
        "solver",
        "max_iter",
        "tol",
        "class_weight",
        "fit_intercept",
        "l1_ratio",
        "random_state",
    },
    "random_forest": _TREE_PARAMS,
    "extra_trees": _TREE_PARAMS,
    "hist_gradient_boosting": {
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "max_depth",
        "min_samples_leaf",
        "l2_regularization",
        "max_bins",
        "early_stopping",
        "random_state",
        "class_weight",
    },
    "svm": {
        "C",
        "kernel",
        "gamma",
        "degree",
        "coef0",
        "tol",
        "max_iter",
        "cache_size",
        "class_weight",
        "shrinking",
        "decision_function_shape",
        "break_ties",
        "random_state",
    },
    "knn": {"n_neighbors", "weights", "algorithm", "leaf_size", "p", "metric", "n_jobs"},
    "gaussian_nb": {"var_smoothing", "priors"},
    "sklearn_mlp": {
        "hidden_layer_sizes",
        "activation",
        "solver",
        "alpha",
        "batch_size",
        "learning_rate",
        "learning_rate_init",
        "max_iter",
        "shuffle",
        "tol",
        "early_stopping",
        "random_state",
        "n_iter_no_change",
    },
    "xgboost": {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "max_bin",
        "tree_method",
        "device",
        "n_jobs",
        "random_state",
        "verbosity",
    },
    "lightgbm": {
        "n_estimators",
        "num_leaves",
        "max_depth",
        "learning_rate",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "max_bin",
        "device_type",
        "n_jobs",
        "random_state",
        "verbosity",
        "force_col_wise",
        "deterministic",
    },
    "catboost": {
        "iterations",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "border_count",
        "random_strength",
        "class_weights",
        "task_type",
        "thread_count",
        "random_seed",
        "verbose",
        "allow_writing_files",
    },
}
_DISTRIBUTIONS = {
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}


def _json_safe(value, *, label="metadata"):
    """Accept data, never callables, opaque objects, or nonfinite numbers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain finite JSON values")
        return value
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, label=label) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_safe(item, label=label) for key, item in value.items()}
    raise TypeError(f"{label} must contain only JSON data with string keys")


def _canonical(name):
    if not isinstance(name, str):
        raise TypeError("Classifier name must be a string")
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in _PROFILES:
        raise ValueError("Unknown classifier; call list_classifiers() for supported names")
    return normalized


def classifier_info(name):
    name = _canonical(name)
    backend, module, estimator, scale = _PROFILES[name]
    return {
        "name": name,
        "backend": backend,
        "estimator": module + "." + estimator,
        "input_kind": "frozen_slide_vectors",
        "backpropagation": False,
        "device": "cpu",
        "scaling_default": scale,
        "probabilities": "explicit calibration required" if name == "svm" else "native",
        "persistence": "restricted_skops" if backend == "sklearn" else "native_model_format",
        "retains_training_vectors": name in {"svm", "knn"},
    }


def list_classifiers():
    return [classifier_info(name) for name in _PROFILES]


def _features(value, dimension=None):
    # Avoid implicit conversion/detachment of tensors that still require gradients.
    if hasattr(value, "requires_grad") and value.requires_grad:
        raise ValueError("Export frozen vectors explicitly before fitting a classical classifier")
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) < 1 or array.dtype.kind not in "fiu":
        raise ValueError("Expected finite numeric slide vectors [n_slides, n_features]")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(f"Expected {dimension} features, received {array.shape[1]}")
    array = np.asarray(array, dtype=np.float64, order="C")
    if not np.isfinite(array).all():
        raise ValueError("Slide vectors must be finite")
    return array


def _labels(value, length):
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != length or array.dtype.kind not in "biufUS":
        raise ValueError("Expected one numeric or string class label per slide")
    if array.dtype.kind in "f" and not np.isfinite(array).all():
        raise ValueError("Class labels must be finite")
    return array


def _class_metadata(classes):
    """JSON class mapping, retaining arbitrary fixed-width byte labels exactly."""
    result = {"classes": classes.tolist(), "class_dtype": classes.dtype.str}
    if classes.dtype.kind == "S":
        result["classes"] = [base64.b64encode(value).decode("ascii") for value in classes.tolist()]
        result["class_encoding"] = "base64"
    return result


def _groups(value, length):
    if value is None:
        return None
    array = _labels(value, length)
    return array


def _folds(value, labels, groups):
    if isinstance(value, (int, str)) or hasattr(value, "split"):
        raise ValueError(
            "calibration_cv must be explicit (train_indices, calibration_indices) pairs; automatic folds are disabled"
        )
    try:
        folds = list(value)
    except TypeError as exc:
        raise ValueError("calibration_cv must contain explicit index pairs") from exc
    if len(folds) < 2:
        raise ValueError("Provide at least two explicit calibration folds")
    coverage = np.zeros(len(labels), dtype=np.int64)
    classes = np.unique(labels)
    validated = []
    for pair in folds:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each calibration fold must contain train and calibration indices")
        parts = []
        for raw in pair:
            part = np.asarray(raw)
            if (
                part.ndim != 1
                or not len(part)
                or part.dtype.kind not in "iu"
                or part.min() < 0
                or part.max() >= len(labels)
                or len(np.unique(part)) != len(part)
            ):
                raise ValueError("Calibration indices must be nonempty, unique, in-range integers")
            parts.append(part.astype(np.int64))
        train, calibration = parts
        if np.intersect1d(train, calibration).size:
            raise ValueError("Calibration train/validation indices must be disjoint")
        if groups is not None and np.intersect1d(groups[train], groups[calibration]).size:
            raise ValueError("Calibration folds must keep supplied groups disjoint")
        if not np.array_equal(np.unique(labels[train]), classes) or not np.array_equal(
            np.unique(labels[calibration]), classes
        ):
            raise ValueError(
                "Every calibration training and validation fold must contain all classes"
            )
        coverage[calibration] += 1
        validated.append((train, calibration))
    if not np.all(coverage == 1):
        raise ValueError(
            "Explicit calibration validation folds must cover every training row exactly once"
        )
    return validated


class FrozenVectorClassifier:
    def __init__(self, name, *, scale="auto", calibration=None, **options):
        self.name = _canonical(name)
        if scale != "auto" and not isinstance(scale, bool):
            raise ValueError("scale must be 'auto' or bool")
        if calibration not in {None, "sigmoid", "isotonic"} or (
            calibration is not None and self.name != "svm"
        ):
            raise ValueError("Explicit sigmoid/isotonic calibration is supported for SVM only")
        unknown = set(options) - _PARAMS[self.name]
        if unknown:
            raise TypeError(f"Unsupported {self.name} options: {sorted(unknown)}")
        self.options = _json_safe(options, label="Classifier options")
        self.scale = _PROFILES[self.name][3] if scale == "auto" else scale
        self.calibration = calibration
        if isinstance(options.get("class_weight"), dict):
            raise ValueError(
                "Use class_weight='balanced' or None; original-label weight dictionaries are not supported"
            )
        if options.get("early_stopping", False):
            raise ValueError(
                "Implicit validation splits are disabled; early_stopping must be False"
            )
        if (
            options.get("device", "cpu") != "cpu"
            or options.get("device_type", "cpu") != "cpu"
            or options.get("task_type", "CPU") != "CPU"
            or options.get("tree_method") == "gpu_hist"
        ):
            raise ValueError("Classical classifiers support CPU execution only")
        if options.get("allow_writing_files", False):
            raise ValueError("CatBoost allow_writing_files must be False")
        if options.get("kernel") == "precomputed" or options.get("metric") == "precomputed":
            raise ValueError("Precomputed kernels/distances are not fixed-width slide vectors")
        self.metadata = classifier_info(self.name)

    def _estimator(self, class_count):
        backend, module, name, _ = _PROFILES[self.name]
        try:
            cls = getattr(import_module(module), name)
        except Exception as exc:
            if (
                not isinstance(exc, (ImportError, OSError))
                and type(exc).__module__ != "xgboost._c_api"
            ):
                raise
            raise ImportError(
                f"Install openhisto[mil-classical{('-' + backend) if backend != 'sklearn' else ''}] in an isolated CPU environment; {backend} is unavailable"
            ) from exc
        defaults = {
            "logistic_regression": {"max_iter": 1000, "random_state": 0},
            "random_forest": {"n_estimators": 100, "n_jobs": 1, "random_state": 0},
            "extra_trees": {"n_estimators": 100, "n_jobs": 1, "random_state": 0},
            "hist_gradient_boosting": {"early_stopping": False, "random_state": 0},
            "svm": {"probability": False, "random_state": 0},
            "knn": {"n_jobs": 1},
            "gaussian_nb": {},
            "sklearn_mlp": {"early_stopping": False, "random_state": 0, "max_iter": 500},
            "xgboost": {
                "n_jobs": 1,
                "random_state": 0,
                "device": "cpu",
                "tree_method": "hist",
                "verbosity": 0,
            },
            "lightgbm": {
                "n_jobs": 1,
                "random_state": 0,
                "verbosity": -1,
                "device_type": "cpu",
                "deterministic": True,
                "force_col_wise": True,
            },
            "catboost": {
                "thread_count": 1,
                "random_seed": 0,
                "task_type": "CPU",
                "verbose": False,
                "allow_writing_files": False,
            },
        }[self.name]
        parameters = {**defaults, **self.options}
        if self.name == "xgboost":
            parameters["objective"] = "binary:logistic" if class_count == 2 else "multi:softprob"
            parameters["eval_metric"] = "logloss" if class_count == 2 else "mlogloss"
        if self.name == "catboost":
            parameters["loss_function"] = "Logloss" if class_count == 2 else "MultiClass"
        return cls(**parameters)

    def fit(
        self,
        X,
        y,
        *,
        groups=None,
        calibration_cv=None,
        calibration_data=None,
        calibration_groups=None,
        provenance=None,
    ):
        X = _features(X)
        y = _labels(y, len(X))
        classes, encoded = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("Classification requires at least two observed training classes")
        groups = _groups(groups, len(X))
        provenance = _json_safe(provenance or {}, label="Provenance")
        if calibration_cv is not None and calibration_data is not None:
            raise ValueError("Choose calibration_cv or calibration_data, not both")
        calibrating = calibration_cv is not None or calibration_data is not None
        if (
            calibrating or self.calibration is not None or calibration_groups is not None
        ) and self.name != "svm":
            raise ValueError("Calibration arguments are supported for SVM only")
        if self.calibration is not None and not calibrating:
            raise ValueError(
                "SVM calibration requires explicit calibration_cv folds or held-out calibration_data"
            )
        method = self.calibration or "sigmoid"
        estimator = self._estimator(len(classes))
        scaler = None
        calibration_info = None
        if calibrating:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            steps = [("scale", StandardScaler())] if self.scale else []
            steps.append(("estimator", estimator))
            pipeline = Pipeline(steps)
            if calibration_cv is not None:
                if calibration_groups is not None:
                    raise ValueError(
                        "Use groups for CV; calibration_groups applies to held-out data"
                    )
                folds = _folds(calibration_cv, encoded, groups)
                estimator = CalibratedClassifierCV(
                    estimator=pipeline, method=method, cv=folds, ensemble=True, n_jobs=1
                )
                estimator.fit(X, encoded)
                calibration_info = {
                    "method": method,
                    "mode": "explicit_cv",
                    "fold_count": len(folds),
                    "groups_checked": groups is not None,
                }
            else:
                if not isinstance(calibration_data, (tuple, list)) or len(calibration_data) != 2:
                    raise ValueError("calibration_data must be (held_out_X, held_out_y)")
                calX = _features(calibration_data[0], X.shape[1])
                caly = _labels(calibration_data[1], len(calX))
                if not np.array_equal(np.unique(caly), classes):
                    raise ValueError(
                        "Held-out calibration data must contain exactly the training classes"
                    )
                calencoded = np.searchsorted(classes, caly)
                calgroups = _groups(calibration_groups, len(calX))
                if (groups is None) != (calgroups is None):
                    raise ValueError(
                        "Provide both training groups and calibration_groups to check held-out group separation"
                    )
                if groups is not None and np.intersect1d(groups, calgroups).size:
                    raise ValueError("Training and held-out calibration groups must be disjoint")
                pipeline.fit(X, encoded)
                estimator = CalibratedClassifierCV(
                    estimator=pipeline, method=method, cv="prefit", n_jobs=1
                )
                estimator.fit(calX, calencoded)
                calibration_info = {
                    "method": method,
                    "mode": "explicit_holdout",
                    "groups_checked": groups is not None,
                }
        else:
            if calibration_groups is not None:
                raise ValueError("calibration_groups requires calibration_data")
            if self.scale:
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler().fit(X)
                transformed = scaler.transform(X)
            else:
                transformed = X
            estimator.fit(transformed, encoded)
        backend = _PROFILES[self.name][0]
        versions = {
            name: package_metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")
        }
        if backend != "sklearn":
            versions[backend] = package_metadata.version(backend)
        self.estimator_, self.scaler_ = estimator, scaler
        self.classes_, self.n_features_in_ = classes, X.shape[1]
        self.metadata = {
            **classifier_info(self.name),
            "feature_dim": self.n_features_in_,
            **_class_metadata(classes),
            "options": deepcopy(self.options),
            "scale": self.scale,
            "calibration": calibration_info,
            "versions": versions,
            "provenance": provenance,
            "scaling_scope": "fold_training_only"
            if calibration_cv is not None
            else "training_only",
        }
        self._native_loaded = False
        return self

    def _ready(self, X):
        if not hasattr(self, "estimator_"):
            raise ValueError("Classifier is not fitted")
        X = _features(X, self.n_features_in_)
        return self.scaler_.transform(X) if self.scaler_ is not None else X

    def predict_proba(self, X):
        X = self._ready(X)
        if self.name == "svm" and self.metadata.get("calibration") is None:
            raise ValueError(
                "SVM probabilities require explicit calibration data or folds at fit time"
            )
        if self._native_loaded and self.name == "lightgbm":
            probability = np.asarray(self.estimator_.predict(X, num_threads=1))
            if probability.ndim == 1:
                probability = np.column_stack((1 - probability, probability))
        else:
            probability = np.asarray(self.estimator_.predict_proba(X))
        if probability.shape != (len(X), len(self.classes_)) or not np.isfinite(probability).all():
            raise ValueError("Backend returned invalid class probabilities")
        return probability

    def predict(self, X):
        if self.name == "svm" and self.metadata.get("calibration") is None:
            transformed = self._ready(X)
            indices = np.asarray(self.estimator_.predict(transformed), dtype=np.int64).reshape(-1)
        else:
            indices = self.predict_proba(X).argmax(axis=1)
        return self.classes_[indices]

    def decision_function(self, X):
        transformed = self._ready(X)
        if not hasattr(self.estimator_, "decision_function"):
            raise ValueError("This estimator does not expose a decision function")
        return np.asarray(self.estimator_.decision_function(transformed))

    def save(self, path, *, overwrite=False):
        from .classical_io import save_classifier

        return save_classifier(self, path, overwrite=overwrite)


def create_classifier(name, **options):
    return FrozenVectorClassifier(name, **options)


def load_classifier(path, *, max_uncompressed_bytes=2_147_483_648, version_policy="strict"):
    from .classical_io import load_classifier as load

    return load(path, max_uncompressed_bytes=max_uncompressed_bytes, version_policy=version_policy)
