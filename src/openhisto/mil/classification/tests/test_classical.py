"""Real CPU fits and non-pickle roundtrips on synthetic slide vectors only."""

import json
from pathlib import Path
import tempfile
import unittest
import warnings
from zipfile import ZipFile, ZIP_STORED

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from openhisto.mil.classification.classical import (
    create_classifier,
    classifier_info,
    list_classifiers,
    load_classifier,
)


NAMES = (
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "svm",
    "knn",
    "gaussian_nb",
    "sklearn_mlp",
    "xgboost",
    "lightgbm",
    "catboost",
)


def synthetic(classes=3):
    rng = np.random.default_rng(0)
    labels = np.array(["synthetic_c", "synthetic_a", "synthetic_b"][:classes])
    centers = np.arange(classes)[:, None] * 5 + np.arange(6)[None, :] * 0.2
    X = np.concatenate([rng.normal(center, 0.25, size=(18, 6)) for center in centers])
    y = np.repeat(labels, 18)
    test = np.concatenate([rng.normal(center, 0.25, size=(6, 6)) for center in centers])
    truth = np.repeat(labels, 6)
    groups = np.arange(len(X)) // 2
    folds = [
        (np.flatnonzero(groups % 3 != fold), np.flatnonzero(groups % 3 == fold))
        for fold in range(3)
    ]
    return X, y, test, truth, groups, folds


def options(name):
    return {
        "random_forest": {"n_estimators": 8, "max_depth": 3},
        "extra_trees": {"n_estimators": 8, "max_depth": 3},
        "hist_gradient_boosting": {"max_iter": 8, "min_samples_leaf": 3},
        "sklearn_mlp": {"hidden_layer_sizes": [12], "solver": "lbfgs", "max_iter": 100},
        "xgboost": {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.3},
        "lightgbm": {"n_estimators": 8, "num_leaves": 5, "min_child_samples": 3},
        "catboost": {"iterations": 8, "depth": 2, "learning_rate": 0.3},
    }.get(name, {})


class UnexpectedArtifactType:
    """A harmless test object outside the fixed persistence trust policy."""


class ClassicalTests(unittest.TestCase):
    def test_supported_label_dtypes_roundtrip_including_arbitrary_bytes(self):
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [2.0, 3.0], [3.0, 2.0]])
        labels = (
            np.array([False] * 3 + [True] * 3),
            np.array([-10] * 3 + [50] * 3),
            np.array([0.25] * 3 + [1.5] * 3),
            np.array(["negative"] * 3 + ["positive"] * 3),
            np.array([b"negative"] * 3 + [b"positive"] * 3, dtype="S8"),
            np.array([b"\xff\x00a"] * 3 + [b"\xfe\x00b"] * 3, dtype="S4"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, y in enumerate(labels):
                with self.subTest(dtype=y.dtype, case=index):
                    model = create_classifier("logistic_regression").fit(X, y)
                    path = Path(temporary) / f"{index}.classifier"
                    model.save(path)
                    loaded = load_classifier(path)
                    self.assertEqual(loaded.classes_.dtype, model.classes_.dtype)
                    np.testing.assert_array_equal(loaded.classes_, model.classes_)
                    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
                    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
                    if y.dtype.kind == "S":
                        self.assertEqual(loaded.metadata["class_encoding"], "base64")

    def test_byte_label_encoding_is_validated_and_sklearn_versions_stay_strict(self):
        X = np.array([[0.0], [1.0], [4.0], [5.0]])
        model = create_classifier("logistic_regression").fit(X, np.array([b"a", b"a", b"b", b"b"]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "original.classifier"
            model.save(path)
            with self.assertRaisesRegex(ValueError, "sklearn requires strict"):
                load_classifier(path, version_policy="native")
            with self.assertRaisesRegex(ValueError, "version_policy"):
                load_classifier(path, version_policy="ignore")
            with ZipFile(path) as archive:
                manifest = json.loads(archive.read("metadata.json"))
                payload = archive.read(manifest["payload"])
            manifest["metadata"]["classes"][0] = "invalid!"
            broken = Path(temporary) / "invalid-labels.classifier"
            with ZipFile(broken, "w") as archive:
                archive.writestr("metadata.json", json.dumps(manifest))
                archive.writestr(manifest["payload"], payload)
            with self.assertRaisesRegex(ValueError, "Invalid base64 class mapping"):
                load_classifier(broken)

    def test_native_version_policy_is_explicit_and_resaving_records_current_versions(self):
        X, y, test, _, _, _ = synthetic(2)
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as temporary:
            for name in ("xgboost", "lightgbm", "catboost"):
                with self.subTest(backend=name):
                    model = create_classifier(name, **options(name)).fit(X, y)
                    original = Path(temporary) / f"{name}-original.classifier"
                    model.save(original)
                    with ZipFile(original) as archive:
                        manifest = json.loads(archive.read("metadata.json"))
                        payload = archive.read(manifest["payload"])
                    current_numpy = manifest["versions"]["numpy"]
                    manifest["versions"]["numpy"] = "0.0.0"
                    manifest["metadata"]["versions"]["numpy"] = "0.0.0"
                    historical = Path(temporary) / f"{name}-historical.classifier"
                    with ZipFile(historical, "w") as archive:
                        archive.writestr("metadata.json", json.dumps(manifest))
                        archive.writestr(manifest["payload"], payload)
                    with self.assertRaisesRegex(ValueError, "numpy==0.0.0"):
                        load_classifier(historical)
                    with self.assertWarnsRegex(RuntimeWarning, "different dependency versions"):
                        loaded = load_classifier(historical, version_policy="native")
                    np.testing.assert_allclose(
                        loaded.predict_proba(test), model.predict_proba(test)
                    )
                    resaved = Path(temporary) / f"{name}-resaved.classifier"
                    loaded.save(resaved)
                    restored = load_classifier(resaved)
                    self.assertEqual(restored.metadata["versions"]["numpy"], current_numpy)
                    self.assertEqual(restored.metadata["training_versions"]["numpy"], "0.0.0")
                    np.testing.assert_allclose(
                        restored.predict_proba(test), model.predict_proba(test)
                    )

    def test_all_eleven_binary_multiclass_and_real_roundtrip(self):
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as temporary:
            for count in (2, 3):
                X, y, test, truth, groups, folds = synthetic(count)
                for name in NAMES:
                    with self.subTest(name=name, classes=count):
                        classifier = create_classifier(name, **options(name))
                        fit = {"calibration_cv": folds, "groups": groups} if name == "svm" else {}
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            classifier.fit(
                                X,
                                y,
                                provenance={"representation": "synthetic_mean", "seed": 0},
                                **fit,
                            )
                        np.testing.assert_array_equal(classifier.classes_, np.unique(y))
                        probabilities = classifier.predict_proba(test)
                        self.assertEqual(probabilities.shape, (len(test), count))
                        self.assertTrue(np.isfinite(probabilities).all())
                        self.assertTrue((probabilities >= 0).all())
                        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
                        predictions = classifier.predict(test)
                        self.assertGreaterEqual(np.mean(predictions == truth), 0.8)
                        if classifier.scaler_ is not None:
                            np.testing.assert_allclose(classifier.scaler_.mean_, X.mean(0))
                            mean = classifier.scaler_.mean_.copy()
                            classifier.predict(test + 0.5)
                            np.testing.assert_array_equal(classifier.scaler_.mean_, mean)
                        path = Path(temporary) / f"{name}-{count}.classifier"
                        classifier.save(path)
                        loaded = load_classifier(path)
                        np.testing.assert_array_equal(loaded.classes_, classifier.classes_)
                        np.testing.assert_array_equal(loaded.predict(test), predictions)
                        np.testing.assert_allclose(
                            loaded.predict_proba(test), probabilities, rtol=1e-7, atol=1e-10
                        )
                        self.assertEqual(
                            loaded.metadata["provenance"], classifier.metadata["provenance"]
                        )
                        self.assertEqual(loaded.n_features_in_, 6)
                        # Also support saving a loaded native model, not just first-fit wrappers.
                        second = Path(temporary) / f"second-{name}-{count}.classifier"
                        loaded.save(second)
                        np.testing.assert_allclose(
                            load_classifier(second).predict_proba(test), probabilities, atol=1e-10
                        )
                        with ZipFile(path) as archive:
                            self.assertFalse(
                                any(
                                    name.endswith((".pkl", ".pickle", ".joblib"))
                                    for name in archive.namelist()
                                )
                            )

    def test_svm_explicit_cv_scalers_are_fitted_inside_each_training_fold(self):
        X, y, _, _, groups, folds = synthetic()
        model = create_classifier("svm").fit(X, y, calibration_cv=folds, groups=groups)
        self.assertIsNone(model.scaler_)
        for calibrated, (training, _) in zip(model.estimator_.calibrated_classifiers_, folds):
            scaler = calibrated.estimator.named_steps["scale"]
            np.testing.assert_allclose(scaler.mean_, X[training].mean(0))
        self.assertEqual(model.metadata["scaling_scope"], "fold_training_only")
        # Explicit folds are valid when no grouping is available; none is inferred.
        create_classifier("svm").fit(X, y, calibration_cv=folds)

    def test_svm_holdout_calibration_scaling_group_checks_and_roundtrip(self):
        X, y, test, truth, groups, folds = synthetic()
        with threadpool_limits(limits=1):
            model = create_classifier("svm", calibration="sigmoid").fit(
                X,
                y,
                groups=groups,
                calibration_data=(test, truth),
                calibration_groups=np.arange(len(test)) + 100,
            )
        np.testing.assert_allclose(model.estimator_.estimator.named_steps["scale"].mean_, X.mean(0))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "holdout.classifier"
            model.save(path)
            np.testing.assert_allclose(
                load_classifier(path).predict_proba(test), model.predict_proba(test)
            )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            create_classifier("svm").fit(
                X,
                y,
                groups=groups,
                calibration_data=(test, truth),
                calibration_groups=np.arange(len(test)),
            )
        with self.assertRaisesRegex(ValueError, "both"):
            create_classifier("svm").fit(X, y, calibration_cv=folds, calibration_data=(test, truth))
        with self.assertRaisesRegex(ValueError, "both training groups"):
            create_classifier("svm").fit(X, y, groups=groups, calibration_data=(test, truth))
        with self.assertRaisesRegex(ValueError, "training classes"):
            create_classifier("svm").fit(X, y, calibration_data=(test[:6], truth[:6]))

    def test_svm_does_not_silently_create_calibration_folds(self):
        X, y, test, _, groups, folds = synthetic()
        uncalibrated = create_classifier("svm").fit(X, y)
        self.assertEqual(uncalibrated.predict(test).shape, (len(test),))
        self.assertEqual(uncalibrated.decision_function(test).shape[0], len(test))
        with self.assertRaisesRegex(ValueError, "explicit calibration"):
            uncalibrated.predict_proba(test)
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            create_classifier("svm", calibration="sigmoid").fit(X, y)
        with self.assertRaisesRegex(ValueError, "automatic folds"):
            create_classifier("svm").fit(X, y, calibration_cv=3)
        badgroups = np.zeros(len(X), dtype=int)
        with self.assertRaisesRegex(ValueError, "groups disjoint"):
            create_classifier("svm").fit(X, y, groups=badgroups, calibration_cv=folds)
        bad = [(folds[0][0], np.concatenate((folds[0][1], folds[0][0][:1]))), *folds[1:]]
        with self.assertRaisesRegex(ValueError, "indices must be disjoint"):
            create_classifier("svm").fit(X, y, calibration_cv=bad)

    def test_isotonic_and_uncalibrated_svm_roundtrip(self):
        X, y, test, _, groups, folds = synthetic()
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as temporary:
            for method in (None, "isotonic"):
                model = create_classifier("svm", calibration=method)
                model.fit(X, y, **({"groups": groups, "calibration_cv": folds} if method else {}))
                path = Path(temporary) / f"{method}.classifier"
                model.save(path)
                loaded = load_classifier(path)
                np.testing.assert_array_equal(loaded.predict(test), model.predict(test))
                if method:
                    np.testing.assert_allclose(
                        loaded.predict_proba(test), model.predict_proba(test)
                    )
                else:
                    np.testing.assert_allclose(
                        loaded.decision_function(test), model.decision_function(test)
                    )

    def test_input_validation_aliases_and_original_numeric_labels(self):
        self.assertEqual(len(list_classifiers()), 11)
        self.assertEqual(classifier_info("rf")["name"], "random_forest")
        X, y, test, _, _, _ = synthetic(2)
        original = np.where(y == "synthetic_c", 50, -10)
        model = create_classifier("logreg").fit(X, original)
        np.testing.assert_array_equal(model.classes_, [-10, 50])
        self.assertTrue(set(model.predict(test)) <= {-10, 50})
        for bad in (np.zeros((2, 3, 4)), np.full((4, 6), np.nan), np.array([["text"]])):
            with self.assertRaises(ValueError):
                create_classifier("rf").fit(bad, np.arange(len(bad)))
        with self.assertRaisesRegex(ValueError, "at least two"):
            create_classifier("rf").fit(X, np.zeros(len(X)))
        with self.assertRaisesRegex(ValueError, "Expected 6"):
            model.predict(test[:, :3])
        with self.assertRaisesRegex(ValueError, "CPU"):
            create_classifier("xgboost", device="cuda")
        with self.assertRaisesRegex(ValueError, "validation splits"):
            create_classifier("sklearn_mlp", early_stopping=True)
        with self.assertRaisesRegex(TypeError, "Unsupported"):
            create_classifier("svm", probability=True)
        with self.assertRaises(TypeError):
            create_classifier("svm", kernel=lambda x: x)

    def test_atomic_overwrite_checksum_and_strict_skops_trust(self):
        X, y, test, _, _, _ = synthetic(2)
        model = create_classifier("logreg").fit(X, y)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "artifact.classifier"
            model.save(path)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                model.save(path)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(TypeError):
                model.save(path, overwrite="False")
            link = root / "linked.classifier"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                model.save(link, overwrite=True)
            self.assertTrue(link.is_symlink())
            self.assertEqual(path.read_bytes(), original)
            model.save(path, overwrite=True)
            np.testing.assert_array_equal(load_classifier(path).predict(test), model.predict(test))
            with ZipFile(path) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
            files["estimator.skops"] += b"corruption"
            with ZipFile(root / "corrupt.classifier", "w", compression=ZIP_STORED) as archive:
                for name, data in files.items():
                    archive.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_classifier(root / "corrupt.classifier")
            model.estimator_.unexpected = UnexpectedArtifactType()
            model.save(root / "untrusted.classifier")
            with self.assertRaisesRegex(ValueError, "Unsupported serialized estimator types"):
                load_classifier(root / "untrusted.classifier")
            self.assertFalse(list(root.glob(".*.tmp")))

    def test_artifact_uncompressed_size_budget_is_validated_before_loading(self):
        X, y, test, _, _, _ = synthetic(2)
        model = create_classifier("logreg").fit(X, y)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.classifier"
            model.save(path)
            with self.assertRaisesRegex(ValueError, "uncompressed size limit"):
                load_classifier(path, max_uncompressed_bytes=64)
            for budget in (True, False, -1, 0, 1.5, "100"):
                with (
                    self.subTest(budget=budget),
                    self.assertRaisesRegex(ValueError, "positive integer"),
                ):
                    load_classifier(path, max_uncompressed_bytes=budget)
            with ZipFile(path) as archive:
                size = sum(entry.file_size for entry in archive.infolist())
            np.testing.assert_array_equal(
                load_classifier(path, max_uncompressed_bytes=size).predict(test),
                model.predict(test),
            )


if __name__ == "__main__":
    unittest.main()
