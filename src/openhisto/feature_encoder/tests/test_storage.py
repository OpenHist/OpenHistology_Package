"""Synthetic, CPU-only storage tests with ordinary AnnData interoperability."""

import hashlib
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import patch
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from openhisto.feature_encoder.storage import FeatureReader, FeatureStore


def observations(ids=("p0", "p1"), mags=None):
    mags = [20.0] * len(ids) if mags is None else mags
    rows = []
    for identifier, mag in zip(ids, mags):
        index = int(identifier.removeprefix("p"))
        rows.append(
            dict(
                patch_id=identifier,
                slide_id="synthetic-slide",
                source_id="synthetic-source",
                source_type="archive",
                source_index=index,
                mag=float(mag),
                content_hash=f"hash-{identifier}",
                x_level0=float(index * 16),
                y_level0=0.0,
                width_level0=16.0,
                height_level0=16.0,
                patch_width=16,
                patch_height=16,
                valid_x0=0,
                valid_y0=0,
                valid_x1=16,
                valid_y1=16,
                source_mag=40.0,
                metadata_json="{}",
            )
        )
    if not rows:
        result = observations().iloc[:0].copy()
        return result
    return pd.DataFrame(rows, index=pd.Index(ids, dtype=str))


def _spawned_reader_check(reader, result_queue):
    """Spawn entry point: the serialized reader must not carry an open handle."""
    try:
        handle_was_absent = reader._file is None
        with reader:
            result_queue.put(
                {
                    "pid": os.getpid(),
                    "reader_pid": reader._pid,
                    "handle_was_absent": handle_was_absent,
                    "features": reader.features("model-a", 20, rows=[1, 0]).tolist(),
                    "ids": reader.patch_metadata(rows=[1, 0]).index.tolist(),
                }
            )
    except BaseException as exc:
        result_queue.put({"error": type(exc).__name__ + ": " + str(exc)})


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(warnings.catch_warnings())
        warnings.filterwarnings(
            "ignore",
            message="The 'future.infer_string' option is deprecated",
            module=r"anndata\..*",
        )
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "slide.h5ad"
        self.sources = [{"source_id": "synthetic-source", "type": "archive"}]

    def reconcile(self, store, obs=None):
        return store.reconcile(
            observations() if obs is None else obs, slide_id="synthetic-slide", sources=self.sources
        )

    def create(self, *, model="model-a", dimension=3, obs=None):
        with FeatureStore(self.output) as store:
            self.reconcile(store, obs)
            key = store.ensure_run(model, 20, "request-a", {"revision": "fixed-v1"}, dimension)
            positions = store.pending_rows(key)
            store.write_batch(
                key,
                positions,
                np.arange(len(positions) * dimension).reshape(len(positions), dimension),
            )
        return key

    def test_standard_anndata_roundtrip_with_multiple_mags_and_dimensions(self):
        obs = observations(("p0", "p1", "p2", "p3"), [20, 20, 10, 10])
        with FeatureStore(self.output) as store:
            self.reconcile(store, obs)
            a = store.ensure_run("model-a", 20, "request-a", {"revision": "a1"}, 3)
            b = store.ensure_run("model-b", 20, "request-b", {"revision": "b1"}, 5)
            c = store.ensure_run("model-a", 10, "request-c", {"revision": "a1"}, 3)
            store.write_batch(a, [0, 1], np.ones((2, 3)))
            store.write_batch(b, [0, 1], np.full((2, 5), 2))
            store.write_batch(c, [2, 3], np.full((2, 3), 3))
        result = ad.read_h5ad(self.output)
        self.assertEqual(result.shape, (4, 0))
        self.assertIsNone(result.X)
        self.assertEqual(result.obsm[a].shape, (4, 3))
        self.assertEqual(result.obsm[b].shape, (4, 5))
        self.assertTrue(np.isnan(result.obsm[a][2:]).all())
        self.assertTrue(np.isnan(result.obsm[c][:2]).all())
        np.testing.assert_array_equal(
            result.obs["encoded_" + a].to_numpy(), [True, True, False, False]
        )
        self.assertIn(a, result.uns["feats"]["20x"]["model-a"]["runs"])
        with FeatureReader(self.output) as reader:
            self.assertEqual(len(reader.list_runs()), 3)
            np.testing.assert_array_equal(reader.features("model-b", 20), np.full((2, 5), 2))
            self.assertEqual(reader.model_metadata("model-b", 20)["feature_index"], list(range(5)))
            self.assertEqual(reader.patch_metadata(10).index.tolist(), ["p2", "p3"])

    def test_noop_preserves_published_bytes_and_mtime(self):
        key = self.create()
        before = self.output.read_bytes()
        mtime = self.output.stat().st_mtime_ns
        with FeatureStore(self.output) as store:
            self.reconcile(store)
            self.assertEqual(
                store.ensure_run("model-a", 20, "request-a", {"revision": "fixed-v1"}, 3), key
            )
            self.assertEqual(store.pending_rows(key).tolist(), [])
            store.set_usage(key, {})
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self.output.stat().st_mtime_ns, mtime)
        self.assertEqual([p.name for p in self.output.parent.iterdir()], [self.output.name])

    def test_append_preserves_rows_and_marks_nonselected_models_pending(self):
        a = self.create()
        with FeatureStore(self.output) as store:
            b = store.ensure_run("model-b", 20, "request-b", {}, 2)
            store.write_batch(b, [0, 1], np.ones((2, 2)))
        with FeatureStore(self.output) as store:
            # Previously seen rows may be omitted or reordered; old order stays.
            unified = self.reconcile(store, observations(("p2", "p0")))
            self.assertEqual(unified.index.tolist(), ["p0", "p1", "p2"])
            self.assertEqual(store.pending_rows(a).tolist(), [2])
            self.assertEqual(store.pending_rows(b).tolist(), [2])
            store.write_batch(a, [2], np.full((1, 3), 8))
        with FeatureReader(self.output) as reader:
            self.assertEqual(reader.features("model-a", 20).shape, (3, 3))
            with self.assertRaisesRegex(ValueError, "pending"):
                reader.features("model-b", 20)
            pending = reader.features("model-b", 20, require_complete=False)
            self.assertTrue(np.isnan(pending[2]).all())
            self.assertEqual(reader.features("model-b", 20, rows=[0, 1]).shape, (2, 2))

    def test_other_magnification_rows_are_not_pending_for_existing_run(self):
        key = self.create()
        with FeatureStore(self.output) as store:
            self.reconcile(store, observations(("p2",), [10]))
            self.assertEqual(store.pending_rows(key).tolist(), [])
        with FeatureReader(self.output) as reader:
            self.assertEqual(reader.features("model-a", 20).shape, (2, 3))
            self.assertEqual(reader.patch_metadata().shape[0], 3)

    def test_partial_commits_resume_without_recomputing_or_replacing_published(self):
        key = self.create()
        baseline = self.output.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            with FeatureStore(self.output) as store:
                self.reconcile(store, observations(("p0", "p1", "p2", "p3")))
                store.write_batch(key, [2], np.ones((1, 3)))
                raise RuntimeError("synthetic interruption")
        self.assertEqual(self.output.read_bytes(), baseline)
        self.assertTrue(store.partial_path.exists())
        self.assertFalse(store.lock_path.exists())
        with FeatureStore(self.output) as resumed:
            self.reconcile(resumed, observations(("p0", "p1", "p2", "p3")))
            self.assertEqual(resumed.pending_rows(key).tolist(), [3])
            resumed.write_batch(key, [3], np.full((1, 3), 9))
        self.assertFalse(resumed.partial_path.exists())
        with FeatureReader(self.output) as reader:
            self.assertEqual(reader.features("model-a", 20).shape, (4, 3))

    def test_failed_new_store_keeps_resumable_partial_and_rejects_reader(self):
        with self.assertRaises(RuntimeError):
            with FeatureStore(self.output) as store:
                self.reconcile(store)
                key = store.ensure_run("model-a", 20, "request-a", {}, 2)
                store.write_batch(key, [0], np.ones((1, 2)))
                raise RuntimeError("interruption")
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "partial"):
            with FeatureReader(store.partial_path):
                pass
        with FeatureStore(self.output) as store:
            self.assertEqual(store.pending_rows(key).tolist(), [1])
            store.write_batch(key, [1], np.ones((1, 2)))
        self.assertTrue(self.output.exists())

    def test_completed_stage_is_resumed_after_publication_failure(self):
        key = self.create()
        baseline = self.output.read_bytes()
        original = Path.replace

        def interrupted_replace(source, target):
            if source.name.endswith(".partial.h5ad"):
                raise OSError("synthetic publication failure")
            return original(source, target)

        with self.assertRaisesRegex(OSError, "publication failure"):
            with patch.object(Path, "replace", interrupted_replace):
                with FeatureStore(self.output) as store:
                    self.reconcile(store, observations(("p0", "p1", "p2")))
                    store.write_batch(key, [2], np.ones((1, 3)))
        self.assertEqual(self.output.read_bytes(), baseline)
        with FeatureStore(self.output) as resumed:
            self.assertEqual(resumed.pending_rows(key).tolist(), [])
        with FeatureReader(self.output) as reader:
            self.assertEqual(reader.features("model-a", 20).shape, (3, 3))

    def test_reconcile_rejects_changed_content_geometry_or_slide(self):
        self.create()
        before = self.output.read_bytes()
        for column, value in (
            ("content_hash", "changed"),
            ("x_level0", 999.0),
            ("source_id", "different"),
            ("source_mag", np.nan),
        ):
            with self.subTest(column=column):
                changed = observations()
                changed.loc["p0", column] = value
                with self.assertRaisesRegex(ValueError, "immutable"):
                    with FeatureStore(self.output) as store:
                        self.reconcile(store, changed)
        with self.assertRaisesRegex(ValueError, "slide"):
            with FeatureStore(self.output) as store:
                store.reconcile(observations(), slide_id="other-slide", sources=self.sources)
        self.assertEqual(self.output.read_bytes(), before)

    def test_optional_metadata_is_preserved_without_reencoding(self):
        self.create()
        changed = observations()
        changed["metadata_json"] = '{"new":true}'
        with FeatureStore(self.output) as store:
            self.reconcile(store, changed)
            self.assertEqual(store.obs.loc["p0", "metadata_json"], "{}")

    def test_lock_is_exclusive_and_stale_locks_are_not_deleted(self):
        with FeatureStore(self.output) as first:
            with self.assertRaisesRegex(FileExistsError, "locked"):
                with FeatureStore(self.output):
                    pass
            self.assertTrue(first.lock_path.exists())
        first.lock_path.write_text("synthetic stale owner")
        with self.assertRaises(FileExistsError):
            with FeatureStore(self.output):
                pass
        self.assertEqual(first.lock_path.read_text(), "synthetic stale owner")

    def test_incompatible_partial_baseline_is_preserved(self):
        self.create()
        with self.assertRaises(RuntimeError):
            with FeatureStore(self.output) as store:
                self.reconcile(store, observations(("p2",)))
                raise RuntimeError("interruption")
        replacement = self.output.with_name("other.h5ad")
        with FeatureStore(replacement) as alternative:
            self.reconcile(alternative)
        shutil.copyfile(replacement, self.output)
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "baseline"):
            with FeatureStore(self.output):
                pass
        self.assertEqual(self.output.read_bytes(), before)
        self.assertTrue(store.partial_path.exists())
        self.assertFalse(store.lock_path.exists())

    def test_empty_input_is_standard_and_does_not_require_vectors(self):
        with FeatureStore(self.output) as store:
            self.reconcile(store, observations(()))
            key = store.ensure_run("model-a", 20, "empty-request", {}, 7)
            self.assertEqual(store.pending_rows(key).tolist(), [])
        result = ad.read_h5ad(self.output)
        self.assertEqual(result.shape, (0, 0))
        self.assertEqual(result.obsm[key].shape, (0, 7))
        with FeatureReader(self.output) as reader:
            self.assertEqual(reader.features("model-a", 20).shape, (0, 7))

    def test_metadata_promotion_identity_guard_and_usage(self):
        with FeatureStore(self.output) as store:
            self.reconcile(store)
            key = store.ensure_run("model-a", 20, "request-a", {"resolved": False}, 2)
            store.set_run_metadata(key, {"resolved": True, "revision": "fixed"})
            store.write_batch(key, [0, 1], np.ones((2, 2)))
            store.set_run_metadata(key, {"revision": "fixed", "resolved": True})
            with self.assertRaisesRegex(ValueError, "identity"):
                store.set_run_metadata(key, {"revision": "changed"})
            store.set_usage(key, {"images": 2, "input_tokens": 100})
        with FeatureReader(self.output) as reader:
            info = reader.model_metadata("model-a", 20)
            self.assertEqual(info["metadata"]["revision"], "fixed")
            self.assertEqual(info["usage"]["images"], 2)

    def test_multiple_intents_require_explicit_run_selection(self):
        a = self.create()
        with FeatureStore(self.output) as store:
            b = store.ensure_run("model-a", 20, "another-request", {}, 2)
            self.assertEqual(len(store.find_runs("model-a", 20, "another-request")), 1)
        with FeatureReader(self.output) as reader:
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                reader.features("model-a", 20)
            self.assertEqual(reader.features("model-a", 20, run_key=a).shape, (2, 3))
            with self.assertRaisesRegex(ValueError, "pending"):
                reader.features("model-a", 20, run_key=b)

    def test_batch_validation_and_unsorted_writes(self):
        with FeatureStore(self.output) as store:
            self.reconcile(store)
            key = store.ensure_run("model-a", 20, "request-a", {}, 2)
            for rows, features in (
                ([0, 0], np.ones((2, 2))),
                ([2], np.ones((1, 2))),
                ([0], [[np.nan, 1]]),
                ([0], [[1e300, 1]]),
                ([0], np.ones((1, 3))),
            ):
                with self.subTest(rows=rows, shape=np.shape(features)):
                    with self.assertRaises(ValueError):
                        store.write_batch(key, rows, features)
            store.write_batch(key, [1, 0], [[3, 4], [1, 2]])
            with self.assertRaisesRegex(ValueError, "overwritten"):
                store.write_batch(key, [0], [[9, 9]])
        with FeatureReader(self.output) as reader:
            np.testing.assert_array_equal(reader.features("model-a", 20), [[1, 2], [3, 4]])

    def test_selective_reader_never_uses_backed_anndata_or_unrequested_features(self):
        key = self.create(obs=observations(("p0", "p1", "p2")))
        with FeatureStore(self.output) as store:
            store.ensure_run("model-b", 20, "request-b", {}, 100)
        original = h5py.Dataset.__getitem__
        feature_reads = []

        def checked_getitem(dataset, selection):
            if dataset.name.startswith("/obsm/"):
                self.assertEqual(dataset.name, "/obsm/" + key)
                feature_reads.append(selection)
            return original(dataset, selection)

        with (
            patch.object(h5py.Dataset, "__getitem__", checked_getitem),
            patch("anndata.read_h5ad", side_effect=AssertionError("no eager AnnData reads")),
        ):
            with FeatureReader(self.output) as reader:
                values = reader.features("model-a", 20, rows=[2, 0, 2])
                self.assertEqual(values.shape, (3, 3))
                np.testing.assert_array_equal(values[0], values[2])
                batches = list(reader.iter_batches("model-a", 20, batch_size=2, rows=[2, 0, 1]))
                self.assertEqual([len(frame) for _, frame in batches], [2, 1])
                self.assertEqual(
                    reader.patch_metadata(rows=["p2", "p0"]).index.tolist(), ["p2", "p0"]
                )
        self.assertTrue(feature_reads)

    def test_adding_model_does_not_read_older_feature_payloads(self):
        self.create()
        original = h5py.Dataset.__getitem__

        def checked_getitem(dataset, selection):
            if dataset.name.startswith("/obsm/"):
                raise AssertionError("Updating a model must not load older feature payloads")
            return original(dataset, selection)

        with patch.object(h5py.Dataset, "__getitem__", checked_getitem):
            with FeatureStore(self.output) as store:
                key = store.ensure_run("model-b", 20, "request-b", {}, 2)
                store.write_batch(key, [0, 1], np.ones((2, 2)))

    def test_reader_pickle_excludes_live_handles(self):
        self.create()
        with FeatureReader(self.output) as reader:
            clone = pickle.loads(pickle.dumps(reader))
            self.assertIsNone(clone._file)
            with clone:
                self.assertEqual(clone.features("model-a", 20).shape, (2, 3))

    def test_spawned_reader_uses_independent_handle_and_preserves_order(self):
        self.create()
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        process = None
        try:
            with FeatureReader(self.output) as parent:
                original_handle = parent._file
                expected = parent.features("model-a", 20)
                process = context.Process(
                    target=_spawned_reader_check, args=(parent, result_queue), daemon=True
                )
                process.start()
                result = result_queue.get(timeout=15)
                process.join(timeout=15)
                self.assertFalse(
                    process.is_alive(), "Spawned reader did not exit within its timeout"
                )
                self.assertEqual(process.exitcode, 0)
                self.assertNotIn("error", result)
                self.assertTrue(result["handle_was_absent"])
                self.assertNotEqual(result["pid"], os.getpid())
                self.assertEqual(result["reader_pid"], result["pid"])
                self.assertEqual(parent._pid, os.getpid())
                self.assertIs(parent._file, original_handle)
                self.assertTrue(original_handle.id.valid)
                self.assertEqual(result["ids"], ["p1", "p0"])
                np.testing.assert_array_equal(result["features"], expected[[1, 0]])
                np.testing.assert_array_equal(parent.features("model-a", 20), expected)
        finally:
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)
            result_queue.close()
            result_queue.join_thread()

    def test_rejects_unrelated_h5ad_and_symlink_destinations(self):
        ad.AnnData(X=np.ones((2, 3))).write_h5ad(self.output)
        digest = hashlib.sha256(self.output.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "compatible"):
            with FeatureStore(self.output):
                pass
        self.assertEqual(hashlib.sha256(self.output.read_bytes()).hexdigest(), digest)
        alias = self.output.with_name("alias.h5ad")
        alias.symlink_to(self.output)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            with FeatureStore(alias):
                pass


if __name__ == "__main__":
    unittest.main()
