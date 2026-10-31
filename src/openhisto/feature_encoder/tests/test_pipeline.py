from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import anndata
import numpy as np
from PIL import Image

from openhisto.feature_encoder import (
    EncodeConfig,
    EncoderError,
    EncoderSpec,
    FeatureReader,
    ModelDescriptor,
    SourceSpec,
    encode_slide,
    register_encoder,
)
from openhisto.patch_extraction import PatchArchiveWriter


STATE = {"prepared": 0, "calls": 0, "closed": 0, "fail_after": None, "bad": False}


class DummyEncoder:
    def __init__(self, spec, *, config, auth):
        STATE["prepared"] += 1
        self.dimension = 2 if spec.model == "test_two" else 3
        self.metadata = {
            "model_id": spec.model,
            "resolved_revision": spec.revision or "fixture-revision",
            "pooling": "mean_rgb",
            "options": spec.options,
        }
        self.usage = {}

    def encode(self, images):
        if STATE["fail_after"] is not None and STATE["calls"] >= STATE["fail_after"]:
            raise EncoderError("Synthetic interruption")
        STATE["calls"] += 1
        if STATE["bad"]:
            return np.zeros((1, self.dimension), dtype=np.float32)
        return np.asarray(
            [image.mean(axis=(0, 1))[: self.dimension] for image in images], dtype=np.float32
        )

    def close(self):
        STATE["closed"] += 1


for name, dimension in (("test_rgb", 3), ("test_two", 2)):
    register_encoder(
        ModelDescriptor(name, name, "test", "public", dimension, "synthetic-model"), DummyEncoder
    )


def archive(path, colors, *, mag=20, coordinate_offset=0):
    scale = 40 / mag
    with PatchArchiveWriter(
        path, patch_size=4, metadata={"target_mag": mag, "source_mag": 40}, overwrite=path.exists()
    ) as writer:
        for index, color in enumerate(colors):
            writer.append(
                np.full((4, 4, 3), color, dtype=np.uint8),
                (index * 8 + coordinate_offset, 0),
                (round(4 * scale), round(4 * scale)),
                (0, 0, 4, 4),
            )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="cpath-feature-pipeline-")
        self.root = Path(self.temporary.name)
        self.input = self.root / "patches.h5"
        self.output = self.root / "slide.h5ad"
        self.colors = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
        archive(self.input, self.colors)
        STATE.update(prepared=0, calls=0, closed=0, fail_after=None, bad=False)
        self.config = EncodeConfig(slide_id="synthetic-slide", batch_size=2)

    def tearDown(self):
        self.temporary.cleanup()

    def test_roundtrip_and_cached_rerun_never_prepares_or_requests_credentials(self):
        report = encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual((report.patches, report.completed, report.reused), (3, 3, 0))
        initial = self.output.read_bytes()
        initial_mtime = self.output.stat().st_mtime_ns
        with patch(
            "openhisto.feature_encoder.pipeline.prepare_model",
            side_effect=AssertionError("cached run prepared a model"),
        ):
            report = encode_slide(self.input, self.output, ["test_rgb"], self.config, auth=object())
        self.assertEqual((report.completed, report.reused), (0, 3))
        self.assertEqual(self.output.read_bytes(), initial)
        self.assertEqual(self.output.stat().st_mtime_ns, initial_mtime)
        adata = anndata.read_h5ad(self.output)
        self.assertIsNone(adata.X)
        self.assertEqual(adata.shape, (3, 0))
        with FeatureReader(self.output) as reader:
            np.testing.assert_array_equal(
                reader.features("test_rgb", 20), np.asarray(self.colors, dtype=np.float32)
            )
            metadata = reader.patch_metadata()
            self.assertEqual(metadata["mag"].tolist(), [20, 20, 20])
            self.assertEqual(metadata["x_level0"].tolist(), [0, 8, 16])
            run_metadata = reader.model_metadata("test_rgb", 20)["metadata"]
            self.assertIn("first_prepared_at_utc", run_metadata["execution"])
            self.assertEqual(
                run_metadata["execution"]["core_versions"]["anndata"], anndata.__version__
            )

    def test_new_model_and_patches_preserve_old_features_and_pending_masks(self):
        encode_slide(self.input, self.output, ["test_rgb", "test_two"], self.config)
        archive(self.input, self.colors + [(100, 110, 120), (130, 140, 150)])
        report = encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual((report.completed, report.reused, report.pending), (2, 3, 2))
        with FeatureReader(self.output) as reader:
            self.assertEqual(reader.features("test_rgb", 20).shape, (5, 3))
            with self.assertRaises(ValueError):
                reader.features("test_two", 20)
            values = reader.features("test_two", 20, require_complete=False)
            np.testing.assert_array_equal(
                values[:3], np.asarray(self.colors, dtype=np.float32)[:, :2]
            )
            self.assertTrue(np.isnan(values[3:]).all())
        report = encode_slide(self.input, self.output, ["test_two"], self.config)
        self.assertEqual((report.completed, report.reused, report.pending), (2, 3, 0))

    def test_two_magnifications_share_one_file_and_keep_query_alignment(self):
        other = self.root / "lower_mag.h5"
        archive(other, [(1, 2, 3), (4, 5, 6)], mag=10)
        report = encode_slide([self.input, other], self.output, ["test_rgb"], self.config)
        self.assertEqual(report.patches, 5)
        adata = anndata.read_h5ad(self.output)
        self.assertEqual(set(adata.uns["feats"]), {"10x", "20x"})
        self.assertEqual(len(adata.obsm), 2)
        with FeatureReader(self.output) as reader:
            np.testing.assert_array_equal(
                reader.features("test_rgb", 10),
                np.asarray([(1, 2, 3), (4, 5, 6)], dtype=np.float32),
            )
            self.assertEqual(reader.features("test_rgb", 20).shape, (3, 3))
            self.assertEqual(reader.patch_metadata(mag=10)["width_level0"].tolist(), [16, 16])
        self.assertEqual(list(self.root.glob("*.h5ad")), [self.output])
        self.assertFalse(Path(str(self.output) + ".lock").exists())

    def test_interruption_resumes_committed_rows_without_publishing_partial(self):
        archive(self.input, self.colors + [(100, 110, 120), (130, 140, 150)])
        STATE["fail_after"] = 1
        with self.assertRaisesRegex(EncoderError, "Synthetic interruption") as failure:
            encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual(
            (failure.exception.report.completed, failure.exception.report.pending), (2, 3)
        )
        self.assertEqual(failure.exception.report.failed, 2)
        self.assertFalse(self.output.exists())
        STATE["fail_after"] = None
        report = encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual((report.completed, report.reused), (3, 2))
        with FeatureReader(self.output) as reader:
            self.assertEqual(reader.features("test_rgb", 20).shape, (5, 3))

    def test_changed_pixels_or_geometry_stop_before_existing_results_change(self):
        encode_slide(self.input, self.output, ["test_rgb"], self.config)
        baseline = hashlib.sha256(self.output.read_bytes()).hexdigest()
        archive(self.input, [(11, 20, 30)] + self.colors[1:])
        with self.assertRaisesRegex(EncoderError, "changed"):
            encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual(hashlib.sha256(self.output.read_bytes()).hexdigest(), baseline)
        archive(self.input, self.colors, coordinate_offset=1)
        with self.assertRaisesRegex(EncoderError, "changed"):
            encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual(hashlib.sha256(self.output.read_bytes()).hexdigest(), baseline)

    def test_revision_and_normalization_create_separate_runs(self):
        encode_slide(self.input, self.output, ["test_rgb"], self.config)
        encode_slide(
            self.input,
            self.output,
            [EncoderSpec("test_rgb", revision="second-revision")],
            self.config,
        )
        report = encode_slide(
            self.input, self.output, [EncoderSpec("test_rgb", normalize=True)], self.config
        )
        self.assertEqual(report.completed, 3)
        with FeatureReader(self.output) as reader:
            self.assertEqual(len(reader.list_runs()), 3)
            with self.assertRaises(ValueError):
                reader.features("test_rgb", 20)
            vectors = reader.features("test_rgb", 20, run_key=report.runs[0])
            np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)

    def test_changed_magnification_keeps_identity_and_rejects_update(self):
        encode_slide(self.input, self.output, ["test_rgb"], self.config)
        baseline = self.output.read_bytes()
        archive(self.input, self.colors, mag=10)
        with self.assertRaisesRegex(EncoderError, "changed"):
            encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual(self.output.read_bytes(), baseline)

    def test_runtime_identity_change_cannot_mix_new_vectors_with_old(self):
        original_init = DummyEncoder.__init__

        def with_precision(precision):
            def initialize(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                instance.metadata.update(precision=precision, adapter_version=1)

            return initialize

        with patch.object(DummyEncoder, "__init__", with_precision("float32")):
            encode_slide(self.input, self.output, ["test_rgb"], self.config)
        baseline = self.output.read_bytes()
        archive(self.input, self.colors + [(100, 110, 120)])
        with patch.object(DummyEncoder, "__init__", with_precision("bfloat16")):
            with self.assertRaisesRegex(EncoderError, "runtime identity"):
                encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual(self.output.read_bytes(), baseline)
        # A new explicit intent creates a separate run and keeps the old one.
        with patch.object(DummyEncoder, "__init__", with_precision("bfloat16")):
            report = encode_slide(
                self.input,
                self.output,
                [EncoderSpec("test_rgb", options={"cache_tag": "new-runtime"})],
                self.config,
            )
        self.assertEqual(report.completed, 4)

    def test_midrun_source_change_is_detected_and_never_published(self):
        image = self.root / "image.png"
        Image.fromarray(np.full((4, 4, 3), 20, dtype=np.uint8)).save(image)
        manifest = self.root / "patches.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "patch_id": "a",
                    "image_path": image.name,
                    "slide_id": "synthetic-slide",
                    "mag": 20,
                }
            )
            + "\n"
        )

        def mutate(spec, *, config, auth):
            Image.fromarray(np.full((4, 4, 3), 21, dtype=np.uint8)).save(image)
            return DummyEncoder(spec, config=config, auth=auth)

        with patch("openhisto.feature_encoder.pipeline.prepare_model", side_effect=mutate):
            with self.assertRaisesRegex(EncoderError, "changed during"):
                encode_slide(manifest, self.output, ["test_rgb"], self.config)
        self.assertFalse(self.output.exists())

    def test_checkpoint_bundle_hash_tracks_preprocessing_and_weights(self):
        from openhisto.feature_encoder.pipeline import _checkpoint_hash

        directory = self.root / "checkpoint"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "preprocessor_config.json").write_text('{"size":224}')
        (directory / "model.safetensors").write_bytes(b"synthetic-weights")
        original = _checkpoint_hash(directory, "phikon")
        (directory / "preprocessor_config.json").write_text('{"size":448}')
        self.assertNotEqual(_checkpoint_hash(directory, "phikon"), original)

    def test_secret_aliases_and_symlink_outputs_are_rejected(self):
        for key in ("apiKey", "x-goog-api-key", "x-api-key", "apiToken", "awsSecretAccessKey"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Credentials"):
                EncoderSpec("test_rgb", options={key: "synthetic-secret"})
        target = self.root / "target.h5ad"
        target.write_bytes(b"unrelated-file")
        self.output.symlink_to(target)
        with self.assertRaisesRegex(EncoderError, "symlinks"):
            encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual(target.read_bytes(), b"unrelated-file")

    def test_preparation_failure_reports_all_pending_rows(self):
        with patch(
            "openhisto.feature_encoder.pipeline.prepare_model",
            side_effect=EncoderError("Synthetic unavailable backend"),
        ):
            with self.assertRaises(EncoderError) as failure:
                encode_slide(self.input, self.output, ["test_rgb", "test_two"], self.config)
        self.assertEqual(failure.exception.report.pending, 6)

    def test_empty_archive_creates_valid_h5ad_without_loading_a_model(self):
        archive(self.input, [])
        with patch(
            "openhisto.feature_encoder.pipeline.prepare_model",
            side_effect=AssertionError("empty input prepared model"),
        ):
            report = encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual(report.patches, 0)
        adata = anndata.read_h5ad(self.output)
        self.assertEqual(adata.shape, (0, 0))
        self.assertEqual(next(iter(adata.obsm.values())).shape, (0, 3))
        archive(self.input, self.colors)
        report = encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertEqual(report.completed, 3)

    def test_fused_or_misaligned_encoder_response_is_rejected(self):
        STATE["bad"] = True
        with self.assertRaisesRegex(EncoderError, "expected"):
            encode_slide(self.input, self.output, ["test_rgb"], self.config)
        self.assertFalse(self.output.exists())

    def test_manifest_reordering_and_renaming_with_stable_source_ids_reuses_features(self):
        images = []
        for index, color in enumerate(self.colors):
            path = self.root / f"image_{index}.png"
            Image.fromarray(np.full((4, 4, 3), color, dtype=np.uint8)).save(path)
            images.append(
                {
                    "patch_id": f"patch-{index}",
                    "image_path": path.name,
                    "slide_id": "synthetic-slide",
                    "mag": 20,
                    "description": f"synthetic-extra-{index}",
                }
            )
        manifest = self.root / "patches.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in images) + "\n")
        source = SourceSpec(manifest, source_id="stable-source")
        encode_slide(source, self.output, ["test_rgb"], self.config)
        manifest.write_text("\n".join(json.dumps(row) for row in reversed(images)) + "\n")
        report = encode_slide(source, self.output, ["test_rgb"], self.config)
        self.assertEqual((report.completed, report.reused), (0, 3))
        with FeatureReader(self.output) as reader:
            np.testing.assert_array_equal(
                reader.features("test_rgb", 20), np.asarray(self.colors, dtype=np.float32)
            )
            frame = reader.patch_metadata()
            self.assertTrue(frame["x_level0"].isna().all())
            self.assertIn("synthetic-extra-0", frame.iloc[0]["metadata_json"])

    def test_manifest_metadata_policy_corrects_metadata_without_reencoding(self):
        image = self.root / "patch.png"
        Image.fromarray(np.full((4, 4, 3), self.colors[0], dtype=np.uint8)).save(image)
        row = {
            "patch_id": "patch-a",
            "image_path": image.name,
            "mag": 20,
            "quality": "unchecked",
            "x_level0": 0,
        }
        manifest = self.root / "patches.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        encode_slide(manifest, self.output, ["test_rgb"], self.config)
        baseline = self.output.read_bytes()
        with FeatureReader(self.output) as reader:
            original_features = reader.features("test_rgb", 20)
            original_runs = reader.list_runs()
        row["quality"] = "reviewed"
        manifest.write_text(json.dumps(row) + "\n")
        with patch(
            "openhisto.feature_encoder.pipeline.prepare_model",
            side_effect=AssertionError("metadata correction loaded a model"),
        ):
            preserved = encode_slide(manifest, self.output, ["test_rgb"], self.config)
            self.assertEqual((preserved.metadata_preserved, preserved.metadata_updated), (1, 0))
            self.assertEqual(self.output.read_bytes(), baseline)
            with self.assertRaisesRegex(EncoderError, "Optional patch metadata changed"):
                encode_slide(
                    manifest,
                    self.output,
                    ["test_rgb"],
                    replace(self.config, metadata_policy="error"),
                )
            self.assertEqual(self.output.read_bytes(), baseline)
            updated = encode_slide(
                manifest, self.output, ["test_rgb"], replace(self.config, metadata_policy="update")
            )
        self.assertEqual(
            (
                updated.completed,
                updated.reused,
                updated.metadata_updated,
                updated.metadata_preserved,
            ),
            (0, 1, 1, 0),
        )
        with FeatureReader(self.output) as reader:
            self.assertEqual(
                json.loads(reader.patch_metadata().iloc[0]["metadata_json"])["quality"], "reviewed"
            )
            np.testing.assert_array_equal(reader.features("test_rgb", 20), original_features)
            self.assertEqual(reader.list_runs(), original_runs)
        corrected = self.output.read_bytes()
        row["x_level0"] = 1
        manifest.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(EncoderError, "immutable geometry"):
            encode_slide(
                manifest, self.output, ["test_rgb"], replace(self.config, metadata_policy="update")
            )
        self.assertEqual(self.output.read_bytes(), corrected)

    def test_missing_magnification_and_slide_identity_are_not_guessed(self):
        other = self.root / "other.h5"
        archive(other, self.colors, mag=10)
        with self.assertRaisesRegex(ValueError, "slide_id"):
            encode_slide([self.input, other], self.output, ["test_rgb"])
        path = self.root / "missing-mag.csv"
        path.write_text("patch_id,image_path,slide_id\na,missing.png,synthetic-slide\n")
        with self.assertRaisesRegex(ValueError, "mag"):
            encode_slide(path, self.output, ["test_rgb"], self.config)

    def test_out_of_memory_halves_the_batch_and_retries_the_same_rows(self):
        class OutOfMemoryError(RuntimeError):
            pass

        original_encode = DummyEncoder.encode
        batches = []

        def encode(instance, images):
            batches.append(len(images))
            if len(images) > 1:
                raise OutOfMemoryError("synthetic")
            return original_encode(instance, images)

        with patch.object(DummyEncoder, "encode", encode):
            report = encode_slide(
                self.input,
                self.output,
                ["test_rgb"],
                EncodeConfig(slide_id="synthetic-slide", batch_size=4),
            )
        self.assertEqual(report.completed, 3)
        self.assertEqual(batches, [3, 1, 1, 1])
        with FeatureReader(self.output) as reader:
            np.testing.assert_array_equal(
                reader.features("test_rgb", 20), np.asarray(self.colors, dtype=np.float32)
            )

    def test_usage_accumulates_across_resumed_runs(self):
        original_encode = DummyEncoder.encode

        def counting(instance, images):
            instance.usage["images"] = instance.usage.get("images", 0) + len(images)
            return original_encode(instance, images)

        with patch.object(DummyEncoder, "encode", counting):
            encode_slide(self.input, self.output, ["test_rgb"], self.config)
            archive(self.input, self.colors + [(100, 110, 120), (130, 140, 150)])
            encode_slide(self.input, self.output, ["test_rgb"], self.config)
        with FeatureReader(self.output) as reader:
            self.assertEqual(reader.model_metadata("test_rgb", 20)["usage"]["images"], 5)

    def test_threaded_patch_loading_preserves_batch_order(self):
        config = EncodeConfig(slide_id="synthetic-slide", workers=2)
        encode_slide(self.input, self.output, ["test_rgb"], config)
        with FeatureReader(self.output) as reader:
            np.testing.assert_array_equal(
                reader.features("test_rgb", 20), np.asarray(self.colors, dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()
