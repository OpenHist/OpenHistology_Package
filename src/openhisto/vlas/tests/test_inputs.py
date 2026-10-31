import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from openhisto.patch_extraction import PatchArchiveWriter
from openhisto.vlas.config import SourceSpec
from openhisto.vlas.inputs import prepare_inputs


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="cpath-vlas-inputs-")
        self.root = Path(self.temporary.name)
        self.config = SimpleNamespace(slide_id="TCGA:synthetic")
        self.archive = self.root / "patches.h5"
        with PatchArchiveWriter(
            self.archive, patch_size=4, metadata={"target_mag": 20, "source_mag": 40}
        ) as writer:
            for i, color in enumerate((20, 70, 120)):
                writer.append(
                    np.full((4, 4, 3), color, dtype=np.uint8), (i * 8 - 2, 10), (8, 8), (1, 0, 4, 4)
                )

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self, rows, suffix=".jsonl"):
        path = self.root / ("patches" + suffix)
        if suffix == ".csv":
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def image_row(self, patch_id="patch-a", color=30, **extra):
        image = self.root / (patch_id + ".png")
        Image.fromarray(np.full((3, 5, 3), color, dtype=np.uint8)).save(image)
        return {
            "patch_id": patch_id,
            "image_path": image.name,
            "slide_id": self.config.slide_id,
            "mag": 20,
            **extra,
        }

    def test_archive_geometry_identity_and_hash_are_feature_compatible(self):
        with prepare_inputs(SourceSpec(self.archive, source_id="grid-20x"), self.config) as patches:
            self.assertEqual(len(patches), 3)
            self.assertEqual(patches.slide_id, self.config.slide_id)
            row = patches.rows[1]
            identity = json.dumps(
                ["TCGA:synthetic", "grid-20x", "1"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            self.assertEqual(row["patch_key"], "p_" + hashlib.sha256(identity.encode()).hexdigest())
            image = np.full((4, 4, 3), 70, dtype=np.uint8)
            expected_hash = hashlib.sha256(
                np.asarray(image.shape, dtype="<i8").tobytes() + image.tobytes()
            ).hexdigest()
            self.assertEqual(row["content_hash"], expected_hash)
            self.assertEqual((row["patch_id"], row["source_index"], row["mag"]), ("1", 1, 20))
            self.assertEqual(
                tuple(
                    row[name] for name in ("x_level0", "y_level0", "width_level0", "height_level0")
                ),
                (6, 10, 8, 8),
            )
            self.assertEqual(
                tuple(row[name] for name in ("valid_x0", "valid_y0", "valid_x1", "valid_y1")),
                (1, 0, 4, 4),
            )
            self.assertEqual(patches.sources[0]["extraction"]["source_mag"], 40)

    def test_requested_order_and_threaded_loading_keep_patch_alignment(self):
        with prepare_inputs(self.archive, self.config) as patches:
            keys = [patches.rows[i]["patch_key"] for i in (2, 0, 1, 2)]
            images = patches.read_images(keys, workers=3)
            self.assertEqual([int(image[0, 0, 0]) for image in images], [120, 20, 70, 120])
            self.assertEqual(patches.read_images([], workers=3), [])
            with self.assertRaisesRegex(ValueError, "Unknown patch key"):
                patches.read_images(["absent"])

    def test_manifest_reordering_and_renaming_keep_stable_keys(self):
        rows = [self.image_row("é-a", color=10, quality="synthetic"), self.image_row("b", color=50)]
        path = self.manifest(rows)
        with prepare_inputs(SourceSpec(path, source_id="stable"), self.config) as patches:
            original = {row["patch_id"]: row for row in patches.rows}
            self.assertIsNone(patches.rows[0]["x_level0"])
            self.assertIsNone(patches.rows[0]["source_mag"])
            self.assertEqual(patches.rows[0]["source_index"], -1)
            self.assertEqual(json.loads(patches.rows[0]["metadata_json"])["quality"], "synthetic")
        path.write_text("\n".join(json.dumps(row) for row in reversed(rows)), encoding="utf-8")
        renamed = path.with_name("renamed.jsonl")
        path.rename(renamed)
        with prepare_inputs(SourceSpec(renamed, source_id="stable"), self.config) as patches:
            self.assertEqual([row["patch_id"] for row in patches.rows], ["b", "é-a"])
            for row in patches.rows:
                self.assertEqual(row, original[row["patch_id"]])

    def test_csv_source_defaults_and_manifest_geometry(self):
        row = self.image_row(
            x_level0=-8,
            y_level0=16,
            width_level0=10,
            height_level0=6,
            patch_index=42,
            valid_x0=1,
            valid_y0=0,
            valid_x1=5,
            valid_y1=3,
        )
        row.pop("slide_id")
        row.pop("mag")
        path = self.manifest([row], ".csv")
        with prepare_inputs(SourceSpec(path, source_id="csv", mag=10), self.config) as patches:
            item = patches.rows[0]
            self.assertEqual((item["source_index"], item["mag"], item["x_level0"]), (42, 10, -8))
            self.assertEqual((item["valid_x0"], item["valid_x1"]), (1, 5))
            json.dumps(patches.rows, allow_nan=False)

    def test_alpha_composites_on_white(self):
        image = self.root / "alpha.png"
        Image.new("RGBA", (3, 2), (0, 0, 0, 0)).save(image)
        path = self.manifest([{"patch_id": "alpha", "image_path": image.name, "mag": 20}])
        with prepare_inputs(path, self.config) as patches:
            rgb = patches.read_images(patches.rows[0]["patch_key"])[0]
            np.testing.assert_array_equal(rgb, np.full((2, 3, 3), 255, dtype=np.uint8))

    def test_structural_source_spec_needs_no_feature_encoder_import(self):
        source = SimpleNamespace(path=self.archive, source_id="structural", mag=20)
        with prepare_inputs(source, self.config) as patches:
            self.assertEqual(patches.rows[0]["source_id"], "structural")

    def test_midrun_manifest_content_change_is_rejected(self):
        row = self.image_row()
        with prepare_inputs(self.manifest([row]), self.config) as patches:
            Image.new("RGB", (5, 3), (200, 200, 200)).save(self.root / row["image_path"])
            with self.assertRaisesRegex(ValueError, "changed during"):
                patches.read_images([patches.rows[0]["patch_key"]])

    def test_duplicate_patch_identity_is_rejected(self):
        row = self.image_row()
        with self.assertRaisesRegex(ValueError, "Duplicate patch identity"):
            prepare_inputs(self.manifest([row, row]), self.config)

    def test_slide_and_magnification_mismatches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Archive magnification disagrees"):
            prepare_inputs(SourceSpec(self.archive, mag=10), self.config)
        row = self.image_row(slide_id="another-slide")
        with self.assertRaisesRegex(ValueError, "slide_id disagrees"):
            prepare_inputs(self.manifest([row]), self.config)
        row = self.image_row()
        with self.assertRaisesRegex(ValueError, "Manifest magnification disagrees"):
            prepare_inputs(SourceSpec(self.manifest([row]), mag=10), self.config)

    def test_multiple_archives_require_explicit_slide_id(self):
        with self.assertRaisesRegex(ValueError, "Combining patch archives requires"):
            prepare_inputs([self.archive, self.archive], SimpleNamespace(slide_id=None))

    def test_high_bit_depth_and_invalid_bounds_are_rejected(self):
        high = self.root / "high.tif"
        Image.fromarray(np.full((2, 2), 2000, dtype=np.uint16)).save(high)
        with self.assertRaisesRegex(ValueError, "high-bit-depth"):
            prepare_inputs(
                self.manifest([{"patch_id": "high", "image_path": high.name, "mag": 20}]),
                self.config,
            )
        row = self.image_row(valid_bounds=[0, 0, 6, 3])
        with self.assertRaisesRegex(ValueError, "valid bounds"):
            prepare_inputs(self.manifest([row]), self.config)

    def test_whole_slide_manifest_is_not_treated_as_patch_input(self):
        slide = {
            "slide_id": "sample",
            "namespaced_slide_id": "TCGA:sample",
            "local_path": "slides/sample.svs",
            "status": "complete",
        }
        with self.assertRaisesRegex(ValueError, "whole-slide dataset manifests"):
            prepare_inputs(self.manifest([slide]), self.config)
        with self.assertRaisesRegex(ValueError, "patch archives or CSV/JSONL"):
            prepare_inputs(self.root / self.image_row()["image_path"], self.config)

    def test_empty_archive_preserves_slide_source_and_magnification(self):
        empty = self.root / "empty.h5"
        with PatchArchiveWriter(empty, patch_size=4, metadata={"target_mag": 10}):
            pass
        with prepare_inputs(empty, self.config) as patches:
            self.assertEqual(patches.rows, [])
            self.assertEqual(patches.slide_id, self.config.slide_id)
            self.assertEqual(patches.sources[0]["magnifications"], [10])

    def test_optional_feature_encoder_identity_hash_geometry_parity(self):
        try:
            from openhisto.feature_encoder import EncodeConfig, SourceSpec as FeatureSourceSpec
            from openhisto.feature_encoder.inputs import prepare_inputs as prepare_features
        except ImportError as exc:
            self.skipTest(f"Optional feature_encoder core dependencies unavailable: {exc}")
        manifests = [
            self.image_row("é-a", color=10, x_level0=4, y_level0=8),
            self.image_row("b", color=50),
        ]
        path = self.manifest(manifests)
        for source in (self.archive, path):
            with self.subTest(source=source.name):
                feature_spec = FeatureSourceSpec(source, source_id="stable-source")
                with prepare_features(
                    feature_spec, EncodeConfig(slide_id=self.config.slide_id)
                ) as features:
                    # Reordering is intentional: joins must use keys, not position.
                    if source == path:
                        path.write_text(
                            "\n".join(json.dumps(row) for row in reversed(manifests)),
                            encoding="utf-8",
                        )
                    with prepare_inputs(feature_spec, self.config) as descriptions:
                        self.assertEqual(
                            set(features.obs.index), {row["patch_key"] for row in descriptions.rows}
                        )
                        for row in descriptions.rows:
                            feature = features.obs.loc[row["patch_key"]]
                            for name, value in row.items():
                                if name == "patch_key":
                                    continue
                                if value is None:
                                    self.assertTrue(np.isnan(feature[name]))
                                else:
                                    self.assertEqual(value, feature[name], name)

    def test_optional_feature_reader_joins_actual_descriptions_after_reordering(self):
        try:
            from openhisto.feature_encoder import (
                EncodeConfig,
                FeatureReader,
                ModelDescriptor as FeatureDescriptor,
                SourceSpec as FeatureSourceSpec,
                encode_slide,
                register_encoder,
            )
        except ImportError as exc:
            self.skipTest(f"Optional feature_encoder core dependencies unavailable: {exc}")
        from openhisto.vlas.base import GenerationResult, ModelDescriptor
        from openhisto.vlas.config import DescribeConfig
        from openhisto.vlas.pipeline import describe_slide
        from openhisto.vlas.registry import register_model
        from openhisto.vlas.storage import DescriptionReader

        class FeatureBackend:
            def __init__(self, spec, *, config, auth):
                self.dimension = 1
                self.metadata = {"model_id": spec.model, "resolved_revision": "synthetic"}
                self.usage = {}

            def encode(self, images):
                return np.asarray([[image[0, 0, 0]] for image in images], dtype=np.float32)

            def close(self):
                pass

        class DescriptionBackend:
            def __init__(self, spec, config):
                self.metadata = {"model_id": spec.model, "resolved_revision": spec.revision}

            def generate(self, images, prompt):
                return [GenerationResult(str(int(image[0, 0, 0]))) for image in images]

            def close(self):
                pass

        register_encoder(
            FeatureDescriptor(
                "test_vla_feature_join", "Synthetic join", "test", "public", 1, "synthetic"
            ),
            FeatureBackend,
        )
        register_model(
            ModelDescriptor(
                "test_feature_description_join",
                "Synthetic join",
                "synthetic",
                "a" * 40,
                "https://example.invalid",
                "synthetic",
                "test",
                "synthetic",
            ),
            DescriptionBackend,
        )
        rows = [
            self.image_row("a", color=20, x_level0=100),
            self.image_row("b", color=80, x_level0=200),
        ]
        manifest = self.manifest(rows)
        source = FeatureSourceSpec(manifest, source_id="stable-join")
        feature_path = self.root / "features.h5ad"
        encode_slide(
            source,
            feature_path,
            ["test_vla_feature_join"],
            EncodeConfig(slide_id=self.config.slide_id),
        )
        manifest.write_text("\n".join(json.dumps(row) for row in reversed(rows)), encoding="utf-8")
        output = self.root / "descriptions"
        describe_slide(
            source,
            output,
            ["test_feature_description_join"],
            "Describe the patch.",
            DescribeConfig(slide_id=self.config.slide_id, device="cpu"),
        )
        with FeatureReader(feature_path) as features, DescriptionReader(output) as descriptions:
            records = descriptions.records()
            keys = [record["patch_key"] for record in records]
            self.assertEqual([record["patch_id"] for record in records], ["b", "a"])
            self.assertNotEqual(keys, features.patch_metadata().index.tolist())
            vectors = features.features("test_vla_feature_join", 20, rows=keys)
            self.assertEqual(
                vectors[:, 0].tolist(), [float(record["description"]) for record in records]
            )
            geometry = features.patch_metadata(rows=keys)
            self.assertEqual(
                geometry["x_level0"].tolist(), [record["x_level0"] for record in records]
            )


if __name__ == "__main__":
    unittest.main()
