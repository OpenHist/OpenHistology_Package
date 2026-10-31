import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from openhisto.stain_transfer.config import DomainSpec, TransferConfig
from openhisto.stain_transfer.imageio import load_rgb, save_rgb, validate_rgb
from openhisto.stain_transfer.normalizer import is_blank
from openhisto.stain_transfer.pipeline import normalize


class _Normalizer:
    provenance = {"schema_version": 1, "backends": [{"method": "test"}]}

    def __init__(self):
        self.seen = []

    def transform(self, image, valid_bounds=None, patch_id=None):
        self.seen.append((patch_id, valid_bounds))
        result = image.copy()
        if is_blank(image, valid_bounds):
            return result
        x0, y0, x1, y1 = valid_bounds or (0, 0, image.shape[1], image.shape[0])
        result[y0:y1, x0:x1] = 255 - result[y0:y1, x0:x1]
        return result


class ImageIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_lossless_atomic_png_and_no_overwrite(self):
        image = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
        output = self.root / "image.png"
        save_rgb(image, output)
        np.testing.assert_array_equal(load_rgb(output), image)
        with self.assertRaises(FileExistsError):
            save_rgb(255 - image, output)
        np.testing.assert_array_equal(load_rgb(output), image)
        self.assertEqual(list(self.root.glob(".*tmp*")), [])

    def test_alpha_white_grayscale_and_high_depth(self):
        rgba = np.array([[[1, 2, 3, 0], [10, 20, 30, 255]]], dtype=np.uint8)
        Image.fromarray(rgba).save(self.root / "alpha.png")
        np.testing.assert_array_equal(
            load_rgb(self.root / "alpha.png"), [[[255, 255, 255], [10, 20, 30]]]
        )
        Image.fromarray(np.full((3, 2), 60, dtype=np.uint8)).save(self.root / "gray.png")
        self.assertEqual(load_rgb(self.root / "gray.png").shape, (3, 2, 3))
        Image.fromarray(np.full((3, 2), 1024, dtype=np.uint16)).save(self.root / "depth.png")
        with self.assertRaisesRegex(ValueError, "8-bit"):
            load_rgb(self.root / "depth.png")
        with self.assertRaises(ValueError):
            validate_rgb(np.zeros((2, 2, 3), dtype=np.float32))

    def test_failed_encoding_leaves_existing_file(self):
        image = np.full((3, 4, 3), 40, dtype=np.uint8)
        output = self.root / "safe.png"
        save_rgb(image, output)
        before = output.read_bytes()
        with patch("PIL.Image.Image.save", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                save_rgb(255 - image, output, overwrite=True)
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(list(self.root.glob(".*tmp*")), [])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.domain = DomainSpec(method="reinhard", references=("unused-reference.png",))
        self.normalizer = _Normalizer()
        self.factory = patch(
            "openhisto.stain_transfer.pipeline.create_normalizer", return_value=self.normalizer
        )
        self.factory.start()

    def tearDown(self):
        self.factory.stop()
        self.temporary.cleanup()

    def image(self, path, value=60, shape=(4, 4, 3)):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full(shape, value, dtype=np.uint8)
        Image.fromarray(image).save(path)
        return image

    def test_image_png_and_blank_count(self):
        source = self.root / "input.png"
        expected = self.image(source)
        output = self.root / "result.png"
        report = normalize(source, output, self.domain)
        self.assertEqual((report.processed, report.blank, report.failed), (1, 0, 0))
        np.testing.assert_array_equal(load_rgb(output), 255 - expected)
        provenance = json.loads((self.root / "result.png.json").read_text())
        self.assertEqual(provenance["stain_transfer"], self.normalizer.provenance)
        self.assertEqual(len(provenance["input"]["rgb_sha256"]), 64)
        self.image(source, 255)
        report = normalize(source, output, self.domain, TransferConfig(overwrite=True))
        self.assertEqual(report.blank, 1)
        np.testing.assert_array_equal(load_rgb(output), 255)
        with self.assertRaisesRegex(ValueError, ".png"):
            normalize(source, self.root / "wrong.jpg", self.domain)

    def test_single_image_sidecar_preflight_and_mixture_report(self):
        source = self.root / "input.png"
        self.image(source)
        output = self.root / "result.png"
        sidecar = self.root / "result.png.json"
        sidecar.write_text("existing provenance")
        with self.assertRaises(FileExistsError):
            normalize(source, output, self.domain)
        self.assertFalse(output.exists())
        mixed = DomainSpec(
            candidates=(self.domain, DomainSpec(method="histogram", references=("other.png",)))
        )
        report = normalize(source, output, mixed, TransferConfig(mix=True, overwrite=True))
        self.assertEqual(report.method, "mixture")

    def test_unsupported_parallel_settings_fail_explicitly(self):
        source = self.root / "input.png"
        self.image(source)
        for config in (TransferConfig(workers=1), TransferConfig(batch_size=2)):
            with self.assertRaisesRegex(ValueError, "stream sequentially"):
                normalize(source, self.root / "out.png", self.domain, config)

    def test_normalizers_are_closed_on_success_and_failure_for_every_adapter(self):
        from openhisto.patch_extraction import PatchArchiveWriter

        image = self.root / "input.png"
        self.image(image)
        directory = self.root / "input-directory"
        self.image(directory / "patch.png")
        manifest = self.root / "input.jsonl"
        manifest.write_text(json.dumps({"image_path": "input.png", "patch_id": "p0"}) + "\n")
        archive = self.root / "input.h5"
        with PatchArchiveWriter(archive, patch_size=4, metadata={"target_mag": 20}) as writer:
            writer.append(np.full((4, 4, 3), 60, dtype=np.uint8), (0, 0), (4, 4), (0, 0, 4, 4))
        for name, source, extension in (
            ("image", image, ".png"),
            ("directory", directory, ""),
            ("manifest", manifest, ""),
            ("archive", archive, ".h5"),
        ):
            for fails in (False, True):
                with self.subTest(adapter=name, fails=fails):
                    backend = _Normalizer()
                    backend.close = Mock()
                    if fails:
                        backend.transform = Mock(side_effect=RuntimeError("backend failed"))
                    output = self.root / f"{name}-{fails}{extension}"
                    with patch(
                        "openhisto.stain_transfer.pipeline.create_normalizer", return_value=backend
                    ):
                        if fails:
                            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                                normalize(source, output, self.domain)
                        else:
                            normalize(source, output, self.domain)
                    backend.close.assert_called_once_with()

    def test_recursive_directory_and_provenance(self):
        source = self.root / "input"
        first = self.image(source / "slide" / "256" / "20" / "10_20.png")
        self.image(source / "blank.jpg", 255)
        (source / "coordinates.csv").write_text("0,0\n")
        output = self.root / "normalized"
        report = normalize(source, output, self.domain)
        self.assertEqual((report.processed, report.blank), (2, 1))
        np.testing.assert_array_equal(load_rgb(output / "slide/256/20/10_20.png"), 255 - first)
        self.assertTrue((output / "blank.png").exists())
        payload = json.loads((output / "stain_transfer.json").read_text())
        self.assertEqual(payload["stain_transfer"], self.normalizer.provenance)
        self.assertEqual(payload["report"]["processed"], 2)
        self.assertFalse((output / "coordinates.csv").exists())

    def test_directory_preflight_collision_and_nesting(self):
        source = self.root / "input"
        self.image(source / "same.png")
        self.image(source / "same.jpg")
        output = self.root / "normalized"
        with self.assertRaisesRegex(ValueError, "same output"):
            normalize(source, output, self.domain)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "contain"):
            normalize(source, source / "normalized", self.domain)

    def test_existing_later_output_is_detected_before_any_image_writes(self):
        source = self.root / "input"
        self.image(source / "a.png")
        self.image(source / "z.png")
        output = self.root / "normalized"
        self.image(output / "z.png", 85)
        with self.assertRaises(FileExistsError):
            normalize(source, output, self.domain)
        self.assertFalse((output / "a.png").exists())
        self.assertEqual(self.normalizer.seen, [])

    def test_csv_alias_and_metadata_are_preserved(self):
        self.image(self.root / "input.png")
        manifest = self.root / "patches.csv"
        manifest.write_text("patch_id,path,slide_id,mag,custom\np0,input.png,slide,20,value\n")
        output = self.root / "normalized"
        normalize(manifest, output, self.domain)
        with (output / manifest.name).open(newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["custom"], "value")
        self.assertEqual(row["patch_id"], "p0")
        self.assertEqual(row["path"], row["image_path"])
        self.assertTrue((output / row["image_path"]).exists())

    def test_jsonl_bounds_metadata_and_stable_paths(self):
        image = self.image(self.root / "input.png")
        manifest = self.root / "patches.jsonl"
        row = {
            "patch_id": "p0",
            "image_path": "input.png",
            "valid_bounds": [1, 1, 3, 3],
            "custom": {"labels": ["x"]},
        }
        row2 = {"patch_id": "p1", "image_path": "input.png"}
        manifest.write_text(json.dumps(row) + "\n" + json.dumps(row2) + "\n")
        output = self.root / "normalized"
        normalize(manifest, output, self.domain)
        results = [json.loads(line) for line in (output / manifest.name).read_text().splitlines()]
        self.assertEqual(results[0]["custom"], row["custom"])
        expected = image.copy()
        expected[1:3, 1:3] = 255 - expected[1:3, 1:3]
        np.testing.assert_array_equal(load_rgb(output / results[0]["image_path"]), expected)
        manifest.write_text(json.dumps(row2) + "\n" + json.dumps(row) + "\n")
        second = self.root / "second"
        normalize(manifest, second, self.domain)
        reordered = [json.loads(line) for line in (second / manifest.name).read_text().splitlines()]
        self.assertEqual(results[0]["image_path"], reordered[1]["image_path"])

    def test_duplicate_manifest_ids_are_rejected_before_writing(self):
        self.image(self.root / "input.png")
        manifest = self.root / "patches.jsonl"
        row = {"patch_id": "p0", "image_path": "input.png"}
        manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        output = self.root / "normalized"
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize(manifest, output, self.domain)
        self.assertFalse(output.exists())

    def test_archive_geometry_metadata_padding_and_counts(self):
        from openhisto.patch_extraction import PatchArchiveWriter, PatchReader

        source = self.root / "input.h5"
        first = np.full((4, 4, 3), 255, dtype=np.uint8)
        first[1:4, 0:3] = 60
        blank = np.full((4, 4, 3), 255, dtype=np.uint8)
        with PatchArchiveWriter(
            source,
            patch_size=4,
            metadata={"target_mag": 20, "source_mag": 40, "config": {"target_mag": 20}},
            codec="jpeg",
        ) as writer:
            writer.append(first, (-2, 5), (8, 8), (0, 1, 3, 4))
            writer.append(blank, (6, 5), (8, 8), (0, 0, 4, 4))
            writer.set_counts({"candidate_count": 3, "patch_count": 2, "skipped_background": 1})
        output = self.root / "output.h5"
        report = normalize(source, output, self.domain)
        self.assertEqual((report.processed, report.blank), (2, 1))
        with PatchReader(source) as original, PatchReader(output) as result:
            self.assertEqual(result.metadata["codec"], "png")
            self.assertEqual(result.metadata["target_mag"], 20)
            self.assertEqual(result.metadata["counts"], original.metadata["counts"])
            self.assertEqual(result.metadata["stain_transfer"], self.normalizer.provenance)
            for old, new in zip(original, result):
                self.assertEqual(
                    (new.index, new.xy_level0, new.footprint_level0, new.valid_bounds),
                    (old.index, old.xy_level0, old.footprint_level0, old.valid_bounds),
                )
                expected = old.image.copy()
                if old.index == 0:
                    expected[1:4, 0:3] = 255 - expected[1:4, 0:3]
                np.testing.assert_array_equal(new.image, expected)

    def test_archive_failure_preserves_published_file_and_removes_temporary(self):
        from openhisto.patch_extraction import PatchArchiveWriter

        source = self.root / "input.h5"
        with PatchArchiveWriter(source, patch_size=4, metadata={"target_mag": 20}) as writer:
            for index in range(2):
                writer.append(
                    np.full((4, 4, 3), 60, dtype=np.uint8), (index * 4, 0), (4, 4), (0, 0, 4, 4)
                )
        output = self.root / "output.h5"
        output.write_bytes(b"previous published file")
        self.normalizer.transform = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("backend failed")
        )
        with self.assertRaises(RuntimeError):
            normalize(source, output, self.domain, TransferConfig(overwrite=True))
        self.assertEqual(output.read_bytes(), b"previous published file")
        self.assertEqual(list(self.root.glob(".*tmp*")), [])

    @unittest.skipUnless(
        importlib.util.find_spec("anndata"),
        "feature_encoder's optional AnnData dependency is unavailable",
    )
    def test_normalized_archive_is_consumed_by_feature_encoder(self):
        from openhisto.feature_encoder.config import EncodeConfig, SourceSpec
        from openhisto.feature_encoder.inputs import prepare_inputs
        from openhisto.patch_extraction import PatchArchiveWriter

        source = self.root / "input.h5"
        with PatchArchiveWriter(
            source, patch_size=4, metadata={"target_mag": 20, "source_mag": 40}
        ) as writer:
            writer.append(np.full((4, 4, 3), 60, dtype=np.uint8), (8, 12), (8, 8), (0, 0, 4, 4))
        output = self.root / "normalized.h5"
        normalize(source, output, self.domain)
        with prepare_inputs(
            [SourceSpec(output, source_id="normalized")], EncodeConfig(slide_id="synthetic")
        ) as patches:
            self.assertEqual(len(patches.obs), 1)
            row = patches.obs.iloc[0]
            self.assertEqual(
                (row["mag"], row["source_index"], row["x_level0"], row["y_level0"]), (20, 0, 8, 12)
            )
            image = patches.read_images(patches.obs.index)[0]
            np.testing.assert_array_equal(image, 195)

    def test_empty_archive_and_directory(self):
        from openhisto.patch_extraction import PatchArchiveWriter, PatchReader

        source = self.root / "empty.h5"
        with PatchArchiveWriter(source, patch_size=4, metadata={"target_mag": 20}):
            pass
        output = self.root / "output.h5"
        self.assertEqual(normalize(source, output, self.domain).processed, 0)
        with PatchReader(output) as reader:
            self.assertEqual(len(reader), 0)
        directory = self.root / "empty"
        directory.mkdir()
        self.assertEqual(
            normalize(directory, self.root / "directory-output", self.domain).processed, 0
        )


if __name__ == "__main__":
    unittest.main()
