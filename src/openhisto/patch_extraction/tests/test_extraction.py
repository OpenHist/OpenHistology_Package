"""Synthetic extraction tests; no slide files or clinical data are required."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from openhisto.patch_extraction.archive import PatchReader
from openhisto.patch_extraction.config import ExtractionConfig
from openhisto.patch_extraction.extraction import (
    extract_grid,
    extract_mask_centered,
    _is_foreground,
)


class FakeSlide:
    def __init__(
        self, dimensions=(16, 16), downsamples=(1, 2, 4), source_mag="40", color=(150, 30, 70)
    ):
        self.dimensions = dimensions
        self.level_downsamples = downsamples
        self.properties = {} if source_mag is None else {"openslide.objective-power": source_mag}
        self.color = color
        self.reads = []
        self.closed = False
        self.fail_read = False

    def read_region(self, location, level, size):
        self.reads.append((location, level, size))
        if self.fail_read:
            raise RuntimeError("synthetic read failure")
        downsample = self.level_downsamples[level]
        xs = location[0] + np.arange(size[0]) * downsample
        ys = location[1] + np.arange(size[1]) * downsample
        valid = (
            (ys[:, None] >= 0)
            & (ys[:, None] < self.dimensions[1])
            & (xs[None, :] >= 0)
            & (xs[None, :] < self.dimensions[0])
        )
        rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        rgba[..., :3] = self.color
        rgba[..., 3] = valid * 255
        return Image.fromarray(rgba, "RGBA")

    def close(self):
        self.closed = True


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "patches.h5"

    def config(self, **kwargs):
        return ExtractionConfig(patch_size=4, foreground_method="none", **kwargs)

    def extract(self, slide, config=None):
        with patch("openhisto.patch_extraction.extraction._open_slide", return_value=slide):
            return extract_grid("synthetic-input", self.output, config or self.config())

    def test_grid_level0_coordinates_counts_and_pyramid(self):
        slide = FakeSlide(dimensions=(10, 7))
        summary = self.extract(slide, ExtractionConfig(patch_size=2, foreground_method="none"))
        self.assertTrue(slide.closed)
        self.assertEqual(
            (summary.candidate_count, summary.patch_count, summary.skipped_edges), (6, 2, 4)
        )
        self.assertEqual(slide.reads, [((0, 0), 1, (2, 2)), ((4, 0), 1, (2, 2))])
        with PatchReader(self.output) as reader:
            np.testing.assert_array_equal(reader.get_coordinates(), [[0, 0], [4, 0]])
            self.assertEqual(reader[0].footprint_level0, (4, 4))
            self.assertEqual(reader[0].valid_bounds, (0, 0, 2, 2))
            self.assertEqual(reader.metadata["counts"]["patch_count"], 2)
            self.assertNotIn("synthetic-input", str(reader.metadata))

    def test_noninteger_ratio_does_not_truncate_or_accumulate_stride(self):
        slide = FakeSlide(dimensions=(18, 9), downsamples=(1, 1.5, 3), source_mag="25")
        summary = self.extract(
            slide, ExtractionConfig(patch_size=3, stride=2, target_mag=10, foreground_method="none")
        )
        self.assertEqual(summary.patch_count, 3)
        self.assertEqual(slide.reads, [((x, 0), 1, (6, 6)) for x in (0, 5, 10)])
        with PatchReader(self.output) as reader:
            np.testing.assert_array_equal(reader.get_coordinates(), [[0, 0], [5, 0], [10, 0]])
            self.assertEqual(reader[0].footprint_level0, (8, 8))
            self.assertEqual(reader.metadata["scale"], 2.5)
            self.assertEqual(reader.metadata["selected_level_downsample"], 1.5)
            self.assertEqual(reader[0].image.shape, (3, 3, 3))

    def test_rounding_small_stride_keeps_distinct_row_major_origins(self):
        slide = FakeSlide(dimensions=(9, 3), source_mag="30")
        self.extract(slide, ExtractionConfig(patch_size=2, stride=1, foreground_method="none"))
        with PatchReader(self.output) as reader:
            np.testing.assert_array_equal(
                reader.get_coordinates(), [[0, 0], [2, 0], [3, 0], [4, 0], [6, 0]]
            )

    def test_pad_is_white_and_filter_uses_only_valid_pixels(self):
        slide = FakeSlide(dimensions=(5, 3), source_mag="20")
        config = ExtractionConfig(patch_size=4, edge="pad", tissue_fraction=0.9)
        summary = self.extract(slide, config)
        self.assertEqual(summary.patch_count, 2)
        with PatchReader(self.output) as reader:
            first, second = reader[0], reader[1]
            self.assertEqual(first.valid_bounds, (0, 0, 4, 3))
            self.assertEqual(second.valid_bounds, (0, 0, 1, 3))
            np.testing.assert_array_equal(second.image[:3, 0], np.tile(slide.color, (3, 1)))
            self.assertTrue(np.all(second.image[:, 1:] == 255))
            self.assertTrue(np.all(second.image[3] == 255))

    def test_background_rejection_and_empty_archive(self):
        slide = FakeSlide(dimensions=(4, 4), source_mag="20", color=(255, 255, 255))
        summary = self.extract(slide, ExtractionConfig(patch_size=4))
        self.assertEqual(
            (summary.candidate_count, summary.patch_count, summary.skipped_background), (1, 0, 1)
        )
        with PatchReader(self.output) as reader:
            self.assertEqual(len(reader), 0)
            self.assertEqual(reader.get_coordinates().shape, (0, 2))

    def test_source_override_and_missing_or_finer_resolution(self):
        slide = FakeSlide(dimensions=(4, 4), source_mag=None)
        self.extract(slide, self.config(source_mag=20))
        self.assertTrue(slide.closed)
        for source in (None, "not-a-number", "nan", "0", "10"):
            with self.subTest(source=source):
                slide = FakeSlide(source_mag=source)
                with self.assertRaises(ValueError):
                    self.extract(slide)
                self.assertTrue(slide.closed)
                self.assertEqual(slide.reads, [])

    def test_slide_closes_and_archive_is_not_published_on_read_failure(self):
        slide = FakeSlide()
        slide.fail_read = True
        with self.assertRaisesRegex(RuntimeError, "synthetic read failure"):
            self.extract(slide)
        self.assertTrue(slide.closed)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])

    def test_output_cannot_replace_slide_mask_or_input_hardlink(self):
        source = Path(self.temporary.name) / "source.tiff"
        source.write_bytes(b"synthetic source")
        alias = Path(self.temporary.name) / "source-alias.tiff"
        alias.hardlink_to(source)
        config = self.config(overwrite=True)
        with patch("openhisto.patch_extraction.extraction._open_slide") as opener:
            for destination in (source, alias):
                with self.subTest(destination=destination.name):
                    with self.assertRaisesRegex(ValueError, "differ from all input"):
                        extract_grid(source, destination, config)
            with self.assertRaisesRegex(ValueError, "differ from all input"):
                extract_mask_centered("synthetic-input", source, source, config, mask_downsample=1)
            opener.assert_not_called()
        self.assertEqual(source.read_bytes(), b"synthetic source")

    def test_overflowing_magnification_or_footprint_is_rejected(self):
        for target_mag in (1e-308, 20):
            with self.subTest(target_mag=target_mag):
                slide = FakeSlide(source_mag="1e308")
                with self.assertRaises(ValueError):
                    self.extract(slide, self.config(target_mag=target_mag))
                self.assertTrue(slide.closed)
                self.assertFalse(self.output.exists())

    def test_existing_output_refused_and_slide_closed(self):
        self.output.write_bytes(b"existing output")
        slide = FakeSlide()
        with self.assertRaises(FileExistsError):
            self.extract(slide)
        self.assertTrue(slide.closed)
        self.assertEqual(self.output.read_bytes(), b"existing output")

    def test_mask_bounding_centers_offsets_order_and_size_cutoff(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5:7, 1:3] = 255
        mask[1:3, 5:7] = 255
        mask[1:3, 1:3] = 255
        mask[9, 9] = 255
        original = mask.copy()
        slide = FakeSlide(dimensions=(64, 64), source_mag="20", color=(255, 255, 255))
        config = self.config(min_box_size=2)
        with patch("openhisto.patch_extraction.extraction._open_slide", return_value=slide):
            summary = extract_mask_centered(
                "synthetic-input",
                mask,
                self.output,
                config,
                mask_downsample=(2, 3),
                mask_origin=(10, 20),
            )
        self.assertEqual(
            (summary.candidate_count, summary.patch_count, summary.skipped_size), (4, 3, 1)
        )
        self.assertEqual(summary.skipped_background, 0)
        np.testing.assert_array_equal(mask, original)
        with PatchReader(self.output) as reader:
            np.testing.assert_array_equal(reader.get_coordinates(), [[12, 24], [20, 24], [12, 36]])
            self.assertEqual(
                reader.metadata["mask_transform"],
                {"downsample": [2.0, 3.0], "origin_level0": [10.0, 20.0]},
            )

    def test_mask_negative_origin_padding_and_optional_filter(self):
        mask = np.ones((2, 2), dtype=np.uint8)
        slide = FakeSlide(dimensions=(8, 8), source_mag="20", color=(255, 255, 255))
        config = self.config(edge="pad")
        with patch("openhisto.patch_extraction.extraction._open_slide", return_value=slide):
            summary = extract_mask_centered(
                "synthetic-input", mask, self.output, config, mask_downsample=1
            )
        self.assertEqual(summary.patch_count, 1)
        with PatchReader(self.output) as reader:
            self.assertEqual(reader[0].xy_level0, (-1, -1))
            self.assertEqual(reader[0].valid_bounds, (1, 1, 4, 4))
            self.assertTrue(np.all(reader[0].image == 255))
        slide = FakeSlide(dimensions=(8, 8), source_mag="20", color=(255, 255, 255))
        filtered = replace(
            config, overwrite=True, filter_mask_patches=True, foreground_method="HSV"
        )
        with patch("openhisto.patch_extraction.extraction._open_slide", return_value=slide):
            summary = extract_mask_centered(
                "synthetic-input", mask, self.output, filtered, mask_downsample=1
            )
        self.assertEqual((summary.patch_count, summary.skipped_background), (0, 1))

    def test_empty_mask_and_tiff_path(self):
        import tifffile

        mask_path = Path(self.temporary.name) / "synthetic-mask.tiff"
        tifffile.imwrite(mask_path, np.zeros((3, 3), dtype=np.uint8))
        slide = FakeSlide(source_mag="20")
        with patch("openhisto.patch_extraction.extraction._open_slide", return_value=slide):
            summary = extract_mask_centered(
                "synthetic-input", mask_path, self.output, self.config(), mask_downsample=1
            )
        self.assertEqual((summary.candidate_count, summary.patch_count), (0, 0))
        self.assertTrue(slide.closed)
        with PatchReader(self.output) as reader:
            self.assertEqual(len(reader), 0)

    def test_mask_validation_happens_before_slide_open(self):
        bad_masks = [
            np.zeros((2, 2, 3)),
            np.array([[0, 2]]),
            np.array([[1, 255]]),
            np.array([[float("nan")]]),
            np.zeros((0, 2)),
            np.array([["0"]]),
        ]
        with patch("openhisto.patch_extraction.extraction._open_slide") as opener:
            for mask in bad_masks:
                with self.subTest(mask_shape=mask.shape, dtype=mask.dtype):
                    with self.assertRaises(ValueError):
                        extract_mask_centered(
                            "synthetic-input", mask, self.output, self.config(), mask_downsample=1
                        )
            for scale in (0, -1, float("inf"), (1, 0), (1, 2, 3), True, "2"):
                with self.subTest(scale=scale):
                    with self.assertRaises(ValueError):
                        extract_mask_centered(
                            "synthetic-input",
                            np.ones((2, 2)),
                            self.output,
                            self.config(),
                            mask_downsample=scale,
                        )
            opener.assert_not_called()

    def test_tissue_filters_and_legacy_otsu_aliases(self):
        red = Image.new("RGB", (4, 4), (150, 30, 70))
        white = Image.new("RGB", (4, 4), "white")
        bounds = (0, 0, 4, 4)
        for method in ("HSV", "SL"):
            config = ExtractionConfig(foreground_method=method)
            self.assertTrue(_is_foreground(red, bounds, config))
            self.assertFalse(_is_foreground(white, bounds, config))
        for method in ("luminesence", "luminance", "otsu"):
            with patch("skimage.filters.threshold_otsu", return_value=128):
                self.assertTrue(
                    _is_foreground(red, bounds, ExtractionConfig(foreground_method=method))
                )
            self.assertFalse(
                _is_foreground(white, bounds, ExtractionConfig(foreground_method=method))
            )


if __name__ == "__main__":
    unittest.main()
