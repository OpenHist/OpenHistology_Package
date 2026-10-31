import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from openhisto.segmentation.config import MaskSpec, Sample
from openhisto.segmentation.data import decode_mask, prepare_dataset, read_sample
from openhisto.segmentation.annotations import rasterize_geojson


class MaskTests(unittest.TestCase):
    def test_palette_image_colors_and_indexed_mask_ids_are_distinct(self):
        indices = np.array([[0, 1]], dtype=np.uint8)
        image = Image.fromarray(indices).convert("P")
        image.putpalette([255, 0, 0, 0, 255, 0] + [0, 0, 0] * 254)
        expected = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "palette.png"
            image.save(path)
            for source in (image, path):
                with self.subTest(source=type(source).__name__):
                    sample = read_sample(Sample(source, source, mask_spec=MaskSpec(kind="indexed")))
                    np.testing.assert_array_equal(sample.image, expected)
                    np.testing.assert_array_equal(sample.mask, indices)
                    colors = MaskSpec(kind="rgb", palette={"255,0,0": 1, "0,255,0": 0})
                    np.testing.assert_array_equal(decode_mask(source, colors), [[1, 0]])

    def test_palette_transparency_is_composited_onto_white(self):
        image = Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).convert("P")
        image.putpalette([255, 0, 0, 0, 255, 0] + [0, 0, 0] * 254)
        image.info["transparency"] = 0
        expected = np.array([[[255, 255, 255], [0, 255, 0]]], dtype=np.uint8)
        np.testing.assert_array_equal(read_sample(Sample(image)).image, expected)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transparent.png"
            image.save(path)
            np.testing.assert_array_equal(read_sample(path).image, expected)

    def test_binary_encodings_and_background_are_retained(self):
        expected = np.array([[0, 1], [1, 0]])
        for array in (
            expected.astype(bool),
            expected.astype(np.uint8),
            expected.astype(np.uint8) * 255,
        ):
            np.testing.assert_array_equal(decode_mask(array), expected)
        samples = prepare_dataset(
            [Sample(np.zeros((2, 3, 3), np.uint8), np.zeros((2, 3), np.uint8))]
        )
        self.assertEqual(len(samples), 1)
        self.assertFalse(read_sample(samples[0]).mask.any())

    def test_indexed_precision_ignore_and_instance_mapping(self):
        array = np.array([[0, 257], [300, 65535]], np.uint16)
        expected = np.array([[0, 257], [300, -100]])
        np.testing.assert_array_equal(
            decode_mask(array, MaskSpec(ignore_values=[65535]), num_classes=301), expected
        )
        with self.assertRaisesRegex(ValueError, "explicit"):
            decode_mask(array, MaskSpec(kind="instance"))
        mapped = decode_mask(
            array, MaskSpec(kind="instance", value_map={0: 0, 257: 1, 300: 1, 65535: 0})
        )
        np.testing.assert_array_equal(mapped, [[0, 1], [1, 0]])

    def test_rgb_explicit_mapping_and_unknown(self):
        colors = np.array([[[0, 0, 0], [255, 0, 0]]], np.uint8)
        spec = MaskSpec(kind="rgb", palette={"0,0,0": 0, "255,0,0": 2})
        np.testing.assert_array_equal(decode_mask(colors, spec, num_classes=3), [[0, 2]])
        colors[0, 1] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "Unknown RGB"):
            decode_mask(colors, spec, num_classes=3)
        with self.assertRaisesRegex(ValueError, "Multichannel"):
            decode_mask(colors)

    def test_channel_axes_and_overlap_priority(self):
        channels = np.zeros((2, 3, 4), np.uint8)
        channels[0, 0, 1] = 1
        channels[1, 0, 1] = 255
        spec = MaskSpec(kind="channels", channel_axis=0, channel_map={0: 1, 1: 2})
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            decode_mask(channels, spec, num_classes=3)
        spec.class_priority = [2, 1]
        expected = decode_mask(channels, spec, num_classes=3)
        self.assertEqual(expected[0, 1], 2)
        spec.channel_axis = -1
        np.testing.assert_array_equal(
            decode_mask(np.moveaxis(channels, 0, -1), spec, num_classes=3), expected
        )

    def test_native_rectangular_alignment_and_mask_frame(self):
        image = np.zeros((2, 4, 3), np.uint8)
        mask = np.array([[0, 1, 2], [2, 1, 0]], np.uint8)
        sample = Sample(
            image,
            mask,
            xy_level0=(12, 22),
            footprint_level0=(8, 4),
            mask_spec=MaskSpec(kind="indexed", origin=(10, 20), downsample=(2, 2)),
        )
        loaded = read_sample(sample, num_classes=3)
        self.assertEqual(loaded.image.shape, (2, 4, 3))
        np.testing.assert_array_equal(loaded.mask, [[1, 0, -100, -100], [-100, -100, -100, -100]])
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            read_sample(Sample(image, mask), num_classes=3)

    def test_valid_bounds_are_ignored(self):
        sample = Sample(
            np.zeros((3, 4, 3), np.uint8), np.ones((3, 4), np.uint8), valid_bounds=(1, 0, 3, 2)
        )
        mask = read_sample(sample).mask
        np.testing.assert_array_equal(
            mask, [[-100, 1, 1, -100], [-100, 1, 1, -100], [-100, -100, -100, -100]]
        )
        sample.valid_bounds = (1.5, 0, 3, 2)
        with self.assertRaisesRegex(ValueError, "integer"):
            read_sample(sample)

    def test_relative_csv_and_legacy_pair_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.fromarray(np.zeros((2, 3, 3), np.uint8)).save(root / "rgb.png")
            Image.fromarray(np.ones((2, 3), np.uint8)).save(root / "mask.png")
            (root / "samples.csv").write_text(
                "image_path,mask_path,sample_id\nrgb.png,mask.png,example\n"
            )
            sample = prepare_dataset(root / "samples.csv")[0]
            self.assertEqual(sample.sample_id, "example")
            self.assertTrue(read_sample(sample).mask.all())
            (root / "images").mkdir()
            (root / "masks").mkdir()
            Image.fromarray(np.zeros((2, 3, 3), np.uint8)).save(root / "images" / "a.PNG")
            Image.fromarray(np.zeros((2, 3), np.uint8)).save(root / "masks" / "a_mask.tif")
            (root / "legacy.csv").write_text(
                "slide_dir,mask_dir,slide_name\nimages,masks,slide-A\n"
            )
            self.assertEqual(len(prepare_dataset(root / "legacy.csv")), 1)
            (root / "masks" / "a_mask.tif").unlink()
            with self.assertRaisesRegex(ValueError, "Missing mask"):
                prepare_dataset(root / "legacy.csv")

    def test_archive_reader_preserves_index_geometry(self):
        from openhisto.patch_extraction.archive import PatchArchiveWriter

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "slide.h5"
            with PatchArchiveWriter(path, patch_size=4, metadata={}) as writer:
                writer.append(np.full((4, 4, 3), 75, np.uint8), (-2, 6), (8, 8), (1, 0, 4, 4))
            samples = prepare_dataset(
                {
                    "archive_path": path,
                    "mask": np.ones((10, 10), np.uint8),
                    "mask_spec": {"kind": "binary", "origin": (0, 0), "downsample": (2, 2)},
                }
            )
            loaded = read_sample(samples[0])
            self.assertEqual(loaded.xy_level0, (-2, 6))
            self.assertEqual(loaded.footprint_level0, (8, 8))
            self.assertEqual(loaded.valid_bounds, (1, 0, 4, 4))
            self.assertTrue(np.all(loaded.mask[:, 0] == -100))
            self.assertTrue(np.all(loaded.mask[:, 1:] == 1))

    def test_selected_archive_csv_nested_paths_and_embedding(self):
        from openhisto.patch_extraction.archive import PatchArchiveWriter

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "patches.h5"
            with PatchArchiveWriter(archive, patch_size=2, metadata={}) as writer:
                for i in range(2):
                    writer.append(np.full((2, 2, 3), i, np.uint8), (2 * i, 0), (2, 2), (0, 0, 2, 2))
            np.save(root / "embedding.npy", np.array([1.0, 2.0]))
            with (root / "selected.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["archive_path", "patch_index", "sample_id", "embedding"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "archive_path": "patches.h5",
                        "patch_index": 1,
                        "sample_id": "chosen",
                        "embedding": json.dumps({"path": "embedding.npy", "key": "unused"}),
                    }
                )
            samples = prepare_dataset(root / "selected.csv")
            self.assertEqual(len(samples), 1)
            loaded = read_sample(samples[0])
            self.assertEqual(loaded.sample_id, "chosen")
            self.assertEqual(loaded.xy_level0, (2, 0))
            self.assertEqual(loaded.embedding["path"], root / "embedding.npy")
            with (root / "nested.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image", "sample_id"])
                writer.writeheader()
                writer.writerow(
                    {
                        "image": json.dumps({"archive_path": "patches.h5", "index": 0}),
                        "sample_id": "nested",
                    }
                )
            self.assertEqual(read_sample(prepare_dataset(root / "nested.csv")[0]).xy_level0, (0, 0))
            np.testing.assert_array_equal(decode_mask(np.array([[0, 255, -100]])), [[0, 1, -100]])
            (root / "dataset.json").write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "image": {"archive_path": "patches.h5", "index": 1},
                                "sample_id": "json",
                            }
                        ]
                    }
                )
            )
            self.assertEqual(
                read_sample(prepare_dataset(root / "dataset.json")[0]).xy_level0, (2, 0)
            )


class GeoJSONTests(unittest.TestCase):
    def test_holes_multipolygon_classes_and_frame(self):
        polygon = {
            "type": "Polygon",
            "coordinates": [
                [[10, 20], [18, 20], [18, 28], [10, 28], [10, 20]],
                [[12, 22], [16, 22], [16, 26], [12, 26], [12, 22]],
            ],
        }
        multi = {
            "type": "MultiPolygon",
            "coordinates": [[[[10, 20], [12, 20], [12, 22], [10, 22], [10, 20]]]],
        }
        features = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": polygon,
                    "properties": {"classification": {"name": "tissue"}},
                },
                {
                    "type": "Feature",
                    "geometry": multi,
                    "properties": {"classification": {"name": "object"}},
                },
            ],
        }
        spec = MaskSpec(kind="geojson", class_map={"tissue": 1, "object": 2})
        mask = rasterize_geojson(
            features, (4, 4), spec, xy_level0=(10, 20), footprint_level0=(8, 8), num_classes=3
        )
        np.testing.assert_array_equal(
            mask, [[2, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]]
        )
        spec.class_priority = [1, 2]
        self.assertEqual(
            rasterize_geojson(
                features, (4, 4), spec, xy_level0=(10, 20), footprint_level0=(8, 8), num_classes=3
            )[0, 0],
            1,
        )
        spec.class_map = {"tissue": 1}
        with self.assertRaisesRegex(ValueError, "Unknown annotation"):
            rasterize_geojson(features, (4, 4), spec)


if __name__ == "__main__":
    unittest.main()
