import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from openhisto.segmentation.config import Prediction, MaskSpec, CapabilityError
from openhisto.segmentation.exports import export_prediction
from openhisto.segmentation.annotations import rasterize_geojson


class ExportTests(unittest.TestCase):
    def test_lossless_multiclass_artifacts_and_atomic_refusal(self):
        labels = np.array([[0, 1, 2], [2, 1, 0]], np.int64)
        image = np.full((2, 3, 3), 100, np.uint8)
        prediction = Prediction(
            labels,
            "example",
            probabilities=np.ones((3, 2, 3), np.float32) / 3,
            image=image,
            class_names=["background", "A", "B"],
        )
        with tempfile.TemporaryDirectory() as temp:
            files = export_prediction(
                prediction,
                temp,
                formats=("labels", "binary", "probabilities", "palette", "overlay"),
            )
            np.testing.assert_array_equal(
                np.asarray(Image.open(Path(temp) / "example_labels.png")), labels
            )
            np.testing.assert_array_equal(
                np.load(Path(temp) / "example_probabilities.npy"), prediction.probabilities
            )
            self.assertIn(Path(temp) / "example_class_2.png", files)
            previous = (Path(temp) / "example_labels.png").read_bytes()
            with self.assertRaises(FileExistsError):
                export_prediction(prediction, temp)
            self.assertEqual((Path(temp) / "example_labels.png").read_bytes(), previous)
            self.assertFalse(any(p.name.endswith(".tmp") for p in Path(temp).iterdir()))

    def test_uint16_and_uint32_lossless_labels(self):
        import tifffile

        with tempfile.TemporaryDirectory() as temp:
            labels = np.array([[0, 300, 65535]], np.int64)
            export_prediction(Prediction(labels, "wide"), temp)
            np.testing.assert_array_equal(
                np.asarray(Image.open(Path(temp) / "wide_labels.png")), labels
            )
            labels = np.array([[0, 65536]], np.int64)
            export_prediction(Prediction(labels, "wider"), temp)
            np.testing.assert_array_equal(tifffile.imread(Path(temp) / "wider_labels.tiff"), labels)
            with self.assertRaisesRegex(ValueError, "PNG labels"):
                export_prediction(Prediction(labels, "invalid"), temp, formats=("labels_png",))

    def test_polygon_holes_roundtrip_with_coordinates(self):
        labels = np.array(
            [[1, 1, 1, 1, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 2], [1, 1, 1, 1, 2]], np.int64
        )
        prediction = Prediction(
            labels,
            "shapes",
            xy_level0=(100, 220),
            footprint_level0=(15, 8),
            class_names=["background", "ring", "other"],
        )
        with tempfile.TemporaryDirectory() as temp:
            export_prediction(prediction, temp, formats=("geojson", "qupath"))
            geojson = json.loads((Path(temp) / "shapes_geojson.geojson").read_text())
            self.assertTrue(
                any(len(feature["geometry"]["coordinates"]) > 1 for feature in geojson["features"])
            )
            spec = MaskSpec(kind="geojson", class_map={"ring": 1, "other": 2})
            actual = rasterize_geojson(
                geojson,
                labels.shape,
                spec,
                xy_level0=(100, 220),
                footprint_level0=(15, 8),
                num_classes=3,
            )
            np.testing.assert_array_equal(actual, labels)
            qupath = json.loads((Path(temp) / "shapes_qupath.geojson").read_text())
            self.assertTrue(
                all(f["properties"]["objectType"] == "annotation" for f in qupath["features"])
            )
            meta = json.loads((Path(temp) / "shapes_metadata.json").read_text())
            self.assertEqual(meta["coordinate_space"], "level0")

    def test_valid_bounds_clip_foreground_products(self):
        labels = np.ones((3, 4), np.int64)
        image = np.full((3, 4, 3), 120, np.uint8)
        pred = Prediction(labels, "bounded", image=image, valid_bounds=(1, 0, 3, 2))
        with tempfile.TemporaryDirectory() as temp:
            export_prediction(pred, temp, formats=("binary", "overlay", "geojson"))
            expected = np.zeros((3, 4), np.uint8)
            expected[:2, 1:3] = 255
            np.testing.assert_array_equal(
                np.asarray(Image.open(Path(temp) / "bounded_class_1.png")), expected
            )
            polygons = json.loads((Path(temp) / "bounded_geojson.geojson").read_text())
            actual = rasterize_geojson(polygons, (3, 4))
            np.testing.assert_array_equal(actual, expected // 255)
            overlay = np.asarray(Image.open(Path(temp) / "bounded_overlay.png"))
            np.testing.assert_array_equal(overlay[2], image[2])

    def test_capability_error_has_no_partial_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            prediction = Prediction(np.zeros((3, 4), np.int64), "example")
            with self.assertRaises(CapabilityError):
                export_prediction(prediction, temp, formats=("labels", "scores"))
            self.assertEqual(list(Path(temp).iterdir()), [])
            with self.assertRaises(CapabilityError):
                export_prediction(prediction, temp, formats=("overlay",))
            with self.assertRaises(ValueError):
                export_prediction(Prediction(np.zeros((2, 2), np.int64), "../escape"), temp)


if __name__ == "__main__":
    unittest.main()
