"""Exercise the actual native backend with generated, non-patient tiled TIFFs."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import tifffile

from openhisto._common import worker_environment
from openhisto.patch_extraction import ExtractionConfig, PatchReader, extract_grid


class NativeOpenSlideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import openslide
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest("OpenSlide native library is unavailable") from exc
        cls.openslide = openslide

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="cpath-native-test-")
        self.root = Path(self.temporary.name)
        self.slide = self.root / "synthetic.tiff"
        y, x = np.indices((192, 256))
        self.pixels = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
        with tifffile.TiffWriter(self.slide) as writer:
            writer.write(
                self.pixels, tile=(32, 32), photometric="rgb", compression="deflate", metadata=None
            )
            writer.write(
                self.pixels[::2, ::2],
                tile=(32, 32),
                photometric="rgb",
                compression="deflate",
                metadata=None,
                subfiletype=1,
            )

    def tearDown(self):
        self.temporary.cleanup()

    def test_native_pyramid_grid_roundtrip(self):
        self.assertEqual(self.openslide.OpenSlide.detect_format(str(self.slide)), "generic-tiff")
        output = self.root / "grid.h5"
        summary = extract_grid(
            self.slide,
            output,
            ExtractionConfig(
                patch_size=32,
                source_mag=40,
                target_mag=20,
                foreground_method="none",
            ),
        )
        self.assertEqual(summary.patch_count, 12)
        with PatchReader(output) as reader:
            self.assertEqual(reader.metadata["selected_level"], 1)
            self.assertEqual(reader[0].xy_level0, (0, 0))
            self.assertEqual(reader[-1].xy_level0, (192, 128))
            np.testing.assert_array_equal(reader[0].image, self.pixels[::2, ::2][:32, :32])

    def test_native_mask_cli_and_error_exit(self):
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[20:28, 28:36] = 255
        mask_path = self.root / "mask.png"
        Image.fromarray(mask).save(mask_path)
        output = self.root / "objects.h5"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "openhisto.patch_extraction",
                "mask",
                str(self.slide),
                str(mask_path),
                str(output),
                "--mask-downsample",
                "4",
                "--patch-size",
                "32",
                "--source-mag",
                "40",
            ],
            env=worker_environment(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["patch_count"], 1)
        with PatchReader(output) as reader:
            self.assertEqual(reader[0].xy_level0, (96, 64))
            np.testing.assert_array_equal(reader[0].image, self.pixels[::2, ::2][32:64, 48:80])
        unsupported = self.root / "unsupported.txt"
        unsupported.write_text("synthetic unsupported input", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "openhisto.patch_extraction",
                "grid",
                str(unsupported),
                str(self.root / "invalid.h5"),
                "--source-mag",
                "40",
            ],
            env=worker_environment(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.root / "invalid.h5").exists())


if __name__ == "__main__":
    unittest.main()
