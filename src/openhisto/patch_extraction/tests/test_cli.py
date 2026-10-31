import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from openhisto.patch_extraction.__main__ import main
from openhisto.patch_extraction.config import ExtractionConfig, ExtractionSummary


class CommandLineTests(unittest.TestCase):
    @patch("openhisto.patch_extraction.__main__.extract_grid")
    def test_grid_defaults_and_json_summary(self, extract):
        extract.return_value = ExtractionSummary(Path("output.h5"), 5, 3, 1, 1)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["grid", "synthetic.svs", "output.h5"]), 0)
        self.assertEqual(extract.call_args.args[:2], ("synthetic.svs", "output.h5"))
        self.assertEqual(extract.call_args.args[2], ExtractionConfig())
        self.assertEqual(json.loads(stdout.getvalue())["patch_count"], 3)

    @patch("openhisto.patch_extraction.__main__.extract_mask_centered")
    def test_mask_transform_and_explicit_jpeg(self, extract):
        extract.return_value = ExtractionSummary(Path("output.h5"), 1, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            main(
                [
                    "mask",
                    "synthetic.svs",
                    "synthetic.png",
                    "output.h5",
                    "--mask-downsample",
                    "2",
                    "3",
                    "--mask-origin",
                    "5",
                    "7",
                    "--codec",
                    "jpeg",
                    "--patch-size",
                    "32",
                    "--filter-mask-patches",
                ]
            )
        self.assertEqual(extract.call_args.kwargs["mask_downsample"], (2.0, 3.0))
        self.assertEqual(extract.call_args.kwargs["mask_origin"], (5.0, 7.0))
        config = extract.call_args.args[3]
        self.assertEqual(config.patch_size, 32)
        self.assertEqual(config.codec, "jpeg")
        self.assertTrue(config.filter_mask_patches)

    @patch("openhisto.patch_extraction.__main__.extract_mask_centered")
    def test_invalid_transform_fails_before_extraction(self, extract):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(
                [
                    "mask",
                    "synthetic.svs",
                    "synthetic.png",
                    "output.h5",
                    "--mask-downsample",
                    "1",
                    "2",
                    "3",
                ]
            )
        self.assertEqual(error.exception.code, 2)
        extract.assert_not_called()

    @patch("openhisto.patch_extraction.__main__.extract_grid")
    def test_extraction_error_has_nonzero_exit(self, extract):
        extract.side_effect = ValueError("source magnification required")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(["grid", "synthetic.svs", "output.h5"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("source magnification required", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
