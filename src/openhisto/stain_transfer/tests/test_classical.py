"""Synthetic classical-normalization contracts; no tissue files required."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import subprocess
import sys

import numpy as np
from PIL import Image

from openhisto._common import worker_environment
from openhisto.stain_transfer import classical


def tissue(seed=0, shape=(40, 44), stains=None):
    rng = np.random.default_rng(seed)
    if stains is None:
        stains = np.array(((0.65, 0.70, 0.29), (0.07, 0.99, 0.11)))
    concentrations = rng.gamma(1.6, 0.6, size=(*shape, 2))
    return np.clip(255 * np.exp(-(concentrations @ stains)), 0, 255).astype(np.uint8)


class ClassicalTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.reference = Path(self.directory.name) / "reference.png"
        self.target = tissue(3, stains=np.array(((0.5, 0.75, 0.3), (0.14, 0.9, 0.2))))
        Image.fromarray(self.target).save(self.reference)
        self.config = SimpleNamespace(seed=256, standardize=False, histogram_clip=(0.01, 0.99))

    def build(self, method, **options):
        return classical.build(
            SimpleNamespace(method=method, references=(str(self.reference),), options=options),
            self.config,
        )

    def test_reinhard_moves_channel_statistics_toward_reference(self):
        source = tissue(2)
        original = source.copy()
        normalizer = self.build("reinhard")
        result = normalizer.transform(source)
        before = np.linalg.norm(
            classical.ClassicalNormalizer._lab(source).mean((0, 1)) - normalizer.target_mean
        )
        after = np.linalg.norm(
            classical.ClassicalNormalizer._lab(result).mean((0, 1)) - normalizer.target_mean
        )
        self.assertLess(after, before)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, source.shape)
        np.testing.assert_array_equal(source, original)

    def test_histogram_matches_reference_distribution(self):
        self.config.histogram_clip = None
        source = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (16, 1, 3))
        target = np.tile((np.arange(256) // 2 + 40).astype(np.uint8)[None, :, None], (16, 1, 3))
        Image.fromarray(target).save(self.reference)
        result = self.build("histogram").transform(source)
        self.assertLess(abs(float(result.mean()) - float(target.mean())), 2)
        self.assertTrue(np.all(np.diff(result[0, :, 0].astype(int)) >= 0))

    def test_histogram_self_match_preserves_sparse_reference_intensities(self):
        image = np.tile(np.array([10, 100, 200], dtype=np.uint8)[None, :, None], (8, 1, 3))
        Image.fromarray(image).save(self.reference)
        for clipping in (None, (0.01, 0.99)):
            with self.subTest(clipping=clipping):
                self.config.histogram_clip = clipping
                np.testing.assert_array_equal(self.build("histogram").transform(image), image)

    def test_histogram_maps_sparse_source_quantiles_to_reference_values(self):
        source = np.tile(np.array([20, 80, 140], dtype=np.uint8)[None, :, None], (8, 1, 3))
        target = np.tile(np.array([10, 100, 200], dtype=np.uint8)[None, :, None], (8, 1, 3))
        Image.fromarray(target).save(self.reference)
        self.config.histogram_clip = None
        np.testing.assert_array_equal(self.build("histogram").transform(source), target)

    def test_padding_excluded_and_restored_with_standardization(self):
        roi = tissue(4, (24, 28))
        image = np.full((40, 44, 3), 255, dtype=np.uint8)
        image[8:32, 7:35] = roi
        self.config.standardize = True
        for method in ("reinhard", "histogram"):
            with self.subTest(method=method):
                normalizer = self.build(method)
                expected = image.copy()
                expected[8:32, 7:35] = normalizer.transform(roi)
                actual = normalizer.transform(image, valid_bounds=(7, 8, 35, 32))
                np.testing.assert_array_equal(actual, expected)

    def test_blank_constant_empty_bounds_preserve_source(self):
        normalizer = self.build("reinhard")
        for value in (0, 150, 255):
            source = np.full((10, 12, 3), value, dtype=np.uint8)
            np.testing.assert_array_equal(normalizer.transform(source), source)
        source = tissue(4)
        np.testing.assert_array_equal(normalizer.transform(source, (2, 2, 2, 10)), source)

    def test_reference_is_fitted_once(self):
        normalizer = self.build("histogram")
        self.reference.unlink()
        normalizer.transform(tissue(2))
        normalizer.transform(tissue(3))
        self.assertIn("reference_sha256", normalizer.provenance)

    def test_constant_reference_rejected(self):
        Image.fromarray(np.full((20, 20, 3), 255, dtype=np.uint8)).save(self.reference)
        with self.assertRaisesRegex(ValueError, "blank or constant"):
            self.build("histogram")

    def test_nonconstant_nearwhite_reference_rejected(self):
        nearwhite = np.random.default_rng(3).integers(245, 256, size=(20, 20, 3), dtype=np.uint8)
        Image.fromarray(nearwhite).save(self.reference)
        for method in ("histogram", "reinhard", "macenko", "vahadane"):
            with (
                self.subTest(method=method),
                self.assertRaisesRegex(ValueError, "blank or constant"),
            ):
                self.build(method)

    def test_unknown_option_rejected_and_versions_recorded(self):
        with self.assertRaisesRegex(ValueError, "luminosity_thresold"):
            self.build("reinhard", luminosity_thresold=0.8)
        self.assertEqual(self.build("reinhard").provenance["versions"]["numpy"], np.__version__)
        self.assertIn("opencv", self.build("reinhard").provenance["versions"])

    def test_high_bit_depth_reference_rejected(self):
        Image.fromarray(np.arange(100, dtype=np.uint16).reshape(10, 10)).save(self.reference)
        with self.assertRaisesRegex(ValueError, "high-bit-depth"):
            self.build("reinhard")

    def test_transparent_reference_composites_onto_white(self):
        rgba = np.dstack((self.target, np.full(self.target.shape[:2], 128, dtype=np.uint8)))
        Image.fromarray(rgba).save(self.reference)
        loaded = classical._load_rgb(self.reference)
        expected = np.rint(self.target.astype(float) * 128 / 255 + 255 * 127 / 255).astype(np.uint8)
        np.testing.assert_array_equal(loaded, expected)

    def test_invalid_input_and_bounds_rejected(self):
        normalizer = self.build("reinhard")
        with self.assertRaises(ValueError):
            normalizer.transform(np.zeros((10, 10, 3), dtype=float))
        for bounds in ((-1, 0, 2, 2), (0, 0, 999, 2), (0.0, 0, 2, 2), (0, 0, 1)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                normalizer.transform(tissue(), bounds)

    def test_spams_is_lazy_and_installation_message_is_clear(self):
        real_import = classical.importlib.import_module

        def missing(name):
            if name == "spams":
                raise ImportError("missing")
            return real_import(name)

        with (
            patch.object(classical.sys, "platform", "linux"),
            patch.object(classical.importlib, "import_module", side_effect=missing),
        ):
            self.build("reinhard")
            self.build("histogram")
            with self.assertRaisesRegex(ImportError, "spams-bin==2.6.14"):
                self.build("macenko")

    @unittest.skipUnless(importlib.util.find_spec("spams"), "SPAMS is not installed")
    def test_real_macenko_projection_has_no_floating_point_exceptions(self):
        # NumPy 2 with macOS Accelerate previously raised spurious divide,
        # overflow and invalid flags while projecting this bounded RGB OD data.
        with np.errstate(all="raise"):
            normalizer = self.build("macenko")
            result = normalizer.transform(tissue(12, (24, 28)))
        self.assertTrue(np.isfinite(result).all())

    @unittest.skipUnless(importlib.util.find_spec("spams"), "SPAMS is not installed")
    def test_real_stain_backends_reject_singular_refs_preserve_singular_inputs(self):
        singular = np.repeat(np.arange(30, 130, dtype=np.uint8).reshape(10, 10, 1), 3, axis=2)
        # This source varies spatially, but cannot separate two independent stains.
        for method in ("macenko", "vahadane"):
            with self.subTest(method=method):
                normalizer = self.build(method, iterations=15)
                np.testing.assert_array_equal(normalizer.transform(singular), singular)
                Image.fromarray(singular).save(self.reference)
                with self.assertRaisesRegex(ValueError, "independent tissue colors"):
                    self.build(method)
                Image.fromarray(self.target).save(self.reference)

    @unittest.skipUnless(importlib.util.find_spec("spams"), "SPAMS is not installed")
    def test_real_stain_backends_shape_padding_and_determinism(self):
        roi = tissue(12, (24, 28))
        source = np.full((32, 36, 3), 255, dtype=np.uint8)
        source[4:28, 4:32] = roi
        original = source.copy()
        for method, standardize in (
            ("macenko", False),
            ("vahadane", False),
            ("macenko", True),
            ("vahadane", True),
        ):
            with self.subTest(method=method, standardize=standardize):
                self.config.standardize = standardize
                normalizer = self.build(method, iterations=15)
                result = normalizer.transform(source, (4, 4, 32, 28))
                expected = source.copy()
                expected[4:28, 4:32] = normalizer.transform(roi)
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(result, normalizer.transform(source, (4, 4, 32, 28)))
                second = self.build(method, iterations=15)
                np.testing.assert_array_equal(result, second.transform(source, (4, 4, 32, 28)))
                np.testing.assert_array_equal(source, original)
                self.assertEqual(result.dtype, np.uint8)
                self.assertTrue(np.isfinite(result).all())
                self.assertFalse(np.array_equal(result, source))
                blank = np.full((12, 12, 3), 255, dtype=np.uint8)
                np.testing.assert_array_equal(normalizer.transform(blank), blank)
                self.assertEqual(normalizer.provenance["solver"], "spams")

    @unittest.skipUnless(importlib.util.find_spec("spams"), "SPAMS is not installed")
    def test_worker_persists_across_calls_and_cleans_up(self):
        from openhisto.stain_transfer.spams_process import SpamsClient

        dictionary = np.asfortranarray(np.array(((1.0, 0.0), (0.0, 1.0), (0.1, 0.1))))
        data = np.asfortranarray(np.array(((1.0, 0.2), (0.3, 1.0), (0.13, 0.12))))
        client = SpamsClient()
        process = client._process
        try:
            first = client.lasso(
                X=data, D=dictionary, mode=2, lambda1=0.01, pos=True, numThreads=1
            ).toarray()
            second = client.lasso(
                X=data, D=dictionary, mode=2, lambda1=0.01, pos=True, numThreads=1
            ).toarray()
            np.testing.assert_array_equal(first, second)
            self.assertIsNone(process.poll())
            with self.assertRaisesRegex(RuntimeError, "Unsupported SPAMS worker operation"):
                client._request("unknown")
            self.assertEqual(client._request("hello"), {"ready": True})
        finally:
            client.close()
        self.assertIsNotNone(process.poll())
        client.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            client._request("hello")

    @unittest.skipUnless(importlib.util.find_spec("spams"), "SPAMS is not installed")
    def test_worker_crash_is_actionable_and_closes_client(self):
        from openhisto.stain_transfer.spams_process import SpamsClient

        client = SpamsClient()
        client._process.kill()
        client._process.wait(timeout=5)
        with self.assertRaisesRegex(RuntimeError, "SPAMS worker communication failed"):
            client._request("hello")
        self.assertTrue(client._closed)

    @unittest.skipUnless(
        sys.platform == "darwin"
        and importlib.util.find_spec("spams")
        and importlib.util.find_spec("torch"),
        "macOS SPAMS/Torch interoperability regression",
    )
    def test_mac_spams_then_torch_and_mixed_backend_do_not_conflict(self):
        # An OpenMP abort must fail this child, not take down the test runner.
        code = """
import json, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from openhisto.stain_transfer import DomainSpec, TransferConfig, create_normalizer
from openhisto.stain_transfer.tests.test_classical import tissue
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    rgb = tissue(3)
    ref = root / 'ref.png'
    Image.fromarray(rgb).save(ref)
    normalizer = create_normalizer(DomainSpec(references=(str(ref),)))
    normalizer.transform(rgb)
    assert normalizer.provenance['backends'][0]['execution'] == 'subprocess'
    assert 'spams' not in __import__('sys').modules
    import torch
    torch.set_num_threads(1)
    torch.nn.Linear(3, 2)(torch.ones(1, 3))
    from openhisto.stain_transfer.networks import generator
    weights = root / 'generator.pth'
    torch.save(generator('cyclegan', 'unet_128', ngf=2).state_dict(), weights)
    gan = DomainSpec(method='cyclegan', checkpoint=str(weights), architecture='unet_128',
                     input_mode='grayscale', direction='AtoB', options={'ngf': 2})
    with create_normalizer(DomainSpec(candidates=(DomainSpec(references=(str(ref),)), gan)),
                           TransferConfig(mix=True, device='cpu')) as mixed:
        mixed.transform(rgb, patch_id='one')
    normalizer.close()
print('ok')
"""
        env = worker_environment()
        completed = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
