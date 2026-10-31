"""Public API/CLI behavior with real lightweight normalizers."""

import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from openhisto._common import worker_environment
from openhisto.stain_transfer import (
    DomainSpec,
    TransferConfig,
    TrainConfig,
    create_normalizer,
    list_methods,
)
from openhisto.stain_transfer.__main__ import main


class PublicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(5)
        self.image = rng.integers(30, 200, size=(30, 35, 3), dtype=np.uint8)
        self.ref = self.root / "ref.png"
        Image.fromarray(self.image[:, :, ::-1]).save(self.ref)

    def tearDown(self):
        self.temp.cleanup()

    def test_defaults_validation_and_aliases(self):
        self.assertEqual(DomainSpec(method="default").method, "vahadane")
        self.assertEqual(DomainSpec(method="cycleGAN").method, "cyclegan")
        self.assertEqual(len(list_methods()), 6)
        self.assertEqual(TransferConfig().seed, 256)
        self.assertFalse(TransferConfig().standardize)
        for value in (
            {"histogram_clip": (0.5, 0.2)},
            {"seed": True},
            {"device": "cuda:bad"},
            {"png_compression": 10},
        ):
            with self.assertRaises(ValueError):
                TransferConfig(**value)
        with self.assertRaises(ValueError):
            TrainConfig(method="reinhard")
        with self.assertRaises(ValueError):
            TrainConfig(resume="a", warm_start="b")

    def test_missing_or_ambiguous_destination(self):
        with self.assertRaises(ValueError):
            create_normalizer(DomainSpec())
        with self.assertRaises(ValueError):
            create_normalizer(DomainSpec(method="reinhard", references=(self.ref, self.ref)))
        with self.assertRaises(ValueError):
            create_normalizer(
                DomainSpec(method="reinhard", references=(self.ref,)), TransferConfig(mix=True)
            )
        with self.assertRaises(ValueError):
            create_normalizer(
                DomainSpec(method="reinhard", references=(self.ref,), input_mode="grayscale")
            )
        with self.assertRaises(ValueError):
            DomainSpec(
                candidates=(DomainSpec(method="reinhard", references=(self.ref,)),),
                options={"unused": 1},
            )

    def test_normalizer_context_releases_resources(self):
        with create_normalizer(DomainSpec(method="reinhard", references=(self.ref,))) as normalizer:
            self.assertEqual(normalizer.transform(self.image).shape, self.image.shape)
        normalizer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            normalizer.transform(self.image)

    def test_mixture_stable_patch_identity_and_bounds(self):
        domain = DomainSpec(
            candidates=(
                DomainSpec(method="reinhard", references=(self.ref,)),
                DomainSpec(method="histogram", references=(self.ref,)),
            )
        )
        norm = create_normalizer(domain, TransferConfig(mix=True))
        identities = ["patch" + str(i) for i in range(30)]
        forward = {key: norm.transform(self.image, patch_id=key) for key in identities}
        for key in reversed(identities):
            np.testing.assert_array_equal(forward[key], norm.transform(self.image, patch_id=key))
        self.assertEqual(set(norm.selected_index(key) for key in identities), {0, 1})
        with self.assertRaises(ValueError):
            norm.transform(self.image)
        bounds = (2, 3, 32, 25)
        result = norm.transform(self.image, bounds, "test")
        np.testing.assert_array_equal(result[:3], self.image[:3])
        np.testing.assert_array_equal(result[:, :2], self.image[:, :2])
        blank = np.full_like(self.image, 255)
        np.testing.assert_array_equal(norm.transform(blank, patch_id="blank"), blank)

    def test_cli_job_relative_paths_and_no_cwd_dependence(self):
        Image.fromarray(self.image).save(self.root / "input.png")
        job = self.root / "job.json"
        job.write_text(
            json.dumps(
                {
                    "input": "input.png",
                    "output": "out.png",
                    "domain": {"method": "reinhard", "references": ["ref.png"]},
                }
            )
        )
        env = worker_environment()
        result = subprocess.run(
            [sys.executable, "-m", "openhisto.stain_transfer", "normalize", "--job", str(job)],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["processed"], 1)
        self.assertTrue((self.root / "out.png").is_file())
        # Existing output must fail explicitly.
        result = subprocess.run(
            [sys.executable, "-m", "openhisto.stain_transfer", "normalize", "--job", str(job)],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stderr)["type"], "FileExistsError")

    def test_cli_flags_errors_and_json_results(self):
        with patch("sys.stdout") as stdout:
            self.assertEqual(main(["methods"]), 0)
            payload = "".join(call.args[0] for call in stdout.write.call_args_list)
            self.assertEqual(json.loads(payload)["default"], "vahadane")
        with patch("sys.stderr") as stderr:
            self.assertEqual(main(["normalize"]), 1)
            payload = "".join(call.args[0] for call in stderr.write.call_args_list)
            self.assertIn("requires input", json.loads(payload)["error"])
        job = self.root / "job.json"
        job.write_text("{}")
        with patch("sys.stderr"):
            self.assertEqual(main(["normalize", "--job", str(job), "--overwrite"]), 1)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            main(["normalize", "--job", str(job), "--met", "histogram"])

    def test_training_default_output_is_job_relative(self):
        job = self.root / "train.json"
        job.write_text(
            json.dumps({"method": "pix2pix", "target": "images", "grayscale_source": True})
        )
        with patch("openhisto.stain_transfer.__main__.train") as run, patch("sys.stdout"):
            run.return_value.to_dict.return_value = {}
            self.assertEqual(main(["train", "--job", str(job)]), 0)
            config = run.call_args.args[0]
            self.assertEqual(config.output, str((self.root / "stain_training").resolve()))
            self.assertEqual(config.target, str((self.root / "images").resolve()))

    def test_import_and_classical_run_do_not_load_torch_or_spams(self):
        code = "\n".join(
            [
                "import sys, numpy as np",
                "from openhisto.stain_transfer import *",
                f"normalizer = create_normalizer(DomainSpec(method='reinhard', references=({str(self.ref)!r},)))",
                "assert 'torch' not in sys.modules and 'spams' not in sys.modules",
                "assert set(list_methods()) == {'vahadane','macenko','reinhard','histogram','cyclegan','pix2pix'}",
            ]
        )
        result = subprocess.run(
            [sys.executable, "-c", code], env=worker_environment(), capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency unavailable")
    def test_gan_training_and_exported_normalization_jobs(self):
        images = self.root / "images"
        images.mkdir()
        Image.fromarray(self.image).save(images / "patch.png")
        training = self.root / "train.json"
        training.write_text(
            json.dumps(
                {
                    "method": "pix2pix",
                    "target": "images",
                    "output": "trained",
                    "grayscale_source": True,
                    "device": "cpu",
                    "ngf": 4,
                    "ndf": 4,
                    "epochs": 1,
                    "decay_epochs": 0,
                    "batch_size": 1,
                    "sample_every": 0,
                }
            )
        )
        env = dict(worker_environment(), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run(
            [sys.executable, "-m", "openhisto.stain_transfer", "train", "--job", str(training)],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["global_step"], 1)
        job = self.root / "infer.json"
        job.write_text(
            json.dumps(
                {
                    "input": "images/patch.png",
                    "output": "normalized.png",
                    "domain": {"method": "pix2pix", "checkpoint": "trained/generator_AtoB.pth"},
                    "config": {"device": "cpu"},
                }
            )
        )
        result = subprocess.run(
            [sys.executable, "-m", "openhisto.stain_transfer", "normalize", "--job", str(job)],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["processed"], 1)
        with Image.open(self.root / "normalized.png") as output:
            self.assertEqual(np.asarray(output).shape, self.image.shape)


if __name__ == "__main__":
    unittest.main()
