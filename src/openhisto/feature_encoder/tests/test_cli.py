import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from openhisto.feature_encoder import FeatureReader, ModelDescriptor, register_encoder
from openhisto.feature_encoder.__main__ import load_job, main
from openhisto.patch_extraction import PatchArchiveWriter


class DummyEncoder:
    def __init__(self, spec, *, config, auth):
        self.dimension = 3
        self.metadata = {"model_id": spec.model, "resolved_revision": "synthetic"}
        self.usage = {}

    def encode(self, images):
        return np.asarray([image.mean(axis=(0, 1)) for image in images], dtype=np.float32)

    def close(self):
        pass


register_encoder(
    ModelDescriptor("test_cli", "Synthetic CLI encoder", "test", "public", 3, "synthetic"),
    DummyEncoder,
)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="cpath-feature-cli-")
        self.root = Path(self.temporary.name)
        with PatchArchiveWriter(
            self.root / "patches.h5", patch_size=4, metadata={"target_mag": 20}
        ) as writer:
            for index, color in enumerate([(1, 2, 3), (4, 5, 6)]):
                writer.append(
                    np.full((4, 4, 3), color, dtype=np.uint8), (index * 4, 0), (4, 4), (0, 0, 4, 4)
                )
        self.job = self.root / "job.json"
        self.job.write_text(
            json.dumps(
                {
                    "slide_id": "synthetic-slide",
                    "inputs": [{"path": "patches.h5", "source_id": "source-20x"}],
                    "output": "features.h5ad",
                    "models": ["test_cli"],
                    "config": {"batch_size": 1},
                }
            )
        )

    def tearDown(self):
        self.temporary.cleanup()

    def run_cli(self, args):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(args)
        return code, json.loads(stdout.getvalue())

    def test_job_relative_paths_encode_status_and_cached_rerun(self):
        code, report = self.run_cli(["encode", "--job", str(self.job)])
        self.assertEqual(code, 0)
        self.assertEqual(report["completed"], 2)
        code, status = self.run_cli(["status", str(self.root / "features.h5ad")])
        self.assertEqual((status["patches"], status["runs"][0]["completed"]), (2, 2))
        self.assertFalse(status["resumable_partial"])
        with patch(
            "openhisto.feature_encoder.pipeline.prepare_model",
            side_effect=AssertionError("cached model load"),
        ):
            code, report = self.run_cli(["encode", "--job", str(self.job), "--offline"])
        self.assertEqual(report["reused"], 2)

    def test_secrets_are_not_accepted_in_job_files(self):
        value = json.loads(self.job.read_text())
        value["auth"] = {"gemini_api_key": "synthetic-secret-not-for-files"}
        self.job.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "never put secret"):
            load_job(self.job)

    def test_metadata_update_flag_overrides_job_policy_without_model_loading(self):
        Image.fromarray(np.full((4, 4, 3), 42, dtype=np.uint8)).save(self.root / "patch.png")
        row = {"patch_id": "patch-a", "image_path": "patch.png", "mag": 20, "quality": "unchecked"}
        manifest = self.root / "patches.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        job = json.loads(self.job.read_text())
        job["inputs"] = ["patches.jsonl"]
        job["config"]["metadata_policy"] = "error"
        self.job.write_text(json.dumps(job))
        self.run_cli(["encode", "--job", str(self.job)])
        row["quality"] = "reviewed"
        manifest.write_text(json.dumps(row) + "\n")
        with patch(
            "openhisto.feature_encoder.pipeline.prepare_model",
            side_effect=AssertionError("metadata update loaded a model"),
        ):
            code, report = self.run_cli(
                ["encode", "--job", str(self.job), "--metadata-policy", "update"]
            )
        self.assertEqual(
            (code, report["completed"], report["reused"], report["metadata_updated"]), (0, 0, 1, 1)
        )
        with FeatureReader(self.root / "features.h5ad") as reader:
            self.assertEqual(
                json.loads(reader.patch_metadata().iloc[0]["metadata_json"])["quality"], "reviewed"
            )

    def test_missing_output_status_reports_partial_name(self):
        (self.root / "features.partial.h5ad").write_bytes(b"synthetic-placeholder")
        code, status = self.run_cli(["status", str(self.root / "features.h5ad")])
        self.assertEqual(code, 0)
        self.assertFalse(status["published"])
        self.assertTrue(status["resumable_partial"])

    def test_models_catalogue_does_not_construct_encoders(self):
        with patch(
            "openhisto.feature_encoder.__main__.prepare_model",
            side_effect=AssertionError("model prepared"),
        ):
            code, result = self.run_cli(["models", "phikon_v2"])
        self.assertEqual(code, 0)
        self.assertEqual(result["dimension"], 1024)
        self.assertIn("approval_url", result)

    def test_prompt_flag_is_runtime_only_and_respected(self):
        value = json.loads(self.job.read_text())
        value["auth"] = {"prompt": True, "hf_token_env": "SYNTHETIC_HF_TOKEN"}
        self.job.write_text(json.dumps(value))
        _, _, _, _, auth = load_job(self.job)
        self.assertTrue(auth.prompt)
        self.assertEqual(auth.hf_token_env, "SYNTHETIC_HF_TOKEN")


if __name__ == "__main__":
    unittest.main()
