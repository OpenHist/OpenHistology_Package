import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from openhisto._common import worker_environment
from openhisto.patch_extraction import PatchArchiveWriter
from openhisto.vlas.base import GenerationResult, ModelDescriptor, VLAError
from openhisto.vlas.__main__ import load_job, main
from openhisto.vlas.registry import register_model
from openhisto.vlas.storage import DescriptionReader


STATE = {"prepared": 0, "calls": 0, "closed": 0, "fail": False}


class CliBackend:
    def __init__(self, spec, config):
        STATE["prepared"] += 1
        self.metadata = {
            "model_id": spec.model,
            "resolved_revision": spec.revision or "a" * 40,
            "precision": config.precision,
        }

    def generate(self, images, prompt):
        STATE["calls"] += 1
        if STATE["fail"]:
            raise VLAError("Synthetic CLI failure")
        return [GenerationResult(f"{prompt} {int(image[0, 0, 0])}") for image in images]

    def close(self):
        STATE["closed"] += 1


register_model(
    ModelDescriptor(
        "test_cli_descriptions",
        "Synthetic CLI",
        "synthetic-checkpoint",
        "a" * 40,
        "https://example.invalid",
        "fixture-code",
        "test",
        "synthetic",
    ),
    CliBackend,
)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="cpath-vlas-cli-")
        self.root = Path(self.temporary.name).resolve()
        with PatchArchiveWriter(
            self.root / "patches.h5", patch_size=4, metadata={"target_mag": 20}
        ) as writer:
            for i, color in enumerate((20, 80)):
                writer.append(
                    np.full((4, 4, 3), color, dtype=np.uint8), (i * 4, 0), (4, 4), (0, 0, 4, 4)
                )
        self.job = self.root / "job.json"
        self.value = {
            "slide_id": "synthetic-slide",
            "inputs": [{"path": "patches.h5", "source_id": "grid20"}],
            "output": "answers",
            "models": ["test_cli_descriptions"],
            "prompt": "Describe morphology.",
            "config": {"device": "cpu", "batch_size": 1},
        }
        self.write_job()
        STATE.update(prepared=0, calls=0, closed=0, fail=False)

    def tearDown(self):
        self.temporary.cleanup()

    def write_job(self):
        self.job.write_text(json.dumps(self.value), encoding="utf-8")

    def run_cli(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = main(args)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_relative_job_describe_status_and_offline_cache(self):
        inputs, output, models, prompt, config = load_job(self.job)
        self.assertEqual(Path(inputs[0].path), self.root / "patches.h5")
        self.assertEqual(Path(output), self.root / "answers")
        self.assertEqual(prompt, self.value["prompt"])
        self.assertEqual(config.slide_id, "synthetic-slide")
        code, stdout, stderr = self.run_cli(["describe", "--job", str(self.job)])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["completed"], 2)
        code, stdout, stderr = self.run_cli(["status", str(self.root / "answers")])
        self.assertEqual(code, 0, stderr)
        status = json.loads(stdout)
        self.assertEqual(status["patches"], 2)
        self.assertEqual(status["runs"][0]["completed"], 2)
        with patch(
            "openhisto.vlas.pipeline.prepare_model",
            side_effect=AssertionError("Cached CLI run prepared a backend"),
        ):
            code, stdout, stderr = self.run_cli(["describe", "--job", str(self.job), "--offline"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["reused"], 2)

    @unittest.skipUnless(os.name == "posix", "Executable fixture wrapper uses a POSIX shebang")
    def test_worker_diagnostics_are_opt_in_redacted_and_not_saved_in_result_metadata(self):
        wrapper = self.root / "failing-python"
        wrapper.write_text(
            "#!" + sys.executable + "\n"
            "import os, sys\n"
            'print("libcuda.so.1 missing " + os.environ["FIXTURE_API_TOKEN"], file=sys.stderr, flush=True)\n'
            "raise SystemExit(42)\n"
        )
        wrapper.chmod(0o755)
        args = ["prepare", "patho_r1_7b", "--python-executable", str(wrapper), "--device", "cpu"]
        with patch.dict(os.environ, {"FIXTURE_API_TOKEN": "private-cli-credential"}):
            code, _, stderr = self.run_cli(args)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stderr)["worker"], {"exit_status": 42})
            self.assertNotIn("libcuda.so.1", stderr)
            code, _, stderr = self.run_cli(args + ["--worker-diagnostics"])
            self.assertEqual(code, 2)
            self.assertIn("libcuda.so.1", json.loads(stderr)["worker"]["stderr_tail"])
            self.assertNotIn("private-cli-credential", stderr)
            self.value["models"] = [{"model": "patho_r1_7b", "python_executable": str(wrapper)}]
            self.write_job()
            code, _, stderr = self.run_cli(
                ["describe", "--job", str(self.job), "--worker-diagnostics"]
            )
        self.assertEqual(code, 1)
        self.assertIn("libcuda.so.1", json.loads(stderr)["worker"]["stderr_tail"])
        self.assertNotIn("private-cli-credential", stderr)
        saved = (self.root / "answers/metadata.json").read_text()
        self.assertNotIn("libcuda.so.1", saved)
        self.assertNotIn("private-cli-credential", saved)

    def test_diagnostics_setting_does_not_change_description_run_identity(self):
        from openhisto.vlas.config import DescribeConfig

        self.assertEqual(
            DescribeConfig().identity(), DescribeConfig(worker_diagnostics=True).identity()
        )

    def test_job_resolves_runtime_artifacts_relative_to_job_file(self):
        self.value["models"] = [
            {
                "model": "test_cli_descriptions",
                "checkpoint": "weights/checkpoint",
                "python_executable": "env/bin/python",
                "options": {"upstream_path": "native", "vision_checkpoint": "weights/vision"},
            }
        ]
        self.value["config"]["cache_dir"] = "cache"
        self.write_job()
        _, _, models, _, config = load_job(self.job)
        spec = models[0]
        self.assertEqual(Path(spec.checkpoint), self.root / "weights/checkpoint")
        self.assertEqual(Path(spec.python_executable), self.root / "env/bin/python")
        self.assertEqual(Path(spec.options["upstream_path"]), self.root / "native")
        self.assertEqual(Path(spec.options["vision_checkpoint"]), self.root / "weights/vision")
        self.assertEqual(Path(config.cache_dir), self.root / "cache")
        self.value["models"][0]["python_executable"] = "python3"
        self.write_job()
        self.assertEqual(load_job(self.job)[2][0].python_executable, "python3")

    def test_unknown_job_fields_missing_prompt_and_embedded_credentials_fail(self):
        for change in (
            {"unexpected": True},
            {"prompt": ""},
            {
                "models": [
                    {"model": "test_cli_descriptions", "options": {"hf_token": "synthetic-secret"}}
                ]
            },
        ):
            with self.subTest(change=change):
                original = dict(self.value)
                self.value.update(change)
                self.write_job()
                code, _, _ = self.run_cli(["describe", "--job", str(self.job)])
                self.assertEqual(code, 2)
                self.assertEqual(STATE["prepared"], 0)
                self.value = original

    def test_generation_failure_returns_one_and_keeps_reviewable_failure_state(self):
        STATE["fail"] = True
        code, stdout, stderr = self.run_cli(["describe", "--job", str(self.job)])
        self.assertEqual(code, 1)
        self.assertIn("Synthetic CLI failure", stdout + stderr)
        with DescriptionReader(self.root / "answers") as reader:
            self.assertEqual(reader.records(), [])
            self.assertEqual(reader.status()["runs"][0]["status"], "failed")
        self.assertFalse(Path(str(self.root / "answers") + ".lock").exists())

    def test_prepare_initializes_selected_model_without_image_generation(self):
        code, stdout, stderr = self.run_cli(
            ["prepare", "test_cli_descriptions", "--device", "cpu", "--options", "{}"]
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["metadata"]["model_id"], "test_cli_descriptions")
        self.assertEqual((STATE["prepared"], STATE["calls"], STATE["closed"]), (1, 0, 1))

    def test_models_and_help_work_when_model_libraries_are_blocked(self):
        environment = worker_environment()
        program = """
import builtins, sys
original_import = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'accelerate', 'llava', 'xtuner', 'pallava'}:
        raise AssertionError('Catalogue imported optional model library: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked
from openhisto.vlas.__main__ import main
raise SystemExit(main(sys.argv[1:]))
"""
        for command in (["models"], ["models", "patho_r1_7b"], ["--help"]):
            with self.subTest(command=command):
                process = subprocess.run(
                    [sys.executable, "-c", program, *command],
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                if command[0] == "models":
                    result = json.loads(process.stdout)
                    self.assertTrue(result)
                else:
                    self.assertIn("describe", process.stdout)


if __name__ == "__main__":
    unittest.main()
