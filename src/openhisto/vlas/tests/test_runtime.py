"""Exercise the real subprocess and PNG/JSONL protocol without model weights."""

from dataclasses import replace
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import textwrap
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from openhisto.vlas.base import VLAError
from openhisto.vlas.config import DescribeConfig, ModelSpec
from openhisto.vlas.runtime import WorkerClient


FIXTURE_WORKER = r"""
import os
import sys
import time

from openhisto.vlas.base import GenerationResult, ModelDescriptor
from openhisto.vlas.registry import register_model
from openhisto.vlas.worker import main


class FixtureBackend:
    def __init__(self, spec, config):
        mode = spec.options.get("mode")
        if mode == "startup_sleep":
            time.sleep(20)
        if mode == "startup_exit":
            print("native startup diagnostic " + os.environ.get("FIXTURE_API_TOKEN", ""), file=sys.stderr, flush=True)
            os._exit(19)
        if mode == "startup_error":
            raise RuntimeError("fixture preparation failed")
        self.requests = 0
        print("vendor startup stdout", flush=True)
        if spec.options.get("log_burst"):
            for index in range(100):
                print("vendor noisy stdout %s " % index + "x" * 4096, flush=True)
        self.metadata = {
            "model": spec.model,
            "resolved_revision": "a" * 40,
            "source_revision": "b" * 40,
            "pid": os.getpid(),
            "offline": config.offline,
            "cache_dir": config.cache_dir,
            "hub_cache": os.environ.get("HF_HUB_CACHE"),
            "offline_environment": os.environ.get("HF_HUB_OFFLINE"),
        }

    def generate(self, images, prompt):
        if prompt == "raise":
            raise RuntimeError("fixture generation failed")
        if prompt == "exit":
            print("native generation diagnostic", file=sys.stderr, flush=True)
            os._exit(17)
        if prompt == "sleep":
            time.sleep(20)
        if prompt == "protocol_noise":
            os.write(1, b"not a JSON protocol response\n")
        self.requests += 1
        print("vendor generate stdout", flush=True)
        results = []
        for image in images:
            assert image.mode == "RGB"
            # Image.format survives PNG protocol decoding only until copy();
            # exact RGB pixels and dimensions prove lossless transport.
            red, green, blue = image.getpixel((0, 0))
            value = "%s:%s,%s,%s:%sx%s:request=%s" % (
                prompt, red, green, blue, image.width, image.height, self.requests
            )
            results.append(GenerationResult(value))
        return results

    def close(self):
        print("vendor close stdout", flush=True)


register_model(
    ModelDescriptor("fixture", "Fixture", "fixture-checkpoint", "a" * 40,
                    "fixture-source", "b" * 40, "test", "test"),
    FixtureBackend,
)
raise SystemExit(main())
"""


@unittest.skipUnless(os.name == "posix", "Executable fixture wrapper uses a POSIX shebang")
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="cpath-vlas-worker-")
        self.root = Path(self.temporary.name)
        self.wrapper = self.root / "fixture-python"
        self.wrapper.write_text(
            "#!" + sys.executable + "\n" + textwrap.dedent(FIXTURE_WORKER), encoding="utf-8"
        )
        self.wrapper.chmod(0o755)
        self.config = DescribeConfig(device="cpu", startup_timeout=5, timeout=5)
        self.image = np.full((3, 5, 3), (17, 89, 231), dtype=np.uint8)

    def tearDown(self):
        self.temporary.cleanup()

    def spec(self, **options):
        return ModelSpec("fixture", python_executable=str(self.wrapper), options=options)

    def assert_cleaned_up(self, worker):
        self.assertTrue(worker._closed)
        self.assertIsNotNone(worker.process.poll())
        self.assertFalse(worker._stdout_thread.is_alive())
        self.assertFalse(worker._stderr_thread.is_alive())
        self.assertTrue(
            all(
                stream.closed
                for stream in (worker.process.stdin, worker.process.stdout, worker.process.stderr)
            )
        )

    def test_persistent_worker_round_trip_metadata_pixels_order_and_vendor_logs(self):
        second = Image.new("RGB", (2, 4), (201, 2, 77))
        with WorkerClient(self.spec(log_burst=True), self.config) as worker:
            pid = worker.process.pid
            self.assertEqual(worker.metadata["pid"], pid)
            self.assertEqual(worker.metadata["resolved_revision"], "a" * 40)
            first = worker.generate([self.image, second], "Describe")
            self.assertEqual(
                [row.text for row in first],
                [
                    "Describe:17,89,231:5x3:request=1",
                    "Describe:201,2,77:2x4:request=1",
                ],
            )
            # One process persists, while every request carries only its own prompt/images.
            repeated = worker.generate([second, self.image], "New prompt")
            self.assertEqual(worker.process.pid, pid)
            self.assertEqual(
                [row.text for row in repeated],
                [
                    "New prompt:201,2,77:2x4:request=2",
                    "New prompt:17,89,231:5x3:request=2",
                ],
            )
            self.assertTrue(
                all(not row.truncated and row.finish_reason == "stop" for row in repeated)
            )
            self.assertTrue(any("vendor" in line for line in worker._logs))
            self.assertLessEqual(len(worker._logs), 20)
        self.assert_cleaned_up(worker)
        worker.close()  # Idempotent even after child pipes have been closed.

    def test_offline_flag_and_cache_configuration_reach_real_worker(self):
        config = replace(self.config, offline=True, cache_dir=str(self.root / "cache"))
        with WorkerClient(self.spec(), config) as worker:
            self.assertTrue(worker.metadata["offline"])
            self.assertEqual(worker.metadata["offline_environment"], "1")
            self.assertEqual(worker.metadata["cache_dir"], config.cache_dir)
            self.assertEqual(worker.metadata["hub_cache"], config.cache_dir)
            self.assertEqual(
                worker.generate([self.image], "Describe")[0].text,
                "Describe:17,89,231:5x3:request=1",
            )
        self.assert_cleaned_up(worker)

    def test_native_stdout_noise_cannot_be_mistaken_for_a_protocol_result(self):
        worker = WorkerClient(self.spec(), self.config)
        with self.assertRaisesRegex(VLAError, "Invalid worker protocol"):
            worker.generate([self.image], "protocol_noise")
        self.assert_cleaned_up(worker)

    def test_generation_exception_is_actionable_and_cleans_up(self):
        worker = WorkerClient(self.spec(), self.config)
        with self.assertRaisesRegex(VLAError, "fixture generation failed"):
            worker.generate([self.image], "raise")
        self.assert_cleaned_up(worker)

    def test_abrupt_generation_exit_is_actionable_and_cleans_up(self):
        worker = WorkerClient(self.spec(), self.config)
        with self.assertRaisesRegex(VLAError, "exited before returning a response"):
            worker.generate([self.image], "exit")
        self.assert_cleaned_up(worker)

    def test_opt_in_startup_diagnostics_are_redacted_and_separate_from_error_text(self):
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.dict(os.environ, {"FIXTURE_API_TOKEN": "private-fixture-credential"}),
            ):
                worker = WorkerClient.__new__(WorkerClient)
                with self.assertRaises(VLAError) as failure:
                    worker.__init__(
                        self.spec(mode="startup_exit"),
                        replace(self.config, worker_diagnostics=enabled),
                    )
                error = failure.exception
                self.assertEqual(error.worker_diagnostics["exit_status"], 19)
                self.assertNotIn("native startup diagnostic", str(error))
                if enabled:
                    tail = error.worker_diagnostics["stderr_tail"]
                    self.assertIn("native startup diagnostic", tail)
                    self.assertIn("[redacted]", tail)
                    self.assertNotIn("private-fixture-credential", tail)
                else:
                    self.assertNotIn("stderr_tail", error.worker_diagnostics)
                self.assert_cleaned_up(worker)

    def test_opt_in_generation_diagnostics_survive_worker_cleanup(self):
        worker = WorkerClient(self.spec(), replace(self.config, worker_diagnostics=True))
        with self.assertRaises(VLAError) as failure:
            worker.generate([self.image], "exit")
        self.assertEqual(failure.exception.worker_diagnostics["exit_status"], 17)
        self.assertIn(
            "native generation diagnostic", failure.exception.worker_diagnostics["stderr_tail"]
        )
        self.assertNotIn("vendor", worker.metadata)
        self.assert_cleaned_up(worker)

    def test_startup_error_and_abrupt_exit_clean_up_partial_initialization(self):
        for mode, message in (
            ("startup_error", "fixture preparation failed"),
            ("startup_exit", "exited before returning a response"),
        ):
            with self.subTest(mode=mode):
                worker = WorkerClient.__new__(WorkerClient)
                with self.assertRaisesRegex(VLAError, message):
                    worker.__init__(self.spec(mode=mode), self.config)
                self.assert_cleaned_up(worker)

    def test_startup_timeout_is_bounded_and_terminates_child(self):
        worker = WorkerClient.__new__(WorkerClient)
        start = time.monotonic()
        with self.assertRaisesRegex(VLAError, "timed out"):
            worker.__init__(
                self.spec(mode="startup_sleep"), replace(self.config, startup_timeout=0.2)
            )
        self.assertLess(time.monotonic() - start, 5)
        self.assert_cleaned_up(worker)

    def test_generation_timeout_is_bounded_and_terminates_child(self):
        worker = WorkerClient(self.spec(), replace(self.config, timeout=0.2))
        start = time.monotonic()
        with self.assertRaisesRegex(VLAError, "timed out"):
            worker.generate([self.image], "sleep")
        self.assertLess(time.monotonic() - start, 5)
        self.assert_cleaned_up(worker)


if __name__ == "__main__":
    unittest.main()
