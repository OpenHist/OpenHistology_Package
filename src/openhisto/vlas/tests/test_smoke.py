"""Test smoke-check acceptance decisions with deterministic synthetic patches."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from openhisto.patch_extraction import PatchArchiveWriter
from openhisto.vlas.base import GenerationResult, ModelDescriptor, VLAError
from openhisto.vlas import registry
from openhisto.vlas.smoke import smoke_job


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="cpath-vlas-smoke-")
        self.root = Path(self.temporary.name)
        archive = self.root / "patches.h5"
        with PatchArchiveWriter(
            archive, patch_size=4, metadata={"target_mag": 20, "source_mag": 40}
        ) as writer:
            for index, color in enumerate((20, 80)):
                writer.append(
                    np.full((4, 4, 3), color, dtype=np.uint8), (index * 8, 0), (8, 8), (0, 0, 4, 4)
                )
        self.job = self.root / "job.json"
        self.job.write_text(
            json.dumps(
                {
                    "inputs": "patches.h5",
                    "output": "descriptions",
                    "models": ["smoke_fixture"],
                    "prompt": "Describe morphology",
                    "slide_id": "smoke-slide",
                    "config": {"device": "cpu"},
                }
            ),
            encoding="utf-8",
        )
        self.state = {
            "prepared": [],
            "closed": 0,
            "requests": [],
            "leak": False,
            "truncated": False,
            "metadata_drift": False,
        }
        state = self.state

        class Backend:
            def __init__(self, spec, config):
                self.offline = config.offline
                self.requests = 0
                state["prepared"].append(config.offline)
                self.metadata = {"model": spec.model, "resolved_revision": "a" * 40}
                if state["metadata_drift"] and config.offline:
                    self.metadata["resolved_revision"] = "b" * 40

            def generate(self, images, prompt):
                self.requests += 1
                state["requests"].append((self.offline, int(images[0][0, 0, 0]), prompt))
                suffix = f" request={self.requests}" if state["leak"] else ""
                return [
                    GenerationResult(
                        f"{prompt}: pixel={int(image[0, 0, 0])}{suffix}",
                        truncated=state["truncated"],
                        finish_reason="length" if state["truncated"] else "stop",
                    )
                    for image in images
                ]

            def close(self):
                state["closed"] += 1

        self.descriptors = patch.dict(registry._DESCRIPTORS)
        self.factories = patch.dict(registry._FACTORIES)
        self.descriptors.start()
        self.factories.start()
        registry.register_model(
            ModelDescriptor(
                "smoke_fixture", "Fixture", "fixture", "a" * 40, "fixture", "b" * 40, "test", "test"
            ),
            Backend,
        )

    def tearDown(self):
        self.factories.stop()
        self.descriptors.stop()
        self.temporary.cleanup()

    def test_repeat_requests_and_offline_reload_are_required_but_not_native_verification(self):
        report = smoke_job(self.job)
        self.assertFalse(report["verified"])
        self.assertFalse(report["native_reference_checked"])
        self.assertTrue(report["independent_conversations"])
        self.assertTrue(report["offline_reload"])
        self.assertEqual(self.state["prepared"], [False, True])
        self.assertEqual(self.state["closed"], 2)
        self.assertEqual(
            [row[:2] for row in self.state["requests"]],
            [(False, 20), (False, 80), (False, 20), (True, 80)],
        )
        self.assertEqual(len(report["records"]), 2)

    def test_matching_reference_sets_verified_and_mismatch_or_duplicates_fail(self):
        report = smoke_job(self.job)
        reference = self.root / "native-reference.jsonl"
        reference.write_text(
            "\n".join(json.dumps(row) for row in report["records"]) + "\n", encoding="utf-8"
        )
        verified = smoke_job(self.job, reference)
        self.assertTrue(verified["verified"])
        self.assertTrue(verified["native_reference_checked"])
        rows = [dict(row) for row in report["records"]]
        rows[0]["description"] = "different native result"
        reference.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(VLAError, "Native inference reference differs"):
            smoke_job(self.job, reference)
        reference.write_text(
            json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "duplicate patch keys"):
            smoke_job(self.job, reference)

    def test_state_leakage_and_offline_runtime_identity_drift_fail(self):
        self.state["leak"] = True
        with self.assertRaisesRegex(VLAError, "Repeated independent patch requests differed"):
            smoke_job(self.job)
        self.state.update(leak=False, metadata_drift=True)
        with self.assertRaisesRegex(VLAError, "Offline reload changed"):
            smoke_job(self.job)
        self.assertEqual(self.state["closed"], 4)

    def test_truncated_answer_fails_acceptance(self):
        self.state["truncated"] = True
        with self.assertRaisesRegex(VLAError, "truncated or unfinished"):
            smoke_job(self.job)


if __name__ == "__main__":
    unittest.main()
