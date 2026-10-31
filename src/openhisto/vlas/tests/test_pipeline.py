import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from openhisto.patch_extraction import PatchArchiveWriter
from openhisto.vlas.base import GenerationResult, ModelDescriptor, VLAError
from openhisto.vlas.config import DescribeConfig, ModelSpec, SourceSpec
from openhisto.vlas.pipeline import DescriptionFailure, describe_slide
from openhisto.vlas.registry import register_model
from openhisto.vlas.storage import DescriptionReader


STATE = {
    "prepared": 0,
    "calls": 0,
    "closed": 0,
    "fail_after": None,
    "truncated": False,
    "bad": False,
    "interrupt": False,
}


class DummyBackend:
    def __init__(self, spec, config):
        STATE["prepared"] += 1
        self.spec = spec
        self.metadata = {
            "model_id": spec.model,
            "resolved_revision": spec.revision or "synthetic",
            "adapter_version": 1,
            "precision": config.precision,
        }

    def generate(self, images, prompt):
        if STATE["interrupt"]:
            raise KeyboardInterrupt()
        if STATE["fail_after"] is not None and STATE["calls"] >= STATE["fail_after"]:
            raise VLAError("Synthetic generation interruption")
        STATE["calls"] += 1
        if STATE["bad"]:
            return [GenerationResult("")]
        return [
            GenerationResult(
                f"{self.spec.model}: {prompt}: pixel={int(image[0, 0, 0])}",
                truncated=STATE["truncated"],
                finish_reason="length" if STATE["truncated"] else "stop",
            )
            for image in images
        ]

    def close(self):
        STATE["closed"] += 1


for name in ("test_description", "test_comparison"):
    register_model(
        ModelDescriptor(
            name,
            name,
            "synthetic-checkpoint",
            "a" * 40,
            "https://example.invalid",
            "synthetic-code",
            "test",
            "synthetic",
        ),
        DummyBackend,
    )


def archive(path, colors, *, mag=20, coordinate_offset=0):
    with PatchArchiveWriter(
        path, patch_size=4, metadata={"target_mag": mag, "source_mag": 40}, overwrite=path.exists()
    ) as writer:
        for index, color in enumerate(colors):
            writer.append(
                np.full((4, 4, 3), color, dtype=np.uint8),
                (index * 8 + coordinate_offset, 0),
                (8, 8),
                (0, 0, 4, 4),
            )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="cpath-vlas-pipeline-")
        self.root = Path(self.temporary.name)
        self.input = self.root / "patches.h5"
        self.output = self.root / "descriptions"
        self.colors = [20, 50, 80]
        archive(self.input, self.colors)
        self.config = DescribeConfig(slide_id="synthetic-slide", batch_size=2, device="cpu")
        self.prompt = "Describe the visible microscopic morphology."
        STATE.update(
            prepared=0,
            calls=0,
            closed=0,
            fail_after=None,
            truncated=False,
            bad=False,
            interrupt=False,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def describe(self, models=None, prompt=None, **kwargs):
        return describe_slide(
            kwargs.pop("inputs", self.input),
            self.output,
            models or ["test_description"],
            prompt or self.prompt,
            config=kwargs.pop("config", self.config),
            **kwargs,
        )

    def journal(self):
        path = self.output / "descriptions.jsonl"
        return (
            [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if path.exists()
            else []
        )

    def test_round_trip_preserves_patch_alignment_and_cached_run_skips_model(self):
        report = self.describe()
        self.assertEqual(
            (report.patches, report.completed, report.reused, report.failed), (3, 3, 0, 0)
        )
        journal_before = (self.output / "descriptions.jsonl").read_bytes()
        with patch(
            "openhisto.vlas.pipeline.prepare_model",
            side_effect=AssertionError("Cached descriptions must not prepare a model"),
        ):
            cached = self.describe(config=replace(self.config, offline=True))
        self.assertEqual((cached.completed, cached.reused), (0, 3))
        self.assertEqual((self.output / "descriptions.jsonl").read_bytes(), journal_before)
        with DescriptionReader(self.output) as reader:
            metadata = reader.patch_metadata()
            records = reader.records()
            self.assertEqual([row["x_level0"] for row in metadata], [0, 8, 16])
            self.assertEqual([row["patch_id"] for row in metadata], ["0", "1", "2"])
            self.assertEqual(
                [row["patch_key"] for row in records], [row["patch_key"] for row in metadata]
            )
            self.assertEqual(
                [row["description"].split("pixel=")[-1] for row in records], ["20", "50", "80"]
            )
            self.assertEqual(reader.list_runs()[0]["prompt"], self.prompt)
        self.assertEqual(STATE["prepared"], STATE["closed"])
        self.assertFalse(Path(str(self.output) + ".lock").exists())

    def test_multiple_models_and_magnifications_share_one_output_and_join_keys(self):
        other = self.root / "patches-10x.h5"
        archive(other, [100, 130], mag=10)
        report = self.describe(
            models=["test_description", "test_comparison"],
            inputs=[
                SourceSpec(self.input, source_id="grid20"),
                SourceSpec(other, source_id="grid10"),
            ],
        )
        self.assertEqual((report.patches, report.completed), (5, 10))
        with DescriptionReader(self.output) as reader:
            metadata = reader.patch_metadata()
            self.assertEqual([row["mag"] for row in metadata], [20, 20, 20, 10, 10])
            keys = {row["patch_key"] for row in metadata}
            self.assertEqual(len(keys), 5)
            self.assertEqual(len(reader.records()), 10)
            for run in reader.list_runs():
                records = reader.records(run["run_key"])
                self.assertEqual({row["patch_key"] for row in records}, keys)
            selected = [metadata[-1]["patch_key"], metadata[0]["patch_key"]]
            chosen_run = reader.list_runs()[0]["run_key"]
            records = reader.records(chosen_run, patch_keys=selected)
            self.assertEqual({row["patch_key"] for row in records}, set(selected))

    def test_new_prompt_revision_and_generation_settings_create_separate_runs(self):
        self.describe()
        self.describe(prompt="Describe nuclear features only.")
        self.describe(models=[ModelSpec("test_description", revision="b" * 40)])
        self.describe(config=replace(self.config, max_new_tokens=128))
        with DescriptionReader(self.output) as reader:
            self.assertEqual(len(reader.list_runs()), 4)
            self.assertEqual(len(reader.records()), 12)
        self.assertEqual(STATE["prepared"], 4)

    def test_appended_patches_keep_old_order_and_fill_only_missing_pairs(self):
        self.describe(models=["test_description", "test_comparison"])
        with DescriptionReader(self.output) as reader:
            keys_before = [row["patch_key"] for row in reader.patch_metadata()]
        archive(self.input, self.colors + [110, 140])
        report = self.describe()
        self.assertEqual((report.patches, report.completed, report.reused), (5, 2, 3))
        with DescriptionReader(self.output) as reader:
            keys_after = [row["patch_key"] for row in reader.patch_metadata()]
            self.assertEqual(keys_after[:3], keys_before)
            runs = {run["model"]: run for run in reader.list_runs()}
            self.assertEqual(len(reader.records(runs["test_description"]["run_key"])), 5)
            self.assertEqual(len(reader.records(runs["test_comparison"]["run_key"])), 3)
        followup = self.describe(models=["test_comparison"])
        self.assertEqual((followup.completed, followup.reused), (2, 3))
        pairs = [(row["run_key"], row["patch_key"]) for row in self.journal()]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_failure_resumes_committed_rows_without_duplicate_journal_pairs(self):
        archive(self.input, self.colors + [110, 140])
        STATE["fail_after"] = 1
        with self.assertRaisesRegex(
            DescriptionFailure, "Synthetic generation interruption"
        ) as caught:
            self.describe()
        self.assertEqual(caught.exception.report.completed, 2)
        self.assertGreater(caught.exception.report.failed, 0)
        self.assertEqual(len(self.journal()), 2)
        with DescriptionReader(self.output) as reader:
            self.assertEqual(reader.list_runs()[0]["status"], "failed")
            self.assertEqual(len(reader.records()), 2)
        STATE["fail_after"] = None
        report = self.describe()
        self.assertEqual((report.completed, report.reused, report.failed), (3, 2, 0))
        self.assertEqual(len(self.journal()), 5)
        self.describe()
        pairs = [(row["run_key"], row["patch_key"]) for row in self.journal()]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(STATE["prepared"], STATE["closed"])

    def test_truncated_answers_remain_visible_and_are_reused(self):
        STATE["truncated"] = True
        report = self.describe()
        self.assertEqual(report.truncated, 3)
        with DescriptionReader(self.output) as reader:
            records = reader.records()
            self.assertEqual(len(records), 3)
            self.assertTrue(
                all(
                    row["status"] == "truncated" and row["finish_reason"] == "length"
                    for row in records
                )
            )
        with patch(
            "openhisto.vlas.pipeline.prepare_model",
            side_effect=AssertionError("Truncated answers should stay cached"),
        ):
            cached = self.describe()
        self.assertEqual(cached.reused, 3)

    def test_existing_pixel_or_geometry_mutation_fails_before_model_preparation(self):
        self.describe()
        journal_hash = hashlib.sha256((self.output / "descriptions.jsonl").read_bytes()).hexdigest()
        for colors, offset in (([21, 50, 80], 0), (self.colors, 1)):
            with self.subTest(offset=offset):
                archive(self.input, colors, coordinate_offset=offset)
                with patch(
                    "openhisto.vlas.pipeline.prepare_model",
                    side_effect=AssertionError("Changed inputs must fail before inference"),
                ):
                    with self.assertRaisesRegex(
                        (DescriptionFailure, VLAError, ValueError), "changed|immutable"
                    ):
                        self.describe()
                self.assertEqual(
                    hashlib.sha256((self.output / "descriptions.jsonl").read_bytes()).hexdigest(),
                    journal_hash,
                )

    def test_csv_snapshot_retains_multiline_prompt_and_description(self):
        prompt = 'Describe morphology, then explain.\nInclude "nuclear" features.'
        self.describe(prompt=prompt)
        with (self.output / "descriptions.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(prompt in row["description"] for row in rows))
        self.assertEqual(
            {row["patch_key"] for row in rows}, {row["patch_key"] for row in self.journal()}
        )
        self.assertEqual(
            {row["description"] for row in rows}, {row["description"] for row in self.journal()}
        )

    def test_incomplete_journal_tail_recovers_without_repeating_completed_work(self):
        self.describe()
        with (self.output / "descriptions.jsonl").open("ab") as stream:
            stream.write(b'{"patch_key": "incomplete')
        with patch(
            "openhisto.vlas.pipeline.prepare_model",
            side_effect=AssertionError("Recovered cached run must not load a model"),
        ):
            report = self.describe()
        self.assertEqual(report.reused, 3)
        self.assertEqual(len(self.journal()), 3)

    def test_writer_lock_prevents_mutation_and_is_not_silently_removed(self):
        self.describe()
        before = (self.output / "descriptions.jsonl").read_bytes()
        lock = Path(str(self.output) + ".lock")
        lock.write_text(json.dumps({"pid": 123456789}), encoding="utf-8")
        with self.assertRaisesRegex(
            (DescriptionFailure, VLAError, ValueError, FileExistsError), "lock|writer"
        ):
            self.describe()
        self.assertTrue(lock.exists())
        self.assertEqual((self.output / "descriptions.jsonl").read_bytes(), before)

    def test_invalid_backend_answer_is_not_committed_as_success(self):
        STATE["bad"] = True
        with self.assertRaises(DescriptionFailure) as caught:
            self.describe()
        self.assertGreater(caught.exception.report.failed, 0)
        self.assertEqual(self.journal(), [])
        self.assertFalse(Path(str(self.output) + ".lock").exists())

    def test_keyboard_interrupt_persists_state_and_releases_lock(self):
        STATE["interrupt"] = True
        with self.assertRaises(KeyboardInterrupt):
            self.describe()
        self.assertFalse(Path(str(self.output) + ".lock").exists())
        with DescriptionReader(self.output) as reader:
            self.assertEqual(reader.list_runs()[0]["status"], "interrupted")
            self.assertEqual(reader.records(), [])
        STATE["interrupt"] = False
        self.assertEqual(self.describe().completed, 3)

    def test_all_selected_runs_are_registered_before_first_model_failure(self):
        STATE["fail_after"] = 0
        with self.assertRaises(DescriptionFailure) as caught:
            self.describe(models=["test_description", "test_comparison"])
        with DescriptionReader(self.output) as reader:
            runs = {run["model"]: run for run in reader.list_runs()}
            self.assertEqual(set(runs), {"test_description", "test_comparison"})
            self.assertEqual(runs["test_description"]["status"], "failed")
            self.assertEqual(runs["test_comparison"]["status"], "pending")
            self.assertEqual(
                {run["run_key"] for run in runs.values()}, set(caught.exception.report.runs)
            )
            self.assertTrue(all(len(run["patch_keys"]) == 3 for run in runs.values()))
            self.assertEqual(
                sum(run["pending"] for run in reader.status()["runs"]),
                caught.exception.report.pending,
            )
        self.assertEqual(STATE["prepared"], 1)

    def test_cached_rerun_normalizes_stale_status_and_terminal_failure_metadata(self):
        self.describe()
        path = self.output / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        run = next(iter(metadata["runs"].values()))
        run["status"] = "running"
        run["failures"] = {
            key: {"status": "failed", "error": "stale failure preceding committed answer"}
            for key in run["patch_keys"]
        }
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with patch(
            "openhisto.vlas.pipeline.prepare_model",
            side_effect=AssertionError("Recovery must reuse terminal answers"),
        ):
            report = self.describe()
        self.assertEqual(
            (report.completed, report.reused, report.failed, report.pending), (0, 3, 0, 0)
        )
        with DescriptionReader(self.output) as reader:
            run = reader.list_runs()[0]
            self.assertEqual(run["status"], "complete")
            self.assertEqual(run["failures"], {})
            self.assertEqual(reader.status()["runs"][0]["status"], "complete")

    def test_cleanup_error_preserves_original_generation_failure(self):
        STATE["fail_after"] = 0
        with patch.object(
            DummyBackend, "close", side_effect=RuntimeError("Synthetic cleanup failure")
        ):
            with self.assertRaisesRegex(
                DescriptionFailure, "Synthetic generation interruption"
            ) as caught:
                self.describe()
        self.assertGreater(caught.exception.report.failed, 0)
        self.assertFalse(Path(str(self.output) + ".lock").exists())
        with DescriptionReader(self.output) as reader:
            run = reader.list_runs()[0]
            self.assertEqual(run["status"], "failed")
            self.assertIn("Synthetic cleanup failure", run["cleanup_error"])
            self.assertEqual(reader.records(), [])

    def test_cleanup_error_after_success_keeps_answers_csv_and_report(self):
        with patch.object(
            DummyBackend, "close", side_effect=RuntimeError("Synthetic cleanup failure")
        ):
            report = self.describe()
        self.assertEqual((report.completed, report.failed, report.pending), (3, 0, 0))
        with DescriptionReader(self.output) as reader:
            run = reader.list_runs()[0]
            self.assertEqual(run["status"], "complete")
            self.assertIn("Synthetic cleanup failure", run["cleanup_error"])
            self.assertEqual(len(reader.records()), 3)
        with (self.output / "descriptions.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)
        self.assertFalse(Path(str(self.output) + ".lock").exists())

    def test_local_checkpoint_content_change_at_same_path_creates_new_run(self):
        checkpoint = self.root / "local-model.bin"
        checkpoint.write_bytes(b"synthetic weights version one")
        spec = ModelSpec("test_description", checkpoint=str(checkpoint))
        first = self.describe(models=[spec])
        checkpoint.write_bytes(b"synthetic weights version two")
        second = self.describe(models=[spec])
        self.assertNotEqual(first.runs, second.runs)
        self.assertEqual((first.completed, second.completed), (3, 3))
        with DescriptionReader(self.output) as reader:
            self.assertEqual(len(reader.list_runs()), 2)
            self.assertEqual(len(reader.records()), 6)
        with patch(
            "openhisto.vlas.pipeline.prepare_model",
            side_effect=AssertionError("Unchanged checkpoint must stay cached"),
        ):
            cached = self.describe(models=[spec])
        self.assertEqual((cached.completed, cached.reused), (0, 3))


if __name__ == "__main__":
    unittest.main()
