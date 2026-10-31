"""Acquisition/idempotence tests against a tiny local Git origin, never GitHub."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from openhisto.mil.models.code_bases import sources
from openhisto.mil.models.code_bases.snapshot import build_manifest, MANIFEST
from openhisto.mil.models.code_bases.source_selection import SOURCE_EXCLUSIONS


def _git(directory, *arguments):
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=" + os.devnull, "-c", "commit.gpgsign=false", *arguments],
        cwd=directory,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    return result.stdout.strip()


def _commit(directory, message):
    _git(directory, "add", ".")
    _git(
        directory,
        "-c",
        "user.name=Synthetic Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    return _git(directory, "rev-parse", "HEAD")


class SourceAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cpath-source-fixture-")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.origin = root / "origin"
        self.origin.mkdir()
        _git(self.origin, "init", "--quiet", "--initial-branch=main")
        (self.origin / "model.py").write_text("VALUE = 1\n")
        (self.origin / "README.md").write_text("Synthetic source fixture.\n")
        (self.origin / "data").mkdir()
        (self.origin / "data/table.csv").write_text("item,value\nsynthetic,1\n")
        (self.origin / "weights.pt").write_bytes(b"synthetic excluded artifact")
        self.original_commit = _commit(self.origin, "Synthetic first revision")
        self.root = root / "managed"
        self.root.mkdir()
        self.checkout = self.root / "upstream/fixture"
        self.record = self.root / "compat/sources/fixture.json"
        self.calls = []
        original_git = sources._git
        expected_url = "https://github.com/synthetic/fixture.git"

        def local_only_git(*arguments, cwd=None):
            translated = []
            for argument in arguments:
                value = str(argument)
                if value == expected_url:
                    value = str(self.origin)
                elif value.startswith(("https://", "http://", "git@")):
                    raise AssertionError("Acquisition fixture attempted an external remote")
                translated.append(value)
            self.calls.append((tuple(translated), cwd))
            return original_git(*translated, cwd=cwd)

        patches = (
            patch.object(sources, "ROOT", self.root),
            patch.object(sources, "REPOSITORIES", {"fixture": ("synthetic/fixture", "main")}),
            patch.object(sources, "_git", side_effect=local_only_git),
            patch.dict("os.environ", {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    def _advance_origin(self):
        (self.origin / "model.py").write_text("VALUE = 2\n")
        (self.origin / "new_model.py").write_text("NEW_REVISION_ONLY = True\n")
        return _commit(self.origin, "Synthetic second revision")

    def _publish_vendor(self):
        record = sources.acquire_one("fixture")
        (self.root / MANIFEST).write_text(json.dumps(build_manifest(self.root)))
        shutil.rmtree(self.checkout / ".git")
        _git(self.root, "init", "--quiet")
        _commit(self.root, "Synthetic outer publication")
        self.calls.clear()
        return record

    def test_existing_vendor_under_outer_git_is_verified_without_network(self):
        record = self._publish_vendor()
        self.assertNotEqual(_git(self.root, "rev-parse", "HEAD"), record["commit"])
        self.assertEqual(sources.acquire_one("fixture"), record)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.checkout / ".git").exists())

    def test_explicit_source_exclusions_survive_publication_and_pinned_restoration(self):
        excluded = "models/incomplete.py"
        (self.origin / "models").mkdir()
        (self.origin / excluded).write_text("def incomplete():\n")
        self.original_commit = _commit(self.origin, "Add excluded incomplete reference")
        policy = {excluded: "Unused incomplete synthetic reference module"}
        with patch.dict(SOURCE_EXCLUSIONS, {"fixture": policy}):
            record = sources.acquire_one("fixture")
            self.assertEqual(record["commit"], self.original_commit)
            self.assertEqual(record["excluded_files"], policy)
            self.assertFalse((self.checkout / excluded).exists())
            self.assertEqual(_git(self.checkout, "diff", "HEAD", "--name-only"), "")
            manifest = build_manifest(self.root)
            self.assertEqual(manifest["snapshots"]["fixture"]["excluded_files"], policy)
            self.assertNotIn(excluded, manifest["snapshots"]["fixture"]["files"])
            (self.root / MANIFEST).write_text(json.dumps(manifest))
            shutil.rmtree(self.checkout / ".git")
            self.calls.clear()
            self.assertEqual(sources.acquire_one("fixture"), record)
            self.assertEqual(self.calls, [])
            self._advance_origin()
            shutil.rmtree(self.checkout)
            restored = sources.acquire_one("fixture")
            self.assertEqual(restored["commit"], self.original_commit)
            self.assertEqual(restored["excluded_files"], policy)
            self.assertFalse((self.checkout / excluded).exists())
            self.assertEqual(_git(self.checkout, "diff", "HEAD", "--name-only"), "")

    def test_existing_vendor_rejects_edited_missing_or_added_files_without_fetch(self):
        self._publish_vendor()
        path = self.checkout / "model.py"
        original = path.read_bytes()
        for change in ("missing", "edited", "added"):
            with self.subTest(change=change):
                extra = self.checkout / "extra.py"
                if change == "missing":
                    path.unlink()
                elif change == "edited":
                    path.write_text('VALUE = "local edit"\n')
                else:
                    extra.write_text('VALUE = "unmanaged source"\n')
                try:
                    with self.assertRaisesRegex(RuntimeError, "snapshot source"):
                        sources.acquire_one("fixture")
                    self.assertEqual(self.calls, [])
                finally:
                    path.write_bytes(original)
                    extra.unlink(missing_ok=True)

    def _assert_no_remote_mutations(self):
        commands = [arguments[0] for arguments, _ in self.calls]
        self.assertFalse({"clone", "fetch", "pull", "reset", "checkout"}.intersection(commands))

    def test_initial_acquisition_and_rerun_are_code_only_and_idempotent(self):
        record = sources.acquire_one("fixture")
        self.assertEqual(record["commit"], self.original_commit)
        self.assertEqual(_git(self.checkout, "rev-parse", "HEAD"), self.original_commit)
        self.assertEqual((self.checkout / "model.py").read_text(), "VALUE = 1\n")
        self.assertFalse((self.checkout / "data/table.csv").exists())
        self.assertFalse((self.checkout / "weights.pt").exists())
        self.assertTrue(record["code_only"])
        self.assertFalse(record["lfs_downloaded"])
        self.assertFalse(record["datasets_downloaded"])
        saved_bytes = self.record.read_bytes()
        self.calls.clear()
        self.assertEqual(sources.acquire_one("fixture"), record)
        self.assertEqual(self.record.read_bytes(), saved_bytes)
        self._assert_no_remote_mutations()

    def test_deleted_managed_checkout_restores_old_pin_after_origin_advances(self):
        original = sources.acquire_one("fixture")
        latest = self._advance_origin()
        self.assertNotEqual(latest, self.original_commit)
        shutil.rmtree(self.checkout)
        self.calls.clear()
        restored = sources.acquire_one("fixture")
        self.assertEqual(restored["commit"], self.original_commit)
        self.assertEqual(_git(self.checkout, "rev-parse", "HEAD"), self.original_commit)
        self.assertEqual(_git(self.origin, "rev-parse", "HEAD"), latest)
        self.assertEqual((self.checkout / "model.py").read_text(), "VALUE = 1\n")
        self.assertFalse((self.checkout / "new_model.py").exists())
        self.assertEqual(restored["source_files"], original["source_files"])
        self.assertEqual(json.loads(self.record.read_text())["commit"], self.original_commit)
        self.calls.clear()
        self.assertEqual(sources.acquire_one("fixture"), restored)
        self._assert_no_remote_mutations()

    def test_existing_foreign_directory_is_refused_without_git_or_fetch(self):
        self.checkout.mkdir(parents=True)
        marker = self.checkout / "README.md"
        marker.write_text("Foreign fixture content must remain.\n")
        with self.assertRaisesRegex(RuntimeError, "unmanaged"):
            sources.acquire_one("fixture")
        self.assertEqual(marker.read_text(), "Foreign fixture content must remain.\n")
        self.assertEqual(self.calls, [])
        self.assertFalse(self.record.exists())

    def test_dirty_managed_checkout_is_refused_without_fetching(self):
        sources.acquire_one("fixture")
        path = self.checkout / "model.py"
        path.write_text("VALUE = 'synthetic local edit'\n")
        self.calls.clear()
        with self.assertRaisesRegex(RuntimeError, "local changes"):
            sources.acquire_one("fixture")
        self.assertEqual(path.read_text(), "VALUE = 'synthetic local edit'\n")
        self._assert_no_remote_mutations()

    def test_changed_managed_revision_is_refused_without_fetching(self):
        sources.acquire_one("fixture")
        (self.checkout / "model.py").write_text("VALUE = 'synthetic local commit'\n")
        changed = _commit(self.checkout, "Synthetic divergent local revision")
        self.assertNotEqual(changed, self.original_commit)
        self.calls.clear()
        with self.assertRaisesRegex(RuntimeError, "revision changed"):
            sources.acquire_one("fixture")
        self._assert_no_remote_mutations()


if __name__ == "__main__":
    unittest.main()
