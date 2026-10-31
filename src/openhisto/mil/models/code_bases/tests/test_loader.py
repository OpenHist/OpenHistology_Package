"""Private import and provenance guards using tiny local Git repositories only."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid

from openhisto.mil.models.code_bases.compat import loader
from openhisto.mil.models.code_bases.snapshot import build_manifest, MANIFEST


def _git(directory, *arguments):
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=" + os.devnull, *arguments],
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


def _repository(root, slug, tag):
    directory = root / "upstream" / slug
    (directory / "models").mkdir(parents=True)
    (directory / "utils").mkdir()
    (directory / "utils/__init__.py").write_text(f"TAG = {tag!r}\n")
    (directory / "utils/helper.py").write_text(
        "from . import TAG\ndef label():\n    return TAG + ':helper'\n"
    )
    (directory / "models/__init__.py").write_text("from utils import TAG\nMODEL = TAG + ':model'\n")
    (directory / "models/layers.py").write_text("from utils.helper import label\nLAYER = label()\n")
    (directory / "entry.py").write_text(
        "import models, utils\n"
        "import models.layers\n"
        "import models.layers as layers_alias\n"
        "from models import MODEL as model_alias\n"
        "from utils.helper import label\n"
        "def values():\n"
        "    return (models.MODEL, utils.TAG, models.layers.LAYER, "
        "layers_alias.LAYER, model_alias, label())\n"
    )
    _git(directory, "init", "--quiet")
    commit = _commit(directory, "Synthetic source baseline")
    record = root / "compat/sources" / (slug + ".json")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"commit": commit, "checkout": "upstream/" + slug}))
    return directory, record


@contextmanager
def _isolated(root):
    # An independent private prefix prevents fixtures from touching native model
    # modules that may already have been loaded by other acceptance tests.
    prefix = "openhisto.mil.models.code_bases._loader_fixture_" + uuid.uuid4().hex
    meta_path = list(sys.meta_path)
    before_path = list(sys.path)
    before_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    global_models = types.ModuleType("models")
    global_utils = types.ModuleType("utils")
    global_models.MODEL = "global-sentinel"
    global_utils.TAG = "global-sentinel"
    with (
        patch.object(loader, "ROOT", root),
        patch.object(loader, "_PREFIX", prefix),
        patch.object(loader, "_CONFIG", {}),
        patch.dict(sys.modules, {"models": global_models, "utils": global_utils}),
    ):
        try:
            yield prefix, global_models, global_utils
        finally:
            for name in list(sys.modules):
                if name == prefix or name.startswith(prefix + "."):
                    del sys.modules[name]
            sys.meta_path[:] = meta_path
            if sys.path != before_path or os.environ.get("CUDA_VISIBLE_DEVICES") != before_cuda:
                raise AssertionError("Private loader changed sys.path or CUDA routing")


class LoaderTests(unittest.TestCase):
    def _vendor(self, root):
        directory, record = _repository(root, "vendor_fixture", "V")
        (root / MANIFEST).write_text(json.dumps(build_manifest(root)))
        shutil.rmtree(directory / ".git")
        # This outer Git repository must never supply an upstream's HEAD.
        _git(root, "init", "--quiet")
        _commit(root, "Synthetic outer snapshot publication")
        return directory, record

    def test_vendored_snapshot_loads_inside_outer_git_without_inner_git(self):
        with tempfile.TemporaryDirectory(prefix="cpath-vendor-fixture-") as temporary:
            root = Path(temporary)
            directory, record = self._vendor(root)
            self.assertNotEqual(
                _git(root, "rev-parse", "HEAD"), json.loads(record.read_text())["commit"]
            )
            (directory / "__pycache__").mkdir()
            (directory / "__pycache__/generated.pyc").write_bytes(b"ignored synthetic cache")
            (directory / ".DS_Store").write_bytes(b"ignored synthetic metadata")
            with _isolated(root):
                native = loader.load_native("vendor_fixture", "entry")
                self.assertEqual(native.values()[1], "V")
            with self.assertRaisesRegex(RuntimeError, "own Git checkout"):
                build_manifest(root)

    def test_vendor_missing_changed_and_added_source_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="cpath-vendor-fixture-") as temporary:
            root = Path(temporary)
            directory, _ = self._vendor(root)
            entry = directory / "entry.py"
            original = entry.read_bytes()
            for change in ("missing", "changed", "added"):
                with self.subTest(change=change):
                    extra = directory / "injected.py"
                    if change == "missing":
                        entry.unlink()
                    elif change == "changed":
                        entry.write_bytes(
                            original + b'\nraise AssertionError("Must not execute")\n'
                        )
                    else:
                        extra.write_text('VALUE = "unexpected source"\n')
                    try:
                        with (
                            _isolated(root),
                            self.assertRaisesRegex(ImportError, "snapshot source"),
                        ):
                            loader.load_native("vendor_fixture", "entry")
                    finally:
                        entry.write_bytes(original)
                        extra.unlink(missing_ok=True)

    def test_vendor_requires_matching_manifest_and_pin(self):
        with tempfile.TemporaryDirectory(prefix="cpath-vendor-fixture-") as temporary:
            root = Path(temporary)
            _, record = self._vendor(root)
            manifest = root / MANIFEST
            original = manifest.read_bytes()
            manifest.unlink()
            with _isolated(root), self.assertRaisesRegex(ImportError, "manifest is missing"):
                loader.load_native("vendor_fixture", "entry")
            manifest.write_bytes(original)
            values = json.loads(record.read_text())
            values["commit"] = _git(root, "rev-parse", "HEAD")
            record.write_text(json.dumps(values))
            with _isolated(root), self.assertRaisesRegex(ImportError, "pin or manifest is invalid"):
                loader.load_native("vendor_fixture", "entry")

    def test_generic_models_utils_are_isolated_in_both_import_orders(self):
        with tempfile.TemporaryDirectory(prefix="cpath-loader-fixture-") as temporary:
            root = Path(temporary)
            _repository(root, "fixture_a", "A")
            _repository(root, "fixture_b", "B")
            for order in (("fixture_a", "fixture_b"), ("fixture_b", "fixture_a")):
                with self.subTest(order=order), _isolated(root) as (prefix, models, utils):
                    before_path = list(sys.path)
                    before_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
                    modules = {slug: loader.load_native(slug, "entry") for slug in order}
                    for slug, tag in (("fixture_a", "A"), ("fixture_b", "B")):
                        self.assertEqual(
                            modules[slug].values(),
                            (
                                tag + ":model",
                                tag,
                                tag + ":helper",
                                tag + ":helper",
                                tag + ":model",
                                tag + ":helper",
                            ),
                        )
                        self.assertTrue(modules[slug].__name__.startswith(prefix + "." + slug))
                    self.assertIsNot(
                        sys.modules[prefix + ".fixture_a.models"],
                        sys.modules[prefix + ".fixture_b.models"],
                    )
                    self.assertIsNot(
                        sys.modules[prefix + ".fixture_a.utils"],
                        sys.modules[prefix + ".fixture_b.utils"],
                    )
                    self.assertIs(sys.modules["models"], models)
                    self.assertIs(sys.modules["utils"], utils)
                    self.assertEqual(models.MODEL, "global-sentinel")
                    self.assertEqual(utils.TAG, "global-sentinel")
                    self.assertEqual(sys.path, before_path)
                    self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), before_cuda)

    def test_changed_head_is_rejected_before_native_import(self):
        with tempfile.TemporaryDirectory(prefix="cpath-loader-fixture-") as temporary:
            root = Path(temporary)
            directory, record = _repository(root, "changed_head", "A")
            expected = json.loads(record.read_text())["commit"]
            with (directory / "entry.py").open("a") as stream:
                stream.write("\nraise AssertionError('Changed source must never execute')\n")
            actual = _commit(directory, "Synthetic different revision")
            self.assertNotEqual(expected, actual)
            self.assertEqual(_git(directory, "diff", "HEAD", "--name-only"), "")
            with _isolated(root) as (prefix, _, _):
                with self.assertRaisesRegex(ImportError, "Pinned source has changed"):
                    loader.load_native("changed_head", "entry")
                self.assertNotIn(prefix + ".changed_head", loader._CONFIG)
                self.assertNotIn(prefix + ".changed_head.entry", sys.modules)

    def test_unstaged_and_staged_tracked_edits_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="cpath-loader-fixture-") as temporary:
            root = Path(temporary)
            for staged in (False, True):
                slug = "edited_staged" if staged else "edited_unstaged"
                directory, _ = _repository(root, slug, "A")
                with (directory / "entry.py").open("a") as stream:
                    stream.write("\nraise AssertionError('Modified source must never execute')\n")
                if staged:
                    _git(directory, "add", "entry.py")
                with self.subTest(staged=staged), _isolated(root) as (prefix, _, _):
                    with self.assertRaisesRegex(ImportError, "Pinned source has changed"):
                        loader.load_native(slug, "entry")
                    self.assertNotIn(prefix + "." + slug, loader._CONFIG)
                    self.assertNotIn(prefix + "." + slug + ".entry", sys.modules)

    def test_missing_source_record_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="cpath-loader-fixture-") as temporary:
            root = Path(temporary)
            _, record = _repository(root, "unmanaged", "A")
            record.unlink()
            with _isolated(root), self.assertRaisesRegex(ImportError, "unmanaged"):
                loader.load_native("unmanaged", "entry")


if __name__ == "__main__":
    unittest.main()
