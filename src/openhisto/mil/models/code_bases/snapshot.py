"""Integrity for pristine Git checkouts and published source-only snapshots.

The outer package vendors source files, not nested Git metadata. A snapshot is
accepted only against the separately tracked pin and exact per-file inventory.
Generating a new inventory requires real, pristine repository-local checkouts.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from .source_selection import source_exclusions

ROOT = Path(__file__).resolve().parent
MANIFEST = "compat/snapshot_integrity.json"
_CACHE_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def source_inventory(directory, *, excluded=()):
    """Hash regular source files; never follow source-directory symlinks."""
    directory = Path(directory)
    excluded = set(excluded)
    result = {}
    for current, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in _CACHE_DIRS)
        for name in dirs:
            if (Path(current) / name).is_symlink():
                raise RuntimeError("Source inventory contains a symlink directory")
        for name in sorted(files):
            if name == ".DS_Store" or name.endswith((".pyc", ".pyo")) or name == ".git":
                continue
            path = Path(current) / name
            relative = path.relative_to(directory).as_posix()
            if relative in excluded:
                continue
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("Source inventory contains a nonregular file")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[relative] = digest.hexdigest()
    return result


def _git(directory, *args):
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Cannot validate source Git checkout") from exc
    if result.returncode:
        raise RuntimeError("Cannot validate source Git checkout")
    return result.stdout.strip()


def _verify_git(directory, record):
    # Never let rev-parse silently discover the outer publication repository.
    if not (directory / ".git").exists():
        raise RuntimeError("Source directory is not its own Git checkout")
    actual_root = Path(_git(directory, "rev-parse", "--show-toplevel")).resolve()
    if actual_root != directory.resolve():
        raise RuntimeError("Source directory is not its own Git checkout")
    if _git(directory, "rev-parse", "HEAD") != record["commit"]:
        raise RuntimeError("Source revision changed")
    if _git(directory, "diff", "HEAD", "--name-only"):
        raise RuntimeError("Source checkout has local changes")


def _entry(root, slug, record, *, required):
    manifest = Path(root) / MANIFEST
    if not manifest.is_file():
        if required:
            raise RuntimeError("Vendored snapshot manifest is missing")
        return None
    try:
        document = json.loads(manifest.read_text())
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != 1
            or document.get("algorithm") != "sha256"
            or not isinstance(document.get("snapshots"), dict)
        ):
            raise ValueError("Unsupported snapshot schema")
        entry = document["snapshots"].get(slug)
        if entry is None and not required:
            return None
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), dict):
            raise ValueError("Missing snapshot entry")
        if (
            entry.get("commit") != record["commit"]
            or entry.get("checkout") != record.get("checkout")
            or entry.get("repository") != record.get("repository")
            or entry.get("excluded_files", {}) != record.get("excluded_files", {})
        ):
            raise ValueError("Snapshot pin does not match source record")
        for name, digest in entry["files"].items():
            parts = name.split("/")
            if (
                not name
                or name.startswith("/")
                or any(part in {"", ".", ".."} for part in parts)
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("Invalid snapshot file entry")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Vendored snapshot pin or manifest is invalid") from exc
    return entry


def validate_source_tree(root, slug, record, *, directory=None):
    """Validate a local checkout or vendored tree without network or mutation."""
    root = Path(root)
    if record.get("excluded_files", {}) != source_exclusions(slug):
        raise RuntimeError("Source record exclusions differ from the explicit selection policy")
    directory = Path(directory) if directory is not None else root / "upstream" / slug
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError("Source directory is absent or is a symlink")
    has_git = (directory / ".git").exists() or (directory / ".git").is_symlink()
    if has_git:
        _verify_git(directory, record)
    entry = _entry(root, slug, record, required=not has_git)
    if entry is not None:
        actual = source_inventory(directory, excluded=source_exclusions(slug))
        if actual.keys() != entry["files"].keys():
            raise RuntimeError("Vendored snapshot source inventory changed")
        if actual != entry["files"]:
            raise RuntimeError("Vendored snapshot source bytes changed")
    return record["commit"]


def build_manifest(root=ROOT):
    """Build inventories only from verified original commits, never vendors."""
    root = Path(root)
    snapshots = {}
    records = sorted((root / "compat/sources").glob("*.json"))
    if not records:
        raise RuntimeError("No pinned source records found")
    for path in records:
        record = json.loads(path.read_text())
        slug = path.stem
        if record.get("checkout") != "upstream/" + slug:
            raise RuntimeError("Source record checkout is not canonical")
        if record.get("excluded_files", {}) != source_exclusions(slug):
            raise RuntimeError("Source record exclusions differ from the explicit selection policy")
        directory = root / record["checkout"]
        _verify_git(directory, record)
        files = source_inventory(directory, excluded=source_exclusions(slug))
        tracked = set(_git(directory, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
        if not files.keys() <= tracked:
            raise RuntimeError("Source checkout contains unmanaged files")
        if "source_files" in record and len(files) != record["source_files"]:
            raise RuntimeError("Source inventory differs from the selected source count")
        snapshots[slug] = {
            "repository": record.get("repository"),
            "commit": record["commit"],
            "checkout": record["checkout"],
            "files": files,
        }
        if source_exclusions(slug):
            snapshots[slug]["excluded_files"] = source_exclusions(slug)
    return {"schema_version": 1, "algorithm": "sha256", "snapshots": snapshots}


def main():
    document = build_manifest()
    output = ROOT / MANIFEST
    output.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    print(f"Verified {len(document['snapshots'])} pinned source inventories")


if __name__ == "__main__":
    main()
