"""Small SQLite acquisition ledger and root-contained asset paths."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .models import AcquisitionPlan, canonical_dataset, validate_relative_path


_RESERVED = {
    "acquisition.sqlite",
    "acquisition_plan.json",
    "manifest.jsonl",
    "slides.csv",
    "clinical.jsonl",
    "provenance.json",
    "source_metadata/records.jsonl",
}
_STATUS = {"pending", "downloading", "complete", "failed", "missing"}
_INTEGRITY = {None, "checksum_verified", "size_verified", "local_hash_only", "failed", "missing"}
_UPDATABLE = {"status", "integrity", "actual_size", "sha256", "etag", "last_modified", "error"}


def safe_asset_path(root, relative_path):
    """Return a lexical asset path after rejecting escapes and control files."""
    root = Path(root).resolve()
    relative = validate_relative_path(relative_path)
    base = relative.removesuffix(".part")
    if (
        relative in _RESERVED
        or base in _RESERVED
        or relative.startswith("acquisition.sqlite-")
        or relative.endswith(".tmp")
        and relative[:-4] in _RESERVED
        or any(
            relative.startswith(name + "/") or name.startswith(relative + "/") for name in _RESERVED
        )
    ):
        raise ValueError("Asset path conflicts with acquisition metadata")
    candidate = root / relative
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("Asset path escapes the acquisition root")
    # An internal symlink must not alias a reserved output either.
    resolved_relative = resolved.relative_to(root).as_posix()
    if resolved_relative in _RESERVED or resolved_relative.startswith("acquisition.sqlite-"):
        raise ValueError("Asset path aliases acquisition metadata")
    return candidate


def _control_path(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError("Acquisition metadata cannot be a symlink")
    return path


def _identity(asset):
    return json.dumps(
        {
            key: asset.get(key)
            for key in ("dataset", "asset_id", "relative_path", "url", "size", "checksums")
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class StateStore:
    """One dataset per root; short transactions do not retain live handles."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = _control_path(self.root, "acquisition.sqlite")
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute("""CREATE TABLE IF NOT EXISTS assets (
                dataset TEXT NOT NULL, asset_id TEXT NOT NULL, asset_json TEXT NOT NULL,
                identity_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                integrity TEXT, actual_size INTEGER, sha256 TEXT, etag TEXT,
                last_modified TEXT, error TEXT, PRIMARY KEY(dataset, asset_id))""")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def close(self):
        """Compatibility hook; each operation already closes its connection."""
        return None

    @contextmanager
    def _connection(self):
        # Recheck before every open: callers may replace filesystem entries.
        _control_path(self.root, "acquisition.sqlite")
        for suffix in ("-journal", "-wal", "-shm"):
            _control_path(self.root, "acquisition.sqlite" + suffix)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _record(row):
        if row is None:
            return None
        return {
            "asset": json.loads(row["asset_json"]),
            **{
                key: row[key]
                for key in (
                    "status",
                    "integrity",
                    "actual_size",
                    "sha256",
                    "etag",
                    "last_modified",
                    "error",
                )
            },
        }

    def register(self, plan):
        # Revalidate mutable dataclasses before allowing their paths into state.
        plan = AcquisitionPlan.from_dict(plan.to_dict())
        plan_path = _control_path(self.root, "acquisition_plan.json")
        if plan_path.exists() and AcquisitionPlan.load(plan_path).dataset != plan.dataset:
            raise ValueError("An acquisition root can contain only one dataset")
        paths = set()
        final_paths = {
            safe_asset_path(self.root, asset.relative_path).resolve() for asset in plan.assets
        }
        for asset in plan.assets:
            for relative in (asset.relative_path, asset.relative_path + ".part"):
                path = safe_asset_path(self.root, relative)
                if path.resolve() in paths:
                    raise ValueError("Asset and partial paths must not collide")
                paths.add(path.resolve())
        stale_partials = []
        with self._connection() as connection:
            current = connection.execute(
                "SELECT value FROM settings WHERE key='dataset'"
            ).fetchone()
            if current is not None and current["value"] != plan.dataset:
                raise ValueError("An acquisition root can contain only one dataset")
            connection.execute(
                "INSERT OR IGNORE INTO settings(key,value) VALUES('dataset',?)", (plan.dataset,)
            )
            for asset in plan.assets:
                value = asset.to_dict()
                identity = _identity(value)
                existing = connection.execute(
                    "SELECT * FROM assets WHERE dataset=? AND asset_id=?",
                    (plan.dataset, asset.asset_id),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO assets(dataset,asset_id,asset_json,identity_json) VALUES(?,?,?,?)",
                        (
                            plan.dataset,
                            asset.asset_id,
                            json.dumps(value, allow_nan=False),
                            identity,
                        ),
                    )
                elif existing["identity_json"] != identity:
                    old = json.loads(existing["asset_json"])
                    if (
                        safe_asset_path(self.root, old["relative_path"] + ".part").resolve()
                        in final_paths
                    ):
                        raise ValueError("A stale partial path conflicts with a current asset")
                    stale_partials.extend(
                        (old["relative_path"] + ".part", asset.relative_path + ".part")
                    )
                    connection.execute(
                        """UPDATE assets SET asset_json=?,identity_json=?,status='pending',integrity=NULL,
                        actual_size=NULL,sha256=NULL,etag=NULL,last_modified=NULL,error=NULL
                        WHERE dataset=? AND asset_id=?""",
                        (
                            json.dumps(value, allow_nan=False),
                            identity,
                            plan.dataset,
                            asset.asset_id,
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE assets SET asset_json=? WHERE dataset=? AND asset_id=?",
                        (json.dumps(value, allow_nan=False), plan.dataset, asset.asset_id),
                    )
        for relative in set(stale_partials):
            path = safe_asset_path(self.root, relative)
            if path.is_symlink():
                # Removing a link is safe; never follow it to remove a target.
                path.unlink()
            elif path.is_file():
                path.unlink()

    def get(self, dataset, asset_id):
        with self._connection() as connection:
            return self._record(
                connection.execute(
                    "SELECT * FROM assets WHERE dataset=? AND asset_id=?",
                    (canonical_dataset(dataset), asset_id),
                ).fetchone()
            )

    def update(self, dataset, asset_id, **values):
        if not values:
            return
        if set(values) - _UPDATABLE:
            raise ValueError("Unknown acquisition state field")
        if "status" in values and values["status"] not in _STATUS:
            raise ValueError("Unknown acquisition status")
        if "integrity" in values and values["integrity"] not in _INTEGRITY:
            raise ValueError("Unknown integrity level")
        if values.get("actual_size") is not None and (
            isinstance(values["actual_size"], bool)
            or not isinstance(values["actual_size"], int)
            or values["actual_size"] < 0
        ):
            raise ValueError("actual_size must be a nonnegative integer")
        with self._connection() as connection:
            result = connection.execute(
                f"UPDATE assets SET {','.join(key + '=?' for key in values)} WHERE dataset=? AND asset_id=?",
                (*values.values(), canonical_dataset(dataset), asset_id),
            )
            if result.rowcount != 1:
                raise KeyError("Asset is not registered")

    def rows(self):
        with self._connection() as connection:
            return [
                self._record(row)
                for row in connection.execute("SELECT * FROM assets ORDER BY dataset,asset_id")
            ]
