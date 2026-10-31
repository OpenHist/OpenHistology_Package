"""One successful JSONL record per patch/run; atomic metadata and CSV snapshots."""

import csv
import json
import os
from pathlib import Path
import tempfile

from .base import VLAError
from .config import json_safe, reject_secrets


CORE_FIELDS = (
    "patch_key",
    "slide_id",
    "source_id",
    "patch_id",
    "source_type",
    "source_index",
    "mag",
    "content_hash",
    "x_level0",
    "y_level0",
    "width_level0",
    "height_level0",
    "patch_width",
    "patch_height",
    "valid_x0",
    "valid_y0",
    "valid_x1",
    "valid_y1",
    "source_mag",
)
CSV_FIELDS = (
    *CORE_FIELDS,
    "run_key",
    "model",
    "description",
    "status",
    "finish_reason",
    "metadata_json",
)


def _atomic(path, write):
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _ledger(path):
    """Yield committed newline-terminated records; a torn final write is pending."""
    if not path.exists():
        return
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                return
            try:
                value = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Corrupt description ledger at line {number}") from exc
            if (
                not isinstance(value, dict)
                or value.get("status") not in {"completed", "truncated"}
                or not isinstance(value.get("description"), str)
                or not value["description"].strip()
            ):
                raise ValueError(f"Invalid description record at line {number}")
            yield value


class DescriptionReader:
    def __init__(self, output):
        self.output = Path(output).expanduser().resolve()
        self.path = self.output / "descriptions.jsonl"
        self.metadata = json.loads((self.output / "metadata.json").read_text(encoding="utf-8"))
        if (
            self.metadata.get("schema_version") != 1
            or not isinstance(self.metadata.get("patches"), dict)
            or not isinstance(self.metadata.get("runs"), dict)
        ):
            raise ValueError("Not a supported VLA result directory")
        self._results = {}
        for row in _ledger(self.path):
            key = (row.get("patch_key"), row.get("run_key"))
            if key in self._results:
                raise ValueError("Duplicate patch/run in description ledger")
            if key[0] not in self.metadata["patches"] or key[1] not in self.metadata["runs"]:
                raise ValueError("Description record refers to missing patch/run metadata")
            if any(
                row.get(field) != self.metadata["patches"][key[0]].get(field)
                for field in CORE_FIELDS
            ):
                raise ValueError("Description ledger patch metadata disagrees with saved identity")
            self._results[key] = row["status"]

    def iter_records(self, run_key=None, patch_keys=None):
        if run_key is not None and run_key not in self.metadata["runs"]:
            raise KeyError(run_key)
        selected = None if patch_keys is None else set(patch_keys)
        for row in _ledger(self.path):
            if (run_key is None or row["run_key"] == run_key) and (
                selected is None or row["patch_key"] in selected
            ):
                yield row

    def records(self, run_key=None, patch_keys=None):
        return list(self.iter_records(run_key, patch_keys))

    def patch_metadata(self):
        return list(self.metadata["patches"].values())

    def list_runs(self):
        return list(self.metadata["runs"].values())

    def status(self):
        runs = []
        for key, run in self.metadata["runs"].items():
            targets = run["patch_keys"]
            terminal = {patch for patch, saved_run in self._results if saved_run == key}
            completed = sum(self._results.get((patch, key)) == "completed" for patch in targets)
            truncated = sum(self._results.get((patch, key)) == "truncated" for patch in targets)
            failed = sum(patch not in terminal for patch in run.get("failures", {}))
            pending = len(targets) - completed - truncated
            state = run["status"]
            if not pending:
                state = "complete"
            elif state == "running" and not Path(str(self.output) + ".lock").exists():
                state = "interrupted"
            runs.append(
                {
                    "run_key": key,
                    "model": run["model"],
                    "status": state,
                    "patches": len(targets),
                    "completed": completed,
                    "truncated": truncated,
                    "failed": failed,
                    "pending": pending,
                }
            )
        return {
            "output": str(self.output),
            "slide_id": self.metadata["slide_id"],
            "patches": len(self.metadata["patches"]),
            "locked": Path(str(self.output) + ".lock").exists(),
            "runs": runs,
        }

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ResultStore(DescriptionReader):
    def __init__(self, output, slide_id):
        self.output = Path(output).expanduser().resolve()
        self.lock = Path(str(self.output) + ".lock")
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise VLAError(
                f"Result writer is locked: {self.lock}; check its PID before removing a stale lock"
            ) from exc
        self._locked = True
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump({"pid": os.getpid()}, stream)
            self.output.mkdir(parents=True, exist_ok=True)
            self.path = self.output / "descriptions.jsonl"
            if (self.output / "metadata.json").exists():
                super().__init__(self.output)
                if self.metadata["slide_id"] != slide_id:
                    raise ValueError("Result directory belongs to another slide")
            else:
                if self.path.exists() and self.path.stat().st_size:
                    raise ValueError(
                        "Description ledger has no metadata; restore metadata before resuming"
                    )
                self.metadata = {
                    "schema_version": 1,
                    "slide_id": slide_id,
                    "patches": {},
                    "runs": {},
                }
                self._results = {}
                self.save()
            # Only the final unterminated write can be discarded, under the lock.
            if self.path.exists():
                committed = 0
                with self.path.open("rb") as stream:
                    for line in stream:
                        if not line.endswith(b"\n"):
                            break
                        committed += len(line)
                if committed != self.path.stat().st_size:
                    with self.path.open("r+b") as stream:
                        stream.truncate(committed)
                        stream.flush()
                        os.fsync(stream.fileno())
        except BaseException:
            self.close()
            raise

    def save(self):
        reject_secrets(self.metadata)
        _atomic(
            self.output / "metadata.json",
            lambda stream: json.dump(json_safe(self.metadata), stream, allow_nan=False),
        )

    def reconcile(self, rows):
        for row in rows:
            old = self.metadata["patches"].get(row["patch_key"])
            if old is not None and any(old.get(field) != row.get(field) for field in CORE_FIELDS):
                raise ValueError(
                    "Patch content or geometry changed under an existing identity; use a new output for revised patches"
                )
        for row in rows:
            self.metadata["patches"].setdefault(row["patch_key"], row)
        self.save()

    def append(self, records):
        keys = [(row["patch_key"], row["run_key"]) for row in records]
        if len(set(keys)) != len(keys) or any(key in self._results for key in keys):
            raise ValueError("A patch/run result was already committed")
        payload = "".join(
            json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n" for row in records
        )
        with self.path.open("ab") as stream:
            stream.write(payload.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        for key, row in zip(keys, records):
            self._results[key] = row["status"]
            self.metadata["runs"][key[1]].get("failures", {}).pop(key[0], None)

    def export_csv(self):
        def write(stream):
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in _ledger(self.path):
                writer.writerow(
                    {
                        key: json.dumps(value, sort_keys=True)
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in row.items()
                    }
                )

        _atomic(self.output / "descriptions.csv", write)

    def close(self):
        if getattr(self, "_locked", False):
            self._locked = False
            self.lock.unlink(missing_ok=True)
