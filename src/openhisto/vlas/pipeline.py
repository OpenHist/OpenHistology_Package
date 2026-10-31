"""Patch identity reconciliation, selected-model execution, and resumable results."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re

from .base import VLAError, error_text, validate_results
from .config import (
    DescribeConfig,
    DescriptionReport,
    ModelSpec,
    fingerprint,
    json_safe,
    reject_secrets,
)
from .inputs import prepare_inputs
from .registry import get_descriptor, prepare_model
from .storage import ResultStore


class DescriptionFailure(VLAError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _artifact_hash(path):
    path = Path(path).expanduser().resolve(strict=True)
    files = (
        [path]
        if path.is_file()
        else sorted(
            file
            for file in path.rglob("*")
            if file.is_file()
            and not any(
                part in {".git", ".cache", "__pycache__"} for part in file.relative_to(path).parts
            )
            and file.suffix not in {".pyc", ".pyo"}
        )
    )
    if not files:
        raise ValueError(f"Checkpoint/source artifact is empty: {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update((file.name if path.is_file() else file.relative_to(path).as_posix()).encode())
        digest.update(b"\0")
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _request(spec, config, prompt):
    descriptor = get_descriptor(spec.model)
    request = spec.to_dict()
    # Remote mutable branches must not silently reuse descriptions from old weights.
    revision = spec.revision or descriptor.checkpoint_revision
    if revision and not spec.checkpoint and not re.fullmatch(r"[a-fA-F0-9]{40,64}", revision):
        raise ValueError(
            "Use an immutable Hugging Face commit revision, not a mutable branch or tag"
        )
    request["revision"] = revision
    if spec.checkpoint:
        request["checkpoint_sha256"] = _artifact_hash(spec.checkpoint)
    artifacts = {
        key: _artifact_hash(value)
        for key, value in spec.options.items()
        if value and (key.endswith("_checkpoint") or key == "upstream_path")
    }
    return {
        "schema_version": 1,
        "adapter_version": "1",
        "model": request,
        "artifacts": artifacts,
        "descriptor": json_safe(descriptor.to_dict()),
        "prompt": prompt,
        "generation": config.identity(),
    }


def describe_slide(inputs, output, models, prompt, config=None):
    config = config or DescribeConfig()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Supply one nonempty prompt per description run")
    if isinstance(models, (str, ModelSpec, dict)):
        models = [models]
    models = [
        item
        if isinstance(item, ModelSpec)
        else ModelSpec(**item)
        if isinstance(item, dict)
        else ModelSpec(item)
        for item in models
    ]
    if not models:
        raise ValueError("Select at least one model")
    requests = [_request(spec, config, prompt) for spec in models]
    run_keys = ["r_" + fingerprint(request) for request in requests]
    if len(set(run_keys)) != len(run_keys):
        raise ValueError("Duplicate model configuration in one job")
    completed = reused = truncated = failed = 0
    with prepare_inputs(inputs, config) as patches:
        rows = patches.rows
        if config.patch_keys is not None:
            wanted = set(config.patch_keys)
            available = {row["patch_key"] for row in rows}
            if wanted - available:
                raise ValueError("Requested patch_keys are absent from the supplied patch sources")
            rows = [row for row in rows if row["patch_key"] in wanted]
        with ResultStore(output, patches.slide_id) as store:
            # Validate all supplied old rows even when selecting only a subset.
            store.reconcile(patches.rows)

            def report():
                pending = sum(
                    (row["patch_key"], key) not in store._results
                    for row in rows
                    for key in run_keys
                )
                return DescriptionReport(
                    str(store.output),
                    len(rows),
                    completed,
                    reused,
                    truncated,
                    failed,
                    pending,
                    tuple(run_keys),
                )

            # Publish the whole requested work set before any backend can fail.
            for spec, request, key in zip(models, requests, run_keys):
                run = store.metadata["runs"].get(key)
                if run is None:
                    run = {
                        "run_key": key,
                        "model": spec.model,
                        "prompt": prompt,
                        "request": request,
                        "status": "pending",
                        "patch_keys": [],
                        "metadata": None,
                        "failures": {},
                    }
                    store.metadata["runs"][key] = run
                targets = set(run["patch_keys"])
                for row in rows:
                    if row["patch_key"] not in targets:
                        run["patch_keys"].append(row["patch_key"])
                        targets.add(row["patch_key"])
            store.save()

            for spec, request, key in zip(models, requests, run_keys):
                run = store.metadata["runs"][key]
                missing = [row for row in rows if (row["patch_key"], key) not in store._results]
                reused += len(rows) - len(missing)
                store.save()
                if not missing:
                    run["failures"] = {
                        patch: failure
                        for patch, failure in run.get("failures", {}).items()
                        if (patch, key) not in store._results
                    }
                    run["status"] = (
                        "complete"
                        if all((patch, key) in store._results for patch in run["patch_keys"])
                        else "pending"
                    )
                    store.save()
                    continue
                model = None
                active = missing
                try:
                    run.update(
                        status="running", last_attempt_at_utc=datetime.now(timezone.utc).isoformat()
                    )
                    store.save()
                    model = prepare_model(spec, config=config)
                    runtime = json_safe(dict(model.metadata))
                    reject_secrets(runtime)
                    if run["metadata"] is not None and fingerprint(runtime) != fingerprint(
                        run["metadata"]
                    ):
                        raise VLAError(
                            "Model/runtime identity differs from saved descriptions; restore the original environment or choose a new model configuration"
                        )
                    run["metadata"] = runtime
                    store.save()
                    for start in range(0, len(missing), config.batch_size):
                        active = missing[start : start + config.batch_size]
                        images = patches.read_images(
                            [row["patch_key"] for row in active], workers=config.workers
                        )
                        results = validate_results(model.generate(images, prompt), len(active))
                        records = [
                            {
                                **store.metadata["patches"][row["patch_key"]],
                                "run_key": key,
                                "model": spec.model,
                                "description": result.text.strip(),
                                "status": "truncated" if result.truncated else "completed",
                                "finish_reason": result.finish_reason,
                            }
                            for row, result in zip(active, results)
                        ]
                        store.append(records)
                        completed += sum(not result.truncated for result in results)
                        truncated += sum(result.truncated for result in results)
                    run["status"] = (
                        "complete"
                        if all((patch, key) in store._results for patch in run["patch_keys"])
                        else "pending"
                    )
                    store.save()
                except BaseException as exc:
                    interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
                    run["status"] = "interrupted" if interrupted else "failed"
                    for row in active:
                        if (row["patch_key"], key) not in store._results:
                            run["failures"][row["patch_key"]] = {
                                "status": run["status"],
                                "error": error_text(exc),
                            }
                    failed += sum((row["patch_key"], key) not in store._results for row in active)
                    store.save()
                    store.export_csv()
                    if interrupted:
                        raise
                    raise DescriptionFailure(error_text(exc), report()) from exc
                finally:
                    if model is not None:
                        try:
                            model.close()
                        except Exception as cleanup:
                            # Cleanup must not replace an inference error/interrupt
                            # or discard already committed, reviewable answers.
                            run["cleanup_error"] = error_text(cleanup)
                            store.save()
            store.export_csv()
            return report()
