"""Opt-in checkpoint/GPU acceptance check; never part of the offline test suite."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .__main__ import load_job
from .base import VLAError, error_text, validate_results
from .config import fingerprint
from .inputs import prepare_inputs
from .registry import prepare_model


def smoke_job(job, reference=None):
    inputs, _, models, prompt, config = load_job(job)
    if len(models) != 1:
        raise ValueError("Smoke checks require exactly one selected model")
    if config.do_sample:
        raise ValueError("Use greedy generation (do_sample=false) for repeat/reference comparisons")
    with prepare_inputs(inputs, config) as patches:
        rows = patches.rows
        if config.patch_keys is not None:
            selected = set(config.patch_keys)
            rows = [row for row in rows if row["patch_key"] in selected]
        if len(rows) < 2:
            raise ValueError("Smoke checks require two distinct patch records")
        rows = rows[:2]
        images = patches.read_images([row["patch_key"] for row in rows])
        with prepare_model(models[0], config=config) as model:
            metadata = model.metadata
            first = validate_results(model.generate([images[0]], prompt), 1)[0]
            second = validate_results(model.generate([images[1]], prompt), 1)[0]
            repeated = validate_results(model.generate([images[0]], prompt), 1)[0]
        # Restart entirely and verify the warmed cache works without network access.
        with prepare_model(models[0], config=replace(config, offline=True)) as fresh:
            second_fresh = validate_results(fresh.generate([images[1]], prompt), 1)[0]
            if fingerprint(metadata) != fingerprint(fresh.metadata):
                raise VLAError("Offline reload changed the prepared model/runtime identity")
        if first != repeated or second != second_fresh:
            raise VLAError(
                "Repeated independent patch requests differed; investigate state leakage or nondeterministic CUDA execution"
            )
        if any(item.truncated or item.finish_reason != "stop" for item in (first, second)):
            raise VLAError(
                "Smoke check produced truncated or unfinished descriptions; adjust generation settings"
            )
        checked_reference = reference is not None
        if checked_reference:
            with Path(reference).open(encoding="utf-8") as stream:
                expected_rows = [json.loads(line) for line in stream if line.strip()]
            expected = {row["patch_key"]: row["description"] for row in expected_rows}
            if len(expected) != len(expected_rows):
                raise ValueError("Native reference contains duplicate patch keys")
            for row, result in zip(rows, (first, second)):
                if expected.get(row["patch_key"]) != result.text:
                    raise VLAError(
                        "Native inference reference differs; compare checkpoint, prompt, preprocessing and generation settings"
                    )
        return {
            "model": models[0].model,
            "verified": checked_reference,
            "native_reference_checked": checked_reference,
            "independent_conversations": True,
            "offline_reload": True,
            "metadata": metadata,
            "records": [
                {"patch_key": row["patch_key"], "description": result.text, "status": "completed"}
                for row, result in zip(rows, (first, second))
            ],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument(
        "--reference",
        help="JSONL with patch_key/description from independent native inference using identical settings",
    )
    parser.add_argument(
        "--report", help="Optional JSON report path; keep outside model/source artifact directories"
    )
    args = parser.parse_args(argv)
    try:
        report = smoke_job(args.job, args.reference)
        value = json.dumps(report, indent=2, allow_nan=False)
        if args.report:
            Path(args.report).write_text(value + "\n", encoding="utf-8")
        print(value)
        return 0
    except (VLAError, ValueError, TypeError, KeyError, OSError) as exc:
        print(json.dumps({"error": error_text(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
