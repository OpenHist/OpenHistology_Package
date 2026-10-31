"""Command-line entry point; importing this module performs no acquisition."""

import argparse
import json
from pathlib import Path

import requests

from .models import AcquisitionPlan, DATASETS, canonical_dataset
from .pipeline import build_manifest, discover, download, import_local, verify


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration files must contain a JSON object")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Acquire public WSI datasets and preserve linked clinical/label metadata."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    query = commands.add_parser(
        "discover", help="Save a metadata-only acquisition plan; no slide transfers."
    )
    query.add_argument("dataset", type=canonical_dataset, choices=DATASETS)
    query.add_argument("--selection", required=True, help="Provider-specific JSON selection file.")
    query.add_argument("--plan", required=True, help="Destination acquisition-plan JSON.")
    fetch = commands.add_parser(
        "download", help="Download a saved plan, then build linked manifests."
    )
    fetch.add_argument("plan")
    fetch.add_argument("--root", required=True)
    fetch.add_argument("--retries", type=int, default=3)
    fetch.add_argument("--timeout", type=float, default=60.0, help="Per-read timeout in seconds.")
    local = commands.add_parser(
        "import-local",
        help="Register approved local slides in place; no copying or network access.",
    )
    local.add_argument("dataset", type=canonical_dataset, choices=DATASETS)
    local.add_argument("--root", required=True)
    local.add_argument(
        "--metadata",
        required=True,
        help="BRACS workbook, KPMP export CSV, or TCGA/CAMELYON plan JSON.",
    )
    local.add_argument(
        "--options", help="Optional JSON provider options, including explicit column mappings."
    )
    for name, help_text in (
        ("verify", "Recheck local payloads and rebuild manifests."),
        ("manifest", "Rebuild linked exports from saved metadata and acquisition state."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--root", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            plan = discover(args.dataset, _object(args.selection))
            plan.save(args.plan)
            result = {
                "dataset": plan.dataset,
                "plan": str(args.plan),
                "asset_count": len(plan.assets),
                "slide_count": sum(asset.kind == "slide" for asset in plan.assets),
                "expected_bytes": plan.expected_bytes,
                "unknown_size_count": plan.unknown_size_count,
            }
        elif args.command == "download":
            if args.retries < 0 or args.timeout <= 0:
                raise ValueError("retries must be nonnegative and timeout must be positive")
            result = download(
                AcquisitionPlan.load(args.plan),
                args.root,
                retries=args.retries,
                timeout=(10, args.timeout),
            ).to_dict()
        elif args.command == "import-local":
            options = _object(args.options) if args.options else {}
            key = {"BRACS": "path", "KPMP": "repository_csv"}.get(args.dataset, "plan")
            options.setdefault(key, args.metadata)
            result = import_local(args.dataset, args.root, options).to_dict()
        elif args.command == "verify":
            result = verify(args.root).to_dict()
        else:
            result = {"manifest": str(build_manifest(args.root))}
    except requests.RequestException as exc:
        # HTTP exception strings can contain redirected URLs. Keep logs aggregate.
        status = getattr(exc.response, "status_code", None)
        parser.exit(2, f"error: provider request failed ({type(exc).__name__}, status={status})\n")
    except (ValueError, OSError, ImportError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if result.get("failed", 0) or result.get("missing", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
