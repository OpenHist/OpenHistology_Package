"""Command line entrypoint. All CLI paths in job JSON are job-relative."""

import argparse
import json
from pathlib import Path
import sys
from . import DomainSpec, TransferConfig, TrainConfig, list_methods, normalize, train


def _path(value, base):
    if value is None:
        return None
    p = Path(value).expanduser()
    return str((base / p).resolve()) if not p.is_absolute() else str(p)


def _domain(value, base):
    value = dict(value)
    if "reference" in value:
        if "references" in value:
            raise ValueError("Use reference or references, not both")
        value["references"] = [value.pop("reference")]
    refs = value.get("references", [])
    if isinstance(refs, str):
        refs = [refs]
    value["references"] = [_path(p, base) for p in refs]
    if value.get("checkpoint"):
        value["checkpoint"] = _path(value["checkpoint"], base)
    value["candidates"] = [_domain(v, base) for v in value.get("candidates", [])]
    return DomainSpec(**value)


def _job(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Job must be a JSON object")
    return value, path.parent


def parser():
    p = argparse.ArgumentParser(
        description="Normalize pathology patches to a reference or trained destination.",
        allow_abbrev=False,
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("methods", help="List supported backends", allow_abbrev=False)
    n = sub.add_parser(
        "normalize",
        help="Normalize an image, folder, CSV/JSONL manifest, or patch HDF5 archive",
        allow_abbrev=False,
    )
    n.add_argument("input", nargs="?")
    n.add_argument("output", nargs="?")
    n.add_argument("--job", help="JSON containing input, output, domain, and optional config")
    n.add_argument("--method", default="vahadane", help="Backend ID; default aliases vahadane")
    n.add_argument(
        "--reference", action="append", default=[], help="Reference image; repeat with --mix"
    )
    n.add_argument("--checkpoint")
    n.add_argument("--architecture", help="Explicit legacy generator architecture")
    n.add_argument("--input-mode", choices=("rgb", "grayscale"))
    n.add_argument("--direction", choices=("AtoB", "BtoA"))
    n.add_argument(
        "--options", type=json.loads, default={}, help="Backend options as a JSON object"
    )
    n.add_argument("--device", default="auto")
    n.add_argument("--seed", type=int, default=256)
    n.add_argument("--standardize", action="store_true")
    n.add_argument("--mix", action="store_true")
    n.add_argument("--overwrite", action="store_true")
    n.add_argument(
        "--histogram-clip", nargs=2, type=float, default=(0.01, 0.99), metavar=("LOW", "HIGH")
    )
    n.add_argument("--png-compression", type=int, default=6)
    t = sub.add_parser("train", help="Train a GAN from a JSON TrainConfig job", allow_abbrev=False)
    t.add_argument(
        "--job", required=True, help="JSON TrainConfig (or object with a training member)"
    )
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "methods":
            print(json.dumps({"methods": list_methods(), "default": "vahadane"}))
            return 0
        if args.command == "normalize":
            if args.job:
                if args.input or args.output:
                    raise ValueError("Use either --job or positional input/output")
                # Avoid silently ignoring a direct override when a job was supplied.
                supplied = sys.argv[1:] if argv is None else argv
                direct_flags = {
                    "--method",
                    "--reference",
                    "--checkpoint",
                    "--architecture",
                    "--input-mode",
                    "--direction",
                    "--options",
                    "--device",
                    "--seed",
                    "--standardize",
                    "--mix",
                    "--overwrite",
                    "--histogram-clip",
                    "--png-compression",
                }
                if any(token.split("=", 1)[0] in direct_flags for token in supplied):
                    raise ValueError(
                        "Put normalization settings in the job; direct overrides with --job are not supported"
                    )
                job, base = _job(args.job)
                unknown = set(job) - {"input", "output", "domain", "config"}
                if unknown:
                    raise ValueError(f"Unknown normalization job fields: {sorted(unknown)}")
                report = normalize(
                    _path(job["input"], base),
                    _path(job["output"], base),
                    _domain(job["domain"], base),
                    TransferConfig(**job.get("config", {})),
                )
            else:
                if not args.input or not args.output:
                    raise ValueError("normalize requires input and output, or --job")
                domain = DomainSpec(
                    method=args.method,
                    references=tuple(args.reference),
                    checkpoint=args.checkpoint,
                    architecture=args.architecture,
                    input_mode=args.input_mode,
                    direction=args.direction,
                    options=args.options,
                )
                config = TransferConfig(
                    seed=args.seed,
                    device=args.device,
                    standardize=args.standardize,
                    mix=args.mix,
                    overwrite=args.overwrite,
                    histogram_clip=args.histogram_clip,
                    png_compression=args.png_compression,
                )
                report = normalize(args.input, args.output, domain, config)
        else:
            job, base = _job(args.job)
            if "training" in job:
                if set(job) != {"training"}:
                    raise ValueError("A wrapped training job must contain only training")
                job = dict(job["training"])
            job.setdefault("output", TrainConfig().output)
            for field in (
                "source",
                "target",
                "output",
                "pairs",
                "paired_ab",
                "resume",
                "warm_start",
            ):
                if job.get(field) is not None:
                    job[field] = _path(job[field], base)
            report = train(TrainConfig(**job))
        print(json.dumps(report.to_dict(), allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
