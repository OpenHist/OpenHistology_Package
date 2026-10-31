"""JSON-configured package CLI; catalogue/status need no model libraries."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .base import VLAError, error_text
from .config import DescribeConfig, ModelSpec, SourceSpec, reject_secrets
from .pipeline import DescriptionFailure, describe_slide
from .registry import list_models, model_info, prepare_model
from .storage import DescriptionReader


def _relative(value, parent):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else parent / path)


def _model(value, parent):
    value = dict(value) if isinstance(value, dict) else {"model": value}
    if value.get("checkpoint"):
        value["checkpoint"] = _relative(value["checkpoint"], parent)
    executable = value.get("python_executable")
    if executable and ("/" in executable or "\\" in executable or executable.startswith("~")):
        value["python_executable"] = _relative(executable, parent)
    options = dict(value.get("options", {}))
    for key, option in options.items():
        if option and (key.endswith("_checkpoint") or key == "upstream_path"):
            options[key] = _relative(option, parent)
    value["options"] = options
    return ModelSpec(**value)


def load_job(path):
    path = Path(path).expanduser().resolve(strict=True)
    job = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"inputs", "output", "models", "prompt", "slide_id", "config"}
    required = {"inputs", "output", "models", "prompt"}
    if not isinstance(job, dict) or set(job) - allowed or not required <= set(job):
        raise ValueError("Job requires inputs/output/models/prompt and optional slide_id/config")
    reject_secrets(job)
    configuration = dict(job.get("config", {}))
    if "slide_id" in job:
        if configuration.get("slide_id") not in (None, job["slide_id"]):
            raise ValueError("Job and config slide_id disagree")
        configuration["slide_id"] = job["slide_id"]
    if configuration.get("cache_dir"):
        configuration["cache_dir"] = _relative(configuration["cache_dir"], path.parent)
    config = DescribeConfig(**configuration)
    sources = job["inputs"]
    if isinstance(sources, (str, dict)):
        sources = [sources]
    inputs = []
    for source in sources:
        value = dict(source) if isinstance(source, dict) else {"path": source}
        value["path"] = _relative(value["path"], path.parent)
        inputs.append(SourceSpec(**value))
    models = job["models"]
    if isinstance(models, (str, dict)):
        models = [models]
    models = [_model(value, path.parent) for value in models]
    if not isinstance(job["prompt"], str) or not job["prompt"].strip():
        raise ValueError("prompt must be a nonempty string")
    return inputs, _relative(job["output"], path.parent), models, job["prompt"], config


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate feature-aligned pathology patch descriptions."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser(
        "models", help="Inspect model provenance/access without downloading weights."
    )
    models.add_argument("model", nargs="?")
    prepare = commands.add_parser(
        "prepare", help="Explicitly load one selected model in its backend environment."
    )
    prepare.add_argument("model")
    prepare.add_argument("--checkpoint")
    prepare.add_argument("--revision")
    prepare.add_argument("--python-executable")
    prepare.add_argument(
        "--options", default="{}", help="JSON backend options; artifact paths are local."
    )
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--precision", default="auto")
    prepare.add_argument("--cache-dir")
    prepare.add_argument("--offline", action="store_true")
    prepare.add_argument("--startup-timeout", type=float, default=600)
    prepare.add_argument(
        "--worker-diagnostics",
        action="store_true",
        help="Include a redacted worker stderr tail in failure output.",
    )
    describe = commands.add_parser(
        "describe", help="Describe a slide's patch sources; resume completed results."
    )
    describe.add_argument("--job", required=True)
    describe.add_argument("--offline", action="store_true")
    describe.add_argument(
        "--worker-diagnostics",
        action="store_true",
        help="Include a redacted worker stderr tail in failure output.",
    )
    status = commands.add_parser("status", help="Read result counts without loading models.")
    status.add_argument("output")
    status.add_argument("--metadata", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "models":
            result = model_info(args.model) if args.model else list_models()
        elif args.command == "prepare":
            spec = _model(
                {
                    "model": args.model,
                    "checkpoint": args.checkpoint,
                    "revision": args.revision,
                    "python_executable": args.python_executable,
                    "options": json.loads(args.options),
                },
                Path.cwd(),
            )
            config = DescribeConfig(
                device=args.device,
                precision=args.precision,
                cache_dir=args.cache_dir,
                offline=args.offline,
                startup_timeout=args.startup_timeout,
                worker_diagnostics=args.worker_diagnostics,
            )
            with prepare_model(spec, config=config) as model:
                result = {"model": args.model, "metadata": model.metadata}
        elif args.command == "describe":
            inputs, output, models, prompt, config = load_job(args.job)
            if args.offline:
                config = replace(config, offline=True)
            if args.worker_diagnostics:
                config = replace(config, worker_diagnostics=True)
            result = describe_slide(inputs, output, models, prompt, config).to_dict()
        else:
            with DescriptionReader(args.output) as reader:
                result = reader.status()
                if args.metadata:
                    result["metadata"] = reader.metadata
        print(json.dumps(result, allow_nan=False))
        return 0
    except DescriptionFailure as exc:
        result = {"error": error_text(exc), "report": exc.report.to_dict()}
        if exc.worker_diagnostics is not None:
            result["worker"] = exc.worker_diagnostics
        print(json.dumps(result), file=sys.stderr)
        return 1
    except (VLAError, ValueError, TypeError, KeyError, OSError) as exc:
        result = {"error": error_text(exc)}
        if isinstance(exc, VLAError) and exc.worker_diagnostics is not None:
            result["worker"] = exc.worker_diagnostics
        print(json.dumps(result), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            json.dumps({"error": "Interrupted; committed descriptions can be resumed"}),
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
