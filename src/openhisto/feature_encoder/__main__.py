"""CLI for credentials, preparation, encoding and selective store inspection."""

import argparse
import json
from pathlib import Path
import sys

from .auth import AuthConfig
from .base import EncoderError
from .config import EncodeConfig, EncoderSpec, SourceSpec, json_safe
from .pipeline import EncodingFailure, encode_slide
from .registry import check_access, list_models, model_info, prepare_model
from .storage import FeatureReader


def _auth_options(parser):
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Request missing credentials through hidden input, only when needed.",
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--gemini-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--cohere-key-env", default="COHERE_API_KEY")
    parser.add_argument("--aws-profile")
    parser.add_argument("--region")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--revision")
    parser.add_argument("--checkpoint")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Incremental image features in one standard h5ad per slide, across models and magnifications."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser(
        "models", help="List profiles or inspect one model, without downloads."
    )
    models.add_argument("model", nargs="?")
    for name, description in (
        ("access", "Check credential references/model approval without inference."),
        ("prepare", "Explicitly prepare the selected model/client; no image inference."),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("model")
        _auth_options(command)
        command.add_argument("--dimension", type=int)
        command.add_argument("--device", default="auto")
        command.add_argument("--precision", default="auto")
    encode = commands.add_parser(
        "encode", help="Run a JSON job; reuse complete output and resume compatible partials."
    )
    encode.add_argument("--job", required=True)
    encode.add_argument(
        "--prompt", action="store_true", help="Enable hidden entry of missing runtime credentials."
    )
    encode.add_argument("--offline", action="store_true")
    encode.add_argument(
        "--metadata-policy",
        choices=("preserve", "update", "error"),
        help="Handle changed optional patch metadata; overrides the job's config.metadata_policy.",
    )
    for name in ("status", "inspect"):
        command = commands.add_parser(
            name, help="Show model/magnification counts without loading feature matrices."
        )
        command.add_argument("output")
        command.add_argument("--metadata", action="store_true")
    return parser


def _relative(value, parent):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else parent / path)


def load_job(path):
    path = Path(path).expanduser().resolve()
    job = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(job, dict) or set(job) - {
        "inputs",
        "output",
        "models",
        "slide_id",
        "config",
        "auth",
    }:
        raise ValueError(
            "Job must contain inputs, output, models and optional slide_id/config/auth"
        )
    if not all(key in job for key in ("inputs", "output", "models")):
        raise ValueError("Job requires inputs, output and models")
    configuration = dict(job.get("config", {}))
    if "slide_id" in job:
        if configuration.get("slide_id") not in (None, job["slide_id"]):
            raise ValueError("Job and config slide_id disagree")
        configuration["slide_id"] = job["slide_id"]
    if configuration.get("cache_dir"):
        configuration["cache_dir"] = _relative(configuration["cache_dir"], path.parent)
    config = EncodeConfig(**configuration)
    auth_options = dict(job.get("auth", {}))
    allowed = {
        "hf_token_env",
        "gemini_api_key_env",
        "cohere_api_key_env",
        "prompt",
        "prompt_credentials",
        "aws_profile",
        "aws_region",
    }
    if set(auth_options) - allowed:
        raise ValueError(
            "Job auth accepts environment-variable names and AWS profile/region only; never put secret values in job files"
        )
    if "prompt" not in auth_options and "prompt_credentials" not in auth_options:
        auth_options["prompt_credentials"] = config.prompt_credentials
    auth = AuthConfig(**auth_options)
    sources = job["inputs"]
    if isinstance(sources, (str, dict)):
        sources = [sources]
    inputs = []
    for source in sources:
        value = dict(source) if isinstance(source, dict) else {"path": source}
        value["path"] = _relative(value["path"], path.parent)
        inputs.append(SourceSpec(**value))
    model_values = job["models"]
    if isinstance(model_values, (str, dict)):
        model_values = [model_values]
    models = []
    for model in model_values:
        value = dict(model) if isinstance(model, dict) else {"model": model}
        if value.get("checkpoint"):
            value["checkpoint"] = _relative(value["checkpoint"], path.parent)
        models.append(EncoderSpec(**value))
    return inputs, _relative(job["output"], path.parent), models, config, auth


def _status(path, include_metadata):
    path = Path(path).expanduser()
    partial = path.with_name(path.stem + ".partial.h5ad")
    if not path.exists():
        return {"output": str(path), "published": False, "resumable_partial": partial.exists()}
    with FeatureReader(path) as reader:
        obs = reader.patch_metadata()
        runs = []
        for run in reader.list_runs():
            eligible = obs["mag"].to_numpy() == run["mag"]
            encoded = obs["encoded_" + run["run_key"]].to_numpy(dtype=bool)
            value = {key: run[key] for key in ("run_key", "model", "mag", "dimension")}
            value.update(
                eligible=int(eligible.sum()),
                completed=int((eligible & encoded).sum()),
                pending=int((eligible & ~encoded).sum()),
            )
            if include_metadata:
                value["metadata"] = reader.model_metadata(run["model"], run["mag"], run["run_key"])
            runs.append(value)
        return {
            "output": str(path),
            "published": True,
            "patches": len(obs),
            "resumable_partial": partial.exists(),
            "runs": runs,
        }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    code = 0
    try:
        if args.command == "models":
            result = model_info(args.model) if args.model else list_models()
        elif args.command in {"access", "prepare"}:
            config = EncodeConfig(
                device=args.device,
                precision=args.precision,
                offline=args.offline,
                cache_dir=args.cache_dir,
                prompt_credentials=args.prompt,
            )
            auth = AuthConfig(
                hf_token_env=args.hf_token_env,
                gemini_api_key_env=args.gemini_key_env,
                cohere_api_key_env=args.cohere_key_env,
                aws_profile=args.aws_profile,
                aws_region=args.region,
                prompt=args.prompt,
            )
            spec = EncoderSpec(
                args.model,
                revision=args.revision,
                checkpoint=args.checkpoint,
                dimension=args.dimension,
            )
            if args.command == "access":
                result = check_access(spec, auth=auth, config=config)
                code = 1 if result.get("available") is False else 0
            else:
                encoder = prepare_model(spec, config=config, auth=auth)
                try:
                    result = encoder.metadata
                finally:
                    encoder.close()
        elif args.command == "encode":
            from dataclasses import replace

            inputs, output, models, config, auth = load_job(args.job)
            if args.prompt:
                auth.prompt = True
            if args.offline:
                config = replace(config, offline=True)
            if args.metadata_policy is not None:
                config = replace(config, metadata_policy=args.metadata_policy)
            result = encode_slide(inputs, output, models, config, auth).to_dict()
        else:
            result = _status(args.output, args.metadata)
    except EncodingFailure as exc:
        print(
            json.dumps({"error": str(exc), "report": exc.report.to_dict()}, allow_nan=False),
            file=sys.stderr,
        )
        return 1
    except (EncoderError, ValueError, OSError, TypeError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(json_safe(result), indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
