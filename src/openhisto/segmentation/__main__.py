"""JSON-configured command line counterpart of the public Python API."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .config import ModelConfig, TrainConfig, PredictConfig, MaskSpec, json_safe


def _source(value):
    # The dataset reader owns JSON/CSV resolution relative to the manifest.
    return value


def _settings(args):
    result = json.loads(Path(args.config).read_text()) if getattr(args, "config", None) else {}
    allowed = {"model", "train", "predict", "mask", "exports"}
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPath researcher-facing semantic segmentation")
    sub = parser.add_subparsers(dest="command", required=True)
    models = sub.add_parser("models", help="Inspect models without importing ML backends")
    models.add_argument("name", nargs="?")
    prep = sub.add_parser(
        "prepare-model", help="Explicitly download/initialize a local checkpoint bundle"
    )
    prep.add_argument("destination")
    prep.add_argument("--config")
    prep.add_argument("--model")
    prep.add_argument("--random-init", action="store_true")
    prep.add_argument("--overwrite", action="store_true")
    data = sub.add_parser(
        "prepare-data", help="Validate and materialize aligned images/masks into a portable dataset"
    )
    data.add_argument("source")
    data.add_argument("destination")
    data.add_argument("--config")
    train = sub.add_parser("train", help="Train or resume an experiment")
    train.add_argument("train_data")
    train.add_argument("val_data")
    train.add_argument("output")
    train.add_argument("--config")
    train.add_argument("--resume")
    for command in ("evaluate", "predict"):
        cmd = sub.add_parser(command)
        cmd.add_argument("checkpoint")
        cmd.add_argument("data")
        if command == "predict":
            cmd.add_argument("output")
            cmd.add_argument("--formats", nargs="+", default=None)
            cmd.add_argument("--overwrite", action="store_true")
        cmd.add_argument("--config")
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            from . import list_models, model_info

            print(json.dumps(model_info(args.name) if args.name else list_models(), indent=2))
            return 0
        settings = _settings(args)
        if args.command == "prepare-model":
            from . import prepare_model

            options = settings.get("model", {})
            if args.model:
                options["model"] = args.model
            if args.random_init:
                options["pretrained"] = False
            prepare_model(ModelConfig(**options), args.destination, overwrite=args.overwrite)
            print(json.dumps({"checkpoint": str(Path(args.destination).resolve())}))
        elif args.command == "prepare-data":
            from .data import prepare_dataset, read_sample
            from .engine import _publish, _json_write
            from PIL import Image
            import numpy as np
            import csv

            config = ModelConfig(**settings.get("model", {}))
            mask_spec = MaskSpec(**settings["mask"]) if "mask" in settings else None
            items = prepare_dataset(
                _source(args.source),
                mask_spec=mask_spec,
                num_classes=config.num_classes,
                ignore_index=config.ignore_index,
            )

            def write(directory):
                rows = []
                for index, item in enumerate(items):
                    sample = read_sample(
                        item, num_classes=config.num_classes, ignore_index=config.ignore_index
                    )
                    stem = f"{index:08d}"
                    Image.fromarray(sample.image).save(directory / f"{stem}.png")
                    mask_path = ""
                    if sample.mask is not None:
                        np.save(directory / f"{stem}.npy", sample.mask, allow_pickle=False)
                        mask_path = f"{stem}.npy"
                    row = {
                        "sample_id": sample.sample_id,
                        "slide_id": sample.slide_id or "",
                        "image_path": f"{stem}.png",
                        "mask_path": mask_path,
                        "mask_spec": json.dumps(
                            MaskSpec(kind="indexed", ignore_values=[config.ignore_index]).to_dict()
                        ),
                    }
                    for name in (
                        "xy_level0",
                        "footprint_level0",
                        "valid_bounds",
                        "prompts",
                        "embedding",
                    ):
                        value = getattr(sample, name)
                        if value is not None:
                            if name == "embedding":
                                from .semantic.special import load_embedding

                                np.save(
                                    directory / f"{stem}_embedding.npy",
                                    load_embedding(sample),
                                    allow_pickle=False,
                                )
                                value = {"path": f"{stem}_embedding.npy"}
                            row[name] = json.dumps(json_safe(value))
                    rows.append(row)
                keys = [
                    "sample_id",
                    "slide_id",
                    "image_path",
                    "mask_path",
                    "mask_spec",
                    "xy_level0",
                    "footprint_level0",
                    "valid_bounds",
                    "prompts",
                    "embedding",
                ]
                with (directory / "manifest.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(rows)
                _json_write(
                    directory / "dataset.json",
                    {
                        "num_classes": config.num_classes,
                        "ignore_index": config.ignore_index,
                        "mask": MaskSpec(
                            kind="indexed", ignore_values=[config.ignore_index]
                        ).to_dict(),
                        "samples": len(rows),
                    },
                )

            _publish(args.destination, write)
            print(
                json.dumps(
                    {
                        "manifest": str(Path(args.destination).resolve() / "manifest.csv"),
                        "samples": len(items),
                    }
                )
            )
        elif args.command == "train":
            from . import train

            result = train(
                _source(args.train_data),
                _source(args.val_data),
                args.output,
                model_config=ModelConfig(**settings.get("model", {})),
                train_config=TrainConfig(**settings.get("train", {})),
                mask_spec=MaskSpec(**settings["mask"]) if "mask" in settings else None,
                resume=args.resume,
            )
            print(json.dumps(result.to_dict(), indent=2))
        elif args.command == "evaluate":
            from . import evaluate

            print(
                json.dumps(
                    evaluate(
                        args.checkpoint,
                        _source(args.data),
                        config=PredictConfig(**settings.get("predict", {})),
                        mask_spec=MaskSpec(**settings["mask"]) if "mask" in settings else None,
                    ),
                    indent=2,
                )
            )
        elif args.command == "predict":
            from . import predict, export_prediction

            formats = args.formats or settings.get("exports", ["labels"])
            config = dict(settings.get("predict", {}))
            for name in ("probabilities", "scores", "logits"):
                if name in formats:
                    config[name] = True
            count = 0
            for result in predict(
                args.checkpoint, _source(args.data), config=PredictConfig(**config)
            ):
                export_prediction(result, args.output, formats=formats, overwrite=args.overwrite)
                count += 1
            print(json.dumps({"output": str(Path(args.output).resolve()), "samples": count}))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
