"""Run with ``python -m openhisto.patch_extraction``."""

import argparse
from dataclasses import fields
import json
import sys

from .config import ExtractionConfig
from .extraction import extract_grid, extract_mask_centered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract WSI patches into one compressed, indexed HDF5 archive."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("grid", "Extract a tissue-filtered grid in row-major order."),
        ("mask", "Extract square patches centered on binary mask components."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("slide_path", help="Input whole-slide image.")
        if name == "mask":
            command.add_argument("mask", help="Input two-dimensional binary mask.")
        command.add_argument("output_path", help="Output HDF5 file, not a patch directory.")
        command.add_argument("--patch-size", type=int, default=512)
        command.add_argument("--stride", type=int, help="Grid stride in target-resolution pixels.")
        command.add_argument("--target-mag", type=float, default=20.0)
        command.add_argument("--source-mag", type=float, help="Override scanner objective power.")
        command.add_argument(
            "--foreground-method",
            default="HSV",
            choices=("HSV", "SL", "luminance", "luminesence", "otsu", "none"),
        )
        command.add_argument("--sat-thresh", type=float, default=0.05)
        command.add_argument("--tissue-fraction", type=float, default=0.05)
        command.add_argument("--lum-thresh", type=float, default=220.0)
        command.add_argument("--sl-tissue-fraction", type=float, default=0.2)
        command.add_argument("--edge", choices=("drop", "pad"), default="drop")
        command.add_argument("--codec", choices=("png", "jpeg"), default="png")
        command.add_argument("--png-compression", type=int, default=6)
        command.add_argument("--jpeg-quality", type=int, default=95)
        command.add_argument("--overwrite", action="store_true")
        if name == "mask":
            command.add_argument(
                "--mask-downsample",
                type=float,
                nargs="+",
                required=True,
                metavar="SCALE",
                help="Level-0 pixels per mask pixel: one value or X Y.",
            )
            command.add_argument(
                "--mask-origin",
                type=float,
                nargs=2,
                default=(0.0, 0.0),
                metavar=("X", "Y"),
                help="Mask origin in level-0 coordinates.",
            )
            command.add_argument("--min-box-size", type=float, default=0.0)
            command.add_argument(
                "--filter-mask-patches",
                action="store_true",
                help="Also apply the foreground filter to mask-selected patches.",
            )
        else:
            command.set_defaults(min_box_size=0.0, filter_mask_patches=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = ExtractionConfig(
            **{field.name: getattr(args, field.name) for field in fields(ExtractionConfig)}
        )
        if args.command == "grid":
            result = extract_grid(args.slide_path, args.output_path, config)
        else:
            if len(args.mask_downsample) not in (1, 2):
                raise ValueError("--mask-downsample requires one value or X Y")
            downsample = (
                args.mask_downsample[0]
                if len(args.mask_downsample) == 1
                else tuple(args.mask_downsample)
            )
            result = extract_mask_centered(
                args.slide_path,
                args.mask,
                args.output_path,
                config,
                mask_downsample=downsample,
                mask_origin=tuple(args.mask_origin),
            )
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except Exception as exc:
        native_error = getattr(sys.modules.get("openslide"), "OpenSlideError", None)
        if native_error is not None and isinstance(exc, native_error):
            parser.exit(2, f"error: {exc}\n")
        raise
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
