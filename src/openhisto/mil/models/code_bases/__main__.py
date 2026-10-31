"""Read-only catalogue CLI; acquisition is a separate explicit command."""

import argparse
import json
from .registry import list_models, model_info


def main(argv=None):
    parser = argparse.ArgumentParser(description="MIL source and CPU adapter catalogue")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    info = commands.add_parser("info")
    info.add_argument("model")
    args = parser.parse_args(argv)
    try:
        value = list_models() if args.command == "list" else model_info(args.model)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
