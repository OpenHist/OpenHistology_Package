import argparse
import json
from . import head_info, list_heads


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Classification head catalogue; fitting uses the Python API"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--kind", choices=["neural", "classifier"])
    info = commands.add_parser("info")
    info.add_argument("name")
    info.add_argument("--kind", choices=["neural", "classifier"])
    args = parser.parse_args(argv)
    try:
        value = (
            list_heads(kind=args.kind)
            if args.command == "list"
            else head_info(args.name, kind=args.kind)
        )
    except ValueError as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
