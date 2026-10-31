"""Private JSONL protocol. No vendor imports occur before the startup request."""

import base64
from contextlib import redirect_stdout
import io
import json
import os
import sys

from .base import VLAError, error_text, validate_results
from .config import DescribeConfig, ModelSpec, json_safe


def main():
    protocol = sys.stdout
    backend = None

    def send(value):
        protocol.write(json.dumps(json_safe(value), allow_nan=False) + "\n")
        protocol.flush()

    try:
        first = sys.stdin.readline()
        if not first:
            return 0
        request = json.loads(first)
        if request.get("op") != "prepare":
            raise VLAError("Worker requires a prepare request first")
        spec = ModelSpec(**request["spec"])
        config = DescribeConfig(**request["config"])
        if config.offline:
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if config.cache_dir:
            os.environ["HF_HUB_CACHE"] = config.cache_dir
        if spec.options.get("upstream_path"):
            sys.path.insert(0, str(spec.options["upstream_path"]))
        with redirect_stdout(sys.stderr):
            from .registry import create_backend

            backend = create_backend(spec, config)
        send({"op": "ready", "metadata": backend.metadata})
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("op") == "close":
                break
            if (
                request.get("op") != "generate"
                or not isinstance(request.get("prompt"), str)
                or not request["prompt"].strip()
            ):
                raise VLAError("Invalid patch generation request")
            from PIL import Image

            images = []
            for encoded in request["images"]:
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                    if image.mode != "RGB":
                        raise VLAError("Protocol images must be RGB")
                    images.append(image.copy())
            with redirect_stdout(sys.stderr):
                results = validate_results(backend.generate(images, request["prompt"]), len(images))
            send(
                {
                    "op": "result",
                    "id": request["id"],
                    "results": [item.to_dict() for item in results],
                }
            )
        return 0
    except Exception as exc:
        send({"op": "error", "error": f"{type(exc).__name__}: {error_text(exc)}"})
        return 1
    finally:
        if backend is not None:
            with redirect_stdout(sys.stderr):
                backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
