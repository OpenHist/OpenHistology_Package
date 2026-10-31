"""Run the stain-transfer container's CPU-only synthetic acceptance check."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


def main():
    import spams  # noqa: F401 -- verify the native backend is loadable in the image
    import torch

    with tempfile.TemporaryDirectory(prefix="openhisto-container-smoke-") as temporary:
        root = Path(temporary)
        rng = np.random.default_rng(12)
        for name in ("source", "reference"):
            rgb = rng.integers(30, 220, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(root / f"{name}.png")
        command = [
            sys.executable,
            "-m",
            "openhisto.stain_transfer",
            "normalize",
            str(root / "source.png"),
            str(root / "output.png"),
            "--method",
            "reinhard",
            "--reference",
            str(root / "reference.png"),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        json.loads(result.stdout)
        with Image.open(root / "output.png") as output:
            if output.mode != "RGB" or output.size != (32, 32):
                raise RuntimeError("Container normalization changed the output image contract")
        if not (root / "output.png.json").is_file():
            raise RuntimeError("Container normalization did not publish provenance")
    print(json.dumps({"synthetic_normalization": "passed", "torch": torch.__version__}))


if __name__ == "__main__":
    main()
