"""Run synthetic CPU verification; publish coverage only after every test passes."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
import unittest

from openhisto._common import IMPORT_ROOT
from .registry import ROOT, list_models, implementation_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(IMPORT_ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    from .tests.test_acceptance import AcceptanceTests

    names = AcceptanceTests.verified_profiles
    if len(names) != 34:
        raise RuntimeError("Coverage publication requires all 34 real model profiles")
    if args.write_report:
        profiles = list_models()
        report = {
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation_sha256": implementation_hash(),
            "tests_passed": result.testsRun,
            "dependencies": {
                name: version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "numpy",
                    "torch-geometric",
                    "timm",
                    "einops",
                    "nystrom-attention",
                )
            },
            "synthetic_only": True,
            "checkpoints_downloaded": False,
            "datasets_downloaded": False,
            "gpu_parity_tested": False,
            "models": {
                item["name"]: {
                    "status": "cpu_forward_backward_state_roundtrip_passed",
                    "source_commit": item["source"]["commit"] if item["source"] else None,
                    "input_kind": item["input_kind"],
                }
                for item in profiles
                if item["name"] in names
            },
        }
        (ROOT / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [
            "# CPU coverage",
            "",
            f"Verified {len(names)}/34 real profiles in {result.testsRun} tests on Python {report['python']}, PyTorch {report['dependencies']['torch']}.",
            "",
            "Synthetic CPU forward, finite backward and strict state-dictionary reload were tested for every profile. No datasets or pretrained weights were downloaded. GPU parity and published benchmark metrics were not tested.",
            "",
            "| Profile | Native input | CPU forward/backward/state | Source commit |",
            "|---|---|---|---|",
        ]
        for item in profiles:
            sha = item["source"]["commit"][:12] if item["source"] else "Original baseline"
            lines.append(f"| `{item['name']}` | {item['input_kind']} | Passed | {sha} |")
        (ROOT / "COVERAGE.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
