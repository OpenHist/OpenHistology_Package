"""Run real synthetic head/backbone/backend tests and publish coverage."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import unittest

from openhisto._common import IMPORT_ROOT


ROOT = Path(__file__).resolve().parents[1]


def implementation_hash():
    digest = hashlib.sha256()
    for directory in ("heads", "classification", "survival"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for directory in (
        "models/code_bases/tests",
        "classification/tests",
        "survival/tests",
        "heads/tests",
    ):
        suite.addTests(loader.discover(str(ROOT / directory), top_level_dir=str(IMPORT_ROOT)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        return 1
    if args.write_report:
        from openhisto.mil.classification import list_classifiers, list_heads
        from openhisto.mil.models.code_bases import list_models
        from openhisto.mil.models.code_bases.registry import implementation_hash as mil_hash
        from openhisto.mil.survival import list_heads as survival_heads
        from openhisto.mil.heads.tests.test_survival_matrix import SurvivalMatrixTests

        models = list_models()
        if len(models) != 34 or len(SurvivalMatrixTests.verified_profiles) != 34:
            raise RuntimeError("Coverage requires all 34 real MIL attachments")
        report = {
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tests_passed": result.testsRun,
            "tests_skipped": 0,
            "implementation_sha256": implementation_hash(),
            "mil_implementation_sha256": mil_hash(),
            "synthetic_only": True,
            "clinical_records_used": False,
            "dependencies": {
                name: version(name)
                for name in (
                    "torch",
                    "numpy",
                    "scikit-learn",
                    "skops",
                    "xgboost",
                    "lightgbm",
                    "catboost",
                )
            },
            "neural_heads": {
                item["name"]: {"cpu_forward_backward": "passed", "artifact_roundtrip": "passed"}
                for item in list_heads(kind="neural")
            },
            "classifiers": {
                item["name"]: {"cpu_binary_multiclass": "passed", "artifact_roundtrip": "passed"}
                for item in list_classifiers()
            },
            "survival_heads": {
                item["name"]: {
                    "linear_mlp_gradients": "passed",
                    "loss_oracles": "passed",
                    "artifact_roundtrip": "passed",
                }
                for item in survival_heads()
            },
            "mil_profiles": {
                item["name"]: {
                    "representation_gradients_freeze_reload": "passed",
                    "cox_and_discrete_head_gradients": "passed",
                    "source_commit": item["source"]["commit"] if item["source"] else None,
                }
                for item in models
            },
            "limits": [
                "No real-cohort accuracy/clinical validation",
                "No GPU validation",
                "Small synthetic configurations, not every upstream ablation",
            ],
        }
        (ROOT / "heads" / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [
            "# Head and backend CPU coverage",
            "",
            f"{result.testsRun} tests passed without skips on Python {report['python']}, Torch {report['dependencies']['torch']}.",
            "",
            "All tests use synthetic features, labels, durations, prototypes and checkpoints. Existing MIL regressions are included. No clinical records were read.",
            "",
            "| Component | CPU checks |",
            "|---|---|",
        ]
        for name in report["neural_heads"]:
            lines.append(f"| Neural `{name}` | Forward, backward, probabilities, artifact reload |")
        for name in report["classifiers"]:
            lines.append(
                f"| Classifier `{name}` | Real binary/multiclass fit, class order, probabilities, artifact reload |"
            )
        for name in report["survival_heads"]:
            lines.append(
                f"| Survival `{name}` | Linear/MLP gradients, censoring/tie oracles, calibrated artifact reload |"
            )
        lines += [
            "",
            "| MIL profile | Representation / gradient / freeze / reload | Cox + discrete head gradients |",
            "|---|---|---|",
        ]
        for name in report["mil_profiles"]:
            lines.append(f"| `{name}` | Passed | Passed |")
        lines += [
            "",
            "This verifies CPU execution and mathematical regression cases. Real-cohort accuracy, clinical validity, GPU execution and every optional upstream configuration remain untested.",
        ]
        (ROOT / "heads" / "COVERAGE.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
