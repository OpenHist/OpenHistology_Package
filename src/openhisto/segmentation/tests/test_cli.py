"""End-to-end CLI experiment with a real small SegDiff model and RGB labels."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from openhisto.segmentation.__main__ import main
from openhisto.segmentation.data import prepare_dataset, read_sample


class CLITests(unittest.TestCase):
    def command(self, *args):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            result = main(list(args))
        self.assertEqual(result, 0, error.getvalue())
        return output.getvalue()

    def test_materialize_train_resume_evaluate_and_export(self):
        import torch

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = np.zeros((9, 13, 3), dtype=np.uint8)
            rgb[1:5, 2:6] = (255, 0, 0)
            rgb[5:8, 7:11] = (0, 255, 0)
            Image.fromarray(rgb).save(root / "image.png")
            Image.fromarray(rgb).save(root / "mask.png")
            np.save(root / "embedding.npy", np.arange(4, dtype=np.float32))
            with (root / "data.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["sample_id", "image_path", "mask_path", "embedding"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "example",
                        "image_path": "image.png",
                        "mask_path": "mask.png",
                        "embedding": json.dumps({"path": "embedding.npy"}),
                    }
                )
            config = {
                "model": {
                    "model": "SegDiff",
                    "num_classes": 3,
                    "pretrained": False,
                    "image_size": [9, 13],
                    "options": {
                        "feature_dim": 4,
                        "base_channels": 8,
                        "channel_mults": [1, 2],
                        "time_emb_dim": 16,
                        "embed_dim": 8,
                        "attn_dim": 8,
                        "timesteps": 2,
                        "inference_seed": 7,
                    },
                },
                "mask": {"kind": "rgb", "palette": {"0,0,0": 0, "255,0,0": 1, "0,255,0": 2}},
                "train": {"num_epochs": 1, "batch_size": 1, "augment": True},
                "predict": {"batch_size": 2},
                "exports": ["labels", "probabilities", "scores", "logits", "qupath", "binary"],
            }
            path = root / "config.json"
            path.write_text(json.dumps(config))
            self.command("prepare-model", str(root / "prepared_model"), "--config", str(path))
            self.assertTrue((root / "prepared_model" / "bundle.json").exists())
            self.command(
                "prepare-data",
                str(root / "data.csv"),
                str(root / "prepared_data"),
                "--config",
                str(path),
            )
            # Canonical per-row mask spec must override the experiment's RGB default.
            items = prepare_dataset(
                root / "prepared_data" / "manifest.csv", mask_spec=config["mask"], num_classes=3
            )
            loaded = read_sample(items[0], num_classes=3)
            self.assertEqual(set(np.unique(loaded.mask)), {0, 1, 2})
            self.command(
                "train",
                str(root / "data.csv"),
                str(root / "data.csv"),
                str(root / "run"),
                "--config",
                str(path),
            )
            metrics = json.loads(
                self.command(
                    "evaluate",
                    str(root / "run" / "best"),
                    str(root / "prepared_data" / "manifest.csv"),
                    "--config",
                    str(path),
                )
            )
            self.assertEqual(metrics["pixels"], 9 * 13)
            self.command(
                "predict",
                str(root / "run" / "best"),
                str(root / "data.csv"),
                str(root / "predictions"),
                "--config",
                str(path),
            )
            for name in (
                "example_labels.png",
                "example_probabilities.npy",
                "example_logits.npy",
                "example_qupath.geojson",
                "example_class_1.png",
                "example_class_2.png",
            ):
                self.assertTrue((root / "predictions" / name).is_file(), name)
            config["train"]["num_epochs"] = 2
            path.write_text(json.dumps(config))
            self.command(
                "train",
                str(root / "data.csv"),
                str(root / "data.csv"),
                str(root / "run"),
                "--config",
                str(path),
                "--resume",
                str(root / "run" / "latest"),
            )
            self.assertEqual(len(json.loads((root / "run" / "history.json").read_text())), 2)

    def test_actionable_bad_configuration_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"unknown": {}}')
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(
                    main(["prepare-model", str(Path(tmp) / "model"), "--config", str(path)]), 2
                )
            self.assertIn("Unknown configuration", error.getvalue())


if __name__ == "__main__":
    unittest.main()
