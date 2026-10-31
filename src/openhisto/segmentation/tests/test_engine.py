"""Offline integration tests for data -> optimization -> reload -> prediction."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from openhisto._common import worker_environment
from openhisto.segmentation import Sample, ModelConfig, TrainConfig, PredictConfig, MaskSpec
from openhisto.segmentation.config import LoadedSample
from openhisto.segmentation.engine import (
    train,
    predict,
    evaluate,
    save_model,
    load_model,
    transform_sample,
    _Metrics,
)


class ToyAdapter:
    def __init__(self, config, device="cpu", load_dir=None):
        self.config = config
        self.model = torch.nn.Conv2d(3, config.num_classes, 1).to(device)
        self.device = device
        self.modes = []
        if load_dir is not None:
            self.model.load_state_dict(torch.load(Path(load_dir) / "toy.pt", weights_only=True))

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def loss(self, samples):
        self.modes.append((self.model.training, torch.is_grad_enabled()))
        values = []
        for s in samples:
            x = (
                torch.from_numpy(s.image.copy()).permute(2, 0, 1).float()[None].to(self.device)
                / 255
            )
            y = torch.from_numpy(s.mask.copy()).long()[None].to(self.device)
            values.append(
                torch.nn.functional.cross_entropy(
                    self.model(x), y, ignore_index=self.config.ignore_index
                )
            )
        return torch.stack(values).mean()

    def infer(self, samples):
        result = []
        for s in samples:
            assert s.mask is None, "Inference must never receive targets"
            x = (
                torch.from_numpy(s.image.copy()).permute(2, 0, 1).float()[None].to(self.device)
                / 255
            )
            logits = self.model(x)[0]
            result.append(
                {
                    "probabilities": logits.softmax(0).detach().cpu().numpy(),
                    "scores": logits.detach().cpu().numpy(),
                    "logits": logits.detach().cpu().numpy(),
                }
            )
        return result

    def save(self, path):
        torch.save(self.model.state_dict(), Path(path) / "toy.pt")


def samples(count=3):
    result = []
    for i in range(count):
        image = np.zeros((5 + i, 7 + i, 3), dtype=np.uint8)
        image[:, 3:, 0] = 255
        mask = np.zeros(image.shape[:2], dtype=np.int64)
        mask[:, 3:] = 1
        mask[0, 0] = -100
        result.append(
            Sample(image, mask, sample_id=f"case_{i}", mask_spec=MaskSpec(kind="indexed"))
        )
    return result


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = ModelConfig(model="segformer", pretrained=False, image_size=(5, 7))

    def factory(self, config, device="cpu", load_dir=None):
        return ToyAdapter(config, device, load_dir)

    def test_discovery_is_lazy(self):
        code = "import sys; from openhisto import segmentation; assert len(segmentation.list_models())==6; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, env=worker_environment())

    def test_prediction_batches_unlabeled_native_size_and_mode(self):
        model = ToyAdapter(self.config)
        model.model.train()
        items = samples()
        items[0].mask = "/definitely/not/a/mask.png"
        predictions = list(predict(model, items, config=PredictConfig(batch_size=2, logits=True)))
        self.assertTrue(model.model.training)
        self.assertEqual(len(predictions), 3)
        for item, prediction in zip(items, predictions):
            self.assertEqual(prediction.labels.shape, item.image.shape[:2])
            np.testing.assert_allclose(prediction.probabilities.sum(0), 1, atol=1e-6)
            self.assertEqual(prediction.logits.shape, (2, *item.image.shape[:2]))

    def test_train_resume_matches_uninterrupted_training(self):
        data = samples()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("openhisto.segmentation.semantic.registry.create_adapter", self.factory),
        ):
            root = Path(tmp)
            settings = TrainConfig(num_epochs=2, batch_size=2, augment=True, seed=19)
            whole = train(
                data, data, root / "whole", model_config=self.config, train_config=settings
            )
            first = train(
                data,
                data,
                root / "split",
                model_config=self.config,
                train_config=TrainConfig(num_epochs=1, batch_size=2, augment=True, seed=19),
            )
            result = train(
                data, data, root / "split", train_config=settings, resume=root / "split" / "latest"
            )
            left, right = load_model(root / "whole" / "final"), load_model(root / "split" / "final")
            for name, value in left.model.state_dict().items():
                torch.testing.assert_close(value, right.model.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(whole.history, result.history)
            self.assertEqual(result.completed_epochs, 2)
            self.assertTrue((root / "split" / "best" / "training_state.pt").is_file())
            self.assertEqual(
                json.loads((root / "split" / "history.json").read_text()), result.history
            )
            self.assertTrue((root / "split" / "epoch_2" / "bundle.json").is_file())
            self.assertTrue(first.history[0]["metrics"]["pixels"] > 0)

    def test_validation_does_not_switch_to_train_or_enable_grad(self):
        model = ToyAdapter(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            train(
                samples(2),
                samples(2),
                Path(tmp) / "run",
                model=model,
                train_config=TrainConfig(num_epochs=1, batch_size=2, augment=False),
            )
        self.assertIn((True, True), model.modes)
        self.assertIn((False, False), model.modes)

    def test_resume_rejects_optimizer_configuration_change(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("openhisto.segmentation.semantic.registry.create_adapter", self.factory),
        ):
            path = Path(tmp) / "run"
            train(
                samples(1),
                samples(1),
                path,
                model_config=self.config,
                train_config=TrainConfig(num_epochs=1, batch_size=1),
            )
            with self.assertRaisesRegex(ValueError, "batch_size"):
                train(
                    samples(1),
                    samples(1),
                    path,
                    train_config=TrainConfig(num_epochs=2, batch_size=2),
                    resume=path / "latest",
                )

    def test_native_logits_are_resized_before_softmax(self):
        from openhisto.segmentation.engine import _prediction

        model = ToyAdapter(self.config)
        native = LoadedSample(np.zeros((1, 5, 3), dtype=np.uint8), sample_id="native")
        logits = np.array([[[-100.0, 5.0]], [[0.0, 0.0]]], dtype=np.float32)
        probability = torch.from_numpy(logits).softmax(0).numpy()
        result = _prediction(
            model,
            native,
            {"logits": logits, "scores": logits, "probabilities": probability},
            PredictConfig(logits=True),
        )
        expected = torch.from_numpy(result.logits).softmax(0).numpy()
        np.testing.assert_allclose(result.probabilities, expected, atol=1e-7)
        np.testing.assert_array_equal(result.labels, result.logits.argmax(0))

    def test_query_scores_are_resized_before_normalization(self):
        from openhisto.segmentation.engine import _prediction, _restore_array

        model = ToyAdapter(self.config)
        native = LoadedSample(np.zeros((1, 5, 3), dtype=np.uint8), sample_id="native")
        scores = np.array([[[1.0, 100.0]], [[1.0, 0.0]]], dtype=np.float32)
        result = _prediction(
            model,
            native,
            {"scores": scores, "probabilities": scores / scores.sum(0)},
            PredictConfig(scores=True),
        )
        expected = _restore_array(scores, (1, 5))
        expected /= expected.sum(0)
        np.testing.assert_allclose(result.probabilities, expected, atol=1e-7)

    def test_prediction_restores_mixed_module_modes(self):
        model = ToyAdapter(self.config)
        model.model = torch.nn.Sequential(model.model, torch.nn.Dropout(0.2))
        model.model.train()
        model.model[1].eval()
        before = [module.training for module in model.model.modules()]
        list(predict(model, samples(1)))
        self.assertEqual(before, [module.training for module in model.model.modules()])

    def test_supplied_model_retains_saved_mask_defaults(self):
        model = ToyAdapter(self.config)
        model.mask_spec = MaskSpec(kind="rgb", palette={"0,0,0": 0, "255,0,0": 1})
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        image[:, 3:, 0] = 255
        data = [Sample(image, image.copy(), sample_id="rgb")]
        with tempfile.TemporaryDirectory() as tmp:
            train(
                data,
                data,
                Path(tmp) / "run",
                model=model,
                train_config=TrainConfig(num_epochs=1, batch_size=1, augment=False),
            )
            metadata = json.loads((Path(tmp) / "run" / "final" / "bundle.json").read_text())
            self.assertEqual(metadata["mask_spec"]["kind"], "rgb")

    def test_publication_does_not_replace_a_racing_destination(self):
        from openhisto.segmentation.engine import _publish

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "bundle"

            def writer(staging):
                (staging / "new").write_text("new")
                destination.mkdir()
                (destination / "original").write_text("original")

            with self.assertRaises(FileExistsError):
                _publish(destination, writer)
            self.assertEqual((destination / "original").read_text(), "original")
            self.assertFalse((destination / "new").exists())

    def test_metrics_ignore_and_missing_classes(self):
        metrics = _Metrics(3, -100)
        metrics.update(np.array([[0, 1], [2, 2]]), np.array([[0, 1], [-100, 1]]))
        result = metrics.result()
        self.assertEqual(result["pixels"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["per_class"][1]["dice"], 2 / 3)

    def test_evaluate_preserves_mode_and_uses_native_masks(self):
        model = ToyAdapter(self.config)
        model.model.train()
        result = evaluate(model, samples(), config=PredictConfig(batch_size=2))
        self.assertTrue(model.model.training)
        self.assertEqual(result["pixels"], sum(s.mask.size - 1 for s in samples()))

    def test_resize_flip_transforms_image_mask_and_prompts_together(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        image[:, 0] = 255
        loaded = LoadedSample(
            image,
            (image[..., 0] > 0).astype(np.int64),
            prompts={"box": [0, 0, 1, 1], "point_coords": [[0, 0]]},
        )
        with patch("numpy.random.random", side_effect=[0.1, 0.9]):
            output = transform_sample(loaded, (4, 6), augment=True)
        self.assertGreater(output.image[0, -1, 0], output.image[0, 0, 0])
        self.assertEqual(output.mask[0, -1], 1)
        self.assertEqual(output.prompts["point_coords"][0], [5.0, 0.0])
        self.assertEqual(output.prompts["box"], [3.0, 0.0, 5.0, 2.0])

    def test_bundle_refuses_overwrite_and_schema_changes(self):
        model = ToyAdapter(self.config)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("openhisto.segmentation.semantic.registry.create_adapter", self.factory),
        ):
            path = Path(tmp) / "bundle"
            save_model(model, path)
            loaded = load_model(path)
            self.assertEqual(loaded.config.image_size, (5, 7))
            with self.assertRaises(FileExistsError):
                save_model(model, path)
            metadata = json.loads((path / "bundle.json").read_text())
            metadata["schema_version"] = 100
            (path / "bundle.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "schema"):
                load_model(path)

    def test_empty_and_all_ignored_training_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                train([], samples(1), Path(tmp) / "empty", model=ToyAdapter(self.config))
            data = samples(1)
            data[0].mask[:] = -100
            with self.assertRaisesRegex(ValueError, "valid target"):
                train(data, samples(1), Path(tmp) / "ignored", model=ToyAdapter(self.config))


if __name__ == "__main__":
    unittest.main()
