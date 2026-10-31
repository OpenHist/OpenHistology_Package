"""Cohort-aware survival-head gradients through every real MIL profile."""

import importlib
from pathlib import Path
import tempfile
import unittest

import torch

from openhisto.mil.heads import attach_head, representation_dimension, save_head, load_head
from openhisto.mil.models.code_bases import create_model, list_models
from openhisto.mil.survival import create_survival_head, cox_ph_loss, discrete_hazard_loss, TimeGrid


def _different_bag(inputs):
    def features(value):
        return (value.detach().clone() * 1.1 + 0.1).requires_grad_(True)

    if isinstance(inputs, torch.Tensor):
        return features(inputs)
    if isinstance(inputs, dict):
        result = dict(inputs)
        for name in ("features", "high_features"):
            if name in result:
                result[name] = features(result[name])
        for name in ("levels", "clusters"):
            if name in result:
                result[name] = type(result[name])(features(value) for value in result[name])
        return result
    result = inputs.clone()
    result.x = features(result.x)
    return result


class SurvivalMatrixTests(unittest.TestCase):
    verified_profiles = []

    def test_all_34_profiles_cox_and_discrete_joint_gradients(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        verified = []
        try:
            with tempfile.TemporaryDirectory(prefix="cpath-survival-matrix-") as temporary:
                root = Path(temporary)
                for profile in list_models():
                    name, family = profile["name"], profile["family"]
                    with self.subTest(profile=name):
                        if family == "baselines":
                            options, inputs, kwargs = (
                                {"input_dim": 8, "num_classes": 2},
                                torch.randn(6, 8, requires_grad=True),
                                {},
                            )
                        else:
                            fixture = importlib.import_module(
                                "openhisto.mil.models.code_bases.tests.test_" + family
                            )
                            options, inputs, kwargs = fixture.make_case(name, root)
                        other = _different_bag(inputs)
                        backbone = create_model(name, **options)
                        width = representation_dimension(backbone)
                        for kind in ("cox", "discrete_hazard"):
                            head_options = (
                                {"boundaries": [0, 1, 2, 3]} if kind == "discrete_hazard" else {}
                            )
                            head = create_survival_head(
                                kind, width, network="mlp", hidden_dims=[8], **head_options
                            )
                            wrapped = attach_head(backbone, head, train_backbone=True).train()
                            wrapped.zero_grad(set_to_none=True)
                            outputs = [wrapped(current, **kwargs) for current in (inputs, other)]
                            times, events = torch.tensor([1.0, 2.0]), torch.tensor([1, 0])
                            if kind == "cox":
                                predictions = torch.cat([value.head.risk for value in outputs])
                                loss = cox_ph_loss(predictions, times, events)
                            else:
                                predictions = torch.cat([value.head.logits for value in outputs])
                                loss = discrete_hazard_loss(
                                    predictions, times, events, TimeGrid([0, 1, 2, 3])
                                )
                            self.assertTrue(torch.isfinite(loss))
                            loss.backward()
                            gradients = [
                                parameter.grad
                                for parameter in head.parameters()
                                if parameter.grad is not None
                            ]
                            self.assertTrue(gradients)
                            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
                            self.assertTrue(
                                all(value.mil.native_output is not None for value in outputs)
                            )
                        verified.append(name)
        finally:
            torch.set_num_threads(threads)
        self.assertEqual(len(verified), 34)
        type(self).verified_profiles = verified

    def test_attached_artifact_pairs_with_current_backbone_weights(self):
        backbone = create_model("mean_mil", input_dim=4)
        head = create_survival_head("cox", 4)
        wrapped = attach_head(backbone, head)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.milhead"
            save_head(wrapped, path)
            clone = load_head(
                path,
                expected_backbone=backbone,
                expected_representation=wrapped.representation_spec,
            )
            features = torch.randn(2, 4)
            torch.testing.assert_close(clone(features).risk, head(features).risk)
            with torch.no_grad():
                next(backbone.parameters()).add_(1)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_head(path, expected_backbone=backbone)


if __name__ == "__main__":
    unittest.main()
