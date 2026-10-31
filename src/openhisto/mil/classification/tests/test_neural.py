from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from openhisto.mil.classification import create_neural_head, list_heads, classifier_info
from openhisto.mil.classification.neural import BayesianLinear
from openhisto.mil.heads import load_head, save_head


class NeuralTests(unittest.TestCase):
    def test_all_neural_heads_gradients_probabilities_and_artifact(self):
        for name in ("linear", "mlp", "bayesian_mlp"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                options = {} if name == "linear" else {"hidden_dims": [8], "dropout": 0.1}
                head = create_neural_head(name, input_dim=5, num_classes=3, **options)
                values = torch.randn(6, 5, requires_grad=True)
                output = head(values)
                torch.testing.assert_close(output.probabilities.sum(1), torch.ones(6))
                loss = head.loss(output, torch.tensor([0, 1, 2, 0, 1, 2]), num_train_samples=30)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(values.grad).all())
                self.assertTrue(
                    all(
                        torch.isfinite(parameter.grad).all()
                        for parameter in head.parameters()
                        if parameter.grad is not None
                    )
                )
                head.eval()
                expected = head(values).logits
                path = Path(directory) / "head.milhead"
                spec = {"profile": "synthetic", "dimension": 5}
                save_head(head, path, representation_spec=spec)
                clone = load_head(path, expected_representation=spec)
                torch.testing.assert_close(clone(values).logits, expected)
                with self.assertRaises(FileExistsError):
                    save_head(head, path)
                with self.assertRaisesRegex(ValueError, "representation"):
                    load_head(path, expected_representation={"dimension": 6})

    def test_analytic_kl_and_required_dataset_scaling(self):
        layer = BayesianLinear(2, 3)
        actual = layer.kl_divergence()
        values = []
        for mu, rho in ((layer.weight_mu, layer.weight_rho), (layer.bias_mu, layer.bias_rho)):
            q = torch.distributions.Normal(mu, layer._std(rho))
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
            values.append(torch.distributions.kl_divergence(q, p).sum())
        torch.testing.assert_close(actual, sum(values))
        head = create_neural_head("bayesian_mlp", 2, 2, hidden_dims=[])
        output = head(torch.ones(3, 2))
        labels = torch.tensor([0, 1, 0])
        with self.assertRaisesRegex(ValueError, "num_train_samples"):
            head.loss(output, labels)
        ce = torch.nn.functional.cross_entropy(output.logits, labels)
        torch.testing.assert_close(
            head.loss(output, labels, num_train_samples=10), ce + output.kl / 10
        )
        torch.testing.assert_close(
            head.loss(output, labels, num_train_samples=20), ce + output.kl / 20
        )

    def test_monte_carlo_uncertainty_reproducibility_and_modes(self):
        head = create_neural_head("bayesian_mlp", 4, 3, hidden_dims=[8], dropout=0.5)
        head.train()
        values = torch.randn(5, 4)
        before_rng = torch.get_rng_state().clone()
        first = head.predict_distribution(values, mc_samples=30, seed=8)
        second = head.predict_distribution(values, mc_samples=30, seed=8)
        self.assertTrue(head.training)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        torch.testing.assert_close(first.probabilities, second.probabilities)
        self.assertIsNone(first.logits)
        self.assertTrue((first.uncertainty["predictive_variance"] >= 0).all())
        self.assertTrue((first.uncertainty["predictive_variance"] > 0).any())
        self.assertTrue((first.uncertainty["mutual_information"] >= 0).all())
        torch.testing.assert_close(
            first.uncertainty["predictive_entropy"],
            first.uncertainty["expected_entropy"] + first.uncertainty["mutual_information"],
        )
        head.eval()
        torch.testing.assert_close(head(values).logits, head(values).logits)

    def test_initialization_is_seeded_without_resetting_global_rng(self):
        state = torch.get_rng_state().clone()
        one = create_neural_head("mlp", 4, 2, seed=17)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        two = create_neural_head("mlp", 4, 2, seed=17)
        for a, b in zip(one.parameters(), two.parameters()):
            torch.testing.assert_close(a, b)
        # CPU-only sampling must not reseed unrelated CUDA/MPS generators.
        with patch(
            "torch.cuda.manual_seed_all", side_effect=AssertionError("CPU head reseeded CUDA")
        ):
            bayes = create_neural_head("bayesian_mlp", 4, 2, hidden_dims=[4])
            bayes.predict_distribution(torch.ones(2, 4), mc_samples=2)

    def test_validation_and_catalogue(self):
        self.assertEqual(len(list_heads(kind="neural")), 3)
        self.assertEqual(len(list_heads(kind="classifier")), 11)
        self.assertEqual(classifier_info("rf")["name"], "random_forest")
        with self.assertRaises(ValueError):
            create_neural_head("mlp", 3, 1)
        with self.assertRaises(ValueError):
            create_neural_head("mlp", 3, 2, class_weights=[0, 1])
        head = create_neural_head("linear", 3, 2)
        for values in (torch.empty(0, 3), torch.ones(2, 4), torch.full((2, 3), float("nan"))):
            with self.assertRaises(ValueError):
                head(values)
        doubles = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        output = head(doubles)
        self.assertEqual(output.logits.dtype, torch.float32)
        output.logits.sum().backward()
        self.assertIsNotNone(doubles.grad)


if __name__ == "__main__":
    unittest.main()
