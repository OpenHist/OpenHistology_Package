"""Synthetic hand oracles for censoring, tied risk sets and fitted calibration."""

from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from openhisto.mil.heads.types import SurvivalOutput
from openhisto.mil.survival import (
    create_survival_head,
    cox_ph_loss,
    discrete_hazard_loss,
    discrete_hazard_nll,
    fit_cox_baseline,
    fit_time_grid,
    head_info,
    list_heads,
    TimeGrid,
)


class CoxLossTests(unittest.TestCase):
    def test_tied_events_and_same_time_censor_hand_oracle(self):
        risk = torch.tensor([2.0, 3.0, 5.0, 7.0], dtype=torch.float64).log()
        times = torch.tensor([1.0, 1.0, 1.0, 2.0])
        events = torch.tensor([1, 1, 0, 1])
        expected = {
            "breslow": (2 * math.log(17) - math.log(6)) / 3,
            "efron": (math.log(17) + math.log(14.5) - math.log(6)) / 3,
        }
        for method in expected:
            for order in itertools.permutations(range(4)):
                index = torch.tensor(order)
                actual = cox_ph_loss(risk[index], times[index], events[index], ties=method)
                self.assertAlmostEqual(float(actual), expected[method], places=12)

    def test_unique_times_hand_oracle_and_gradient(self):
        risk = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64).log().requires_grad_()
        times = [1.0, 2.0, 3.0]
        events = [1, 0, 1]
        expected = math.log(10 / 2) / 2
        actual = cox_ph_loss(risk, times, events)
        self.assertAlmostEqual(float(actual), expected, places=12)
        actual.backward()
        self.assertTrue(torch.isfinite(risk.grad).all())
        self.assertGreater(float(risk.grad.abs().sum()), 0)

    def test_extreme_risks_and_constant_shift_invariance(self):
        values = torch.tensor([10000.0, -10000.0, 9999.0, -9999.0], requires_grad=True)
        for method in ("efron", "breslow"):
            loss = cox_ph_loss(values, [1.0, 1.0, 2.0, 3.0], [1, 1, 0, 1], ties=method)
            self.assertTrue(torch.isfinite(loss))
            gradient = torch.autograd.grad(loss, values, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            shifted = cox_ph_loss(values + 1000.0, [1.0, 1.0, 2.0, 3.0], [1, 1, 0, 1], ties=method)
            torch.testing.assert_close(loss, shifted)

    def test_reject_degenerate_risk_sets_and_bad_targets(self):
        with self.assertRaisesRegex(ValueError, "two subjects"):
            cox_ph_loss(torch.tensor([0.0]), [1.0], [1])
        with self.assertRaisesRegex(ValueError, "observed event"):
            cox_ph_loss(torch.zeros(2), [1.0, 2.0], [0, 0])
        risk = torch.randn(2, requires_grad=True)
        zero = cox_ph_loss(risk, [1.0, 2.0], [0, 0], allow_no_events=True)
        zero.backward()
        torch.testing.assert_close(risk.grad, torch.zeros_like(risk))
        for times, flags in [
            ([-1.0, 2.0], [0, 1]),
            ([1.0, 2.0], [1, 2]),
            ([1.0, float("nan")], [1, 0]),
        ]:
            with self.assertRaises(ValueError):
                cox_ph_loss(torch.zeros(2), times, flags)


class TimeGridTests(unittest.TestCase):
    def test_default_equal_width_training_grid(self):
        grid = fit_time_grid([1.0, 4.0, 10.0], [1, 0, 1])
        self.assertEqual(grid.num_bins, 20)
        self.assertEqual(len(grid.boundaries), 21)
        torch.testing.assert_close(
            grid.boundaries, torch.linspace(0.0, 10.0, 21, dtype=torch.float64)
        )
        self.assertEqual(grid.metadata["fit_split"], "train")
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "train"):
                fit_time_grid([1.0, 2.0], split=split)

    def test_rounding_exact_boundaries_and_partial_censoring(self):
        grid = TimeGrid([0.0, 2.0, 5.0, 9.0])
        times = [0.0, 1.0, 2.0, 2.0, 2.1, 2.1, 5.0, 5.0, 9.0, 9.0]
        events = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        labels = grid.transform(times, events)
        self.assertEqual(labels.event_bin.tolist(), [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
        self.assertEqual(labels.completed_intervals.tolist(), [0, 0, 1, 1, 1, 1, 2, 2, 3, 3])
        logits = torch.zeros(10, 3, requires_grad=True)
        losses = discrete_hazard_nll(logits, labels, reduction="none")
        torch.testing.assert_close(
            losses,
            torch.tensor([1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=torch.float64)
            * math.log(2),
        )
        losses.sum().backward()
        torch.testing.assert_close(logits.grad[1], torch.zeros(3))

    def test_nonconstant_hazard_hand_oracle(self):
        probabilities = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
        logits = torch.logit(probabilities).repeat(4, 1)
        times, events = [2.1, 2.1, 9.0, 9.0], [1, 0, 1, 0]
        actual = discrete_hazard_loss(logits, times, events, [0.0, 2.0, 5.0, 9.0], reduction="none")
        expected = [
            -math.log(0.8 * 0.3),
            -math.log(0.8),
            -math.log(0.8 * 0.7 * 0.4),
            -math.log(0.8 * 0.7 * 0.6),
        ]
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))

    def test_administrative_truncation_preserves_original_labels(self):
        grid = TimeGrid([0.0, 2.0, 5.0, 9.0])
        with self.assertRaisesRegex(ValueError, "horizon"):
            grid.transform([10.0, 11.0], [1, 0])
        labels = grid.transform([10.0, 11.0], [1, 0], administrative_truncation=True)
        self.assertEqual(labels.durations.tolist(), [9.0, 9.0])
        self.assertEqual(labels.events.tolist(), [False, False])
        self.assertEqual(labels.original_durations.tolist(), [10.0, 11.0])
        self.assertEqual(labels.original_events.tolist(), [True, False])
        self.assertEqual(labels.metadata["administratively_truncated"], 2)
        self.assertEqual(labels.metadata["observed_events_truncated"], 1)
        _, returned = discrete_hazard_loss(
            torch.zeros(2, 3),
            [10.0, 11.0],
            [1, 0],
            grid,
            administrative_truncation=True,
            return_targets=True,
        )
        self.assertEqual(returned.truncated_count, 2)

    def test_extreme_logits_finite_loss_and_gradient(self):
        logits = torch.tensor([[10000.0, -10000.0], [-10000.0, 10000.0]], requires_grad=True)
        loss = discrete_hazard_loss(logits, [1.0, 2.0], [1, 0], [0.0, 1.0, 2.0])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_truncation_override_requires_actual_boolean(self):
        grid = TimeGrid([0.0, 1.0, 2.0])
        for flag in ("False", 1, None):
            with self.assertRaisesRegex(ValueError, "boolean"):
                grid.transform([3.0], [1], administrative_truncation=flag)
            with self.assertRaisesRegex(ValueError, "boolean"):
                discrete_hazard_loss(
                    torch.zeros(1, 2), [3.0], [1], grid, administrative_truncation=flag
                )


class BaselineTests(unittest.TestCase):
    def test_breslow_oracle_and_absolute_survival(self):
        risk = torch.tensor([2.0, 3.0, 5.0, 7.0], dtype=torch.float64).log()
        baseline = fit_cox_baseline(risk, [1.0, 1.0, 1.0, 2.0], [1, 1, 0, 1])
        torch.testing.assert_close(
            baseline.cumulative_hazard, torch.tensor([2 / 17, 2 / 17 + 1 / 7], dtype=torch.float64)
        )
        prediction = baseline.predict(
            torch.tensor([0.0, math.log(2)], dtype=torch.float64), [0.0, 0.5, 1.0, 1.5, 2.0]
        )
        expected_h = torch.tensor([0.0, 0.0, 2 / 17, 2 / 17, 2 / 17 + 1 / 7], dtype=torch.float64)
        torch.testing.assert_close(prediction.cumulative_hazard[0], expected_h)
        torch.testing.assert_close(prediction.survival[1], (-2 * expected_h).exp())
        with self.assertRaisesRegex(ValueError, "horizon"):
            baseline.predict([0.0], [3.0])
        self.assertEqual(
            baseline.predict([0.0], [3.0], allow_extrapolation=True).aux["extrapolation"],
            "carry_last_baseline",
        )

    def test_extreme_offset_invariance_and_finite_predictions(self):
        risks = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
        expected = fit_cox_baseline(risks, [1.0, 2.0, 3.0], [1, 1, 1]).predict(risks).survival
        for offset in (-10000.0, 10000.0):
            fitted = fit_cox_baseline(risks + offset, [1.0, 2.0, 3.0], [1, 1, 1])
            actual = fitted.predict(risks + offset)
            torch.testing.assert_close(actual.survival, expected)
            extreme = fitted.predict(torch.tensor([-100000.0, 100000.0]))
            self.assertTrue(torch.isfinite(extreme.survival).all())
            self.assertTrue(torch.isfinite(extreme.cumulative_hazard).all())

    def test_training_only_and_events_required(self):
        with self.assertRaisesRegex(ValueError, "train"):
            fit_cox_baseline([0.0, 1.0], [1.0, 2.0], [1, 0], split="test")
        with self.assertRaisesRegex(ValueError, "observed training event"):
            fit_cox_baseline([0.0, 1.0], [1.0, 2.0], [0, 0])

    def test_extrapolation_override_requires_actual_boolean(self):
        baseline = fit_cox_baseline([0.0, 1.0], [1.0, 2.0], [1, 0])
        for flag in ("False", 1, None):
            with self.assertRaisesRegex(ValueError, "boolean"):
                baseline.predict([0.0], [3.0], allow_extrapolation=flag)
            with self.assertRaisesRegex(ValueError, "boolean"):
                create_survival_head("cox", 2)(torch.zeros(1, 2), allow_extrapolation=flag)


class SurvivalHeadTests(unittest.TestCase):
    def test_catalogue_and_linear_mlp_forward_backward(self):
        self.assertEqual({entry["name"] for entry in list_heads()}, {"cox", "discrete_hazard"})
        self.assertEqual(head_info("cox")["family"], "survival")
        for name in ("cox", "discrete_hazard"):
            for network in ("linear", "mlp"):
                with self.subTest(name=name, network=network):
                    head = create_survival_head(name, 4, network=network, hidden_dims=[8, 6])
                    inputs = torch.randn(4, 4, requires_grad=True)
                    if name == "discrete_hazard":
                        self.assertIsNone(head(inputs).survival)
                        head.fit_time_grid([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 1])
                    output = head(inputs)
                    self.assertIsInstance(output, SurvivalOutput)
                    loss = head.loss(output, [1.0, 2.0, 3.0, 4.0], [1, 0, 1, 1])
                    loss.backward()
                    self.assertTrue(torch.isfinite(inputs.grad).all())
                    self.assertTrue(
                        any(parameter.grad is not None for parameter in head.parameters())
                    )
                    json.dumps(head.config, allow_nan=False)

    def test_cox_absolute_survival_requires_fit_and_reload_resizes_baseline(self):
        head = create_survival_head("cox", 3, network="mlp", hidden_dims=[5]).eval()
        features = torch.randn(4, 3)
        self.assertIsNone(head(features).survival)
        with self.assertRaisesRegex(ValueError, "baseline"):
            head(features, times=[1.0])
        head.fit_baseline(features, [1.0, 1.0, 2.0, 4.0], [1, 1, 0, 1])
        expected = head(features, times=[0.0, 1.0, 3.0, 4.0])
        state = deepcopy(head.state_dict())
        config = dict(head.config)
        name = config.pop("name")
        config.pop("family")
        restored = create_survival_head(name, **config).eval()
        self.assertEqual(restored.baseline_times.numel(), 0)
        restored.load_state_dict(state, strict=True)
        self.assertEqual(restored.baseline_times.numel(), 2)
        actual = restored(features, times=[0.0, 1.0, 3.0, 4.0])
        torch.testing.assert_close(actual.survival, expected.survival)
        self.assertEqual(set(state), set(restored.state_dict()))

    def test_discrete_fitted_grid_config_and_strict_reload(self):
        head = create_survival_head(
            "discrete_hazard", 3, network="mlp", hidden_dims=[5], num_bins=3
        )
        head.fit_time_grid([1.0, 4.0, 9.0], [1, 0, 1])
        features = torch.randn(4, 3)
        expected = head(features)
        self.assertEqual(head.boundaries.tolist(), [0.0, 3.0, 6.0, 9.0])
        self.assertTrue((expected.survival[:, 1:] <= expected.survival[:, :-1]).all())
        torch.testing.assert_close(expected.risk, -expected.survival.sum(dim=1))
        config = dict(head.config)
        name = config.pop("name")
        config.pop("family")
        restored = create_survival_head(name, **config)
        restored.load_state_dict(head.state_dict(), strict=True)
        torch.testing.assert_close(restored(features).survival, expected.survival)
        # State-only reload into an unfitted but dimension-matching head also works.
        plain = create_survival_head(
            "discrete_hazard", 3, network="mlp", hidden_dims=[5], num_bins=3
        )
        plain.load_state_dict(head.state_dict(), strict=True)
        torch.testing.assert_close(plain(features).survival, expected.survival)

    def test_shared_artifact_roundtrip_preserves_calibration(self):
        from openhisto.mil.heads import save_head, load_head

        features = torch.randn(4, 3)
        for name in ("cox", "discrete_hazard"):
            head = create_survival_head(name, 3).eval()
            if name == "cox":
                head.fit_baseline(features, [1.0, 2.0, 3.0, 4.0], [1, 0, 1, 1])
            else:
                head.fit_time_grid([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 1])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "synthetic_head.cpath"
                spec = {"dimension": 3, "mode": "synthetic_fixture"}
                save_head(head, path, representation_spec=spec)
                restored = load_head(path, expected_representation=spec)
                torch.testing.assert_close(restored(features).survival, head(features).survival)

    def test_invalid_inputs_and_configuration(self):
        for name in ("cox", "discrete_hazard"):
            with self.assertRaises(ValueError):
                create_survival_head(name, 0)
            model = create_survival_head(name, 3)
            for inputs in (
                torch.zeros(2, 4),
                torch.zeros(2, 3, dtype=torch.long),
                torch.full((2, 3), torch.nan),
            ):
                with self.assertRaises(ValueError):
                    model(inputs)
        with self.assertRaises(ValueError):
            create_survival_head("discrete_hazard", 3, num_bins=3, boundaries=[0.0, 1.0, 2.0])
        for bounds in ([1.0, 2.0], [0.0, 0.0, 2.0], [0.0, float("inf")]):
            with self.assertRaises(ValueError):
                TimeGrid(bounds)


if __name__ == "__main__":
    unittest.main()
