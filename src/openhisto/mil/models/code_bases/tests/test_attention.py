"""Real native CPU forwards/backwards with synthetic feature bags only."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from openhisto.mil.models.code_bases.adapters.attention import build


PROFILES = (
    "abmil",
    "gated_abmil",
    "dsmil",
    "dtfd_mil",
    "acmil",
    "asmil",
    "clam_sb",
    "clam_mb",
    "varpool",
    "panther",
)


def make_case(name, root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    options = {"input_dim": 16, "num_classes": 2}
    forward = {}
    if name == "dtfd_mil":
        options.update(hidden_dim=16, num_groups=3)
    elif name == "acmil":
        options.update(hidden_dim=16, n_token=3, n_masked_patch=2)
    elif name == "asmil":
        options.update(hidden_dim=16, n_token=3, n_drop=1)
    elif name in {"clam_sb", "clam_mb"}:
        options.update(k_sample=2)
        forward = {"label": torch.tensor([1]), "instance_eval": True}
    elif name == "varpool":
        options.update(n_var_pools=4)
    elif name == "panther":
        path = root / "synthetic_prototypes.npy"
        np.save(path, np.random.default_rng(3).normal(size=(3, 16)).astype(np.float32))
        options.update(prototype_path=str(path), em_iter=2)
    inputs = torch.randn(12, 16, requires_grad=True)
    return options, inputs, forward


class AttentionTests(unittest.TestCase):
    def test_all_profiles_forward_backward_and_state_roundtrip(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            for name in PROFILES:
                with self.subTest(profile=name):
                    options, inputs, kwargs = make_case(name, Path(temporary))
                    model = build(name, **options)
                    output = model(inputs, **kwargs)
                    self.assertEqual(output.logits.shape, (1, 2))
                    self.assertTrue(torch.isfinite(output.logits).all())
                    self.assertIsNotNone(output.native_output)
                    self.assertIsNotNone(output.embedding)
                    loss = output.logits.square().mean()
                    for key in ("instance_loss", "anchor_loss"):
                        if key in output.aux:
                            loss = loss + output.aux[key]
                    loss.backward()
                    self.assertIsNotNone(inputs.grad)
                    self.assertTrue(torch.isfinite(inputs.grad).all())
                    self.assertTrue(
                        any(
                            parameter.grad is not None
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        )
                    )
                    clone = build(name, **options)
                    clone.load_state_dict(model.state_dict())
                    model.eval()
                    clone.eval()
                    with torch.no_grad():
                        first = model(inputs, **kwargs).logits
                        second = clone(inputs, **kwargs).logits
                    torch.testing.assert_close(first, second)

    def test_native_binary_attention_semantics_and_adaptation_metadata(self):
        bag = torch.randn(5, 500)
        for name in ("abmil", "gated_abmil"):
            model = build(name, input_dim=500, num_classes=2)
            output = model(bag)
            torch.testing.assert_close(output.probabilities[:, 1:], output.native_output[0])
            self.assertIn("identity_projection", model.metadata["adaptations"])

    def test_distillation_variants_keep_native_first_tier_state(self):
        for distill in ("AFS", "MaxS", "MaxMinS"):
            model = build("dtfd_mil", input_dim=8, hidden_dim=16, distill=distill, num_groups=2)
            output = model(torch.randn(8, 8))
            self.assertEqual(output.aux["pseudo_bag_logits"].shape, (2, 2))
            self.assertEqual(len(output.aux["pseudo_bag_indices"]), 2)

    def test_asmil_explicit_anchor_update_and_prototype_requirement(self):
        model = build("asmil", input_dim=8, hidden_dim=16, n_token=2, n_drop=1)
        model.update_anchor(decay=0)
        student = model.native.state_dict()
        for name, value in model.anchor.state_dict().items():
            if name in student:
                torch.testing.assert_close(value, student[name])
        with self.assertRaises(ValueError):
            build("panther", input_dim=8)

    def test_dsmil_and_asmil_preserve_native_probability_and_attention_semantics(self):
        dsmil = build("dsmil", input_dim=8)
        output = dsmil(torch.randn(7, 8))
        instances, logits, _, _ = output.native_output
        expected = (logits.sigmoid() + instances.max(0, keepdim=True).values.sigmoid()) / 2
        torch.testing.assert_close(output.probabilities, expected)
        self.assertEqual(
            dsmil.metadata["probability_semantics"], "independent_per_class_sigmoid_ensemble"
        )
        asmil = build("asmil", input_dim=8, hidden_dim=16, n_token=2, n_drop=1)
        bag = torch.randn(7, 8)
        output = asmil(bag)
        torch.testing.assert_close(output.attention, output.aux["raw_attention"].softmax(-1))
        with torch.no_grad():
            expected = asmil.sigmax(asmil.anchor(bag[None]) / asmil.temperature, dim=-1)
        torch.testing.assert_close(output.aux["anchor_attention"], expected)


if __name__ == "__main__":
    unittest.main()
