"""Synthetic native representations for every supported MIL profile."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from openhisto.mil.classification import create_neural_head
from openhisto.mil.heads.attachment import attach_head, extract_slide_embeddings
from openhisto.mil.heads.bindings import PROFILES, get_representation_spec, representation_dimension
from openhisto.mil.heads.types import ClassificationOutput, SurvivalOutput, TaskOutput
from openhisto.mil.heads.provenance import state_fingerprint
from openhisto.mil.models.code_bases import create_model, list_models


def _case(profile, root):
    family = next(record["family"] for record in list_models() if record["name"] == profile)
    if family == "baselines":
        return {"input_dim": 8, "num_classes": 2}, torch.randn(5, 8, requires_grad=True), {}
    cases = importlib.import_module("openhisto.mil.models.code_bases.tests.test_" + family)
    options, inputs, kwargs = cases.make_case(profile, root)
    if isinstance(inputs, torch.Tensor):
        inputs.requires_grad_(True)
    return options, inputs, kwargs


def _hooks(module):
    return sum(
        len(child._forward_hooks) + len(child._forward_pre_hooks) for child in module.modules()
    )


class AttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_34_profiles_declared_forward_gradient_freeze_and_state(self):
        self.assertEqual(set(PROFILES), {record["name"] for record in list_models()})
        self.assertEqual(len(PROFILES), 34)
        with tempfile.TemporaryDirectory(prefix="cpath-bindings-") as temporary:
            for profile in PROFILES:
                with self.subTest(profile=profile):
                    options, inputs, kwargs = _case(profile, Path(temporary))
                    backbone = create_model(profile, **options)
                    before = deepcopy(backbone.state_dict())
                    # Binding construction may inspect declared module dimensions
                    # but may not execute a probe or alter any native state.
                    with patch.object(
                        backbone, "forward", side_effect=AssertionError("No probe forward")
                    ):
                        spec = get_representation_spec(backbone)
                        dimension = representation_dimension(backbone)
                        head = create_neural_head("linear", dimension, 3)
                        head_config = deepcopy(head.config)
                        model = attach_head(backbone, head)
                    json.dumps(spec, allow_nan=False)
                    self.assertEqual(dimension, spec["dimension"])
                    for key, value in before.items():
                        torch.testing.assert_close(backbone.state_dict()[key], value)
                    self.assertEqual(head.metadata["representation_spec"], spec)
                    self.assertEqual(head.config, head_config)
                    self.assertTrue(any(key.startswith("backbone.") for key in model.state_dict()))
                    hooks_before = _hooks(backbone)
                    result = model(inputs, **kwargs)
                    self.assertIsInstance(result, TaskOutput)
                    self.assertIsInstance(result.head, ClassificationOutput)
                    self.assertEqual(result.representation.shape, (1, dimension))
                    self.assertEqual(result.head.logits.shape, (1, 3))
                    self.assertTrue(result.representation.requires_grad)
                    result.representation.retain_grad()
                    result.head.logits.square().mean().backward()
                    self.assertTrue(torch.isfinite(result.representation.grad).all())
                    self.assertGreater(result.representation.grad.abs().sum(), 0)
                    self.assertTrue(
                        any(
                            parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in head.parameters()
                        )
                    )
                    self.assertEqual(_hooks(backbone), hooks_before)
                    # The original classifier/teacher/auxiliary modules remain
                    # registered even when the new head bypasses their logits.
                    self.assertEqual(set(before), set(model.backbone.state_dict()))
                    state = deepcopy(model.state_dict())
                    model.eval()
                    expected = model(inputs, **kwargs)
                    clone_backbone = create_model(profile, **options)
                    clone = attach_head(
                        clone_backbone, create_neural_head("linear", dimension, 3)
                    ).eval()
                    clone.load_state_dict(state, strict=True)
                    actual = clone(inputs, **kwargs)
                    torch.testing.assert_close(actual.representation, expected.representation)
                    torch.testing.assert_close(actual.head.logits, expected.head.logits)
                    # Frozen backbones must stay in eval and must not mutate
                    # centers, BatchNorm, teacher or merge EMA state on train().
                    frozen = attach_head(
                        clone_backbone,
                        create_neural_head("linear", dimension, 3),
                        train_backbone=False,
                    ).train()
                    self.assertTrue(frozen.head.training)
                    self.assertFalse(frozen.backbone.training)
                    self.assertTrue(
                        all(
                            not parameter.requires_grad
                            for parameter in frozen.backbone.parameters()
                        )
                    )
                    frozen_before = deepcopy(frozen.backbone.state_dict())
                    frozen_result = frozen(inputs, **kwargs)
                    self.assertFalse(frozen_result.representation.requires_grad)
                    frozen_result.head.logits.square().mean().backward()
                    self.assertTrue(
                        any(parameter.grad is not None for parameter in frozen.head.parameters())
                    )
                    frozen.after_optimizer_step()
                    for key, value in frozen_before.items():
                        torch.testing.assert_close(frozen.backbone.state_dict()[key], value)

    def test_class_axes_are_flattened_in_order_and_never_averaged(self):
        for profile in ("dsmil", "clam_mb", "mhim_v1", "mhim_v2"):
            with self.subTest(profile=profile):
                options = {"input_dim": 8, "num_classes": 3}
                if profile.startswith("mhim"):
                    options.update(baseline="dsmil", mlp_dim=32)
                backbone = create_model(profile, **options).eval()
                dim = representation_dimension(backbone)
                model = attach_head(backbone, create_neural_head("linear", dim, 2)).eval()
                result = model(torch.randn(7, 8))
                original = result.mil.embedding
                self.assertEqual(result.representation_spec["class_order"]["values"], [0, 1, 2])
                torch.testing.assert_close(result.representation, original.reshape(1, -1))
                self.assertEqual(dim, original.numel())

    def test_hooks_removed_after_failure_and_native_output_is_unchanged(self):
        backbone = create_model("wikg", input_dim=8, dim_hidden=16, topk=2)
        model = attach_head(backbone, create_neural_head("linear", 16, 2))
        count = _hooks(backbone)
        with patch.object(
            backbone.native, "forward", side_effect=RuntimeError("synthetic failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                model(torch.randn(5, 8))
        self.assertEqual(_hooks(backbone), count)
        saved = []
        original = backbone.forward

        def recording(*args, **kwargs):
            value = original(*args, **kwargs)
            saved.append(value)
            return value

        with patch.object(backbone, "forward", side_effect=recording):
            result = model(torch.randn(5, 8))
        self.assertIs(result.mil, saved[-1])
        self.assertIsNone(result.mil.embedding)

    def test_extraction_order_empty_and_restored_mixed_modes(self):
        backbone = create_model("mean_mil", input_dim=3)
        first, second = torch.ones(2, 3), torch.full((4, 3), 5.0)
        result, spec = extract_slide_embeddings(backbone, [first, second])
        torch.testing.assert_close(result, torch.tensor([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]))
        self.assertEqual(result.device.type, "cpu")
        self.assertFalse(result.requires_grad)
        self.assertEqual(spec["backbone_provenance"]["state_sha256"], state_fingerprint(backbone))
        self.assertNotIn("state_sha256", get_representation_spec(backbone)["backbone_provenance"])
        self.assertTrue(backbone.training)
        empty, _ = extract_slide_embeddings(backbone, [])
        self.assertEqual(empty.shape, (0, 3))
        mhim = create_model("mhim_v1", input_dim=8, mlp_dim=32, baseline="attn").train()
        self.assertFalse(mhim.teacher.training)
        before = [(module, module.training) for module in mhim.modules()]
        with self.assertRaises(ValueError):
            extract_slide_embeddings(mhim, [torch.empty(0, 8)])
        self.assertTrue(all(module.training == mode for module, mode in before))
        self.assertFalse(mhim.teacher.training)

    def test_extraction_rejects_state_change_and_records_configuration(self):
        backbone = create_model("mean_mil", input_dim=3).train()

        def changing_inputs():
            yield torch.ones(2, 3)
            with torch.no_grad():
                backbone.classifier.weight.add_(1)
            yield torch.ones(2, 3)

        with self.assertRaisesRegex(RuntimeError, "state changed"):
            extract_slide_embeddings(backbone, changing_inputs())
        self.assertTrue(backbone.training)
        attention = create_model("rrt_mil", input_dim=8, mlp_dim=32)
        average = create_model("rrt_mil", input_dim=8, mlp_dim=32, pool="avg")
        self.assertNotEqual(
            get_representation_spec(attention)["backbone_provenance"],
            get_representation_spec(average)["backbone_provenance"],
        )
        json.dumps(get_representation_spec(average), allow_nan=False)

    def test_update_hooks_delegate_only_for_trainable_backbone(self):
        for profile, method in (("mhim_v1", "update_teacher"), ("asmil", "update_anchor")):
            options = {"input_dim": 8}
            if profile == "asmil":
                options.update(hidden_dim=16, n_token=2, n_drop=1)
            else:
                options.update(mlp_dim=32, baseline="attn")
            backbone = create_model(profile, **options)
            model = attach_head(
                backbone, create_neural_head("linear", representation_dimension(backbone), 2)
            )
            with patch.object(backbone, method, wraps=getattr(backbone, method)) as update:
                model.after_optimizer_step()
                self.assertEqual(update.call_count, 1)
                getattr(model, method)()
                self.assertEqual(update.call_count, 2)
            frozen = attach_head(
                backbone,
                create_neural_head("linear", representation_dimension(backbone), 2),
                train_backbone=False,
            ).train()
            with patch.object(backbone, method, side_effect=AssertionError("Frozen update")):
                frozen.after_optimizer_step()
                getattr(frozen, method)()

    def test_explicit_adaptation_flags_and_survival_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            for profile in ("slidegraph_plus", "nnmil"):
                options, _, _ = _case(profile, Path(temporary))
                self.assertTrue(
                    get_representation_spec(create_model(profile, **options))["adaptations"]
                )

        class RiskHead(nn.Module):
            input_dim = 8

            def __init__(self):
                super().__init__()
                self.score = nn.Linear(8, 1)

            def forward(self, value):
                return SurvivalOutput(risk=self.score(value).squeeze(-1))

        model = attach_head(create_model("mean_mil", input_dim=8), RiskHead())
        result = model(torch.randn(3, 8))
        self.assertIsInstance(result.head, SurvivalOutput)
        result.head.risk.sum().backward()

    def test_dimension_and_policy_errors_are_early(self):
        backbone = create_model("mean_mil", input_dim=8)
        with self.assertRaises(ValueError):
            attach_head(backbone, create_neural_head("linear", 7, 2))
        with self.assertRaises(ValueError):
            get_representation_spec(backbone, "mean_any_tensor")
        with self.assertRaises(TypeError):
            attach_head(backbone, object())

    def test_freezing_clears_stale_gradients_before_optimizer_step(self):
        backbone = create_model("mean_mil", input_dim=8)
        backbone(torch.randn(4, 8)).logits.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in backbone.parameters()))
        model = attach_head(
            backbone, create_neural_head("linear", 8, 2), train_backbone=False
        ).train()
        self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))
        before = deepcopy(backbone.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.randn(4, 8)).head.logits.sum().backward()
        optimizer.step()
        for key, value in before.items():
            torch.testing.assert_close(backbone.state_dict()[key], value)

    def test_optional_native_variants_have_declared_bindings(self):
        cases = [
            (
                "rrt_mil",
                {"input_dim": 8, "mlp_dim": 32, "n_heads": 4, "crmsa_heads": 4, "pool": "avg"},
                torch.randn(5, 8),
            ),
            ("hipt", {"input_dim": 192, "variant": "gp"}, torch.randn(3, 192)),
            ("hipt", {"input_dim": 8, "variant": "none"}, torch.randn(3, 5, 8)),
        ]
        for profile, options, inputs in cases:
            with self.subTest(profile=profile, options=options):
                backbone = create_model(profile, **options)
                dimension = representation_dimension(backbone)
                model = attach_head(backbone, create_neural_head("linear", dimension, 2))
                result = model(inputs)
                result.head.logits.sum().backward()
                self.assertEqual(result.representation.shape, (1, dimension))
        backbone = create_model("clam_sb", input_dim=8, k_sample=2)
        inputs = torch.randn(7, 8)
        rows, spec = extract_slide_embeddings(
            backbone, [(inputs, {"label": torch.tensor([1]), "instance_eval": True})]
        )
        self.assertEqual(rows.shape, (1, spec["dimension"]))


if __name__ == "__main__":
    unittest.main()
