"""Synthetic CPU tests exercising native transformer/state-space computation."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from openhisto.mil.models.code_bases.adapters.transformers import build
from openhisto.mil.models.code_bases.compat.mamba_reference import (
    selective_scan_ref,
    mamba_inner_fn_no_out_proj,
)


PROFILES = (
    "transmil",
    "ilra",
    "rrt_mil",
    "dgr_mil",
    "caprmil",
    "lin_mil",
    "mamba_mil",
    "nnmil",
    "mhim_v1",
    "mhim_v2",
)


def make_case(name, root: Path):
    """Small native configurations for the package-wide public-API matrix."""
    options = {"input_dim": 16, "num_classes": 3}
    settings = {
        "transmil": {"num_landmarks": 8},
        "ilra": {"num_layers": 1, "hidden_feat": 32, "num_heads": 4, "topk": 2},
        "rrt_mil": {
            "mlp_dim": 32,
            "n_layers": 2,
            "n_heads": 4,
            "crmsa_heads": 4,
            "region_num": 2,
            "epeg_k": 3,
            "crmsa_k": 1,
            "dropout": 0,
            "trans_dropout": 0,
        },
        "dgr_mil": {"L": 32, "n_lesion": 3, "dropout_node": 0},
        "caprmil": {
            "hidden_dim": 32,
            "num_heads": 4,
            "head_dim": 8,
            "cluster_num": 4,
            "T_depth": 1,
            "dropout": 0,
        },
        "lin_mil": {
            "latent_dim": 32,
            "transformer_depth": 1,
            "heads": 4,
            "dropout": 0,
            "emb_dropout": 0,
        },
        "mamba_mil": {"layer": 1, "rate": 4, "dropout": 0},
        "nnmil": {"hidden_dim": 8, "dropout": False},
        "mhim_v1": {
            "mlp_dim": 32,
            "baseline": "attn",
            "head": 4,
            "dropout": 0,
            "mask_ratio_h": 0.2,
            "mask_ratio_hr": 1,
        },
        "mhim_v2": {
            "mlp_dim": 32,
            "baseline": "attn",
            "head": 4,
            "dropout": 0,
            "mask_ratio_h": 0.2,
            "mask_ratio_hr": 1,
            "merge_ratio": 0.5,
            "merge_k": 2,
        },
    }
    options.update(settings[name])
    inputs = torch.randn(9, options["input_dim"])
    return options, inputs, {}


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_native_cpu_forward_backward_and_state(self):
        for name in PROFILES:
            with self.subTest(name=name):
                torch.manual_seed(7)
                options, inputs, kwargs = make_case(name, Path("."))
                inputs.requires_grad_(True)
                model = build(name, **options).train()
                output = model(inputs, **kwargs)
                self.assertEqual(output.logits.shape, (1, 3))
                self.assertTrue(torch.isfinite(output.logits).all())
                self.assertIn("._native.", type(model.native).__module__)
                loss = output.logits.square().mean()
                if "distillation_loss" in output.aux:
                    loss = loss + output.aux["distillation_loss"]
                loss.backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                gradients = [p.grad for p in model.native.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                state = deepcopy(model.state_dict())
                model.eval()
                expected = model(inputs.detach()).logits
                restored = build(name, **options).eval()
                restored.load_state_dict(state)
                actual = restored(inputs.detach()).logits
                torch.testing.assert_close(actual, expected)

    def test_mamba_variants_short_and_nondivisible_sequences(self):
        for variant in ("Mamba", "BiMamba", "SRMamba"):
            for count in (1, 3, 5):
                with self.subTest(variant=variant, count=count):
                    model = build(
                        "mamba_mil", input_dim=8, num_classes=2, layer=1, type=variant, rate=4
                    )
                    inputs = torch.randn(count, 8, requires_grad=True)
                    output = model(inputs)
                    self.assertEqual(output.logits.shape, (1, 2))
                    output.logits.sum().backward()
                    self.assertTrue(torch.isfinite(inputs.grad).all())
                    self.assertEqual(model.metadata["kernel_backend"], "torch_reference")

    def test_selective_scan_matches_independent_recurrence(self):
        torch.manual_seed(13)
        batch, channels, length, state_dim = 2, 3, 5, 4
        u = torch.randn(batch, channels, length, requires_grad=True)
        delta = torch.rand_like(u)
        A = -torch.rand(channels, state_dim)
        B = torch.rand(batch, state_dim, length)
        C = torch.rand(batch, state_dim, length)
        D = torch.rand(channels)
        z = torch.randn_like(u)
        bias = torch.randn(channels)
        state = torch.zeros(batch, channels, state_dim)
        expected = []
        for i in range(length):
            dt = torch.nn.functional.softplus(delta[:, :, i] + bias)
            state = (
                state * torch.exp(dt[:, :, None] * A)
                + dt[:, :, None] * B[:, None, :, i] * u[:, :, i, None]
            )
            expected.append(
                ((state * C[:, None, :, i]).sum(-1) + D * u[:, :, i])
                * torch.nn.functional.silu(z[:, :, i])
            )
        actual, last = selective_scan_ref(u, delta, A, B, C, D, z, bias, True, True)
        torch.testing.assert_close(actual, torch.stack(expected, dim=-1))
        torch.testing.assert_close(last, state)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(u.grad).all())

    def test_nnmil_single_window_dispersion_is_zero(self):
        model = build("nnmil", input_dim=8, hidden_dim=16, num_classes=3).eval()
        output = model(torch.randn(1, 8))
        self.assertEqual(model.native.eval_chunks, 1)
        for key in ("var_logits", "WSI_attn_std"):
            self.assertTrue(torch.isfinite(output.aux[key]).all())
            torch.testing.assert_close(output.aux[key], torch.zeros_like(output.aux[key]))

    def test_mamba_inner_projection_and_causal_convolution(self):
        torch.manual_seed(11)
        channels, length, state_dim, rank, width = 2, 5, 3, 1, 3
        xz = torch.randn(1, channels * 2, length, requires_grad=True)
        conv = torch.randn(channels, 1, width)
        conv_bias = torch.randn(channels)
        projection = torch.randn(rank + 2 * state_dim, channels)
        delta_projection = torch.randn(channels, rank)
        A = -torch.rand(channels, state_dim)
        D, bias = torch.randn(channels), torch.randn(channels)
        actual = mamba_inner_fn_no_out_proj(
            xz, conv, conv_bias, projection, delta_projection, A, D=D, delta_bias=bias
        )
        x, gate = xz.chunk(2, dim=1)
        causal = torch.zeros_like(x)
        for t in range(length):
            for channel in range(channels):
                value = conv_bias[channel]
                for k in range(width):
                    if t - width + 1 + k >= 0:
                        value = value + conv[channel, 0, k] * x[0, channel, t - width + 1 + k]
                causal[0, channel, t] = torch.nn.functional.silu(value)
        projected = causal.transpose(1, 2) @ projection.T
        delta = (projected[..., :rank] @ delta_projection.T).transpose(1, 2)
        B = projected[..., rank : rank + state_dim].transpose(1, 2)
        C = projected[..., -state_dim:].transpose(1, 2)
        expected = selective_scan_ref(causal, delta, A, B, C, D, gate, bias, True)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(xz.grad).all())

    def test_mhim_teacher_ema_masks_and_checkpoint(self):
        for name in ("mhim_v1", "mhim_v2"):
            options, inputs, _ = make_case(name, Path("."))
            model = build(name, **options).train()
            self.assertFalse(model.teacher.training)
            output = model(inputs)
            self.assertIn("distillation_loss", output.aux)
            self.assertLess(output.aux["retained_patch_count"], len(inputs))
            self.assertTrue(all(not p.requires_grad for p in model.teacher.parameters()))
            teacher_before = deepcopy(model.teacher.state_dict())
            with torch.no_grad():
                next(model.native.parameters()).add_(1)
            model.update_teacher(0.0)
            self.assertEqual(model.teacher_updates.item(), 1)
            self.assertTrue(
                any(
                    not torch.equal(teacher_before[k], v)
                    for k, v in model.teacher.state_dict().items()
                )
            )
            for key, value in model.native.state_dict().items():
                torch.testing.assert_close(model.teacher.state_dict()[key], value)
            model.reset_teacher()
            self.assertEqual(model.teacher_updates.item(), 0)
            # Small bags must never leave native attention with an empty bag.
            one = model(torch.randn(1, 16))
            self.assertTrue(torch.isfinite(one.logits).all())

    def test_mhim_v2_zero_mask_and_disabled_merge(self):
        for merge in (False, True):
            model = build(
                "mhim_v2",
                input_dim=8,
                num_classes=2,
                baseline="attn",
                mlp_dim=32,
                mask_ratio_h=0,
                merge_enable=merge,
                merge_mm=1,
            )
            result = model(torch.randn(3, 8))
            result.logits.sum().backward()
            self.assertTrue(torch.isfinite(result.logits).all())

    def test_mhim_default_self_attention_train_and_eval(self):
        for name in ("mhim_v1", "mhim_v2"):
            with self.subTest(name=name):
                model = build(name, input_dim=16, num_classes=3)
                bag = torch.randn(5, 16, requires_grad=True)
                result = model(bag)
                (result.logits.square().mean() + result.aux["distillation_loss"]).backward()
                self.assertTrue(torch.isfinite(bag.grad).all())
                prediction = model.eval()(bag.detach())
                self.assertEqual(prediction.logits.shape, (1, 3))
                self.assertTrue(torch.isfinite(prediction.logits).all())

    def test_mhim_teacher_shared_query_ema_applied_once(self):
        model = build("mhim_v2", input_dim=8, baseline="attn", mlp_dim=32)
        old = model.teacher.merge.global_q.detach().clone()
        with torch.no_grad():
            model.native.merge.global_q.add_(2)
        expected = (old + model.native.merge.global_q.detach()) / 2
        model.update_teacher(0.5)
        torch.testing.assert_close(model.teacher.merge.global_q, expected)

    def test_rrt_average_pool_preserves_feature_axis(self):
        for count in (1, 9):
            options, _, _ = make_case("rrt_mil", Path("."))
            options["pool"] = "avg"
            model = build("rrt_mil", **options)
            encoded = []
            handle = model.native.online_encoder.register_forward_hook(
                lambda _, args, result: encoded.append(result)
            )
            try:
                bag = torch.randn(count, 16, requires_grad=True)
                result = model(bag)
            finally:
                handle.remove()
            self.assertEqual(result.logits.shape, (1, 3))
            torch.testing.assert_close(result.embedding, encoded[-1].mean(dim=1))
            result.logits.square().mean().backward()
            self.assertTrue(torch.isfinite(bag.grad).all())
            self.assertTrue(torch.isfinite(model.eval()(bag.detach()).logits).all())

    def test_rrt_rejects_unsupported_positional_geometry_before_native_import(self):
        with patch(
            "openhisto.mil.models.code_bases.adapters.transformers._native",
            side_effect=AssertionError(
                "Unsupported geometry must fail before importing native code"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "four-dimensional region geometry"):
                build("rrt_mil", pos="sincos")
            with self.assertRaisesRegex(ValueError, "RRT pos must be"):
                build("rrt_mil", pos="unknown")

    def test_mhim_dsmil_preserves_both_heads_and_class_features(self):
        for name in ("mhim_v1", "mhim_v2"):
            for count in (1, 9):
                for masking in (0, 0.2):
                    with self.subTest(name=name, count=count, masking=masking):
                        options, _, _ = make_case(name, Path("."))
                        options.update(baseline="dsmil", mask_ratio_h=masking)
                        model = build(name, **options).train()
                        bag = torch.randn(count, 16, requires_grad=True)
                        result = model(bag)
                        self.assertEqual(result.logits.shape, (1, 3))
                        self.assertEqual(result.aux["max_instance_logits"].shape, (1, 3))
                        self.assertEqual(result.aux["class_embeddings"].shape, (1, 3, 32))
                        target = torch.tensor([1])
                        loss = 0.5 * torch.nn.functional.cross_entropy(result.logits, target)
                        loss = loss + 0.5 * torch.nn.functional.cross_entropy(
                            result.aux["max_instance_logits"], target
                        )
                        loss = loss + result.aux["distillation_loss"]
                        loss.backward()
                        self.assertTrue(torch.isfinite(bag.grad).all())
                        for layer in (
                            model.native.online_encoder.i_classifier[0],
                            model.native.online_encoder.b_classifier.fcc,
                        ):
                            self.assertTrue(torch.isfinite(layer.weight.grad).all())
                            self.assertGreater(layer.weight.grad.abs().sum(), 0)
                        state = deepcopy(model.state_dict())
                        expected = model.eval()(bag.detach())
                        restored = build(name, **options).eval()
                        restored.load_state_dict(state, strict=True)
                        actual = restored(bag.detach())
                        torch.testing.assert_close(expected.logits, actual.logits)
                        torch.testing.assert_close(
                            expected.aux["max_instance_logits"], actual.aux["max_instance_logits"]
                        )

    def test_mhim_combined_masking_keeps_valid_index_axis(self):
        for name in ("mhim_v1", "mhim_v2"):
            for count in (1, 2, 9):
                with self.subTest(name=name, count=count):
                    model = build(
                        name,
                        input_dim=8,
                        mlp_dim=32,
                        baseline="attn",
                        mask_ratio=0.1,
                        mask_ratio_l=0.2,
                        mask_ratio_h=0.2,
                    )
                    bag = torch.randn(count, 8, requires_grad=True)
                    result = model(bag)
                    result.logits.sum().backward()
                    self.assertTrue(torch.isfinite(bag.grad).all())
                    self.assertGreater(result.aux["retained_patch_count"], 0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            build("mamba_mil", layer=0)
        with self.assertRaises(ValueError):
            build("mhim_v2", mask_ratio_hr=0)
        model = build("ilra", input_dim=8, hidden_feat=16, num_heads=4)
        for bag in (torch.empty(0, 8), torch.randn(4, 7), torch.full((3, 8), float("nan"))):
            with self.assertRaises(ValueError):
                model(bag)


if __name__ == "__main__":
    unittest.main()
