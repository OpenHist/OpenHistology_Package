"""Real small native model forwards/backwards; no weights or images downloaded."""

from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from openhisto.segmentation.config import LoadedSample, ModelConfig
from openhisto.segmentation.semantic.base import preprocess_batch
from openhisto.segmentation.semantic.registry import canonical_model, create_adapter, list_models


def tiny_config(name):
    architecture = {
        "segformer": dict(
            hidden_sizes=[8, 16, 32, 64],
            depths=[1, 1, 1, 1],
            num_attention_heads=[1, 2, 4, 8],
            decoder_hidden_size=16,
        ),
        "upernet": dict(
            backbone_config=dict(
                model_type="convnext",
                hidden_sizes=[8, 16, 32, 64],
                depths=[1, 1, 1, 1],
                out_features=["stage1", "stage2", "stage3", "stage4"],
            ),
            hidden_size=16,
            auxiliary_channels=16,
            auxiliary_in_channels=32,
        ),
        "mask2former": dict(
            backbone_config=dict(
                model_type="swin",
                embed_dim=16,
                depths=[1, 1, 1, 1],
                num_heads=[1, 2, 4, 8],
                window_size=2,
                out_features=["stage1", "stage2", "stage3", "stage4"],
            ),
            feature_size=32,
            mask_feature_size=32,
            hidden_dim=32,
            encoder_feedforward_dim=64,
            encoder_layers=1,
            decoder_layers=2,
            num_attention_heads=4,
            dim_feedforward=64,
            num_queries=4,
            train_num_points=32,
        ),
    }
    options = (
        {"backbone": "mobilenet_v3_large", "freeze_batch_norm": True}
        if name == "deeplabv3"
        else {"architecture_config": architecture[name]}
    )
    return ModelConfig(
        model=name, num_classes=3, pretrained=False, image_size=(64, 64), options=options
    )


def samples():
    rng = np.random.default_rng(4)
    result = []
    for i, size in enumerate(((64, 64), (48, 57))):
        mask = rng.integers(0, 3, size, dtype=np.int64)
        mask[:3] = -100
        result.append(
            LoadedSample(
                image=rng.integers(0, 256, (*size, 3), dtype=np.uint8), mask=mask, sample_id=str(i)
            )
        )
    return result


class PreprocessingTests(unittest.TestCase):
    def test_background_ignore_padding_and_normalize_once(self):
        sample = LoadedSample(
            np.full((5, 9, 3), 255, dtype=np.uint8), np.zeros((5, 9), dtype=np.int64)
        )
        images, labels, sizes, valid = preprocess_batch([sample])
        self.assertEqual(images.shape, (1, 3, 32, 32))
        np.testing.assert_allclose(
            images[0, :, 0, 0],
            (np.ones(3) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225],
            rtol=1e-6,
        )
        self.assertTrue((labels[0, :5, :9] == 0).all())
        self.assertTrue((labels[0, 5:] == -100).all())
        self.assertEqual(int(valid.sum()), 45)
        self.assertEqual(sizes, [(5, 9)])

    def test_catalogue_and_aliases(self):
        self.assertEqual(len(list_models()), 6)
        self.assertEqual(canonical_model("M2F"), "mask2former")
        self.assertEqual(canonical_model("SegFormer"), "segformer")
        self.assertEqual(canonical_model("DLV3"), "deeplabv3")
        with self.assertRaises(ValueError):
            canonical_model("unknown")


class NativeBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def check_backend(self, name):
        torch.manual_seed(0)
        config = tiny_config(name)
        batch = samples()
        with patch(
            "torch.hub.download_url_to_file", side_effect=AssertionError("No model downloads")
        ):
            adapter = create_adapter(config)
            adapter.model.train()
            loss = adapter.loss(batch)
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(
                any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in adapter.model.parameters()
                )
            )
            modes_before_infer = [module.training for module in adapter.model.modules()]
            before = adapter.infer(batch)
            self.assertEqual(
                modes_before_infer, [module.training for module in adapter.model.modules()]
            )
            for prediction, sample in zip(before, batch):
                value = prediction["probabilities"]
                self.assertEqual(value.shape, (3, *sample.image.shape[:2]))
                self.assertTrue(np.isfinite(value).all())
                np.testing.assert_allclose(value.sum(0), 1, atol=1e-6)
                self.assertEqual(prediction["logits"] is None, name == "mask2former")
                # Float32 softmax can merge logits separated below one ULP.
                chosen = np.take_along_axis(prediction["scores"], value.argmax(0)[None], axis=0)[0]
                np.testing.assert_allclose(chosen, prediction["scores"].max(0), atol=1e-6)
            with tempfile.TemporaryDirectory() as directory:
                adapter.save(Path(directory))
                restored = create_adapter(config, load_dir=directory)
                after = restored.infer(batch)
                for a, b in zip(before, after):
                    np.testing.assert_allclose(
                        a["probabilities"], b["probabilities"], rtol=1e-6, atol=1e-7
                    )
            for sample in batch:
                sample.mask[:] = -100
            with self.assertRaisesRegex(ValueError, "no labeled"):
                adapter.loss(batch)

    @unittest.skipUnless(find_spec("torchvision"), "torchvision unavailable")
    def test_deeplab_native_backward_and_local_reload(self):
        self.check_backend("deeplabv3")

    @unittest.skipUnless(find_spec("transformers"), "transformers unavailable")
    def test_segformer_native_backward_and_local_reload(self):
        self.check_backend("segformer")

    @unittest.skipUnless(find_spec("transformers"), "transformers unavailable")
    def test_upernet_native_backward_and_local_reload(self):
        self.check_backend("upernet")

    @unittest.skipUnless(
        find_spec("transformers") and find_spec("scipy"), "transformers/scipy unavailable"
    )
    def test_mask2former_native_backward_and_local_reload(self):
        self.check_backend("mask2former")

    @unittest.skipUnless(
        find_spec("transformers") and find_spec("scipy"), "transformers/scipy unavailable"
    )
    def test_mask2former_native_fully_labeled_objective(self):
        batch = samples()
        for sample in batch:
            sample.mask[sample.mask == -100] = 0
            sample.image = np.pad(
                sample.image,
                ((0, 64 - sample.image.shape[0]), (0, 64 - sample.image.shape[1]), (0, 0)),
            )
            sample.mask = np.pad(
                sample.mask, ((0, 64 - sample.mask.shape[0]), (0, 64 - sample.mask.shape[1]))
            )
        adapter = create_adapter(tiny_config("mask2former"))
        loss = adapter.loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    @unittest.skipUnless(
        find_spec("transformers") and find_spec("scipy"), "transformers/scipy unavailable"
    )
    def test_mask2former_chunked_scores_match_native_query_formula(self):
        config = tiny_config("mask2former")
        config.options["query_chunk_size"] = 2
        adapter = create_adapter(config)
        batch = samples()
        native = SimpleNamespace(
            class_queries_logits=torch.randn(2, 7, 4), masks_queries_logits=torch.randn(2, 7, 6, 10)
        )
        with patch.object(adapter.model, "forward", return_value=native):
            results = adapter.infer(batch)
        masks = torch.nn.functional.interpolate(
            native.masks_queries_logits, size=(64, 64), mode="bilinear", align_corners=False
        ).sigmoid()
        scores = torch.einsum(
            "bqc,bqhw->bchw", native.class_queries_logits.softmax(-1)[..., :-1], masks
        )
        for result, sample, expected in zip(results, batch, scores):
            h, w = sample.image.shape[:2]
            np.testing.assert_allclose(result["scores"], expected[:, :h, :w], atol=1e-6)

    @unittest.skipUnless(find_spec("scipy"), "scipy unavailable")
    def test_mask2former_ignored_pixels_cannot_change_mask_loss(self):
        from openhisto.segmentation.semantic.standard import _masked_query_loss

        native = SimpleNamespace(
            num_labels=2,
            ignore_value=-100,
            no_object_weight=0.1,
            train_num_points=128,
            class_weight=2.0,
            mask_weight=5.0,
            dice_weight=5.0,
        )
        labels = torch.zeros((1, 8, 8), dtype=torch.long)
        labels[:, :, 4:] = 1
        labels[:, :2] = -100
        masks = torch.randn((1, 3, 8, 8), requires_grad=True)
        classes = torch.randn((1, 3, 3), requires_grad=True)
        loss = _masked_query_loss(classes, masks, labels, native)
        changed = masks.detach().clone()
        changed[:, :, :2] = 1000
        other = _masked_query_loss(classes, changed, labels, native)
        torch.testing.assert_close(loss, other)
        loss.backward()
        self.assertTrue((masks.grad[:, :, :2] == 0).all())

    @unittest.skipUnless(find_spec("scipy"), "scipy unavailable")
    def test_mask2former_class_ce_has_native_global_weighting(self):
        from openhisto.segmentation.semantic.standard import _masked_query_loss

        native = SimpleNamespace(
            num_labels=2,
            ignore_value=-100,
            no_object_weight=0.1,
            train_num_points=128,
            class_weight=2.0,
            mask_weight=0.0,
            dice_weight=0.0,
        )
        # First image has background only; second has both semantic classes.
        labels = torch.tensor([[[-100, 0], [0, 0]], [[-100, 0], [1, 1]]])
        classes = torch.tensor(
            [
                [[2.0, 0.0, -1.0], [0.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
                [[0.0, 2.0, -1.0], [2.0, 0.0, -1.0], [0.0, 0.0, 3.0]],
            ],
            requires_grad=True,
        )
        masks = torch.zeros(2, 3, 2, 2)
        targets = torch.tensor([[0, 2, 2], [1, 0, 2]])
        weights = torch.tensor([1.0, 1.0, 0.1])
        actual = _masked_query_loss(classes, masks, labels, native)
        expected = 2 * torch.nn.functional.cross_entropy(
            classes.transpose(1, 2), targets, weight=weights
        )
        wrong_per_image = torch.stack(
            [
                2 * torch.nn.functional.cross_entropy(c, t, weight=weights)
                for c, t in zip(classes, targets)
            ]
        ).mean()
        self.assertGreater(float((expected - wrong_per_image).detach().abs()), 1e-3)
        torch.testing.assert_close(actual, expected)
        actual_gradient = torch.autograd.grad(actual, classes, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, classes)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def check_singleton(self, name):
        config = tiny_config(name)
        config.options.pop("freeze_batch_norm", None)
        adapter = create_adapter(config)
        adapter.model.train()
        norms = [
            m
            for m in adapter.model.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        ]
        # A caller may already have frozen selected BatchNorm layers.
        norms[0].eval()
        modes = [m.training for m in adapter.model.modules()]
        seen = []
        hooks = [
            m.register_forward_pre_hook(lambda module, _: seen.append(module.training))
            for m in norms
        ]
        try:
            loss = adapter.loss(samples()[:1])
            loss.backward()
        finally:
            for hook in hooks:
                hook.remove()
        self.assertTrue(seen)
        self.assertFalse(any(seen))
        self.assertEqual(modes, [m.training for m in adapter.model.modules()])
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.model.parameters())
        )
        adapter.model.eval()
        loss = adapter.loss(samples()[:1])
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(any(m.training for m in adapter.model.modules()))
        adapter.model.train()
        adapter.config.options["freeze_batch_norm"] = False
        with self.assertRaisesRegex(ValueError, "Expected more than 1 value"):
            adapter.loss(samples()[:1])
        # Failed native forwards must also leave caller mode choices intact.
        self.assertTrue(all(m.training for m in adapter.model.modules()))

    @unittest.skipUnless(find_spec("torchvision"), "torchvision unavailable")
    def test_deeplab_singleton_auto_bn_scoped_modes_and_gradients(self):
        self.check_singleton("deeplabv3")

    @unittest.skipUnless(find_spec("transformers"), "transformers unavailable")
    def test_upernet_singleton_auto_bn_scoped_modes_and_gradients(self):
        self.check_singleton("upernet")

    @unittest.skipUnless(find_spec("transformers"), "transformers unavailable")
    def test_upernet_default_random_convnext_never_downloads(self):
        config = tiny_config("upernet")
        config.pretrained = True
        import transformers

        with patch.object(
            transformers.AutoImageProcessor,
            "from_pretrained",
            side_effect=AssertionError("No processor download"),
        ):
            with patch.object(
                transformers.UperNetForSemanticSegmentation,
                "from_pretrained",
                side_effect=AssertionError("No pretrained download"),
            ):
                adapter = create_adapter(config)
        self.assertEqual(adapter.model.config.backbone_config.model_type, "convnext")


if __name__ == "__main__":
    unittest.main()
