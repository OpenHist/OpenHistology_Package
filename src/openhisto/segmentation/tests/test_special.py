"""Real small-network checks without downloading checkpoints."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from openhisto.segmentation.config import CapabilityError, LoadedSample, ModelConfig
from openhisto.segmentation.semantic.special import SegDiffAdapter, SAM2Adapter, load_embedding


def diffusion_config(**extra):
    options = dict(
        feature_dim=5,
        base_channels=8,
        channel_mults=[1, 2],
        time_emb_dim=16,
        embed_dim=8,
        attn_dim=8,
        timesteps=2,
        inference_seed=17,
    )
    options.update(extra)
    return ModelConfig(model="segdiff", num_classes=3, pretrained=False, options=options)


def sample(classes=3):
    image = np.zeros((9, 13, 3), np.uint8)
    mask = np.arange(9 * 13).reshape(9, 13).astype(np.int64) % classes
    mask[0] = -100
    return LoadedSample(
        image=image, mask=mask, sample_id="example", embedding=np.arange(5, dtype=np.float32)
    )


class SegDiffTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_single_token_conditioning_matches_source_attention(self):
        from openhisto.segmentation.semantic.segdiff import GlobalCrossAttention

        module = GlobalCrossAttention(3, 5, 8)
        x, embedding = torch.randn(2, 3, 5, 7), torch.randn(2, 5)
        flat = x.permute(0, 2, 3, 1).reshape(2, 35, 3)
        query, key = module.q_proj(flat), module.k_proj(embedding[:, None])
        weights = (query @ key.transpose(-1, -2) / np.sqrt(8)).softmax(-1)
        expected = flat + module.out_proj(weights @ module.v_proj(embedding[:, None]))
        expected = expected.reshape(2, 5, 7, 3).permute(0, 3, 1, 2)
        torch.testing.assert_close(module(x, embedding), expected)

    def test_backward_projector_and_offline_roundtrip(self):
        config, item = diffusion_config(), sample()
        adapter = SegDiffAdapter(config)
        optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=0.001)
        before = adapter.model.feature_projector.weight.detach().clone()
        loss = adapter.loss([item])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(adapter.model.feature_projector.weight.grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, adapter.model.feature_projector.weight))
        adapter.model.eval()
        prediction = adapter.infer([item, item])
        self.assertEqual(len(prediction), 2)
        self.assertEqual(prediction[0]["probabilities"].shape, (3, 9, 13))
        np.testing.assert_allclose(prediction[0]["probabilities"].sum(0), 1, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save(directory)
            loaded = SegDiffAdapter(config, load_dir=directory)
            loaded.model.eval()
            result = loaded.infer([item])[0]
            np.testing.assert_allclose(result["logits"], prediction[0]["logits"], atol=1e-6)
        adapter.loss([item])
        self.assertFalse(adapter.model.training, "Validation loss must preserve eval mode")

    def test_padding_ignore_and_invalid_features(self):
        adapter, item = SegDiffAdapter(diffusion_config()), sample()
        item.mask[:] = -100
        with self.assertRaisesRegex(ValueError, "no valid"):
            adapter.loss([item])
        with self.assertRaisesRegex(ValueError, "dimension"):
            load_embedding(np.ones(6), expected_dim=5)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_embedding([1, float("nan")])
        with self.assertRaises(ValueError):
            SegDiffAdapter(diffusion_config(feature_dim=0))

    def test_variable_shape_training_batch(self):
        adapter, first, second = SegDiffAdapter(diffusion_config()), sample(), sample()
        second.image, second.mask = second.image[:7, :11].copy(), second.mask[:7, :11].copy()
        loss = adapter.loss([first, second])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(adapter.model.feature_projector.weight.grad)

    def test_optional_bottleneck_styles_backward(self):
        for style in ("attention", "low_dim"):
            with self.subTest(style=style):
                adapter = SegDiffAdapter(
                    diffusion_config(network_style=style, low_dim=8, num_heads=2, depth=1)
                )
                adapter.loss([sample()]).backward()
                self.assertTrue(
                    any(p.grad is not None for p in adapter.model.denoiser.inner.parameters())
                )

    def test_explicit_legacy_row_and_coordinates(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "features/20x", data=np.arange(15, dtype=np.float32).reshape(3, 5)
                )
                handle.create_dataset("coords/20x", data=[[0, 0], [10, 20], [30, 40]])
            expected = np.arange(5, 10, dtype=np.float32)
            np.testing.assert_array_equal(load_embedding({"path": str(path), "row": 1}), expected)
            np.testing.assert_array_equal(
                load_embedding({"path": str(path), "xy": [10, 20]}), expected
            )
            with self.assertRaisesRegex(ValueError, "explicit"):
                load_embedding({"path": str(path)})
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_embedding({"path": str(path), "xy": [999, 999]})

    def test_native_feature_store_selection(self):
        try:
            from openhisto.feature_encoder.storage import FeatureStore
            import pandas as pd
        except ImportError:
            self.skipTest("Sibling feature_encoder core dependencies are not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slide.h5ad"
            rows = [
                dict(
                    patch_id=f"p{i}",
                    slide_id="slide",
                    source_id="source",
                    source_type="archive",
                    source_index=i,
                    mag=20.0,
                    content_hash=f"hash{i}",
                    x_level0=float(i * 16),
                    y_level0=0.0,
                    width_level0=16.0,
                    height_level0=16.0,
                    patch_width=16,
                    patch_height=16,
                    valid_x0=0,
                    valid_y0=0,
                    valid_x1=16,
                    valid_y1=16,
                    source_mag=40.0,
                    metadata_json="{}",
                )
                for i in range(2)
            ]
            obs = pd.DataFrame(rows, index=["p0", "p1"])
            with FeatureStore(path) as store:
                store.reconcile(
                    obs, slide_id="slide", sources=[{"source_id": "source", "type": "archive"}]
                )
                key = store.ensure_run("synthetic", 20, "request", {}, 5)
                store.write_batch(key, [0, 1], np.arange(10, dtype=np.float32).reshape(2, 5))
            spec = {"kind": "feature_store", "path": str(path), "model": "synthetic", "mag": 20}
            np.testing.assert_array_equal(
                load_embedding({**spec, "row": "p1"}), np.arange(5, 10, dtype=np.float32)
            )
            np.testing.assert_array_equal(
                load_embedding({**spec, "xy": [16, 0]}), np.arange(5, 10, dtype=np.float32)
            )


_SAM_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("sam2", "hydra", "tensordict")
)


@unittest.skipUnless(_SAM_AVAILABLE, "Official SAM2 and training dependencies are not installed")
class SAM2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib import resources
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            resources.files("sam2").joinpath("configs/sam2.1/sam2.1_hiera_t.yaml").read_text()
        )
        # Keep the official computation and decoder, with a small random Hiera
        # encoder/resolution. No checkpoint download is involved.
        cfg.model.image_size = 64
        cfg.model.image_encoder.trunk.embed_dim = 32
        cfg.model.image_encoder.trunk.stages = [1, 1, 1, 1]
        cfg.model.image_encoder.trunk.global_att_blocks = []
        cfg.model.image_encoder.trunk.window_spec = [2, 2, 2, 2]
        cfg.model.image_encoder.trunk.window_pos_embed_bkg_spatial_size = [2, 2]
        cfg.model.image_encoder.neck.backbone_channel_list = [256, 128, 64, 32]
        cls.config = ModelConfig(
            model="sam2",
            num_classes=2,
            pretrained=False,
            options={"architecture": OmegaConf.to_container(cfg.model, resolve=True)},
        )
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_official_backward_prompt_prediction_and_reload(self):
        item = sample(classes=2)
        item.prompts = {"box": [1, 1, 11, 8], "point_coords": [[5, 5]], "point_labels": [1]}
        adapter = SAM2Adapter(self.config)
        adapter.model.train()
        loss = adapter.loss([item])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in adapter.model.sam_mask_decoder.parameters()
            )
        )
        adapter.model.eval()
        result = adapter.infer([item])[0]
        self.assertEqual(result["probabilities"].shape, (2, 9, 13))
        np.testing.assert_allclose(result["probabilities"].sum(0), 1, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save(directory)
            reloaded = SAM2Adapter(self.config, load_dir=directory)
            reloaded.model.eval()
            np.testing.assert_allclose(
                reloaded.infer([item])[0]["logits"], result["logits"], atol=1e-5
            )
        adapter.loss([item])
        self.assertFalse(adapter.model.training)
        item.prompts = None
        with self.assertRaises(CapabilityError):
            adapter.infer([item])
        self.assertTrue(torch.isfinite(adapter.loss([item])))


if __name__ == "__main__":
    unittest.main()
