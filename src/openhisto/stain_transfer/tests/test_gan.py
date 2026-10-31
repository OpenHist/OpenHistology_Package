import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openhisto.stain_transfer import DomainSpec, TransferConfig, create_normalizer

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "optional torch dependency unavailable")
class GANTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def weights(self, method="cyclegan", sidecar=True, direction="AtoB"):
        import torch
        from openhisto.stain_transfer.networks import generator

        path = self.root / f"{method}.pth"
        torch.manual_seed(7)
        architecture = "unet_256" if method == "cyclegan" else "pix2pix_unet_256"
        model = generator(method, architecture, ngf=4)
        torch.save(model.state_dict(), path)
        if sidecar:
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "method": method,
                        "architecture": architecture,
                        "input_mode": "grayscale",
                        "direction": direction,
                        "ngf": 4,
                    }
                )
            )
        return path

    def test_both_generators_preserve_dimensions_and_are_deterministic(self):
        image = np.random.default_rng(7).integers(20, 230, (49, 71, 3), dtype=np.uint8)
        for method in ("cyclegan", "pix2pix"):
            normalizer = create_normalizer(
                DomainSpec(method=method, checkpoint=self.weights(method)),
                TransferConfig(device="cpu"),
            )
            first = normalizer.transform(image)
            np.testing.assert_array_equal(first, normalizer.transform(image))
            self.assertEqual(first.shape, image.shape)
            self.assertEqual(first.dtype, np.uint8)
            backend = normalizer.provenance["backends"][0]
            self.assertEqual(len(backend["checkpoint_sha256"]), 64)
            self.assertEqual(backend["preprocessing"]["input_mode"], "grayscale")

    def test_single_generator_metadata_cannot_be_relabeled(self):
        path = self.weights()
        for kwargs in (
            {"direction": "BtoA"},
            {"input_mode": "rgb"},
            {"architecture": "resnet_6blocks"},
        ):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                create_normalizer(DomainSpec(method="cyclegan", checkpoint=path, **kwargs))

    def test_raw_checkpoint_requires_explicit_contract(self):
        path = self.weights(sidecar=False)
        with self.assertRaisesRegex(ValueError, "explicit architecture"):
            create_normalizer(DomainSpec(method="cyclegan", checkpoint=path))
        normalizer = create_normalizer(
            DomainSpec(
                method="cyclegan",
                checkpoint=path,
                architecture="unet_256",
                input_mode="grayscale",
                direction="AtoB",
                options={"ngf": 4},
            )
        )
        self.assertEqual(normalizer.provenance["backends"][0]["direction"], "AtoB")

    def test_blank_and_padding_are_preserved(self):
        normalizer = create_normalizer(DomainSpec(method="cyclegan", checkpoint=self.weights()))
        white = np.full((31, 41, 3), 255, np.uint8)
        np.testing.assert_array_equal(normalizer.transform(white), white)
        image = white.copy()
        image[3:19, 7:33] = (120, 70, 100)
        output = normalizer.transform(image, valid_bounds=(7, 3, 33, 19))
        outside = np.ones(image.shape[:2], bool)
        outside[3:19, 7:33] = False
        np.testing.assert_array_equal(output[outside], image[outside])

    def test_unknown_gan_options_fail(self):
        with self.assertRaisesRegex(ValueError, "Unknown GAN options"):
            create_normalizer(
                DomainSpec(method="cyclegan", checkpoint=self.weights(), options={"typo": 1})
            )

    def test_all_cyclegan_architectures_handle_odd_and_tiny_images(self):
        import torch
        from openhisto.stain_transfer.networks import generator

        for architecture in ("unet_128", "resnet_6blocks", "resnet_9blocks"):
            path = self.root / f"{architecture}.pth"
            torch.save(generator("cyclegan", architecture, ngf=4).state_dict(), path)
            normalizer = create_normalizer(
                DomainSpec(
                    method="cyclegan",
                    checkpoint=path,
                    architecture=architecture,
                    input_mode="grayscale",
                    direction="AtoB",
                    options={"ngf": 4},
                )
            )
            for shape in ((1, 1, 3), (17, 23, 3)):
                with self.subTest(architecture=architecture, shape=shape):
                    image = np.full(shape, 100, np.uint8)
                    result = normalizer.transform(image)
                    self.assertEqual(result.shape, image.shape)
                    self.assertEqual(result.dtype, np.uint8)

    def test_pix2pix_dropout_option_is_honored_without_changing_checkpoint_keys(self):
        import torch
        from openhisto.stain_transfer.networks import generator

        enabled = generator("pix2pix", "pix2pix_unet_256", ngf=4, dropout=True)
        disabled = generator("pix2pix", "pix2pix_unet_256", ngf=4, dropout=False)
        self.assertEqual(
            sum(isinstance(module, torch.nn.Dropout) for module in enabled.modules()), 3
        )
        self.assertEqual(
            sum(isinstance(module, torch.nn.Dropout) for module in disabled.modules()), 0
        )
        self.assertEqual(set(enabled.state_dict()), set(disabled.state_dict()))

    def test_full_checkpoint_sidecar_cannot_override_immutable_metadata(self):
        import torch
        from openhisto.stain_transfer.networks import generator

        path = self.root / "full.pth"
        metadata = {
            "method": "cyclegan",
            "architecture": "unet_256",
            "input_mode": "grayscale",
            "ngf": 4,
            "norm": "instance",
            "dropout": False,
            "direction": "AtoB",
        }
        weights = generator("cyclegan", "unet_256", ngf=4).state_dict()
        torch.save(
            {
                "format": "stain_transfer.training.v1",
                "metadata": metadata,
                "networks": {"G_A": weights, "G_B": weights},
            },
            path,
        )
        conflicts = {
            "method": "pix2pix",
            "architecture": "unet_128",
            "input_mode": "rgb",
            "ngf": 8,
            "norm": "batch",
            "dropout": True,
            "direction": "BtoA",
        }
        for field, value in conflicts.items():
            with self.subTest(field=field):
                path.with_suffix(".json").write_text(json.dumps({field: value}))
                with self.assertRaisesRegex(ValueError, f"Sidecar {field} conflicts"):
                    create_normalizer(DomainSpec(method="cyclegan", checkpoint=path))
        path.with_suffix(".json").unlink()
        normalizer = create_normalizer(
            DomainSpec(method="cyclegan", checkpoint=path, direction="BtoA")
        )
        self.assertEqual(normalizer.provenance["backends"][0]["direction"], "BtoA")
