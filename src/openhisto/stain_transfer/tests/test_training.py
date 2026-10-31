import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from openhisto.stain_transfer import (
    DomainSpec,
    TrainConfig,
    TransferConfig,
    create_normalizer,
    train,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "optional torch dependency unavailable")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.target = self.root / "source", self.root / "target"
        self.source.mkdir()
        self.target.mkdir()
        pixels = np.random.default_rng(4).integers(20, 220, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(self.source / "patch.png")
        Image.fromarray(np.roll(pixels, 1, axis=2)).save(self.target / "patch.png")

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, **kwargs):
        values = dict(
            source=self.source,
            target=self.target,
            output=self.root / "trained",
            device="cpu",
            ngf=4,
            ndf=4,
            batch_size=1,
            epochs=1,
            decay_epochs=0,
            load_size=256,
            crop_size=256,
            sample_every=0,
            options={"color_jitter": False, "pool_size": 2},
        )
        values.update(kwargs)
        return TrainConfig(**values)

    def test_resume_honors_explicit_default_values_without_changing_checkpoint_schema(self):
        from openhisto.stain_transfer.training import _resolve

        saved = self.config(workers=2, device="cpu", checkpoint_every=5, sample_every=3)
        state = {"config": saved.to_dict()}
        inherited = _resolve(TrainConfig(resume="checkpoint.pth"), state)
        self.assertEqual(
            (inherited.workers, inherited.device, inherited.checkpoint_every), (2, "cpu", 5)
        )
        explicit = _resolve(
            TrainConfig(
                resume="checkpoint.pth",
                workers=0,
                device="auto",
                checkpoint_every=1,
                sample_every=1,
            ),
            state,
        )
        self.assertEqual(
            (explicit.workers, explicit.device, explicit.checkpoint_every, explicit.sample_every),
            (0, "auto", 1, 1),
        )
        serialized = explicit.to_dict()
        self.assertNotIn("_specified_resume_fields", serialized)
        self.assertEqual(TrainConfig(**json.loads(json.dumps(serialized))).to_dict(), serialized)
        self.assertEqual(TrainConfig().workers, 0)

    def test_partial_checkpoint_with_workers_can_resume_with_explicit_zero(self):
        import torch

        for root in (self.source, self.target):
            with Image.open(root / "patch.png") as image:
                image.save(root / "second.png")
        first = train(self.config(max_steps=1, workers=1))
        state = torch.load(first.checkpoint, weights_only=True)
        self.assertEqual((state["config"]["workers"], state["next_batch"]), (1, 1))
        resumed = train(
            TrainConfig(
                resume=first.checkpoint, output=self.root / "worker_resume", workers=0, max_steps=1
            )
        )
        self.assertEqual((resumed.epochs_completed, resumed.global_step), (1, 2))

    def test_cycle_training_export_and_resume_match_uninterrupted_run(self):
        import torch

        first = train(self.config(epochs=2, max_steps=1))
        self.assertEqual((first.epochs_completed, first.global_step), (1, 1))
        state = torch.load(first.checkpoint, weights_only=True)
        self.assertEqual(set(state["optimizers"]), {"G", "D"})
        self.assertEqual(set(state["schedulers"]), {"G", "D"})
        resumed = train(
            TrainConfig(
                resume=first.checkpoint,
                epochs=2,
                max_steps=1,
                output=self.root / "resumed",
                device="cpu",
                sample_every=0,
            )
        )
        uninterrupted = train(self.config(output=self.root / "uninterrupted", epochs=2))
        actual = torch.load(resumed.checkpoint, weights_only=True)
        expected = torch.load(uninterrupted.checkpoint, weights_only=True)
        self.assertEqual((resumed.epochs_completed, resumed.global_step), (2, 2))
        for name in actual["networks"]:
            for key, tensor in actual["networks"][name].items():
                self.assertTrue(torch.equal(tensor, expected["networks"][name][key]), (name, key))
        normalizer = create_normalizer(
            DomainSpec(method="cyclegan", checkpoint=resumed.exports[1]),
            TransferConfig(device="cpu"),
        )
        image = np.full((33, 49, 3), 100, np.uint8)
        self.assertEqual(normalizer.transform(image).shape, image.shape)
        full = create_normalizer(
            DomainSpec(method="cyclegan", checkpoint=resumed.checkpoint, direction="BtoA")
        )
        self.assertEqual(full.provenance["backends"][0]["direction"], "BtoA")

    def test_partial_epoch_resume_reproduces_training(self):
        import torch

        for root in (self.source, self.target):
            Image.open(root / "patch.png").save(root / "second.png")
        config = self.config(max_steps=1, options={"color_jitter": True, "pool_size": 2})
        first = train(config)
        self.assertEqual(first.epochs_completed, 0)
        resumed = train(
            TrainConfig(resume=first.checkpoint, max_steps=1, output=self.root / "partial_resume")
        )
        expected = train(
            self.config(
                output=self.root / "partial_expected",
                options={"color_jitter": True, "pool_size": 2},
            )
        )
        actual_state = torch.load(resumed.checkpoint, weights_only=True)
        expected_state = torch.load(expected.checkpoint, weights_only=True)
        self.assertEqual((resumed.epochs_completed, resumed.global_step), (1, 2))
        for name, state in actual_state["networks"].items():
            for key, tensor in state.items():
                self.assertTrue(
                    torch.equal(tensor, expected_state["networks"][name][key]), (name, key)
                )

    def test_pix2pix_training_and_self_pairing(self):
        report = train(
            self.config(
                method="pix2pix",
                source=None,
                grayscale_source=True,
                output=self.root / "pix",
                options={},
            )
        )
        self.assertEqual(report.global_step, 1)
        self.assertEqual(len(report.exports), 1)
        metadata = json.loads(report.exports[0].with_suffix(".json").read_text())
        self.assertEqual(metadata["input_mode"], "grayscale")
        self.assertEqual(metadata["architecture"], "pix2pix_unet_256")

    def test_masks_train_and_slide_manifests_expand(self):
        from openhisto.stain_transfer.datasets import image_records

        masks = self.root / "masks"
        masks.mkdir()
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[40:150, 60:190] = 255
        Image.fromarray(mask).save(masks / "patch_mask.png")
        manifest = self.root / "slides.csv"
        manifest.write_text("slide_dir,mask_dir\nsource,masks\n")
        self.assertEqual(len(image_records(manifest)), 1)
        report = train(
            self.config(
                source=manifest,
                lambda_mask=1,
                seg_channels=4,
                seg_depth=2,
                output=self.root / "masked",
            )
        )
        losses = json.loads((report.output / "losses.jsonl").read_text().splitlines()[0])
        self.assertGreater(losses["mask"], 0)
        self.assertGreater(losses["segmentation"], 0)

    def test_mask_weight_requires_available_masks(self):
        with self.assertRaisesRegex(ValueError, "requires usable mask_path"):
            train(self.config(lambda_mask=1))

    def test_paired_filename_mismatch_is_not_silently_ignored(self):
        (self.target / "patch.png").rename(self.target / "different.png")
        with self.assertRaisesRegex(ValueError, "Unmatched Pix2Pix"):
            train(self.config(method="pix2pix"))

    def test_legacy_weights_are_warm_start_not_resume(self):
        import torch

        path = self.root / "raw.pth"
        torch.save({"weight": torch.ones(1)}, path)
        with self.assertRaisesRegex(ValueError, "full stain_transfer training checkpoint"):
            train(self.config(resume=path))

    def test_empty_and_incomplete_resume_checkpoints_fail_before_writing(self):
        import torch

        invalid_states = ({}, [], None, {"format": "stain_transfer.training.v1"})
        for index, state in enumerate(invalid_states):
            with self.subTest(state=state):
                path = self.root / f"invalid_{index}.pth"
                output = self.root / f"invalid_output_{index}"
                torch.save(state, path)
                with self.assertRaisesRegex(
                    ValueError,
                    "full stain_transfer training checkpoint|Incomplete training checkpoint",
                ):
                    train(self.config(resume=path, output=output))
                self.assertFalse(output.exists())

    def test_fresh_training_refuses_previous_artifacts_without_checkpoint(self):
        for name in (
            "training_config.json",
            "losses.jsonl",
            "generator_AtoB.pth",
            "generator_BtoA.json",
        ):
            with self.subTest(name=name):
                output = self.root / name.replace(".", "_")
                output.mkdir()
                artifact = output / name
                artifact.write_bytes(b"existing training evidence")
                with self.assertRaisesRegex(FileExistsError, "training artifacts"):
                    train(self.config(output=output))
                self.assertEqual(artifact.read_bytes(), b"existing training evidence")
                self.assertEqual(list(output.iterdir()), [artifact])

    def test_full_warm_start_requires_same_method_and_compatible_generator(self):
        import torch

        cases = (
            ("cyclegan", "pix2pix", {"G": {}}, "does not match"),
            ("pix2pix", "cyclegan", {"G_A": {}}, "does not match"),
            ("cyclegan", "cyclegan", {"D_A": {}}, "compatible generator"),
        )
        for index, (method, saved_method, networks, message) in enumerate(cases):
            with self.subTest(method=method, saved_method=saved_method):
                path = self.root / f"warm_{index}.pth"
                output = self.root / f"warm_output_{index}"
                torch.save(
                    {
                        "format": "stain_transfer.training.v1",
                        "metadata": {"method": saved_method},
                        "networks": networks,
                    },
                    path,
                )
                with self.assertRaisesRegex(ValueError, message):
                    train(self.config(method=method, warm_start=path, output=output))
                self.assertFalse(output.exists())
