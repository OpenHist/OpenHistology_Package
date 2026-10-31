"""No-download checks for native Patho-R1 response and generation handling."""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from openhisto.vlas.backends.patho_r1 import PathoR1Backend, extract_final_answer
from openhisto.vlas.config import DescribeConfig, ModelSpec
from openhisto.vlas.base import AccessError, GenerationResult, ModelDescriptor, VLAError
from openhisto.vlas import registry


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=[[1, 2, 3]])
        self.input_ids = self["input_ids"]

    def to(self, device):
        return self


class PathoTests(unittest.TestCase):
    def test_final_answer_excludes_reasoning(self):
        cases = [
            ("<think>private reasoning</think><answer>Visible glands.</answer>", "Visible glands."),
            ("<think>unfinished reasoning", ""),
            ("reasoning without opening tag</think>", ""),
            ("<think>reasoning</think>Visible glands.", "Visible glands."),
            ("<answer>Visible glands", "Visible glands"),
            ("Visible glands.", "Visible glands."),
        ]
        for raw, answer in cases:
            with self.subTest(raw=raw):
                self.assertEqual(extract_final_answer(raw), answer)

    def test_generate_trims_prompt_and_keeps_patches_independent(self):
        backend = PathoR1Backend.__new__(PathoR1Backend)
        backend.config = DescribeConfig(device="cpu", max_new_tokens=2)
        backend.device = "cpu"
        backend.torch = MagicMock()
        backend.model = MagicMock()
        backend.model.generation_config.eos_token_id = 9
        backend.model.generate.return_value = [[1, 2, 3, 8, 8]]
        backend.processor = MagicMock()
        backend.processor.return_value = FakeInputs()
        backend.processor.batch_decode.return_value = [
            "<think>internal</think><answer>Glands.</answer>"
        ]
        backend.process_vision_info = MagicMock(return_value=([object()], None))
        image = MagicMock()
        results = backend.generate([image, image], "Describe the morphology")
        self.assertEqual([result.text for result in results], ["Glands.", "Glands."])
        self.assertTrue(
            all(result.truncated and result.finish_reason == "length" for result in results)
        )
        self.assertEqual(backend.processor.batch_decode.call_args.args[0], [[8, 8]])
        self.assertEqual(len(backend.processor.apply_chat_template.call_args_list), 2)
        for call in backend.processor.apply_chat_template.call_args_list:
            self.assertEqual(len(call.args[0]), 2)
        self.assertNotIn("temperature", backend.model.generate.call_args.kwargs)
        backend.model.generate.return_value = [[1, 2, 3, 8, 9]]
        completed = backend.generate([image], "Describe")[0]
        self.assertFalse(completed.truncated)
        self.assertEqual(completed.finish_reason, "stop")
        backend.processor.batch_decode.return_value = ["<think>unfinished"]
        unfinished = backend.generate([image], "Describe")[0]
        self.assertEqual(unfinished.text, "")
        self.assertEqual(unfinished.finish_reason, "no_final_answer")

    def test_load_offline_passes_pinned_revision_and_processor_settings(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = "float32"
        fake_torch.cuda = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        processor = MagicMock()
        model = MagicMock()
        model.config._commit_hash = "resolved-revision"
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = MagicMock()
        transformers.AutoProcessor.from_pretrained.return_value = processor
        transformers.Qwen2_5_VLForConditionalGeneration = MagicMock()
        transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.return_value = (model, {})
        vision = types.ModuleType("qwen_vl_utils")
        vision.process_vision_info = MagicMock()
        modules = {"torch": fake_torch, "transformers": transformers, "qwen_vl_utils": vision}
        spec = ModelSpec(
            model="patho_r1_7b",
            checkpoint="WenchuanZhang/Patho-R1-7B",
            revision="pinned",
            options={"max_pixels": 1024},
        )
        with patch.dict(sys.modules, modules):
            backend = PathoR1Backend(
                spec, DescribeConfig(device="cpu", offline=True, cache_dir="/cache")
            )
            kwargs = transformers.AutoProcessor.from_pretrained.call_args.kwargs
            self.assertEqual(
                kwargs,
                {
                    "local_files_only": True,
                    "cache_dir": "/cache",
                    "revision": "pinned",
                    "max_pixels": 1024,
                },
            )
            self.assertEqual(backend.metadata["resolved_revision"], "resolved-revision")
            self.assertEqual(
                transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.call_args.kwargs[
                    "device_map"
                ],
                "cpu",
            )
            backend.close()
            transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.return_value = (
                model,
                {"missing_keys": ["visual.learned_weight"]},
            )
            with self.assertRaisesRegex(VLAError, "refusing partially initialized weights"):
                PathoR1Backend(spec, DescribeConfig(device="cpu", offline=True))

    def test_registry_pins_defaults_and_rejects_pathchat_before_worker(self):
        with patch("openhisto.vlas.runtime.WorkerClient") as worker:
            with self.assertRaises(AccessError):
                registry.prepare_model("pathchat_plus")
            worker.assert_not_called()
            handle = registry.prepare_model("patho_r1_7b", {"device": "cpu"})
            resolved, config = worker.call_args.args
            self.assertEqual(
                resolved.revision, registry.get_descriptor("patho_r1_7b").checkpoint_revision
            )
            self.assertEqual(config.device, "cpu")
            handle.close()

    def test_registered_backend_loads_once_and_handle_closes_once(self):
        descriptor = ModelDescriptor(
            "fake_patho_test", "Fake", "fixture", "r1", "fixture", "r1", "test", "test"
        )
        backend = MagicMock()
        backend.metadata = {"model": "fake_patho_test"}
        backend.generate.return_value = [GenerationResult("Glands")]
        factory = MagicMock(return_value=backend)
        with patch.dict(registry._DESCRIPTORS), patch.dict(registry._FACTORIES):
            registry.register_model(descriptor, factory)
            with registry.prepare_model("fake_patho_test") as handle:
                self.assertEqual(handle.generate([object()], "Describe")[0].text, "Glands")
                handle.generate([object()], "Describe again")
                self.assertEqual(handle.metadata["model"], "fake_patho_test")
            handle.close()
            factory.assert_called_once()
            backend.close.assert_called_once()
            with self.assertRaises(VLAError):
                handle.generate([], "Describe")


if __name__ == "__main__":
    unittest.main()
