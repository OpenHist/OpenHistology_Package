"""No-download contract checks for native legacy preprocessing and loading."""

import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

from openhisto.vlas.backends.legacy import (
    PathGenBackend,
    QuiltBackend,
    _code_revision,
    _learned_loading_errors,
)
from openhisto.vlas.backends.pallava import PALlavaBackend
from openhisto.vlas.base import VLAError
from openhisto.vlas.config import DescribeConfig, ModelSpec


@unittest.skipUnless(shutil.which("git"), "Git checkout provenance requires git")
class CodeRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "Containing application",
            ],
            check=True,
        )
        self.origin = self.root / ".venv/lib/site-packages/llava/model.py"
        self.origin.parent.mkdir(parents=True)
        self.origin.write_text("# Installed module, outside the application source\n")
        self.module = types.SimpleNamespace(__file__=str(self.origin))

    def test_untracked_virtualenv_module_does_not_inherit_parent_commit(self):
        self.assertIsNone(_code_revision(self.module))

    def test_installed_module_uses_own_distribution_vcs_metadata(self):
        installed = types.SimpleNamespace(
            files=["llava/model.py"],
            locate_file=lambda file: self.origin.parent.parent / file,
            read_text=lambda name: json.dumps({"vcs_info": {"commit_id": "a" * 40}}),
        )
        with patch(
            "openhisto.vlas.backends.legacy.importlib.metadata.distribution", return_value=installed
        ):
            self.assertEqual(_code_revision(self.module, "llava"), "a" * 40)

    def test_tracked_checkout_uses_its_git_revision(self):
        origin = self.root / "llava/model.py"
        origin.parent.mkdir()
        origin.write_text("# Native source checkout\n")
        subprocess.run(["git", "-C", str(self.root), "add", "llava/model.py"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-q",
                "-m",
                "Native module",
            ],
            check=True,
        )
        expected = subprocess.check_output(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True
        ).strip()
        unrelated = types.SimpleNamespace(
            files=["llava/model.py"],
            locate_file=lambda file: self.origin.parent.parent / file,
            read_text=lambda name: json.dumps({"vcs_info": {"commit_id": "a" * 40}}),
        )
        with patch(
            "openhisto.vlas.backends.legacy.importlib.metadata.distribution", return_value=unrelated
        ):
            self.assertEqual(
                _code_revision(types.SimpleNamespace(__file__=str(origin)), "llava"), expected
            )


class Tensor:
    def __init__(self, values):
        self.values = values

    @property
    def shape(self):
        return (
            (len(self.values), len(self.values[0]))
            if self.values and isinstance(self.values[0], list)
            else (len(self.values),)
        )

    def unsqueeze(self, axis):
        return Tensor([self.values])

    def to(self, *args, **kwargs):
        return self

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        value = self.values[item]
        return Tensor(value) if isinstance(value, list) else value


def fake_torch():
    torch = types.ModuleType("torch")
    torch.float16, torch.float32, torch.bfloat16 = "float16", "float32", "bfloat16"
    torch.cuda = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.manual_seed = MagicMock()
    torch.inference_mode = nullcontext
    torch.tensor = MagicMock(side_effect=lambda values, **kwargs: Tensor(values))
    torch.equal = lambda left, right: left.values == right.values
    return torch


class LegacyContracts(unittest.TestCase):
    def test_pathgen_loader_overrides_vision_config_only_in_memory(self):
        torch = fake_torch()
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = MagicMock()
        tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
        tokenizer.__len__.return_value = 10
        native = types.ModuleType("llava.model.language_model.llava_llama")
        native.LlavaConfig = MagicMock()
        config = types.SimpleNamespace(
            mm_vision_tower="remote/vision",
            image_aspect_ratio="pad",
            mm_use_im_start_end=False,
            mm_use_im_patch_token=False,
        )
        native.LlavaConfig.from_pretrained.return_value = config
        native.LlavaLlamaForCausalLM = MagicMock()
        model = MagicMock()
        model.config = config
        model.get_input_embeddings.return_value.weight.shape = (10, 8)
        model.get_vision_tower.return_value.is_loaded = False
        model.get_vision_tower.return_value.vision_tower.load_state_dict.return_value = (
            types.SimpleNamespace(missing_keys=[], unexpected_keys=[])
        )
        native.LlavaLlamaForCausalLM.from_pretrained.return_value = (
            model,
            {"unexpected_keys": ["model.vision_tower.vision_tower.vision_model.weight"]},
        )
        torch.load = MagicMock(
            return_value={
                "model.vision_tower.vision_tower.vision_model.weight": "released-vision-weight"
            }
        )
        modules = {
            "torch": torch,
            "transformers": transformers,
            "llava": types.ModuleType("llava"),
            "llava.model.language_model.llava_llama": native,
            "llava.mm_utils": types.ModuleType("llava.mm_utils"),
            "llava.constants": types.ModuleType("llava.constants"),
            "llava.conversation": types.SimpleNamespace(conv_templates={"llava_v1": object()}),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir, vision_dir = root / "model", root / "vision"
            model_dir.mkdir()
            vision_dir.mkdir()
            original = {"model_type": "llava_llama", "mm_vision_tower": "remote/vision"}
            (model_dir / "config.json").write_text(json.dumps(original))
            (model_dir / "pytorch_model.bin.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.vision_tower.vision_tower.vision_model.weight": "vision-shard.bin"
                        }
                    }
                )
            )
            with (
                patch.dict(sys.modules, modules),
                patch(
                    "openhisto.vlas.backends.legacy._snapshot",
                    side_effect=[(str(model_dir), "model-sha"), (str(vision_dir), "vision-sha")],
                ),
                patch("openhisto.vlas.backends.legacy._code_revision", return_value="actual-code"),
            ):
                backend = PathGenBackend(
                    ModelSpec("pathgen_llava", checkpoint="remote/model", revision="pinned"),
                    DescribeConfig(offline=True, precision="float32"),
                )
                self.assertEqual(config.mm_vision_tower, str(vision_dir))
                self.assertEqual(json.loads((model_dir / "config.json").read_text()), original)
                load_args = native.LlavaLlamaForCausalLM.from_pretrained.call_args.kwargs
                self.assertTrue(load_args["local_files_only"])
                self.assertIs(load_args["config"], config)
                self.assertEqual(load_args["torch_dtype"], "float32")
                model.get_vision_tower.return_value.load_model.assert_called_once()
                model.get_vision_tower.return_value.vision_tower.load_state_dict.assert_called_once_with(
                    {"vision_model.weight": "released-vision-weight"}, strict=False
                )
                self.assertEqual(backend.metadata["source_revision"], "actual-code")
                self.assertEqual(backend.metadata["resolved_revision"], "model-sha")
                self.assertEqual(
                    backend.metadata["vision_weights"], "released multimodal checkpoint"
                )
                backend.close()

    def test_saved_vision_weights_are_restored_instead_of_dropped(self):
        from openhisto.vlas.backends.legacy import _saved_vision_state

        torch = fake_torch()
        torch.load = MagicMock(
            return_value={
                "model.vision_tower.vision_tower.vision_model.weight": "learned",
                "lm_head.weight": "language",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pytorch_model.bin.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.vision_tower.vision_tower.vision_model.weight": "last-shard.bin",
                            "lm_head.weight": "first-shard.bin",
                        }
                    }
                )
            )
            state = _saved_vision_state(root, torch)
            self.assertEqual(state, {"vision_model.weight": "learned"})
            self.assertEqual(torch.load.call_count, 1)
            self.assertEqual(Path(torch.load.call_args.args[0]).name, "last-shard.bin")
            self.assertTrue(torch.load.call_args.kwargs["weights_only"])

    def _generation_backend(self, cls):
        backend = cls.__new__(cls)
        backend.config = DescribeConfig(max_new_tokens=2)
        backend.torch = fake_torch()
        backend.device, backend.dtype = "cuda", "float16"
        backend.constants = types.SimpleNamespace(
            DEFAULT_IMAGE_TOKEN="<image>", IMAGE_TOKEN_INDEX=-200
        )
        conversation = MagicMock()
        conversation.roles = ("user", "assistant")
        conversation.sep, conversation.sep2, conversation.sep_style = "END", "STOP", 1
        backend.conversation = types.SimpleNamespace(
            conv_templates={
                "llava_v1": types.SimpleNamespace(copy=MagicMock(side_effect=lambda: conversation))
            },
            SeparatorStyle=types.SimpleNamespace(TWO=2),
        )
        backend.conversation_mode = "llava_v1"
        backend.mm = types.SimpleNamespace(
            tokenizer_image_token=MagicMock(return_value=Tensor([1, 2, 3])),
            process_images=MagicMock(return_value=Tensor([4])),
            KeywordsStoppingCriteria=MagicMock(),
        )
        backend.tokenizer = MagicMock()
        backend.tokenizer.eos_token_id = 9
        backend.tokenizer.decode.return_value = "Morphology END"
        backend.model = MagicMock()
        backend.model.config.mm_use_im_start_end = False
        backend.model.generate.return_value = [Tensor([1, 2, 3, 8, 8])]
        backend.image_processor = object()
        return backend, conversation

    def test_native_generation_keeps_images_independent_and_decodes_continuation(self):
        for cls in (PathGenBackend, QuiltBackend):
            with self.subTest(backend=cls.__name__):
                backend, conversation = self._generation_backend(cls)
                image = MagicMock()
                image.size = (256, 384)
                results = backend.generate([image, image], "Describe the morphology")
                self.assertEqual([result.text for result in results], ["Morphology", "Morphology"])
                self.assertTrue(all(result.truncated for result in results))
                self.assertEqual(backend.tokenizer.decode.call_args.args[0].values, [8, 8])
                self.assertEqual(backend.mm.process_images.call_count, 2)
                self.assertIs(backend.mm.process_images.call_args.args[2], backend.model.config)
                self.assertEqual(backend.conversation.conv_templates["llava_v1"].copy.call_count, 2)
                self.assertNotIn("temperature", backend.model.generate.call_args.kwargs)
                self.assertEqual(
                    "image_sizes" in backend.model.generate.call_args.kwargs, cls is PathGenBackend
                )
                self.assertEqual(
                    conversation.append_message.call_args_list[0].args,
                    ("user", "<image>\nDescribe the morphology"),
                )

    def test_missing_learned_weights_are_not_accepted(self):
        self.assertTrue(_learned_loading_errors({"missing_keys": ["model.layers.0.weight"]}))
        self.assertFalse(_learned_loading_errors({"missing_keys": ["model.rotary_emb.inv_freq"]}))

    def test_pa_requires_prepared_artifacts_before_network(self):
        with patch("openhisto.vlas.backends.pallava._snapshot") as download:
            with self.assertRaisesRegex(VLAError, "projector_checkpoint"):
                PALlavaBackend(
                    ModelSpec("pa_llava", checkpoint="OpenFace-CQUPT/Pathology-LLaVA"),
                    DescribeConfig(),
                )
            download.assert_not_called()

    def test_pa_cannot_run_plain_base_without_a_trained_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projector, vision, base = root / "projector", root / "plip", root / "base"
            for path in (projector, vision, base):
                path.mkdir()
            for path in (
                projector / "config.json",
                projector / "pytorch_model.bin",
                vision / "config.json",
                vision / "plip_80w_196token.bin",
            ):
                path.write_text("fixture")
            spec = ModelSpec(
                "pa_llava",
                checkpoint=str(root),
                options={"vision_checkpoint": str(vision), "base_checkpoint": str(base)},
            )
            with patch("openhisto.vlas.backends.pallava._snapshot") as download:
                with self.assertRaisesRegex(VLAError, "adapter_checkpoint"):
                    PALlavaBackend(spec, DescribeConfig())
                download.assert_not_called()

    def test_pa_projector_bert_config_uses_local_pinned_assets_and_restores_binding(self):
        from openhisto.vlas.backends.pallava import _local_projector_bert

        class BertConfig:
            from_pretrained = MagicMock(return_value="native-config")

        module = types.SimpleNamespace(BertConfig=BertConfig)
        with self.assertRaisesRegex(RuntimeError, "construction"):
            with _local_projector_bert(module, "/cache/snapshots/pinned-bert"):
                self.assertEqual(
                    module.BertConfig.from_pretrained("google-bert/bert-base-uncased"),
                    "native-config",
                )
                raise RuntimeError("construction")
        self.assertIs(module.BertConfig, BertConfig)
        BertConfig.from_pretrained.assert_called_once_with(
            "/cache/snapshots/pinned-bert", local_files_only=True
        )

    def test_pa_preserves_anyres_global_local_projector_flow(self):
        backend = PALlavaBackend.__new__(PALlavaBackend)
        backend.config = DescribeConfig(max_new_tokens=2)
        backend.device, backend.dtype = "cuda", "float16"
        backend.torch = fake_torch()
        backend.image_token, backend.image_token_index = "<image>", -200
        backend.template = {"INSTRUCTION": "USER:{input} ASSISTANT:", "STOP_WORDS": ["STOP"]}
        backend.tokenizer = MagicMock()
        backend.tokenizer.encode.side_effect = [[1], [2], [1], [2]]
        (
            backend.tokenizer.eos_token_id,
            backend.tokenizer.bos_token_id,
            backend.tokenizer.pad_token_id,
        ) = 9, 1, 0
        backend.tokenizer.decode.return_value = "Glands"
        backend.xtuner_dataset = types.SimpleNamespace(
            get_bos_eos_token_ids=MagicMock(return_value=([1], [9]))
        )
        backend.xtuner_tools = types.SimpleNamespace(get_stop_criteria=MagicMock(return_value=[]))
        backend.xtuner_model = types.SimpleNamespace(
            prepare_inputs_labels_for_multimodal=MagicMock(
                return_value={"inputs_embeds": "multimodal"}
            )
        )
        backend.process_anyres_image = MagicMock()
        backend.image_processor = object()
        hidden = MagicMock()
        hidden.shape = (3, 196, 768)
        selected = hidden.__getitem__.return_value
        selected.shape = (3, 196, 768)
        selected.__getitem__.return_value.reshape.return_value.unsqueeze.return_value.shape = (
            1,
            392,
            768,
        )
        backend.visual_encoder = MagicMock(
            return_value=types.SimpleNamespace(hidden_states=[None, hidden, None])
        )
        backend.projector = MagicMock(return_value="projected")
        backend.llm = MagicMock()
        backend.llm.generate.return_value = [Tensor([8, 8])]
        backend.GenerationConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
        image = MagicMock()
        results = backend.generate([image, image], "Describe")
        self.assertEqual([result.text for result in results], ["Glands", "Glands"])
        self.assertTrue(all(result.truncated for result in results))
        self.assertEqual(backend.process_anyres_image.call_count, 2)
        self.assertIs(backend.process_anyres_image.call_args.args[1], backend.image_processor)
        self.assertEqual(backend.projector.call_count, 2)
        self.assertEqual(
            backend.xtuner_model.prepare_inputs_labels_for_multimodal.call_args.kwargs[
                "pixel_values"
            ],
            "projected",
        )
        self.assertEqual(backend.llm.generate.call_args.kwargs["inputs_embeds"], "multimodal")
        self.assertFalse(
            hasattr(backend.llm.generate.call_args.kwargs["generation_config"], "temperature")
        )


if __name__ == "__main__":
    unittest.main()
