"""Experimental native PA-LLaVA inference from explicitly prepared artifacts."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path

from ..base import GenerationResult, VLAError
from .legacy import (
    _check_module_origin,
    _code_revision,
    _descriptor,
    _field,
    _learned_loading_errors,
    _snapshot,
    _upstream_path,
    _version,
)


def _directory(value, label):
    if not value:
        raise VLAError(f"PA-LLaVA requires {label}; see its prepared-artifact setup instructions.")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise VLAError(f"PA-LLaVA {label} must be an existing prepared directory: {path}")
    return path


def _require_weights(path, label, adapter=False):
    names = (
        ("adapter_model.safetensors", "adapter_model.bin")
        if adapter
        else (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )
    if not any((path / name).is_file() for name in names):
        raise VLAError(
            f"No learned {label} weights found in {path}; raw training .pth files must first be converted with the authors' PA-LLaVA/XTuner workflow."
        )


@contextmanager
def _local_projector_bert(module, config_path):
    """Route the native constructor's hardcoded config lookup to pinned assets.

    The projector itself and its learned weights remain native. Each persistent
    worker constructs one model at a time, and the module binding is restored.
    """
    original = module.BertConfig

    class LocalBertConfig(original):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return original.from_pretrained(str(config_path), local_files_only=True)

    module.BertConfig = LocalBertConfig
    try:
        yield
    finally:
        module.BertConfig = original


class PALlavaBackend:
    """Preserve PA's PLIP -> anyres QFormer -> Llama-3 generation path."""

    def __init__(self, spec, config):
        self.spec, self.config = spec, config
        self.llm = self.visual_encoder = self.projector = self.tokenizer = None
        options = dict(spec.options or {})
        unsupported = set(options) - {
            "upstream_path",
            "vision_checkpoint",
            "base_checkpoint",
            "projector_checkpoint",
            "adapter_checkpoint",
        }
        if unsupported:
            raise VLAError(f"Unsupported PA-LLaVA options: {', '.join(sorted(unsupported))}")
        root = Path(spec.checkpoint).expanduser() if spec.checkpoint else None
        prepared = root if root is not None and root.is_dir() else None
        projector_path = _directory(
            options.get("projector_checkpoint") or (prepared / "projector" if prepared else None),
            "options.projector_checkpoint (converted QFormer projector)",
        )
        _require_weights(projector_path, "projector")
        if not (projector_path / "config.json").is_file():
            raise VLAError(f"Converted PA-LLaVA projector config.json is missing: {projector_path}")
        vision_path = _directory(
            options.get("vision_checkpoint"), "options.vision_checkpoint (extracted authors' PLIP)"
        )
        if not all(
            (vision_path / name).is_file() for name in ("config.json", "plip_80w_196token.bin")
        ):
            raise VLAError(
                f"PLIP directory must contain config.json and plip_80w_196token.bin: {vision_path}"
            )
        adapter_value = options.get("adapter_checkpoint")
        if adapter_value is None and prepared is not None and (prepared / "llm_adapter").is_dir():
            adapter_value = prepared / "llm_adapter"
        adapter_path = _directory(
            adapter_value, "options.adapter_checkpoint (converted PA-LLaVA llm_adapter)"
        )
        _require_weights(adapter_path, "LLM adapter", adapter=True)
        if not (adapter_path / "adapter_config.json").is_file():
            raise VLAError(f"PA-LLaVA PEFT adapter_config.json is missing: {adapter_path}")
        base_value = str(
            _directory(
                options.get("base_checkpoint"),
                "options.base_checkpoint (local original Llama-3 base)",
            )
        )
        upstream = _upstream_path(options, "pallava")
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                CLIPImageProcessor,
                GenerationConfig,
            )

            self.transformers = importlib.import_module("transformers")
            pallava = importlib.import_module("pallava")
            _check_module_origin(pallava, upstream)
            native_vision = importlib.import_module("pallava.flip.fflip")
            native_projector = importlib.import_module("pallava.projector")
            native_utils = importlib.import_module("pallava.utils")
            xtuner_utils = importlib.import_module("xtuner.utils")
            self.xtuner_dataset = importlib.import_module("xtuner.dataset.utils")
            self.xtuner_model = importlib.import_module("xtuner.model.utils")
            self.xtuner_tools = importlib.import_module("xtuner.tools.utils")
            from peft import PeftModel
        except (ImportError, AttributeError) as exc:
            raise VLAError(
                f"PA-LLaVA needs its dedicated XTuner runtime and the authors' pallava package on upstream_path; install requirements-pallava.txt: {exc}"
            ) from exc
        self.torch = torch
        self.device = "cuda" if str(config.device) == "auto" else str(config.device)
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise VLAError("Experimental PA-LLaVA native inference requires an available CUDA GPU.")
        precision = "float16" if str(config.precision) == "auto" else str(config.precision)
        if precision not in {"float16", "float32", "bfloat16"}:
            raise VLAError(
                "PA-LLaVA supports float16, bfloat16, or float32; quantized PA inference is not validated."
            )
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise VLAError(
                "The selected CUDA GPU does not support bfloat16; use precision='float16'."
            )
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[precision]
        self.image_token = xtuner_utils.DEFAULT_IMAGE_TOKEN
        self.image_token_index = xtuner_utils.IMAGE_TOKEN_INDEX
        self.template = xtuner_utils.PROMPT_TEMPLATE["llama3_chat"]
        self.process_anyres_image = native_utils.process_anyres_image
        base_path, base_revision = _snapshot(base_value, config)
        _require_weights(Path(base_path), "base LLM")
        # These are configuration/preprocessor assets only, not substitute encoders.
        clip_path, clip_revision = _snapshot(
            "openai/clip-vit-base-patch16",
            config,
            revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
            patterns=["config.json", "preprocessor_config.json"],
        )
        bert_path, bert_revision = _snapshot(
            "google-bert/bert-base-uncased",
            config,
            revision="86b5e0934494bd15c9632b12f734a8a67f723594",
            patterns=["config.json"],
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_path, local_files_only=True, encode_special_tokens=True
            )
            self.llm, loading_info = AutoModelForCausalLM.from_pretrained(
                base_path,
                torch_dtype=self.dtype,
                device_map={"": self.device},
                low_cpu_mem_usage=True,
                local_files_only=True,
                output_loading_info=True,
            )
            errors = _learned_loading_errors(loading_info)
            if errors:
                raise VLAError(f"PA-LLaVA base LLM did not load completely: {errors}")
            if self.llm.config.model_type != "llama":
                raise VLAError("PA-LLaVA requires its Llama-3 base or merged LLM checkpoint.")
            self.llm = PeftModel.from_pretrained(
                self.llm, str(adapter_path), is_trainable=False, local_files_only=True
            )
            vision_config = native_vision.VisionConfig.from_json_file(
                str(vision_path / "config.json")
            )
            self.visual_encoder = native_vision.VisionModel(vision_config)
            checkpoint = torch.load(
                str(vision_path / "plip_80w_196token.bin"), map_location="cpu", weights_only=True
            )
            self.visual_encoder.load_state_dict(checkpoint, strict=True)
            self.visual_encoder.to(device=self.device, dtype=self.dtype).eval()
            self.image_processor = CLIPImageProcessor.from_pretrained(
                clip_path, local_files_only=True
            )
            projector_config = native_projector.QFormerProjectorConfig.from_pretrained(
                str(projector_path), local_files_only=True
            )
            if (
                projector_config.visual_hidden_size != self.visual_encoder.config.hidden_size
                or projector_config.llm_hidden_size != self.llm.config.hidden_size
            ):
                raise VLAError(
                    "PA-LLaVA PLIP/projector/LLM hidden dimensions disagree; use matching prepared artifacts."
                )
            with _local_projector_bert(native_projector, bert_path):
                self.projector, loading_info = (
                    native_projector.QFormerProjectorModel.from_pretrained(
                        str(projector_path),
                        config=projector_config,
                        torch_dtype=self.dtype,
                        local_files_only=True,
                        output_loading_info=True,
                    )
                )
            errors = _learned_loading_errors(loading_info)
            if errors:
                raise VLAError(f"PA-LLaVA learned projector did not load completely: {errors}")
            self.projector.to(device=self.device, dtype=self.dtype).eval()
            self.llm.eval()
            self.GenerationConfig = GenerationConfig
        except VLAError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise VLAError(f"Failed loading prepared PA-LLaVA artifacts: {exc}") from exc
        descriptor = _descriptor(spec)
        self.metadata = {
            "backend": "pallava",
            "experimental": True,
            "checkpoint": str(spec.checkpoint),
            "resolved_revision": None,
            "base_checkpoint": str(base_value),
            "base_revision": base_revision,
            "adapter_checkpoint": str(adapter_path) if adapter_path else None,
            "projector_checkpoint": str(projector_path),
            "vision_checkpoint": str(vision_path),
            "source_revision": _code_revision(native_projector),
            "expected_source_revision": _field(descriptor, "source_revision"),
            "upstream_module": getattr(native_projector, "__file__", None),
            "preprocessing": {
                "implementation": "pallava.utils.process_anyres_image",
                "processor": "openai/clip-vit-base-patch16",
                "processor_revision": clip_revision,
                "visual_select_layer": -2,
            },
            "projector_bert_config_revision": bert_revision,
            "conversation_mode": "llama3_chat",
            "device": self.device,
            "precision": precision,
            "libraries": {
                name: _version(name)
                for name in ("torch", "transformers", "xtuner", "peft", "mmengine")
            },
        }

    def generate(self, images, prompt):
        if self.llm is None:
            raise VLAError("The PA-LLaVA backend is closed.")
        if self.image_token in prompt:
            raise VLAError(
                "PA-LLaVA inserts its image token automatically; remove it from the prompt."
            )
        results = []
        for image in images:
            if getattr(self.config, "seed", None) is not None:
                self.torch.manual_seed(self.config.seed)
            text = self.template["INSTRUCTION"].format(
                input=self.image_token + "\n" + prompt, round=1, bot_name="BOT"
            )
            bos_ids, _ = self.xtuner_dataset.get_bos_eos_token_ids(self.tokenizer)
            ids = list(bos_ids)
            for index, chunk in enumerate(text.split(self.image_token)):
                ids.extend(
                    self.tokenizer.encode(
                        chunk, **({} if index == 0 else {"add_special_tokens": False})
                    )
                )
                if index == 0:
                    ids.append(self.image_token_index)
            input_ids = self.torch.tensor(ids, device=self.device).unsqueeze(0)
            pixels = self.process_anyres_image(image.convert("RGB"), self.image_processor, None).to(
                device=self.device, dtype=self.dtype
            )
            stop_criteria = self.xtuner_tools.get_stop_criteria(
                tokenizer=self.tokenizer, stop_words=list(self.template.get("STOP_WORDS", []))
            )
            generation = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.do_sample,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id,
            }
            if self.config.do_sample:
                generation["temperature"] = self.config.temperature
            with self.torch.inference_mode():
                outputs = self.visual_encoder(pixels, output_hidden_states=True)
                hidden = outputs.hidden_states[-2][:, 1:]
                global_features = hidden[0].unsqueeze(0)
                local_features = hidden[1:].reshape(-1, hidden.shape[-1]).unsqueeze(0)
                if not local_features.shape[1]:
                    raise VLAError("PA-LLaVA anyres preprocessing produced no local features.")
                projected = self.projector(global_features, local_features)
                multimodal = self.xtuner_model.prepare_inputs_labels_for_multimodal(
                    llm=self.llm, input_ids=input_ids, pixel_values=projected
                )
                generated = self.llm.generate(
                    **multimodal,
                    generation_config=self.GenerationConfig(**generation),
                    streamer=None,
                    bos_token_id=self.tokenizer.bos_token_id,
                    stopping_criteria=stop_criteria,
                )
            tokens = generated[0]
            # Some Transformers releases return the supplied text prefix, others
            # return only generated tokens for inputs_embeds. Handle both forms.
            if len(tokens) >= input_ids.shape[1] and self.torch.equal(
                tokens[: input_ids.shape[1]], input_ids[0]
            ):
                tokens = tokens[input_ids.shape[1] :]
            answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            truncated = len(tokens) >= self.config.max_new_tokens and (
                not len(tokens) or int(tokens[-1]) != self.tokenizer.eos_token_id
            )
            results.append(
                GenerationResult(
                    text=answer,
                    truncated=truncated,
                    finish_reason="length" if truncated else "stop",
                )
            )
        return results

    def close(self):
        self.llm = self.visual_encoder = self.projector = self.tokenizer = None
        torch = getattr(self, "torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
