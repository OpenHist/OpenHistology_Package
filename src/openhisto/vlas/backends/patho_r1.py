"""Native Patho-R1 inference using the released Qwen2.5-VL checkpoint."""

from __future__ import annotations

import gc
import importlib.metadata
import re
from pathlib import Path

from ..base import AccessError, GenerationResult, VLAError


SYSTEM_PROMPT = (
    "You are a pathology image assistant. Follow the user's instruction using only "
    "features visible in the supplied image. Put the final response inside "
    "<answer>...</answer>."
)


def extract_final_answer(raw: str) -> str:
    """Return the answer without exposing model reasoning as a description."""
    answers = re.findall(r"<answer\s*>(.*?)</answer\s*>", raw, flags=re.I | re.S)
    if answers:
        return answers[-1].strip()
    opening = re.search(r"<answer\s*>", raw, flags=re.I)
    if opening:
        answer = raw[opening.end() :]
        return re.sub(r"<[^>]*>", "", answer).strip()
    if re.search(r"</?think\s*>", raw, flags=re.I):
        closing = list(re.finditer(r"</think\s*>", raw, flags=re.I))
        return raw[closing[-1].end() :].strip() if closing else ""
    return raw.strip()


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class PathoR1Backend:
    def __init__(self, spec, config):
        self.spec, self.config = spec, config
        self.model = self.processor = None
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise VLAError(
                "Patho-R1 dependencies are missing. Install openhisto/vlas/requirements-patho-r1.txt "
                "in this backend's Python environment."
            ) from exc
        self.torch, self.process_vision_info = torch, process_vision_info
        self.device = self._device(config.device)
        dtype = self._dtype(config.precision)
        options = dict(spec.options or {})
        unknown = set(options) - {"min_pixels", "max_pixels", "attn_implementation"}
        if unknown:
            raise VLAError(f"Unknown Patho-R1 options: {', '.join(sorted(unknown))}")
        processor_options = {}
        for name in ("min_pixels", "max_pixels"):
            if name in options:
                value = options[name]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise VLAError(f"{name} must be a positive integer")
                processor_options[name] = value
        if processor_options.get("min_pixels", 0) > processor_options.get(
            "max_pixels", float("inf")
        ):
            raise VLAError("min_pixels must be no greater than max_pixels")
        attention = options.get("attn_implementation", "sdpa")
        if attention not in {"sdpa", "eager", "flash_attention_2"}:
            raise VLAError("attn_implementation must be sdpa, eager, or flash_attention_2")
        checkpoint = spec.checkpoint or "WenchuanZhang/Patho-R1-7B"
        load_options = {"local_files_only": config.offline}
        if config.cache_dir:
            load_options["cache_dir"] = str(config.cache_dir)
        if spec.revision and not Path(checkpoint).is_dir():
            load_options["revision"] = spec.revision
        try:
            self.processor = AutoProcessor.from_pretrained(
                checkpoint, **load_options, **processor_options
            )
            self.model, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint,
                torch_dtype=dtype,
                device_map=self.device,
                attn_implementation=attention,
                output_loading_info=True,
                use_safetensors=True,
                **load_options,
            )
            buffers = (".position_ids", ".rotary_emb.inv_freq")
            missing = [key for key in loading.get("missing_keys", []) if not key.endswith(buffers)]
            if missing or any(
                loading.get(key) for key in ("unexpected_keys", "mismatched_keys", "error_msgs")
            ):
                raise VLAError(
                    "Patho-R1 checkpoint does not completely match the native Qwen2.5-VL architecture; refusing partially initialized weights"
                )
            self.model.eval()
        except Exception as exc:
            self.close()
            kind = type(exc).__name__
            if kind in {"GatedRepoError", "RepositoryNotFoundError"} or "403" in str(exc):
                raise AccessError(
                    "Patho-R1 checkpoint access failed. Check your existing Hugging Face authorization and the model-card access terms."
                ) from exc
            raise VLAError(f"Could not load Patho-R1 checkpoint ({kind}): {exc}") from exc
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        resolved = getattr(self.model.config, "_commit_hash", None)
        self.metadata = {
            "model": spec.model,
            "checkpoint": str(checkpoint),
            "requested_revision": spec.revision,
            "resolved_revision": resolved
            or (str(Path(checkpoint).resolve()) if Path(checkpoint).is_dir() else spec.revision),
            "device": self.device,
            "precision": str(dtype),
            "dependencies": {
                name: _version(name)
                for name in ("torch", "transformers", "accelerate", "qwen-vl-utils", "Pillow")
            },
            "preprocessing": {
                "processor": "native Qwen2.5-VL AutoProcessor",
                "min_pixels": getattr(
                    getattr(self.processor, "image_processor", None),
                    "min_pixels",
                    processor_options.get("min_pixels"),
                ),
                "max_pixels": getattr(
                    getattr(self.processor, "image_processor", None),
                    "max_pixels",
                    processor_options.get("max_pixels"),
                ),
                "system_prompt": SYSTEM_PROMPT,
                "attn_implementation": attention,
            },
        }

    def _device(self, requested):
        requested = str(requested)
        if requested == "auto":
            if self.torch.cuda.is_available():
                return "cuda:0"
            mps = getattr(self.torch.backends, "mps", None)
            return "mps" if mps is not None and mps.is_available() else "cpu"
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            raise VLAError("CUDA was requested but is unavailable in the Patho-R1 environment")
        if requested == "mps":
            mps = getattr(self.torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise VLAError("MPS was requested but is unavailable in the Patho-R1 environment")
        return requested

    def _dtype(self, precision):
        name = str(precision).lower()
        if name == "auto":
            if self.device.startswith("cuda"):
                return (
                    self.torch.bfloat16
                    if self.torch.cuda.is_bf16_supported()
                    else self.torch.float16
                )
            return self.torch.float16 if self.device == "mps" else self.torch.float32
        aliases = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
        name = aliases.get(name, name)
        if name not in {"float16", "bfloat16", "float32"}:
            raise VLAError("precision must be auto, fp16, bf16, or fp32")
        return getattr(self.torch, name)

    def generate(self, images, prompt):
        if self.model is None:
            raise VLAError("This Patho-R1 backend is closed")
        if not isinstance(prompt, str) or not prompt.strip():
            raise VLAError("prompt must be a non-empty string")
        results = []
        for image in images:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.convert("RGB")},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            generation = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.do_sample,
            }
            if self.config.do_sample:
                generation["temperature"] = self.config.temperature
            # Seeding is process-local; an independent conversation is built for every patch.
            self.torch.manual_seed(self.config.seed)
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **generation)
            sequences = getattr(output, "sequences", output)
            new_ids = [row[len(source) :] for source, row in zip(inputs.input_ids, sequences)]
            raw = self.processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            eos = getattr(self.model.generation_config, "eos_token_id", None)
            if eos is None:
                eos = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
            count = len(new_ids[0])
            ended = bool(count and int(new_ids[0][-1]) in eos_ids)
            truncated = count >= self.config.max_new_tokens and not ended
            answer = extract_final_answer(raw)
            reason = "length" if truncated else "stop"
            if not answer:
                reason = "no_final_answer"
            results.append(GenerationResult(text=answer, truncated=truncated, finish_reason=reason))
        return results

    def close(self):
        self.model = None
        self.processor = None
        gc.collect()
        torch = getattr(self, "torch", None)
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
