"""Native LLaVA adapters; install each legacy runtime in its own environment."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from ..base import GenerationResult, VLAError


_PRETRAINED_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "model*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model.bin.index.json",
]


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _descriptor(spec):
    from ..registry import get_descriptor

    return get_descriptor(spec.model)


def _field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _snapshot(checkpoint, config, revision=None, patterns=None):
    """Resolve remote assets before native code runs, including offline operation."""
    path = Path(str(checkpoint)).expanduser()
    if path.is_dir():
        return str(path.resolve()), None
    if path.is_file():
        raise VLAError(f"Expected a pretrained model directory, received file: {path}")
    try:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(
            repo_id=str(checkpoint),
            revision=revision,
            cache_dir=getattr(config, "cache_dir", None),
            local_files_only=getattr(config, "offline", False),
            allow_patterns=patterns if patterns is not None else _PRETRAINED_FILES,
        )
    except Exception as exc:
        raise VLAError(f"Cannot resolve checkpoint {checkpoint!r}: {exc}") from exc
    resolved_path = Path(resolved)
    sha = resolved_path.name if resolved_path.parent.name == "snapshots" else revision
    return str(resolved_path.resolve()), sha


def _upstream_path(options, module_name):
    """An explicit checkout must actually supply the imported implementation."""
    path = options.get("upstream_path")
    if not path:
        return None
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise VLAError(f"upstream_path is not a directory: {directory}")
    candidates = [directory, directory / "xtuner_add"]
    for candidate in candidates:
        if (candidate / module_name).is_dir():
            sys.path.insert(0, str(candidate))
            return candidate
    raise VLAError(f"upstream_path must contain the native {module_name!r} package: {directory}")


def _check_module_origin(module, directory):
    if directory is None:
        return
    origin = getattr(module, "__file__", None)
    if not origin or not Path(origin).resolve().is_relative_to(directory):
        raise VLAError(
            f"Another {module.__name__} is already imported. Use a fresh isolated backend worker."
        )


def _code_revision(module, distribution=None):
    origin = getattr(module, "__file__", None)
    origin = Path(origin).resolve() if origin else None
    if distribution and origin is not None:
        try:
            installed = importlib.metadata.distribution(distribution)
            # An explicitly selected source checkout can coexist with an
            # installed distribution. Use its metadata only for its own files.
            owns_module = any(
                Path(installed.locate_file(file)).resolve() == origin
                for file in installed.files or ()
            )
            if owns_module:
                value = installed.read_text("direct_url.json")
                commit = json.loads(value or "{}").get("vcs_info", {}).get("commit_id")
                if commit:
                    return commit
        except (importlib.metadata.PackageNotFoundError, ValueError, OSError):
            pass
    if origin:
        try:
            # Git searches parent directories. A module in a repo-local virtual
            # environment must not inherit the application's unrelated commit.
            tracked = subprocess.run(
                ["git", "-C", str(origin.parent), "ls-files", "--error-unmatch", "--", str(origin)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if tracked.returncode != 0:
                return None
            result = subprocess.run(
                ["git", "-C", str(origin.parent), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def _learned_loading_errors(info):
    # Older Transformers versions persist deterministic positional buffers.
    buffers = (".position_ids", ".rotary_emb.inv_freq")
    missing = [key for key in info.get("missing_keys", []) if not key.endswith(buffers)]
    return {
        key: value
        for key, value in {
            "missing_keys": missing,
            "unexpected_keys": info.get("unexpected_keys", []),
            "mismatched_keys": info.get("mismatched_keys", []),
            "error_msgs": info.get("error_msgs", []),
        }.items()
        if value
    }


_VISION_PREFIX = "model.vision_tower.vision_tower."


def _saved_vision_state(model_path, torch):
    """Read the released tower weights that native delayed loading skips.

    LLaVA's language-model constructor creates only a vision config, so these
    checkpoint tensors appear as unexpected keys during its initial HF load.
    Load only their shards/tensors and restore them once the native tower exists.
    """
    root = Path(model_path)
    weight_map = None
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (root / name).is_file():
            weight_map = json.loads((root / name).read_text())["weight_map"]
            break
    if weight_map is not None:
        names = {key: value for key, value in weight_map.items() if key.startswith(_VISION_PREFIX)}
        shards = sorted(set(names.values()))
    else:
        names = None
        shards = [
            name for name in ("model.safetensors", "pytorch_model.bin") if (root / name).is_file()
        ][:1]
    state = {}
    for filename in shards:
        path = root / filename
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = names if names is not None else handle.keys()
                for key in keys:
                    if key.startswith(_VISION_PREFIX) and (names is None or names[key] == filename):
                        state[key[len(_VISION_PREFIX) :]] = handle.get_tensor(key)
        else:
            weights = torch.load(str(path), map_location="cpu", weights_only=True)
            for key, value in weights.items():
                if key.startswith(_VISION_PREFIX):
                    state[key[len(_VISION_PREFIX) :]] = value
            del weights
    if weight_map is not None and len(state) != len(names):
        raise VLAError("The checkpoint's saved LLaVA vision tower is incomplete.")
    return state


class LegacyLlavaBackend:
    """Load native LLaVA weights without rewriting their downloaded config."""

    kind = None

    def __init__(self, spec, config):
        self.spec, self.config = spec, config
        self.model = self.tokenizer = self.image_processor = None
        options = dict(spec.options or {})
        unsupported = set(options) - {"upstream_path", "vision_checkpoint", "conversation_mode"}
        if unsupported:
            raise VLAError(f"Unsupported {self.kind} options: {', '.join(sorted(unsupported))}")
        if (
            options.get("vision_checkpoint")
            and not Path(options["vision_checkpoint"]).expanduser().is_dir()
        ):
            raise VLAError(
                "options.vision_checkpoint must be a local native vision model directory; omit it to use the pinned published encoder."
            )
        upstream = _upstream_path(options, "llava")
        try:
            import torch
            from transformers import AutoTokenizer

            llava = importlib.import_module("llava")
            _check_module_origin(llava, upstream)
            native = importlib.import_module("llava.model.language_model.llava_llama")
            self.mm = importlib.import_module("llava.mm_utils")
            self.constants = importlib.import_module("llava.constants")
            self.conversation = importlib.import_module("llava.conversation")
        except (ImportError, AttributeError) as exc:
            raise VLAError(
                f"{self.kind} needs its native LLaVA runtime in a separate Python environment; "
                f"install requirements-{self.kind}.txt and the documented upstream checkout: {exc}"
            ) from exc
        self.torch = torch
        device = str(config.device)
        self.device = "cuda" if device == "auto" else device
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise VLAError(
                f"{self.kind} native inference requires an available CUDA GPU; select a CUDA backend Python."
            )
        precision = str(config.precision)
        if precision == "auto":
            precision = "float16"
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if precision not in dtypes:
            raise VLAError(
                f"Unsupported {self.kind} precision: {precision}; use float16, bfloat16, or float32."
            )
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise VLAError(
                "The selected CUDA GPU does not support bfloat16; use precision='float16'."
            )
        self.dtype = dtypes[precision]
        descriptor = _descriptor(spec)
        checkpoint = spec.checkpoint or _field(descriptor, "checkpoint")
        revision = spec.revision if spec.checkpoint else _field(descriptor, "checkpoint_revision")
        if not checkpoint:
            raise VLAError(f"No checkpoint configured for {spec.model}.")
        model_path, resolved_revision = _snapshot(checkpoint, config, revision)
        config_file = Path(model_path) / "config.json"
        try:
            raw_config = json.loads(config_file.read_text())
        except (OSError, ValueError) as exc:
            raise VLAError(f"Cannot read native LLaVA config at {config_file}: {exc}") from exc
        expected_type = "llava_llama" if self.kind == "pathgen" else "llava"
        if raw_config.get("model_type") != expected_type:
            raise VLAError(
                f"{self.kind} requires model_type={expected_type!r}; this checkpoint has {raw_config.get('model_type')!r}."
            )
        native_config = native.LlavaConfig.from_pretrained(model_path, local_files_only=True)
        vision_id = options.get("vision_checkpoint") or raw_config.get("mm_vision_tower")
        if not vision_id:
            raise VLAError(
                "LLaVA checkpoint config has no vision tower; set options.vision_checkpoint."
            )
        vision_pin = {
            "jamessyx/pathgenclip-vit-large-patch14-hf": "33af8295d6e42422f027a61bd1d960ab258488f7",
            "openai/clip-vit-large-patch14-336": "ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        }.get(str(vision_id))
        vision_path, vision_revision = _snapshot(vision_id, config, vision_pin)
        # Native LLaVA accepts existing local paths; arbitrary Hub namespaces
        # are rejected by its vision builder. The override stays in memory.
        native_config.mm_vision_tower = vision_path
        self.conversation_mode = options.get("conversation_mode", "llava_v1")
        if self.conversation_mode not in self.conversation.conv_templates:
            raise VLAError(f"Unknown native LLaVA conversation template: {self.conversation_mode}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, local_files_only=True
            )
            loaded = native.LlavaLlamaForCausalLM.from_pretrained(
                model_path,
                config=native_config,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map={"": self.device},
                local_files_only=True,
                output_loading_info=True,
            )
            self.model, loading_info = loaded
            if getattr(self.model.config, "mm_use_im_patch_token", False):
                self.tokenizer.add_tokens(
                    [self.constants.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True
                )
            if getattr(self.model.config, "mm_use_im_start_end", False):
                self.tokenizer.add_tokens(
                    [self.constants.DEFAULT_IM_START_TOKEN, self.constants.DEFAULT_IM_END_TOKEN],
                    special_tokens=True,
                )
            if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
                raise VLAError(
                    "Checkpoint tokenizer and learned embeddings differ in size; refusing to initialize random token embeddings."
                )
            vision = self.model.get_vision_tower()
            if not vision.is_loaded:
                vision.load_model()
            saved_vision = _saved_vision_state(model_path, torch)
            if saved_vision:
                restored = vision.vision_tower.load_state_dict(saved_vision, strict=False)
                vision_errors = _learned_loading_errors(
                    {
                        "missing_keys": restored.missing_keys,
                        "unexpected_keys": restored.unexpected_keys,
                    }
                )
                if vision_errors:
                    raise VLAError(
                        f"Saved native LLaVA vision weights did not load completely: {vision_errors}"
                    )
                loading_info = dict(loading_info)
                loading_info["unexpected_keys"] = [
                    key
                    for key in loading_info.get("unexpected_keys", [])
                    if not key.startswith(_VISION_PREFIX)
                ]
            errors = _learned_loading_errors(loading_info)
            if errors:
                raise VLAError(f"Native LLaVA checkpoint did not load completely: {errors}")
            vision.to(device=self.device, dtype=self.dtype)
            self.image_processor = vision.image_processor
            self.model.eval()
        except VLAError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise VLAError(
                f"Failed loading native {self.kind} model and vision tower: {exc}"
            ) from exc
        self.metadata = {
            "backend": self.kind,
            "checkpoint": str(checkpoint),
            "resolved_revision": resolved_revision,
            "requested_revision": revision,
            "source_revision": _code_revision(native, "llava"),
            "expected_source_revision": _field(descriptor, "source_revision"),
            "upstream_module": getattr(native, "__file__", None),
            "vision_checkpoint": str(vision_id),
            "vision_revision": vision_revision,
            "vision_weights": "released multimodal checkpoint"
            if saved_vision
            else "configured native vision checkpoint",
            "device": self.device,
            "precision": precision,
            "conversation_mode": self.conversation_mode,
            "preprocessing": {
                "implementation": "llava.mm_utils.process_images",
                "image_aspect_ratio": getattr(native_config, "image_aspect_ratio", None),
            },
            "libraries": {
                name: _version(name) for name in ("torch", "transformers", "llava", "accelerate")
            },
        }

    def generate(self, images, prompt):
        if self.model is None:
            raise VLAError("The LLaVA backend is closed.")
        if self.constants.DEFAULT_IMAGE_TOKEN in prompt:
            raise VLAError(
                "LLaVA inserts its image token automatically; remove it from the prompt."
            )
        results = []
        for image in images:
            if getattr(self.config, "seed", None) is not None:
                self.torch.manual_seed(self.config.seed)
            conversation = self.conversation.conv_templates[self.conversation_mode].copy()
            token = self.constants.DEFAULT_IMAGE_TOKEN
            if getattr(self.model.config, "mm_use_im_start_end", False):
                token = (
                    self.constants.DEFAULT_IM_START_TOKEN
                    + token
                    + self.constants.DEFAULT_IM_END_TOKEN
                )
            conversation.append_message(conversation.roles[0], token + "\n" + prompt)
            conversation.append_message(conversation.roles[1], None)
            input_ids = (
                self.mm.tokenizer_image_token(
                    conversation.get_prompt(),
                    self.tokenizer,
                    self.constants.IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.device)
            )
            pixels = self.mm.process_images(
                [image.convert("RGB")], self.image_processor, self.model.config
            )
            if isinstance(pixels, list):
                pixels = [value.to(device=self.device, dtype=self.dtype) for value in pixels]
            else:
                pixels = pixels.to(device=self.device, dtype=self.dtype)
            stop = conversation.sep
            if conversation.sep_style == self.conversation.SeparatorStyle.TWO:
                stop = conversation.sep2
            kwargs = {
                "images": pixels,
                "do_sample": self.config.do_sample,
                "max_new_tokens": self.config.max_new_tokens,
                "use_cache": True,
                "stopping_criteria": [
                    self.mm.KeywordsStoppingCriteria([stop], self.tokenizer, input_ids)
                ],
            }
            if self.config.do_sample:
                kwargs["temperature"] = self.config.temperature
            if self.kind == "pathgen":
                kwargs["image_sizes"] = [image.size]
            with self.torch.inference_mode():
                output = self.model.generate(input_ids, **kwargs)
            tokens = output[0]
            input_length = input_ids.shape[1]
            if len(tokens) >= input_length and self.torch.equal(
                tokens[:input_length], input_ids[0]
            ):
                tokens = tokens[input_length:]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if stop and text.endswith(stop):
                text = text[: -len(stop)].strip()
            eos = self.tokenizer.eos_token_id
            truncated = len(tokens) >= self.config.max_new_tokens and (
                not len(tokens) or int(tokens[-1]) != eos
            )
            results.append(
                GenerationResult(
                    text=text, truncated=truncated, finish_reason="length" if truncated else "stop"
                )
            )
        return results

    def close(self):
        self.model = self.tokenizer = self.image_processor = None
        torch = getattr(self, "torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class PathGenBackend(LegacyLlavaBackend):
    kind = "pathgen"


class QuiltBackend(LegacyLlavaBackend):
    kind = "quilt"
