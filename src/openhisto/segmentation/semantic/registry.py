"""Model catalogue without imports, model downloads, or credential prompts."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from ..config import ModelConfig


_PROFILES = {
    "mask2former": (
        "Mask2FormerAdapter",
        "standard",
        "transformers",
        "facebook/mask2former-swin-tiny-ade-semantic",
        False,
    ),
    "segformer": (
        "SegFormerAdapter",
        "standard",
        "transformers",
        "nvidia/segformer-b5-finetuned-ade-640-640",
        True,
    ),
    "upernet": ("UPerNetAdapter", "standard", "transformers", "openmmlab/upernet-swin-tiny", True),
    "deeplabv3": (
        "DeepLabV3Adapter",
        "standard",
        "torchvision",
        "DeepLabV3_ResNet50_Weights.DEFAULT",
        True,
    ),
    "sam2": ("SAM2Adapter", "special", "sam2", None, True),
    "segdiff": ("SegDiffAdapter", "special", "torch", None, True),
}
_ALIASES = {
    "m2f": "mask2former",
    "dlv3": "deeplabv3",
    "deep_lab_v3": "deeplabv3",
    "seg_diff": "segdiff",
}


def canonical_model(name):
    if not isinstance(name, str):
        raise ValueError("Model name must be text")
    key = name.lower().replace("-", "").replace(" ", "")
    key = _ALIASES.get(key, key)
    if key not in _PROFILES:
        raise ValueError(f"Unknown segmentation model {name!r}; choose from {sorted(_PROFILES)}")
    return key


def model_info(name):
    name = canonical_model(name)
    _, _, dependency, default, logits = _PROFILES[name]
    result = {
        "name": name,
        "task": "semantic_segmentation",
        "dependency": dependency,
        "dependency_present": find_spec(dependency) is not None,
        "default_pretrained": default,
        "supports_logits": logits,
        "score_semantics": "summed_query_class_mask_scores"
        if name == "mask2former"
        else "class_logits",
        "requires_prompts": name == "sam2",
    }
    if name == "sam2":
        result.update(
            num_classes=2,
            requires_prompts_at_inference=True,
            score_semantics="best_prompt_mask_foreground_sigmoid_merged_by_maximum",
        )
    if name == "segdiff":
        result.update(
            requires_embeddings=True,
            score_semantics="final_diffusion_logits_conditional_on_patch_embedding",
        )
    if name == "upernet":
        result["default_initialization"] = "random_convnext"
    return result


def list_models():
    return [model_info(name) for name in _PROFILES]


def create_adapter(config, device="cpu", load_dir=None):
    if isinstance(config, str):
        config = ModelConfig(model=config)
    elif isinstance(config, dict):
        config = ModelConfig(**config)
    if not isinstance(config, ModelConfig):
        raise TypeError("Expected ModelConfig, configuration dictionary, or model name")
    name = canonical_model(config.model)
    config = replace(config, model=name)
    cls_name, module, _, _, _ = _PROFILES[name]
    cls = getattr(import_module(f".{module}", __package__), cls_name)
    return cls(config, device=device, load_dir=load_dir)


def create_model(config, device="cpu", load_dir=None):
    """Alias returning the complete trainable adapter, including preprocessing."""
    return create_adapter(config, device=device, load_dir=load_dir)


def prepare_model(config, cache_dir=None, *, device="cpu"):
    """Explicitly construct/download a backend, optionally saving backend files.

    This lower-level function returns the prepared adapter. Use the public bundle
    saver when a self-describing bundle is needed for later load_model calls.
    """
    adapter = create_adapter(config, device=device)
    if cache_dir is not None:
        adapter.save(Path(cache_dir))
    return adapter
