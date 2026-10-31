"""Lightweight model catalogue and lazy adapter factories."""

import importlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

from .auth import AuthConfig
from .base import EncoderError, ModelDescriptor
from .config import EncodeConfig, EncoderSpec, json_safe, reject_secrets


_REGISTRY = {}


def register_encoder(descriptor: ModelDescriptor, factory):
    """Register an adapter factory; no model is initialized by registration."""
    if not isinstance(descriptor, ModelDescriptor) or not callable(factory):
        raise TypeError("register_encoder requires a ModelDescriptor and callable factory")
    if not descriptor.model_id or descriptor.dimension < 1:
        raise ValueError("Encoder profiles require an ID and positive dimension")
    if descriptor.model_id in _REGISTRY:
        raise ValueError("An encoder with this profile ID is already registered")
    reject_secrets(descriptor.to_dict())
    json.dumps(json_safe(descriptor.to_dict()), allow_nan=False)
    _REGISTRY[descriptor.model_id] = (descriptor, factory)


def _lazy(module, name):
    def factory(spec, *, config, auth):
        cls = getattr(importlib.import_module(module, __package__), name)
        return cls(spec, config=config, auth=auth)

    return factory


def _builtin_profiles():
    for descriptor, module, name in (
        (
            ModelDescriptor(
                "phikon",
                "Phikon",
                "hf",
                "public",
                768,
                "owkin/phikon",
                license_url="https://huggingface.co/owkin/phikon",
                auth_env="HF_TOKEN",
                dependencies=("torch", "transformers", "huggingface_hub", "PIL"),
                preprocessing={
                    "processor": "checkpoint AutoImageProcessor",
                    "input_size": 224,
                    "pooling": "cls_token",
                    "default_revision": "057cc0295895c2df3dd7681a89680da6015cbefe",
                },
            ),
            ".public.phikon",
            "PhikonEncoder",
        ),
        (
            ModelDescriptor(
                "phikon_v2",
                "Phikon-v2",
                "hf",
                "public",
                1024,
                "owkin/phikon-v2",
                license_url="https://huggingface.co/owkin/phikon-v2",
                auth_env="HF_TOKEN",
                dependencies=("torch", "transformers", "huggingface_hub", "PIL"),
                preprocessing={
                    "processor": "checkpoint AutoImageProcessor",
                    "input_size": 224,
                    "pooling": "cls_token",
                    "default_revision": "2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd",
                },
            ),
            ".public.phikon",
            "PhikonEncoder",
        ),
        (
            ModelDescriptor(
                "uni_v1",
                "UNI v1",
                "hf",
                "gated",
                1024,
                "MahmoodLab/UNI",
                license_url="https://huggingface.co/MahmoodLab/UNI",
                approval_url="https://huggingface.co/MahmoodLab/UNI",
                auth_env="HF_TOKEN",
                dependencies=("torch", "torchvision", "timm", "huggingface_hub", "PIL"),
                preprocessing={
                    "processor": "checkpoint timm pretrained_cfg",
                    "input_size": 224,
                    "pooling": "cls_token",
                    "default_revision": "b55a5ec6cade1a39edfe6534189a9b8ca7a022f0",
                },
            ),
            ".locked.HF.local",
            "UNIEncoder",
        ),
        (
            ModelDescriptor(
                "conch15_titan",
                "CONCH v1.5 (TITAN patch representation)",
                "hf",
                "gated",
                768,
                "MahmoodLab/TITAN",
                license_url="https://huggingface.co/MahmoodLab/TITAN",
                approval_url="https://huggingface.co/MahmoodLab/TITAN",
                auth_env="HF_TOKEN",
                dependencies=(
                    "torch",
                    "torchvision",
                    "timm",
                    "einops",
                    "einops_exts",
                    "huggingface_hub",
                    "PIL",
                ),
                preprocessing={
                    "processor": "TITAN CONCH v1.5 resize/center crop",
                    "input_size": 448,
                    "source_patch_size": 512,
                    "source_mag": 20,
                    "pooling": "attentional_pooler",
                    "default_revision": "dac6773d9961cfc75503440676ff157a2c6e8d2e",
                },
            ),
            ".locked.HF.local",
            "Conch15Encoder",
        ),
        (
            ModelDescriptor(
                "gemini_embedding_2",
                "Gemini Embedding 2",
                "google",
                "api",
                3072,
                "gemini-embedding-2",
                license_url="https://ai.google.dev/gemini-api/terms",
                approval_url="https://aistudio.google.com/api-keys",
                auth_env="GEMINI_API_KEY",
                dependencies=("google.genai", "PIL"),
                preprocessing={
                    "input": "one RGB image per embedding",
                    "dimension_range": [128, 3072],
                },
                dimensions=(768, 1536, 3072),
            ),
            ".non_path.gemini",
            "GeminiEncoder",
        ),
        (
            ModelDescriptor(
                "cohere_embed_5_pro",
                "Cohere Embed v5 Pro",
                "cohere",
                "api",
                2048,
                "embed-v5.0-pro",
                license_url="https://cohere.com/terms-of-use",
                approval_url="https://dashboard.cohere.com/api-keys",
                auth_env="COHERE_API_KEY",
                dependencies=("cohere", "PIL"),
                preprocessing={"input": "one RGB image per embedding"},
                dimensions=(256, 512, 768, 1024, 1536, 2048),
            ),
            ".non_path.cohere",
            "CohereEncoder",
        ),
        (
            ModelDescriptor(
                "cohere_embed_5_fast",
                "Cohere Embed v5 Fast",
                "cohere",
                "api",
                2048,
                "embed-v5.0-fast",
                license_url="https://cohere.com/terms-of-use",
                approval_url="https://dashboard.cohere.com/api-keys",
                auth_env="COHERE_API_KEY",
                dependencies=("cohere", "PIL"),
                preprocessing={"input": "one RGB image per embedding"},
                dimensions=(256, 512, 768, 1024, 1536, 2048),
            ),
            ".non_path.cohere",
            "CohereEncoder",
        ),
        (
            ModelDescriptor(
                "nova_multimodal",
                "Amazon Nova Multimodal Embeddings",
                "aws",
                "api",
                3072,
                "amazon.nova-2-multimodal-embeddings-v1:0",
                license_url="https://aws.amazon.com/service-terms/",
                approval_url="https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html",
                dependencies=("boto3", "PIL"),
                preprocessing={
                    "input": "one RGB image per embedding",
                    "auth": "AWS standard credential chain",
                },
                dimensions=(256, 384, 1024, 3072),
            ),
            ".non_path.nova",
            "NovaEncoder",
        ),
    ):
        register_encoder(descriptor, _lazy(module, name))


def get_descriptor(model) -> ModelDescriptor:
    name = model.model if isinstance(model, EncoderSpec) else model
    try:
        return _REGISTRY[name][0]
    except (KeyError, TypeError):
        raise ValueError(
            "Unknown encoder profile; use list_models() to see available IDs"
        ) from None


def _dependency_available(name):
    if "." in name:
        distribution = {"google.genai": "google-genai"}.get(name)
        if distribution:
            try:
                importlib.metadata.distribution(distribution)
                return True
            except importlib.metadata.PackageNotFoundError:
                return False
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _cache_status(descriptor, *, revision=None, cache_dir=None):
    """Inspect HF cache filenames only: no Hub imports, token reads, or HTTP."""
    requirements = {
        "phikon": (
            ("config.json",),
            ("preprocessor_config.json",),
            ("model.safetensors", "pytorch_model.bin"),
        ),
        "phikon_v2": (
            ("config.json",),
            ("preprocessor_config.json",),
            ("model.safetensors", "pytorch_model.bin"),
        ),
        "uni_v1": (("config.json",), ("model.safetensors", "pytorch_model.bin")),
        "conch15_titan": (("config.json",), ("conch_v1_5.py",), ("conch_v1_5_pytorch_model.bin",)),
    }.get(descriptor.model_id)
    if descriptor.provider != "hf" or requirements is None:
        return {"status": "not_applicable", "verified": False}
    home = Path(
        os.environ.get("HF_HOME")
        or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
    ).expanduser()
    root = Path(
        cache_dir
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or home / "hub"
    ).expanduser()
    repo = root / ("models--" + descriptor.model_name.replace("/", "--"))
    revision = revision or descriptor.preprocessing["default_revision"]
    result = {"status": "not_cached", "verified": False, "revision": revision}
    try:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            reference = repo / "refs" / revision
            if not reference.resolve().is_relative_to((repo / "refs").resolve()):
                return result
            revision = reference.read_text(encoding="utf-8").strip()
            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                return result
        directory = repo / "snapshots" / revision
        present = [
            any((directory / name).is_file() for name in alternatives)
            for alternatives in requirements
        ]
        result.update(
            revision=revision,
            status="present_unverified"
            if all(present)
            else "partial_unverified"
            if any(present)
            else "not_cached",
        )
        return result
    except (OSError, TypeError, ValueError):
        return result


def model_info(model) -> dict:
    descriptor = get_descriptor(model)
    result = json_safe(descriptor.to_dict())
    result["dependency_status"] = {
        name: _dependency_available(name) for name in descriptor.dependencies
    }
    result["credentials_checked"] = False
    result["access_verified"] = False
    result["default_revision"] = descriptor.preprocessing.get("default_revision")
    result["cache_status"] = _cache_status(descriptor)
    result["license_note"] = (
        "Access mode and license terms are separate; consult the linked model terms."
    )
    return result


def list_models() -> list[dict]:
    """Return descriptions and dependency presence without model imports or HTTP."""
    return [model_info(name) for name in _REGISTRY]


def check_access(model, auth=None, config=None) -> dict:
    """Explicit lightweight access check; never downloads weights or calls inference."""
    spec = model if isinstance(model, EncoderSpec) else EncoderSpec(model)
    descriptor = get_descriptor(spec)
    config = config or EncodeConfig()
    auth = auth or AuthConfig(prompt_credentials=config.prompt_credentials)
    result = {
        "model": spec.model,
        "provider": descriptor.provider,
        "access": descriptor.access,
        "approval_url": descriptor.approval_url,
        "license_url": descriptor.license_url,
        "verified": False,
    }
    if spec.checkpoint:
        exists = Path(spec.checkpoint).exists()
        return {
            **result,
            "status": "local_checkpoint_present" if exists else "local_checkpoint_missing",
            "available": exists,
        }
    if descriptor.provider == "aws":
        return {
            **result,
            "status": "sdk_credential_chain_unverified",
            "available": None,
            "profile": auth.aws_profile,
            "region": auth.aws_region,
        }
    if descriptor.provider not in {"hf", "google", "cohere"}:
        return {**result, "status": "custom_adapter_unverified", "available": None}
    if descriptor.provider != "hf":
        present = (
            auth.resolve(descriptor.provider, required=getattr(auth, "prompt", False)) is not None
        )
        return {
            **result,
            "status": "credentials_present_unverified" if present else "credentials_missing",
            "available": present,
            "credential_env": auth.env_name(descriptor.provider),
        }
    cached = _cache_status(descriptor, revision=spec.revision, cache_dir=config.cache_dir)
    if cached["status"] == "present_unverified":
        return {
            **result,
            "status": "checkpoint_cached_unverified",
            "available": True,
            "cache_status": cached,
        }
    if config.offline:
        return {
            **result,
            "status": "checkpoint_not_cached",
            "available": False,
            "cache_status": cached,
        }
    if descriptor.access == "public":
        return {**result, "status": "public_access_unverified", "available": True}
    token = auth.resolve("hf", required=getattr(auth, "prompt", False))
    if not token:
        return {
            **result,
            "status": "credentials_missing",
            "available": False,
            "credential_env": auth.env_name("hf"),
        }
    filename = (
        "conch_v1_5_pytorch_model.bin" if spec.model == "conch15_titan" else "pytorch_model.bin"
    )
    revision = spec.revision or descriptor.preprocessing["default_revision"]
    url = f"https://huggingface.co/{descriptor.model_name}/resolve/{quote(revision, safe='')}/{filename}"
    try:
        import requests

        response = requests.head(
            url,
            headers={"Authorization": "Bearer " + token},
            timeout=config.timeout,
            allow_redirects=False,
        )
        code = response.status_code
        response.close()
    except Exception:
        return {**result, "status": "access_check_failed", "available": None}
    status = (
        "granted"
        if code in {200, 301, 302, 303, 307, 308}
        else "authentication_failed"
        if code == 401
        else "approval_required_or_denied"
        if code == 403
        else "checkpoint_not_found"
        if code == 404
        else "access_check_failed"
    )
    return {
        **result,
        "status": status,
        "available": status == "granted",
        "verified": status == "granted",
    }


def prepare_model(spec, *, config=None, auth=None):
    spec = EncoderSpec(spec) if isinstance(spec, str) else spec
    config = EncodeConfig() if config is None else config
    if not isinstance(spec, EncoderSpec) or not isinstance(config, EncodeConfig):
        raise TypeError("prepare_model requires an EncoderSpec/profile ID and EncodeConfig")
    descriptor = get_descriptor(spec)
    dimension = spec.dimension or descriptor.dimension
    allowed = descriptor.dimensions or (descriptor.dimension,)
    if spec.model == "gemini_embedding_2":
        valid_dimension = 128 <= dimension <= 3072
    else:
        valid_dimension = dimension in allowed
    if not valid_dimension:
        raise ValueError("Requested dimension is not supported by this encoder profile")
    auth = auth or AuthConfig(prompt_credentials=config.prompt_credentials)
    encoder = _REGISTRY[spec.model][1](spec, config=config, auth=auth)
    try:
        if encoder.dimension != dimension:
            raise EncoderError("Prepared encoder dimension does not match the requested profile")
        reject_secrets(encoder.metadata)
        json.dumps(json_safe(encoder.metadata), allow_nan=False)
    except Exception:
        encoder.close()
        raise
    return encoder


_builtin_profiles()
