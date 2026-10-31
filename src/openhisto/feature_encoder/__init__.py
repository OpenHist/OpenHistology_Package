"""Incremental multi-model and multi-magnification patch feature encoding."""

from .auth import AuthConfig
from .base import AccessError, EncoderError, ModelDescriptor
from .config import EncodeConfig, EncoderSpec, EncodingReport, SourceSpec
from .pipeline import EncodingFailure, encode_slide
from .registry import (
    check_access,
    get_descriptor,
    list_models,
    model_info,
    prepare_model,
    register_encoder,
)
from .storage import FeatureReader

__all__ = [
    "AuthConfig",
    "AccessError",
    "EncoderError",
    "EncodingFailure",
    "ModelDescriptor",
    "EncodeConfig",
    "EncoderSpec",
    "EncodingReport",
    "SourceSpec",
    "encode_slide",
    "check_access",
    "get_descriptor",
    "list_models",
    "model_info",
    "prepare_model",
    "register_encoder",
    "FeatureReader",
]
