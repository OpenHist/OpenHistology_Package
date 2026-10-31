"""Prompted pathology patch descriptions with optional isolated model runtimes."""

from .base import AccessError, GenerationResult, ModelDescriptor, VLAError, WorkerError
from .config import DescribeConfig, DescriptionReport, ModelSpec, SourceSpec


def describe_slide(*args, **kwargs):
    from .pipeline import describe_slide as implementation

    return implementation(*args, **kwargs)


def list_models():
    from .registry import list_models as implementation

    return implementation()


def model_info(model):
    from .registry import model_info as implementation

    return implementation(model)


def prepare_model(*args, **kwargs):
    from .registry import prepare_model as implementation

    return implementation(*args, **kwargs)


def register_model(*args, **kwargs):
    from .registry import register_model as implementation

    return implementation(*args, **kwargs)


def __getattr__(name):
    if name == "DescriptionReader":
        from .storage import DescriptionReader

        return DescriptionReader
    if name == "DescriptionFailure":
        from .pipeline import DescriptionFailure

        return DescriptionFailure
    raise AttributeError(name)


__all__ = [
    "SourceSpec",
    "ModelSpec",
    "DescribeConfig",
    "DescriptionReport",
    "GenerationResult",
    "ModelDescriptor",
    "VLAError",
    "WorkerError",
    "AccessError",
    "DescriptionFailure",
    "DescriptionReader",
    "describe_slide",
    "list_models",
    "model_info",
    "prepare_model",
    "register_model",
]
