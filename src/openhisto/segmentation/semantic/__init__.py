"""Semantic segmentation with lazily imported model backends."""

from .registry import create_adapter, create_model, list_models, model_info, prepare_model

__all__ = ["create_adapter", "create_model", "list_models", "model_info", "prepare_model"]


def __getattr__(name):
    if name in {"SegmentationAdapter", "BaseSegAdapter"}:
        from . import base

        return getattr(base, name)
    raise AttributeError(name)
