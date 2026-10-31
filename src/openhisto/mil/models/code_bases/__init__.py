"""Pinned MIL implementations with isolated imports and CPU adapters."""

from .registry import create_model, list_models, model_info

__all__ = ["create_model", "list_models", "model_info", "MILOutput", "MILAdapter"]


def __getattr__(name):
    if name in {"MILOutput", "MILAdapter"}:
        from . import base

        return getattr(base, name)
    raise AttributeError(name)
