"""Task heads attached through explicit native MIL representations."""

from .types import ClassificationOutput, SurvivalOutput, TaskOutput


def attach_head(*args, **kwargs):
    from .attachment import attach_head as attach

    return attach(*args, **kwargs)


def extract_slide_embeddings(*args, **kwargs):
    from .attachment import extract_slide_embeddings as extract

    return extract(*args, **kwargs)


def get_representation_spec(*args, **kwargs):
    from .bindings import get_representation_spec as spec

    return spec(*args, **kwargs)


def representation_dimension(*args, **kwargs):
    from .bindings import representation_dimension as dimension

    return dimension(*args, **kwargs)


def save_head(*args, **kwargs):
    from .persistence import save_head as save

    return save(*args, **kwargs)


def load_head(*args, **kwargs):
    from .persistence import load_head as load

    return load(*args, **kwargs)


__all__ = [
    "attach_head",
    "extract_slide_embeddings",
    "get_representation_spec",
    "representation_dimension",
    "save_head",
    "load_head",
    "TaskOutput",
    "ClassificationOutput",
    "SurvivalOutput",
]
