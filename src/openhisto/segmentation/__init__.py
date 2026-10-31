"""Reusable binary/multiclass semantic segmentation with optional model backends."""

from .config import (
    MaskSpec,
    Sample,
    LoadedSample,
    ModelConfig,
    TrainConfig,
    PredictConfig,
    Prediction,
    TrainingResult,
    CapabilityError,
)


def list_models():
    from .semantic.registry import list_models as implementation

    return implementation()


def model_info(name):
    from .semantic.registry import model_info as implementation

    return implementation(name)


def prepare_model(*args, **kwargs):
    from .engine import prepare_model as implementation

    return implementation(*args, **kwargs)


def load_model(*args, **kwargs):
    from .engine import load_model as implementation

    return implementation(*args, **kwargs)


def prepare_dataset(*args, **kwargs):
    from .data import prepare_dataset as implementation

    return implementation(*args, **kwargs)


def train(*args, **kwargs):
    from .engine import train as implementation

    return implementation(*args, **kwargs)


def evaluate(*args, **kwargs):
    from .engine import evaluate as implementation

    return implementation(*args, **kwargs)


def predict(*args, **kwargs):
    from .engine import predict as implementation

    return implementation(*args, **kwargs)


def export_prediction(*args, **kwargs):
    from .exports import export_prediction as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "MaskSpec",
    "Sample",
    "LoadedSample",
    "ModelConfig",
    "TrainConfig",
    "PredictConfig",
    "Prediction",
    "TrainingResult",
    "CapabilityError",
    "list_models",
    "model_info",
    "prepare_model",
    "load_model",
    "prepare_dataset",
    "train",
    "evaluate",
    "predict",
    "export_prediction",
]
