"""Small adapter contract; heavy model dependencies are imported lazily."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np


class EncoderError(RuntimeError):
    """An actionable, credential-free adapter error."""


class AccessError(EncoderError):
    pass


@contextmanager
def translated_errors(message, error=EncoderError):
    """Report third-party failures as ``error(message)``; EncoderErrors pass through.

    The original exception is not chained, so vendor text that may contain
    credentials or local paths never reaches saved reports.
    """
    try:
        yield
    except EncoderError:
        raise
    except Exception:
        raise error(message) from None


@dataclass(frozen=True)
class ModelDescriptor:
    model_id: str
    display_name: str
    provider: str
    access: str
    dimension: int
    model_name: str
    license_url: str = ""
    approval_url: str = ""
    auth_env: str = ""
    dependencies: tuple[str, ...] = ()
    preprocessing: dict = field(default_factory=dict)
    dimensions: tuple[int, ...] = ()

    def to_dict(self):
        return asdict(self)


class Encoder(Protocol):
    dimension: int
    metadata: dict
    usage: dict

    def encode(self, images: list[np.ndarray]) -> np.ndarray: ...
    def close(self) -> None: ...


def validate_features(features, count, dimension):
    array = np.asarray(features)
    if array.shape != (count, dimension):
        raise EncoderError(f"Encoder returned shape {array.shape}; expected ({count}, {dimension})")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise EncoderError("Encoder returned nonnumeric or nonfinite features")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise EncoderError("Encoder output cannot be represented as finite float32 features")
    return result
