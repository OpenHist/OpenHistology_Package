"""Reference- and model-based stain transfer for pathology patches."""

from .config import DomainSpec, TransferConfig, TrainConfig, TransferReport, TrainReport
from .normalizer import create_normalizer, list_methods


def normalize(inputs, output, domain, config=None):
    """Normalize an image, directory, manifest, or native patch archive."""
    from .pipeline import normalize as run

    return run(inputs, output, domain, config)


def train(config):
    """Train CycleGAN or pix2pix; Torch is imported only on use."""
    from .training import train as run

    if isinstance(config, dict):
        config = TrainConfig(**config)
    return run(config)


__all__ = [
    "DomainSpec",
    "TransferConfig",
    "TrainConfig",
    "TransferReport",
    "TrainReport",
    "create_normalizer",
    "normalize",
    "train",
    "list_methods",
]
