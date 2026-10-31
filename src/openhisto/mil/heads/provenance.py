"""Deterministic tensor fingerprints for fitted head/backbone pairing."""

import hashlib
import json


def state_fingerprint(module):
    """Hash parameter/buffer bytes without retaining another full model copy."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise TypeError("Backbone fingerprinting requires a dense tensor-only state dictionary")
        values = tensor.detach().cpu().contiguous()
        digest.update(
            json.dumps(
                [name, str(values.dtype), list(values.shape)], separators=(",", ":")
            ).encode()
        )
        octets = values.reshape(-1).view(torch.uint8).numpy()
        for start in range(0, len(octets), 1024 * 1024):
            digest.update(octets[start : start + 1024 * 1024].tobytes())
    return digest.hexdigest()
