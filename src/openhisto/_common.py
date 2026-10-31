"""Helpers shared by the OpenHisto subpackages.

Standard library only: isolated model-worker interpreters, which have none of
the scientific dependencies installed, import this module.
"""

import hashlib
import json
import os
from pathlib import Path
import re


# The directory that contains the ``openhisto`` package (``src/`` in a checkout).
IMPORT_ROOT = Path(__file__).resolve().parent.parent

_SECRET_KEYS = frozenset(
    {
        "token",
        "hftoken",
        "apikey",
        "apitoken",
        "accesstoken",
        "authorization",
        "password",
        "secret",
        "credentials",
        "awssecretaccesskey",
        "awsaccesskeyid",
        "awssessiontoken",
        "xapikey",
        "xgoogapikey",
        "geminiapikey",
        "cohereapikey",
        "clientsecret",
        "secretkey",
        "bearertoken",
        "refreshtoken",
    }
)


def json_safe(value):
    """Convert paths, tuples and NumPy/Torch values to plain JSON types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if getattr(value, "ndim", 0) and hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def fingerprint(value):
    """SHA-256 of canonical JSON.

    Saved run identities are compared by these exact bytes, so the encoding
    (sorted keys, compact separators, ASCII escapes) must never change.
    """
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def reject_secrets(value, message):
    """Raise ``ValueError(message)`` if any nested mapping has a credential-like key.

    Keys are compared without case or punctuation, so ``apiKey``, ``api_key``
    and ``x-api-key`` are all rejected.
    """
    if isinstance(value, dict):
        if any(re.sub(r"[^a-z0-9]", "", str(key).lower()) in _SECRET_KEYS for key in value):
            raise ValueError(message)
        for item in value.values():
            reject_secrets(item, message)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_secrets(item, message)


def resolve_device(torch, requested="auto"):
    """Validate a torch device request; ``auto`` selects CUDA when available, else CPU."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def worker_environment():
    """``os.environ`` with PYTHONPATH extended so a child of *this* interpreter can import openhisto."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(IMPORT_ROOT), environment.get("PYTHONPATH")])
    )
    return environment


def isolated_module_command(python, module, *flags):
    """Command that runs ``module`` under another interpreter with only openhisto added.

    The other interpreter (an isolated model environment) need not have openhisto
    installed. Putting IMPORT_ROOT on its PYTHONPATH would also expose every
    package installed next to openhisto, such as a different NumPy or Torch, so
    the package is registered from its own directory instead.
    """
    package = Path(__file__).resolve().parent
    bootstrap = "\n".join(
        [
            "import importlib.util, runpy, sys",
            f"spec = importlib.util.spec_from_file_location('openhisto', {str(package / '__init__.py')!r},"
            f" submodule_search_locations=[{str(package)!r}])",
            "sys.modules['openhisto'] = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(sys.modules['openhisto'])",
            f"runpy.run_module({module!r}, run_name='__main__', alter_sys=True)",
        ]
    )
    return [str(python), *flags, "-c", bootstrap]
