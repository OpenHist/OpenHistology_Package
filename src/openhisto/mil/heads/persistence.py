"""Atomic, single-file head artifacts with registered constructors only."""

import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import zipfile


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def save_head(head, path, *, representation_spec=None, provenance=None, overwrite=False):
    """Save configuration and tensor state, never a pickled executable module."""
    import torch
    from datetime import datetime, timezone

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be an explicit boolean")
    # Saving the attached wrapper fingerprints the current trained backbone,
    # while the payload remains the independently reusable task head.
    attached_backbone = None
    if hasattr(head, "backbone") and hasattr(head, "head") and hasattr(head, "representation_spec"):
        attached_backbone = head.backbone
        if representation_spec is None:
            representation_spec = head.representation_spec
        head = head.head
    config = _json_safe(dict(head.config))
    if config.get("family") not in {"classification", "survival"}:
        raise ValueError("Only registered classification/survival heads can be saved")
    metadata = _json_safe(dict(getattr(head, "metadata", {})))
    if representation_spec is not None:
        metadata["representation_spec"] = _json_safe(representation_spec)
    if provenance is not None:
        metadata["provenance"] = _json_safe(provenance)
    if attached_backbone is not None:
        from .provenance import state_fingerprint

        metadata["backbone_state_sha256"] = state_fingerprint(attached_backbone)
    buffer = io.BytesIO()
    torch.save({key: tensor.detach().cpu() for key, tensor in head.state_dict().items()}, buffer)
    weights = buffer.getvalue()
    manifest = {
        "format": "cpath-mil-head",
        "version": 1,
        "config": config,
        "metadata": metadata,
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
    }
    encoded = json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    destination = Path(path).expanduser().absolute()
    if destination.is_symlink():
        raise ValueError("Head artifact destination cannot be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError("Head artifact exists; choose a new path or overwrite=True")
    fd, temporary = tempfile.mkstemp(
        prefix="." + destination.name + "-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w+b") as file:
            with zipfile.ZipFile(
                file, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as archive:
                archive.writestr("manifest.json", encoded)
                archive.writestr("weights.pt", weights)
            file.flush()
            os.fsync(file.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        return destination
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_head(
    path,
    *,
    expected_representation=None,
    expected_backbone=None,
    max_uncompressed_bytes=2_147_483_648,
):
    """Return a CPU evaluation head; call .train() explicitly to resume training."""
    import torch

    if (
        isinstance(max_uncompressed_bytes, bool)
        or not isinstance(max_uncompressed_bytes, int)
        or max_uncompressed_bytes < 1
    ):
        raise ValueError("max_uncompressed_bytes must be a positive integer")
    try:
        with zipfile.ZipFile(Path(path).expanduser()) as archive:
            infos = archive.infolist()
            if len(infos) != 2 or {entry.filename for entry in infos} != {
                "manifest.json",
                "weights.pt",
            }:
                raise ValueError("Unexpected head artifact members")
            if (
                sum(entry.file_size for entry in infos) > max_uncompressed_bytes
                or archive.getinfo("manifest.json").file_size > 8 * 1024 * 1024
            ):
                raise ValueError("Head artifact exceeds its size limit")
            manifest = json.loads(archive.read("manifest.json"))
            weights = archive.read("weights.pt")
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid head artifact") from exc
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("config"), dict)
        or not isinstance(manifest.get("metadata"), dict)
    ):
        raise ValueError("Head manifest must contain configuration and metadata objects")
    json.dumps(manifest, allow_nan=False)
    if manifest.get("format") != "cpath-mil-head" or manifest.get("version") != 1:
        raise ValueError("Unsupported head artifact format/version")
    if hashlib.sha256(weights).hexdigest() != manifest.get("weights_sha256"):
        raise ValueError("Head tensor checksum mismatch")
    metadata = manifest.get("metadata", {})
    if expected_representation is not None and metadata.get("representation_spec") != _json_safe(
        expected_representation
    ):
        raise ValueError("Head representation does not match the requested MIL backbone")
    if expected_backbone is not None:
        from .provenance import state_fingerprint
        from .bindings import get_representation_spec

        specification = get_representation_spec(expected_backbone)
        if metadata.get("representation_spec") != _json_safe(specification):
            raise ValueError("Head representation/configuration differs from this MIL backbone")
        expected = state_fingerprint(expected_backbone)
        if metadata.get("backbone_state_sha256") != expected:
            raise ValueError(
                "Head was not saved for this backbone checkpoint; its state fingerprint differs"
            )
    config = dict(manifest["config"])
    family, name = config.pop("family"), config.pop("name")
    if family == "classification":
        from openhisto.mil.classification import create_neural_head

        head = create_neural_head(name, **config)
    elif family == "survival":
        from openhisto.mil.survival import create_survival_head

        head = create_survival_head(name, **config)
    else:
        raise ValueError("Unregistered head family")
    state = torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(
        not isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("Head artifact must contain a tensor-only state dictionary")
    head.load_state_dict(state, strict=True)
    head.metadata.update(metadata)
    head.eval()
    return head
