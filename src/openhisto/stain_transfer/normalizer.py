"""Common normalizer facade and deterministic destination selection."""

from dataclasses import replace
import hashlib
import json
import numpy as np
from .config import DomainSpec, TransferConfig, METHODS


def list_methods():
    """Return supported canonical backend IDs; ``default`` aliases Vahadane."""
    return list(METHODS)


def bounds_for(rgb, valid_bounds=None):
    from .imageio import validate_rgb

    validate_rgb(rgb)
    h, w = rgb.shape[:2]
    if valid_bounds is None:
        return (0, 0, w, h)
    bounds = tuple(valid_bounds)
    if len(bounds) != 4 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in bounds
    ):
        raise ValueError("valid_bounds must contain four integer pixel coordinates")
    x0, y0, x1, y1 = map(int, bounds)
    if not (0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h):
        raise ValueError("valid_bounds must lie inside the image")
    return (x0, y0, x1, y1)


def is_blank(rgb, valid_bounds=None):
    """Treat empty bounds and near-white background-only patches as blank."""
    x0, y0, x1, y1 = bounds_for(rgb, valid_bounds)
    region = rgb[y0:y1, x0:x1]
    return not region.size or bool(np.all(region >= 245))


class Normalizer:
    def __init__(self, domains, config):
        self.config = config
        self.backends = []
        self._closed = False
        for domain in domains:
            if domain.method in {"cyclegan", "pix2pix"}:
                from .gan import build

                if config.standardize:
                    raise ValueError(
                        "Brightness standardization is only supported for statistical methods"
                    )
                if not domain.checkpoint or domain.references:
                    raise ValueError("GAN destinations require a checkpoint and no references")
            else:
                from .classical import build

                if len(domain.references) != 1 or domain.checkpoint:
                    raise ValueError(
                        "Statistical destinations require exactly one reference and no checkpoint"
                    )
                if any(
                    value is not None
                    for value in (domain.architecture, domain.input_mode, domain.direction)
                ):
                    raise ValueError(
                        "Statistical destinations do not accept GAN architecture, input_mode or direction"
                    )
            self.backends.append(build(domain, config))
        self.provenance = {
            "schema_version": 1,
            "selection": "seeded_mixture" if len(domains) > 1 else "fixed",
            "seed": config.seed,
            "standardize": config.standardize,
            "backends": [b.provenance for b in self.backends],
        }

    def close(self):
        """Release optional backend workers after a reusable normalizer is done."""
        if not self._closed:
            for backend in self.backends:
                close = getattr(backend, "close", None)
                if close is not None:
                    close()
            self.backends.clear()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Normalizer is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def selected_index(self, patch_id):
        if len(self.backends) == 1:
            return 0
        if patch_id is None or str(patch_id) == "":
            raise ValueError("A stable patch_id is required for a mixture")
        payload = json.dumps([self.config.seed, str(patch_id)], separators=(",", ":")).encode()
        return int.from_bytes(hashlib.sha256(payload).digest(), "big") % len(self.backends)

    def transform(self, rgb, valid_bounds=None, patch_id=None):
        if self._closed:
            raise RuntimeError("Normalizer is closed")
        bounds = bounds_for(rgb, valid_bounds)
        choice = self.selected_index(patch_id)
        if is_blank(rgb, bounds):
            return rgb.copy()
        try:
            result = self.backends[choice].transform(
                rgb.copy(), valid_bounds=bounds, patch_id=patch_id
            )
            from .imageio import validate_rgb

            validate_rgb(result)
            if result.shape != rgb.shape:
                raise ValueError("Normalizer changed image dimensions")
            x0, y0, x1, y1 = bounds
            output = rgb.copy()
            output[y0:y1, x0:x1] = result[y0:y1, x0:x1]
            return output
        except Exception as exc:
            raise RuntimeError(f"Normalization failed for patch {patch_id!r}: {exc}") from exc


def create_normalizer(domain, config=None):
    """Fit/load a reusable destination (or explicit seeded mixture)."""
    if isinstance(domain, dict):
        domain = DomainSpec(**domain)
    if not isinstance(domain, DomainSpec):
        raise TypeError("domain must be DomainSpec or a domain dictionary")
    config = TransferConfig() if config is None else config
    if not isinstance(config, TransferConfig):
        raise TypeError("config must be TransferConfig")
    domains = list(domain.candidates) if domain.candidates else [domain]
    expanded = []
    for candidate in domains:
        if candidate.references:
            expanded.extend(
                replace(candidate, references=(reference,)) for reference in candidate.references
            )
        else:
            expanded.append(candidate)
    if len(expanded) > 1 and not config.mix:
        raise ValueError("Multiple destinations/references require mix=True")
    if config.mix and len(expanded) < 2:
        raise ValueError("mix=True requires at least two method/reference destinations")
    return Normalizer(expanded, config)
