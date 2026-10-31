"""Classical RGB normalization with fitted references and bounded image buffers.

The stain-separation and LAB formulas follow Peter Byfield's MIT-licensed
StainTools. SPAMS is imported only by Macenko and Vahadane, including Macenko's
positive sparse concentration fit. See CLASSICAL_LICENSE.txt for attribution.
"""

from hashlib import sha256
import importlib
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
import sys

import numpy as np

from .imageio import load_rgb


def _load_rgb(path):
    return load_rgb(Path(path).expanduser())


def _cv2():
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise ImportError(
            'Classical stain normalization requires OpenCV; pip install "openhisto[stain-transfer]"'
        ) from exc


def _spams():
    try:
        if sys.platform == "darwin":
            # SPAMS and recent Torch wheels bundle different libomp builds.
            # Keep SPAMS' native runtime outside the caller's address space.
            from .spams_process import SpamsClient

            return SpamsClient()
        return importlib.import_module("spams")
    except (ImportError, OSError, RuntimeError) as exc:
        raise ImportError(
            'Macenko and Vahadane require SPAMS (spams-bin==2.6.14); pip install "openhisto[stain-transfer-spams]". '
            + str(exc)
        ) from exc


def _spams_version():
    for distribution in ("spams-bin", "spams"):
        try:
            return distribution, version(distribution)
        except PackageNotFoundError:
            continue
    return "spams", "unknown"


def _number(value, name, *, minimum=0, maximum=None, exclusive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or exclusive and value == minimum or maximum is not None and value > maximum:
        raise ValueError(f"Invalid {name}")
    return float(value)


def _bounds(image, bounds):
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        raise ValueError("Normalization requires nonempty uint8 HWC RGB images")
    if bounds is None:
        return 0, 0, image.shape[1], image.shape[0]
    try:
        values = tuple(bounds)
    except TypeError as exc:
        raise ValueError("valid_bounds must contain four integers") from exc
    if len(values) != 4 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values
    ):
        raise ValueError("valid_bounds must contain four integers")
    x0, y0, x1, y1 = map(int, values)
    if not 0 <= x0 <= x1 <= image.shape[1] or not 0 <= y0 <= y1 <= image.shape[0]:
        raise ValueError("valid_bounds must lie inside the image")
    return x0, y0, x1, y1


def _flat(image):
    pixels = image.reshape(-1, 3)
    return len(pixels) == 0 or np.max(np.ptp(pixels, axis=0)) == 0


def _standardize(image):
    cv = _cv2()
    lab = cv.cvtColor(np.ascontiguousarray(image), cv.COLOR_RGB2LAB)
    lum = lab[:, :, 0].astype(np.float64)
    percentile = np.percentile(lum, 95)
    if percentile <= 0:
        return image.copy()
    lab[:, :, 0] = np.clip(255 * lum / percentile, 0, 255).astype(np.uint8)
    return cv.cvtColor(lab, cv.COLOR_LAB2RGB)


def _od(image):
    # StainTools clips zero RGB to one and OD to 1e-6. Never mutate caller data.
    return np.maximum(-np.log(np.maximum(image.astype(np.float64), 1) / 255), 1e-6).reshape(-1, 3)


class ClassicalNormalizer:
    def __init__(self, domain, config):
        self.method = domain.method.lower()
        if self.method == "hist":
            self.method = "histogram"
        if self.method not in {"reinhard", "macenko", "vahadane", "histogram"}:
            raise ValueError(f"Unknown classical normalization method: {domain.method}")
        if len(domain.references) != 1:
            raise ValueError("A fitted classical backend requires exactly one reference image")
        self.seed = int(config.seed)
        self.standardize = bool(config.standardize)
        options = domain.options or {}
        allowed_options = {
            "luminosity_threshold",
            "stain_regularizer",
            "concentration_regularizer",
            "iterations",
        }
        unknown = set(options) - allowed_options
        if unknown:
            raise ValueError(
                "Unknown classical normalization options: " + ", ".join(sorted(map(str, unknown)))
            )
        self.threshold = _number(
            options.get("luminosity_threshold", 0.8),
            "luminosity_threshold",
            maximum=1,
            exclusive=True,
        )
        self.stain_regularizer = _number(options.get("stain_regularizer", 0.1), "stain_regularizer")
        self.concentration_regularizer = _number(
            options.get("concentration_regularizer", 0.01), "concentration_regularizer"
        )
        self.iterations = options.get("iterations", 100)
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, Integral)
            or self.iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")
        clip = getattr(config, "histogram_clip", (0.01, 0.99))
        if clip is not None:
            if len(clip) != 2:
                raise ValueError("histogram_clip must contain two quantiles")
            clip = tuple(_number(v, "histogram_clip", maximum=1) for v in clip)
            if clip[0] >= clip[1]:
                raise ValueError("histogram_clip quantiles must increase")
        self.clip = clip
        target = _load_rgb(domain.references[0])
        digest = sha256()
        digest.update(np.asarray(target.shape, dtype="<i8").tobytes())
        digest.update(target.tobytes())
        if _flat(target) or np.all(target >= 245):
            raise ValueError(
                "Reference image is blank or constant and cannot define a normalization domain"
            )
        if self.standardize:
            target = _standardize(target)
        self.provenance = {
            "backend": "classical",
            "method": self.method,
            "reference_sha256": digest.hexdigest(),
            "standardize": self.standardize,
            "reference_hash_encoding": "decoded RGB shape as little-endian int64 then C-order uint8 pixels",
            "versions": {"numpy": np.__version__},
        }
        if self.method != "histogram" or self.standardize:
            self.provenance["versions"]["opencv"] = _cv2().__version__
        if self.method == "reinhard":
            lab = self._lab(target)
            self.target_mean = lab.mean(axis=(0, 1))
            self.target_std = lab.std(axis=(0, 1))
        elif self.method == "histogram":
            self.target_cdfs = [self._cdf(target[:, :, c]) for c in range(3)]
            self.provenance["histogram_clip"] = self.clip
            self.provenance["histogram_mapping"] = "occupied_rgb_quantiles_v1"
        else:
            self.spams = _spams()
            distribution, installed_version = _spams_version()
            self.provenance["versions"][distribution] = installed_version
            self.target_stains = self._stains(target)
            if self.target_stains is None:
                raise ValueError(
                    "Reference image does not contain enough independent tissue colors for stain separation"
                )
            concentrations = self._concentrations(target, self.target_stains)
            self.target_max = np.percentile(concentrations, 99, axis=0)
            if not np.all(np.isfinite(self.target_max)) or np.any(self.target_max <= 1e-10):
                raise ValueError("Reference image has degenerate stain concentrations")
            self.provenance.update(
                solver="spams",
                luminosity_threshold=self.threshold,
                execution=getattr(self.spams, "execution", "inprocess"),
                concentration_regularizer=self.concentration_regularizer,
                concentration_percentile=99,
            )
            if self.method == "vahadane":
                self.provenance.update(
                    stain_regularizer=self.stain_regularizer,
                    iterations=int(self.iterations),
                    seed=self.seed,
                    dictionary_mode="batch",
                    threads=1,
                )

    def close(self):
        """Release a persistent native worker when this backend owns one."""
        close = getattr(getattr(self, "spams", None), "close", None)
        if close is not None:
            close()

    @staticmethod
    def _lab(image):
        cv = _cv2()
        lab = cv.cvtColor(np.ascontiguousarray(image), cv.COLOR_RGB2LAB).astype(np.float64)
        lab[:, :, 0] /= 2.55
        lab[:, :, 1:] -= 128
        return lab

    def _cdf(self, channel):
        # Invert only occupied intensities. Empty histogram bins create CDF
        # plateaus whose arbitrary right edges are not reference intensities.
        values, counts = np.unique(channel, return_counts=True)
        cdf = np.cumsum(counts, dtype=np.float64) / counts.sum()
        if self.clip is not None:
            lo, hi = self.clip
            cdf = (np.clip(cdf, lo, hi) - lo) / (hi - lo)
        return values, cdf

    def _stains(self, image):
        cv = _cv2()
        lab = cv.cvtColor(np.ascontiguousarray(image), cv.COLOR_RGB2LAB)
        tissue = lab[:, :, 0].reshape(-1) / 255 < self.threshold
        od = _od(image)[tissue]
        if len(od) < 3:
            return None
        eigenvalues, vectors = np.linalg.eigh(np.cov(od, rowvar=False))
        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[-2] <= 1e-10:
            return None
        if self.method == "macenko":
            vectors = vectors[:, [2, 1]]
            vectors[:, vectors[0] < 0] *= -1
            # RGB has only three terms. The scalar einsum loop avoids a
            # NumPy 2 / macOS Accelerate matmul kernel that raises spurious
            # floating-point flags on finite, bounded three-channel data.
            projected = np.einsum("ij,jk->ik", od, vectors, optimize=False)
            phi = np.arctan2(projected[:, 1], projected[:, 0])
            low, high = np.percentile(phi, (1, 99))
            stains = np.stack(
                (
                    np.einsum("ij,j->i", vectors, (np.cos(low), np.sin(low)), optimize=False),
                    np.einsum("ij,j->i", vectors, (np.cos(high), np.sin(high)), optimize=False),
                )
            )
        else:
            # Explicit initialization, full-batch updates and one thread avoid
            # dependence on SPAMS' process-global random generator/timing.
            rng = np.random.default_rng(self.seed)
            initial = od[rng.choice(len(od), size=2, replace=False)].copy()
            initial /= np.maximum(np.linalg.norm(initial, axis=1, keepdims=True), 1e-12)
            stains = np.asarray(
                self.spams.trainDL(
                    X=np.asfortranarray(od.T),
                    D=np.asfortranarray(initial.T),
                    K=2,
                    lambda1=self.stain_regularizer,
                    mode=2,
                    modeD=0,
                    posAlpha=True,
                    posD=True,
                    verbose=False,
                    numThreads=1,
                    iter=int(self.iterations),
                    batch=True,
                    clean=False,
                )
            ).T
        if stains.shape != (2, 3) or not np.all(np.isfinite(stains)):
            return None
        norms = np.linalg.norm(stains, axis=1, keepdims=True)
        if np.any(norms <= 1e-10):
            return None
        stains = stains / norms
        if stains[0, 0] < stains[1, 0]:
            stains = stains[[1, 0]]
        if np.linalg.matrix_rank(stains, tol=1e-8) < 2:
            return None
        return stains

    def _concentrations(self, image, stains):
        od = _od(image)
        concentrations = np.empty((len(od), 2), dtype=np.float64)
        dictionary = np.asfortranarray(stains.T, dtype=np.float64)
        for start in range(0, len(od), 65536):
            stop = min(start + 65536, len(od))
            concentrations[start:stop] = (
                self.spams.lasso(
                    X=np.asfortranarray(od[start:stop].T),
                    D=dictionary,
                    mode=2,
                    lambda1=self.concentration_regularizer,
                    pos=True,
                    numThreads=1,
                )
                .toarray()
                .T
            )
        if not np.all(np.isfinite(concentrations)):
            raise RuntimeError("SPAMS concentration fitting returned non-finite values")
        return concentrations

    def transform(self, rgb, valid_bounds=None, patch_id=None):
        """Transform the valid rectangle and preserve every pixel outside it."""
        rgb = np.asarray(rgb)
        x0, y0, x1, y1 = _bounds(rgb, valid_bounds)
        result = rgb.copy()
        original = rgb[y0:y1, x0:x1]
        if x0 == x1 or y0 == y1 or _flat(original):
            return result
        source = _standardize(original) if self.standardize else original.copy()
        if self.method == "reinhard":
            lab = self._lab(source)
            mean, std = lab.mean(axis=(0, 1)), lab.std(axis=(0, 1))
            lab = (lab - mean) * (self.target_std / np.maximum(std, 1e-8)) + self.target_mean
            lab[:, :, 0] *= 2.55
            lab[:, :, 1:] += 128
            cv = _cv2()
            transformed = cv.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv.COLOR_LAB2RGB)
        elif self.method == "histogram":
            transformed = np.empty_like(source)
            for channel in range(3):
                values, quantiles = self._cdf(source[:, :, channel])
                target_values, target_quantiles = self.target_cdfs[channel]
                matched = np.interp(quantiles, target_quantiles, target_values)
                transformed[:, :, channel] = np.rint(
                    np.interp(source[:, :, channel], values, matched)
                ).astype(np.uint8)
        else:
            stains = self._stains(source)
            if stains is None:
                return result
            concentrations = self._concentrations(source, stains)
            source_max = np.percentile(concentrations, 99, axis=0)
            if not np.all(np.isfinite(source_max)) or np.any(source_max <= 1e-10):
                return result
            concentrations *= self.target_max / source_max
            reconstructed = np.einsum(
                "ij,jk->ik", concentrations, self.target_stains, optimize=False
            )
            transformed = (
                np.clip(255 * np.exp(-reconstructed), 0, 255).reshape(source.shape).astype(np.uint8)
            )
        result[y0:y1, x0:x1] = transformed
        return result


def build(domain, config):
    """Fit one reference-backed classical normalizer once for a domain."""
    return ClassicalNormalizer(domain, config)
