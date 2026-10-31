"""Pinned local PyTorch adapters; no model dependencies load at module import."""

from contextlib import nullcontext
from functools import partial
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np

from ...base import AccessError, EncoderError, translated_errors, validate_features
from ...config import json_safe


_COMMIT = re.compile(r"[0-9a-fA-F]{40}")


def dependency(name, distribution=None):
    try:
        return importlib.import_module(name)
    except ImportError:
        raise EncoderError(
            f"Optional model dependency is missing; install {distribution or name} for this adapter"
        ) from None


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise EncoderError(
            "Checkpoint metadata is missing or invalid; provide the matching model and processor configuration"
        ) from None
    if not isinstance(value, dict):
        raise EncoderError("Checkpoint metadata must contain a JSON object")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_commit(value):
    return _COMMIT.fullmatch(value) is not None


def _is_missing_file(exc):
    # Matched by exact name, not isinstance: LocalEntryNotFoundError (Hub
    # unreachable) subclasses EntryNotFoundError but is not a missing file, and
    # RemoteEntryNotFoundError only exists in newer huggingface_hub releases.
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 404 or type(exc).__name__ in {"EntryNotFoundError", "RemoteEntryNotFoundError"}


def _snapshot_revision(path, requested):
    """The commit a hub file belongs to: its snapshot folder, else a pinned request."""
    for candidate in (path.parent.name, requested):
        if _is_commit(candidate):
            return candidate
    raise EncoderError("The checkpoint cache did not expose an immutable revision")


def _local_snapshot(spec, revision):
    checkpoint = Path(spec.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise EncoderError("The selected local checkpoint does not exist")
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    if directory.parent.name == "snapshots" and _is_commit(directory.name):
        resolved = directory.name
    elif spec.revision and _is_commit(revision):
        resolved = revision
    else:
        resolved = None
    return directory, resolved, checkpoint if checkpoint.is_file() else None


class _HubCache:
    """Lookups in the local HF cache only; never authenticates or uses the network."""

    def __init__(self, hub, repo_id, cache_dir):
        self._lookup = partial(hub.try_to_load_from_cache, repo_id, cache_dir=cache_dir)

    def file(self, filename, revision):
        try:
            value = self._lookup(filename, revision=revision)
            return Path(value) if isinstance(value, str) and Path(value).is_file() else None
        except Exception:
            return None  # An unreadable cache entry is a cache miss.

    def has_any(self, alternatives, revision):
        return any(self.file(filename, revision) is not None for filename in alternatives)

    def complete_snapshot(self, groups, revision):
        config_path = self.file("config.json", revision)
        if config_path is None:
            return None
        resolved = _snapshot_revision(config_path, revision)
        if not all(self.has_any(group, resolved) for group in groups):
            return None
        return config_path.parent, resolved, None


def _download_first_available(download, alternatives, revision):
    for filename in alternatives[:-1]:
        try:
            return download(filename=filename, revision=revision)
        except Exception as exc:
            if not _is_missing_file(exc):
                raise
    return download(filename=alternatives[-1], revision=revision)


def _import_conch_source(source):
    """Import the checkpoint's own ``conch_v1_5.py`` once per content hash."""
    name = "_cpath_conch15_" + sha256(source)
    if name in sys.modules:
        return sys.modules[name]
    module_spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise EncoderError(
            "Could not load the pinned CONCH adapter source; install its documented model dependencies"
        ) from None
    return module


def snapshot(spec, descriptor, config, auth, filenames):
    """Resolve an immutable snapshot using metadata first, then exact files only.

    ``filenames`` entries are a filename or a tuple of alternatives, of which the
    first the repository provides is used. Returns ``(directory, revision,
    explicit_weight_file)``.
    """
    revision = spec.revision or descriptor.preprocessing["default_revision"]
    if spec.checkpoint:
        return _local_snapshot(spec, revision)
    hub = dependency("huggingface_hub")
    groups = [
        (filename,) if isinstance(filename, str) else tuple(filename) for filename in filenames
    ]
    cache = _HubCache(hub, descriptor.model_name, config.cache_dir)
    # A complete immutable snapshot needs neither authentication nor a network
    # request, even for gated repositories whose approval was obtained earlier.
    cached = cache.complete_snapshot(groups, revision)
    if cached is not None:
        return cached
    if config.offline:
        raise EncoderError(
            "The complete checkpoint and processor bundle is not cached; prepare it online or provide a matching local checkpoint"
        )
    token = auth.resolve("hf", required=descriptor.access == "gated")
    download = partial(
        hub.hf_hub_download,
        repo_id=descriptor.model_name,
        cache_dir=config.cache_dir,
        local_files_only=False,
        token=token,
    )
    if descriptor.access == "gated":
        error, message = (
            AccessError,
            "Could not resolve the approved checkpoint; review the model access page, HF credentials, revision, and offline cache",
        )
    else:
        error, message = (
            EncoderError,
            "Could not resolve the checkpoint; check network access, revision, and offline cache",
        )
    with translated_errors(message, error):
        config_path = Path(download(filename="config.json", revision=revision))
        resolved = _snapshot_revision(config_path, revision)
        for group in groups:
            if not cache.has_any(group, resolved):
                _download_first_available(download, group, resolved)
    return config_path.parent, resolved, None


def checkpoint_file(
    directory, explicit=None, *, preferred=("model.safetensors", "pytorch_model.bin")
):
    if explicit is not None:
        return explicit
    for filename in preferred:
        path = directory / filename
        if path.is_file():
            return path
    raise EncoderError("The checkpoint bundle does not contain the expected model weight file")


def load_weights(torch, path):
    if path.suffix == ".safetensors":
        module = dependency("safetensors.torch", "safetensors")
        value = module.load_file(str(path), device="cpu")
    else:
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            raise EncoderError(
                "Safe checkpoint loading requires a PyTorch version supporting weights_only=True"
            ) from None
        except Exception:
            raise EncoderError("Could not safely load the selected checkpoint") from None
    if not isinstance(value, dict):
        raise EncoderError("Checkpoint must contain a tensor state dictionary")
    for key in ("state_dict", "model", "net"):
        if isinstance(value.get(key), dict):
            return value[key]
    return value


def device_and_precision(torch, config):
    cuda = torch.cuda.is_available()
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps = bool(mps_backend and mps_backend.is_available())
    name = config.device
    if name == "auto":
        name = "cuda" if cuda else "mps" if mps else "cpu"
    if name.startswith("cuda") and not cuda:
        raise EncoderError("CUDA was requested but is not available")
    if name == "mps" and not mps:
        raise EncoderError("MPS was requested but is not available")
    device = torch.device(name)
    if device.type != "cuda":
        if config.precision not in {"auto", "float32"}:
            raise EncoderError("CPU and MPS adapters require float32 precision")
        return device, "float32"
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    precision = config.precision
    if precision == "auto":
        precision = "bfloat16" if bf16 else "float16"
    if precision == "bfloat16" and not bf16:
        raise EncoderError("bfloat16 was requested but this CUDA device does not support it")
    return device, precision


def versions():
    result = {}
    for name in ("torch", "torchvision", "transformers", "timm", "huggingface_hub", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


class TorchEncoder:
    """Shared inference wrapper preserving (batch, dimension), including batch=1."""

    def _initialize(self, spec, descriptor, config):
        if set(spec.options) - {"cache_tag"}:
            raise ValueError("Local encoder profiles accept only the cache_tag option")
        if "cache_tag" in spec.options and not isinstance(spec.options["cache_tag"], str):
            raise ValueError("cache_tag must be a string")
        self.spec = spec
        self.descriptor = descriptor
        self.dimension = spec.dimension or descriptor.dimension
        self.usage = {}
        self.torch = dependency("torch")
        self.device, self.precision = device_and_precision(self.torch, config)
        self.model = None
        self.processor = None
        self.transform = None

    def _finish(self, directory, revision, weight_path, preprocessing, *, code_paths=()):
        self.model.to(self.device)
        self.model.eval()
        self.metadata = json_safe(
            {
                "model_id": self.spec.model,
                "provider": "hf",
                "model_name": self.descriptor.model_name,
                "dimension": self.dimension,
                "resolved_revision": revision,
                "weights_sha256": {weight_path.name: sha256(weight_path)},
                "checkpoint_config_sha256": sha256(directory / "config.json"),
                "code_sha256": {Path(path).name: sha256(path) for path in code_paths},
                "checkpoint_source": "local" if self.spec.checkpoint else "huggingface",
                "preprocessing": preprocessing,
                "pooling": self.descriptor.preprocessing["pooling"],
                "device": str(self.device),
                "precision": self.precision,
                "output_dtype": "float32",
                "library_versions": versions(),
                "adapter_version": 1,
            }
        )

    def encode(self, images):
        if self.model is None:
            raise EncoderError("Encoder is closed")
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        pil = dependency("PIL.Image", "pillow")
        converted = []
        for image in images:
            image = np.asarray(image)
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or min(image.shape[:2]) < 1
            ):
                raise ValueError("Encoder images must be nonempty uint8 RGB arrays")
            converted.append(pil.fromarray(image))
        if self.processor is not None:
            batch = self.processor(images=converted, return_tensors="pt")
            batch = {key: value.to(self.device) for key, value in batch.items()}
        else:
            batch = self.torch.stack([self.transform(image) for image in converted]).to(self.device)
        mixed = self.precision != "float32"
        context = (
            self.torch.autocast(device_type="cuda", dtype=getattr(self.torch, self.precision))
            if mixed
            else nullcontext()
        )
        with self.torch.inference_mode(), context:
            features = self._forward(batch)
        return validate_features(
            features.detach().float().cpu().numpy(), len(images), self.dimension
        )

    def _forward(self, batch):
        return self.model(batch)

    def close(self):
        self.model = None
        self.processor = None
        self.transform = None


class UNIEncoder(TorchEncoder):
    def __init__(self, spec, *, config, auth):
        from ...registry import get_descriptor

        descriptor = get_descriptor(spec)
        self._initialize(spec, descriptor, config)
        timm = dependency("timm")
        directory, revision, explicit = snapshot(
            spec, descriptor, config, auth, [("pytorch_model.bin", "model.safetensors")]
        )
        metadata = read_json(directory / "config.json")
        pretrained_cfg = metadata.get("pretrained_cfg")
        if not isinstance(pretrained_cfg, dict):
            raise EncoderError(
                "UNI checkpoints require the matching config.json with pretrained_cfg"
            )
        if metadata.get("architecture", "vit_large_patch16_224") != "vit_large_patch16_224":
            raise EncoderError("The selected checkpoint architecture does not match UNI v1")
        self.model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        weight_path = checkpoint_file(
            directory, explicit, preferred=("pytorch_model.bin", "model.safetensors")
        )
        with translated_errors("The selected weights do not match the UNI v1 architecture"):
            self.model.load_state_dict(load_weights(self.torch, weight_path), strict=True)
        data = dependency("timm.data")
        transform_factory = dependency("timm.data.transforms_factory")
        self.model.pretrained_cfg = pretrained_cfg
        transform_config = data.resolve_data_config(pretrained_cfg, model=self.model)
        self.transform = transform_factory.create_transform(**transform_config, is_training=False)
        self._finish(directory, revision, weight_path, {"kind": "timm", "config": transform_config})


class Conch15Encoder(TorchEncoder):
    """TITAN-compatible attentional-pooler features from its pinned patch encoder.

    Only the official CONCH code/config/patch weights are acquired. The unrelated
    TITAN slide and text checkpoint is not needed for patch encoding.
    """

    def __init__(self, spec, *, config, auth):
        from ...registry import get_descriptor

        descriptor = get_descriptor(spec)
        self._initialize(spec, descriptor, config)
        for name in ("timm", "einops", "einops_exts"):
            dependency(name)
        directory, revision, explicit = snapshot(
            spec, descriptor, config, auth, ["conch_v1_5.py", "conch_v1_5_pytorch_model.bin"]
        )
        data = read_json(directory / "config.json")
        conch_config = data.get("conch_config")
        required = (
            "patch_size",
            "context_dim",
            "depth",
            "num_heads",
            "mlp_ratio",
            "qkv_bias",
            "init_values",
            "embed_dim",
            "pooler_n_queries_contrast",
        )
        if not isinstance(conch_config, dict) or any(name not in conch_config for name in required):
            raise EncoderError("CONCH v1.5 requires the matching TITAN conch_config")
        cfg = SimpleNamespace(**conch_config)
        if cfg.embed_dim != self.dimension:
            raise EncoderError(
                "The TITAN patch checkpoint does not declare the required 768-dimensional representation"
            )
        source = directory / "conch_v1_5.py"
        if not source.is_file():
            raise EncoderError("CONCH v1.5 requires its matching official conch_v1_5.py source")
        module = _import_conch_source(source)
        encoder = module.VisionTransformer(
            patch_size=cfg.patch_size,
            embed_dim=cfg.context_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            init_values=cfg.init_values,
        )
        pooler = module.AttentionalPooler(
            d_model=cfg.embed_dim,
            context_dim=cfg.context_dim,
            n_queries=cfg.pooler_n_queries_contrast,
        )
        self.model = module.EncoderWithAttentionalPooler(
            encoder=encoder,
            attn_pooler_contrast=pooler,
            embed_dim=cfg.embed_dim,
        )
        weight_path = checkpoint_file(
            directory, explicit, preferred=("conch_v1_5_pytorch_model.bin",)
        )
        with translated_errors(
            "The checkpoint does not match the CONCH v1.5 attentional-pooler architecture"
        ):
            self.model.load_state_dict(load_weights(self.torch, weight_path), strict=True)
        transforms = dependency("torchvision.transforms", "torchvision")
        self.transform = transforms.Compose(
            [
                transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self._finish(
            directory,
            revision,
            weight_path,
            {
                "kind": "TITAN CONCH v1.5 evaluation transform",
                "resize": 448,
                "crop_size": 448,
                "interpolation": "bilinear",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "source_patch_size": 512,
                "source_magnification": 20,
            },
            code_paths=[source],
        )
