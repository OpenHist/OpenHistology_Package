"""Model provenance and lazy, persistent inference backend selection.

Importing this module never imports torch or downloads model weights.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from .base import AccessError, ModelDescriptor, VLAError, validate_results
from .config import DescribeConfig, ModelSpec


_DESCRIPTORS = {
    item.model_id: item
    for item in (
        ModelDescriptor(
            model_id="patho_r1_7b",
            display_name="Patho-R1-7B",
            checkpoint="WenchuanZhang/Patho-R1-7B",
            checkpoint_revision="7a69eb299bde72a4c3b8ec26fe0b17515346ef73",
            source_url="https://github.com/Wenchuan-Zhang/Patho-R1",
            source_revision="bb34a4e73c6d384aadf51169b877829302a60649",
            license="CC-BY-NC-ND-4.0; additional model-card research-use terms",
            access="Public files currently; model card requires individual institutional registration and prohibits output-derived datasets without written approval.",
            dependencies=("requirements-patho-r1.txt",),
            preprocessing={
                "processor": "checkpoint AutoProcessor",
                "images": "native Qwen2.5-VL processor",
                "history": "independent patch requests",
            },
            required_artifacts=("model and processor checkpoint",),
        ),
        ModelDescriptor(
            model_id="pathgen_llava",
            display_name="PathGen-LLaVA-13B",
            checkpoint="jamessyx/PathGen-LLaVA",
            checkpoint_revision="783890f0e2aca42ad0329660bea48af89829a74d",
            source_url="https://github.com/haotian-liu/LLaVA",
            source_revision="c121f0432da27facab705978f83c4ada465e46fd",
            license="CC-BY-4.0 model card; underlying Vicuna/LLaMA terms also apply",
            access="Public checkpoint; separate PathGen-CLIP visual encoder required.",
            dependencies=("requirements-pathgen.txt",),
            preprocessing={
                "processor": "native PathGen-CLIP/LLaVA image processor",
                "history": "independent patch requests",
                "vision_checkpoint": "jamessyx/pathgenclip-vit-large-patch14-hf",
                "vision_revision": "33af8295d6e42422f027a61bd1d960ab258488f7",
                "parameters": "13B",
            },
            required_artifacts=(
                "PathGen-LLaVA checkpoint",
                "PathGen-CLIP visual encoder",
                "native LLaVA code",
            ),
        ),
        ModelDescriptor(
            model_id="quilt_llava_7b",
            display_name="Quilt-LLaVA-7B",
            checkpoint="wisdomik/Quilt-Llava-v1.5-7b",
            checkpoint_revision="1bdf5f8b75fb26acc80b08aba7f1979e8e9b12bd",
            source_url="https://github.com/aldraus/quilt-llava",
            source_revision="7e70fc39f792ac55de010eb37bff0a6d6f491c13",
            license="CC-BY-NC-3.0 checkpoint; MIT upstream code; underlying base-model terms apply",
            access="Public checkpoint; native Quilt-LLaVA environment required.",
            dependencies=("requirements-quilt.txt",),
            preprocessing={
                "processor": "native Quilt-LLaVA image processor",
                "history": "independent patch requests",
            },
            required_artifacts=(
                "Quilt-LLaVA checkpoint",
                "configured visual encoder",
                "Quilt-LLaVA code",
            ),
        ),
        ModelDescriptor(
            model_id="pa_llava",
            display_name="PA-LLaVA / Pathology-LLaVA",
            checkpoint="OpenFace-CQUPT/Pathology-LLaVA",
            checkpoint_revision="32bb3deb9fe47c53163fc79aac011b81a8031801",
            source_url="https://github.com/ddw2AIGROUP2CQUPT/PA-LLaVA",
            source_revision="a5c71d31e37aa37eec4249e4c0973a80581402ef",
            license="Apache-2.0 checkpoint card; Meta Llama 3 base-model terms apply",
            access="Public XTuner checkpoint; prepared adapters, custom PLIP encoder and authorized Llama 3 base model required.",
            dependencies=("requirements-pallava.txt",),
            preprocessing={
                "processor": "native PA-LLaVA PLIP processor and connector",
                "history": "independent patch requests",
            },
            required_artifacts=(
                "prepared llm_adapter and projector",
                "PA-LLaVA custom PLIP encoder",
                "Meta-Llama-3-8B-Instruct and tokenizer",
                "native PA-LLaVA code",
            ),
            status="experimental",
        ),
        ModelDescriptor(
            model_id="pathchat_plus",
            display_name="PathChat+",
            checkpoint=None,
            checkpoint_revision=None,
            source_url="https://arxiv.org/abs/2506.20964",
            source_revision="2506.20964v2",
            license="Access and use terms must be obtained from the authors",
            access="Limited access from the original authors; no public downloadable inference checkpoint verified.",
            dependencies=(),
            preprocessing={
                "encoder": "CONCH v1.5",
                "llm": "Qwen2.5-14B-Instruct",
                "tiling": "native 448px AnyRes tiles, up to 896px",
            },
            required_artifacts=(
                "authorized complete PathChat+ checkpoint and matching inference implementation",
            ),
            status="unavailable",
        ),
    )
}
_FACTORIES: dict[str, Callable[..., Any]] = {}


def get_descriptor(model: str | ModelSpec) -> ModelDescriptor:
    name = model.model if isinstance(model, ModelSpec) else model
    try:
        return _DESCRIPTORS[name]
    except (KeyError, TypeError) as exc:
        raise VLAError(
            f"Unknown VLA model {name!r}. Available models: {', '.join(_DESCRIPTORS)}"
        ) from exc


def list_models() -> list[dict[str, Any]]:
    return [item.to_dict() for item in _DESCRIPTORS.values()]


def model_info(model: str | ModelSpec) -> dict[str, Any]:
    return get_descriptor(model).to_dict()


def register_model(descriptor: ModelDescriptor, factory: Callable[..., Any]) -> None:
    """Register an in-process backend, primarily for lightweight testing.

    The factory receives ``(ModelSpec, DescribeConfig)`` and returns an object
    implementing ``generate``, ``metadata`` and ``close``.
    """
    if not isinstance(descriptor, ModelDescriptor) or not callable(factory):
        raise TypeError("register_model requires a ModelDescriptor and callable factory")
    _DESCRIPTORS[descriptor.model_id] = descriptor
    _FACTORIES[descriptor.model_id] = factory


def _normalized(spec: str | dict | ModelSpec, config: dict | DescribeConfig | None):
    if isinstance(spec, str):
        spec = ModelSpec(model=spec)
    elif isinstance(spec, dict):
        spec = ModelSpec(**spec)
    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a model ID, ModelSpec, or mapping")
    if config is None:
        config = DescribeConfig()
    elif isinstance(config, dict):
        config = DescribeConfig(**config)
    if not isinstance(config, DescribeConfig):
        raise TypeError("config must be DescribeConfig or mapping")
    descriptor = get_descriptor(spec)
    # A user-supplied checkpoint may have a completely different revision.
    revision = (
        spec.revision if spec.checkpoint else (spec.revision or descriptor.checkpoint_revision)
    )
    return replace(
        spec, checkpoint=spec.checkpoint or descriptor.checkpoint, revision=revision
    ), config


def _check_access(spec: ModelSpec) -> None:
    if spec.model == "pathchat_plus" and spec.model not in _FACTORIES:
        raise AccessError(
            "PathChat+ requires direct access from the original authors. No public complete "
            "checkpoint or supported inference implementation is available for this adapter. "
            "See https://github.com/mahmoodlab/viper/blob/main/docs/REPRODUCIBILITY.md#limited-access-models"
        )


class PreparedModel:
    """Persistent handle; model weights are loaded once until close()."""

    def __init__(self, backend: Any):
        self._backend = backend
        self._closed = False

    @property
    def metadata(self) -> dict[str, Any]:
        return self._backend.metadata

    def generate(self, images, prompt):
        if self._closed:
            raise VLAError("This model handle is closed")
        if not isinstance(prompt, str) or not prompt.strip():
            raise VLAError("prompt must be a non-empty string")
        images = list(images)
        if not images:
            return []
        return validate_results(self._backend.generate(images, prompt), len(images))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._backend.close()

    def __enter__(self):
        if self._closed:
            raise VLAError("This model handle is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def prepare_model(
    spec: str | dict | ModelSpec, config: dict | DescribeConfig | None = None
) -> PreparedModel:
    spec, config = _normalized(spec, config)
    _check_access(spec)
    if spec.model in _FACTORIES:
        return PreparedModel(_FACTORIES[spec.model](spec, config))
    from .runtime import WorkerClient

    return PreparedModel(WorkerClient(spec, config))


def create_backend(spec: str | dict | ModelSpec, config: dict | DescribeConfig | None = None):
    """Instantiate in the worker process without recursively starting workers."""
    spec, config = _normalized(spec, config)
    _check_access(spec)
    if spec.model in _FACTORIES:
        return _FACTORIES[spec.model](spec, config)
    if spec.model == "patho_r1_7b":
        from .backends.patho_r1 import PathoR1Backend

        return PathoR1Backend(spec, config)
    if spec.model in {"pathgen_llava", "quilt_llava_7b"}:
        from .backends.legacy import PathGenBackend, QuiltBackend

        return (PathGenBackend if spec.model == "pathgen_llava" else QuiltBackend)(spec, config)
    if spec.model == "pa_llava":
        from .backends.pallava import PALlavaBackend

        return PALlavaBackend(spec, config)
    raise VLAError(f"No backend implementation for {spec.model!r}")
