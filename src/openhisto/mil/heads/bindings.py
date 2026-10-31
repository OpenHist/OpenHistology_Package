"""Declared representation bindings for the 34 supported MIL profiles.

Dimensions come from constructed modules and their explicit axes, never a
training forward or guessed reduction over an arbitrary tensor.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn


_SLIDE_AXES = {"native": ["slide", "feature"], "result": ["slide", "feature"]}
_DEFAULT_POLICY = "native_slide_embedding"

_PROVENANCE_KEYS = (
    "repository",
    "native_entrypoint",
    "origin",
    "implementation",
    "weights",
    "prototype_sha256",
    "kernel_backend",
    "distill",
    "num_groups",
    "partition_seed",
    "n_proto",
    "out_type",
    "variant",
    "effective_global_width",
)
# A source pin alone cannot distinguish e.g. attention versus mean readout, so
# configuration is recorded too, minus weight locations.
_NON_CONFIGURATION_KEYS = {
    "checkpoint",
    "prototype_path",
    "pretrain_4k",
    "pretrain_WSI",
    "teacher_state_dict",
}
_OPAQUE = object()


def _input_width(module):
    """Input width of the first linear layer at or below ``module``."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child.in_features
    raise ValueError("The declared representation boundary has no linear input dimension")


@dataclass(frozen=True)
class _Boundary:
    """Where one profile's slide representation lives in its native model."""

    dimension: int
    policy: str = _DEFAULT_POLICY
    mode: str = "embedding"
    targets: tuple = ()
    capture_index: int = -1
    axes: dict = None
    class_count: int = None
    adaptation: str = None


def _embedding(module, policy=_DEFAULT_POLICY, *, adaptation=None):
    """The adapter's own ``output.embedding`` feeds ``module``."""
    return _Boundary(_input_width(module), policy, adaptation=adaptation)


def _hook(module, policy, *, capture_index=-1):
    """Capture the input of ``module``, the native classifier, during forward."""
    return _Boundary(
        _input_width(module), policy, mode="hook", targets=(module,), capture_index=capture_index
    )


def _class_embeddings(class_count, width, native_axes, sizes):
    """One native embedding per class, flattened in native class order."""
    axes = {
        "native": native_axes,
        "sizes": sizes,
        "flattened": ["native_class", "feature"],
        "result": ["slide", "feature"],
    }
    return _Boundary(
        class_count * width, "flatten_native_class_embeddings", axes=axes, class_count=class_count
    )


def _dsmil(classifier):
    # DSMIL's bag classifier is a Conv1d whose kernel spans the feature width.
    class_count, width = classifier.in_channels, classifier.kernel_size[0]
    return _class_embeddings(
        class_count, width, ["slide", "native_class", "feature"], [1, class_count, width]
    )


def _clam_mb(native):
    class_count = len(native.classifiers)
    width = _input_width(native.classifiers[0])
    return _class_embeddings(class_count, width, ["native_class", "feature"], [class_count, width])


def _mhim(native):
    if native.baseline == "dsmil":
        return _dsmil(native.online_encoder.b_classifier.fcc)
    return _embedding(native.predictor)


def _slidegraph(native):
    widths = list(native.embeddings_dim)
    return _Boundary(
        sum(widths),
        "concatenate_native_feature_readouts_per_layer",
        mode="slidegraph",
        targets=(native.first_h, *native.convs),
        axes={
            "native": ["slide", "ordered_layer_features"],
            "layer_widths": widths,
            "result": ["slide", "feature"],
        },
        adaptation="Concatenated pooled feature stages are a downstream representation adaptation; "
        "native inference sums per-layer prediction scores",
    )


# Each entry receives the backbone adapter (``b``) and declares its boundary.
_BOUNDARIES = {
    "abmil": lambda b: _embedding(b.native.classifier),
    "gated_abmil": lambda b: _embedding(b.native.classifier),
    "dsmil": lambda b: _dsmil(b.native.b_classifier.fcc),
    "dtfd_mil": lambda b: _embedding(
        b.native["att_classifier"].classifier, "native_second_tier_slide_embedding"
    ),
    "acmil": lambda b: _embedding(b.native.Slide_classifier),
    "asmil": lambda b: _embedding(b.native.Slide_classifier),
    "clam_sb": lambda b: _embedding(b.native.classifiers),
    "clam_mb": lambda b: _clam_mb(b.native),
    "varpool": lambda b: _embedding(b.native.head, "native_attention_mean_and_variance_embedding"),
    "panther": lambda b: _embedding(b.classifier, "native_prototype_representation"),
    "transmil": lambda b: _embedding(b.native._fc2),
    "ilra": lambda b: _embedding(b.native.classifier),
    "rrt_mil": lambda b: _embedding(b.native.predictor),
    "dgr_mil": lambda b: _embedding(b.native.classifier),
    "caprmil": lambda b: _embedding(b.native.classifier),
    "lin_mil": lambda b: _embedding(b.native.mlp_head),
    "mamba_mil": lambda b: _embedding(b.native.classifier),
    "nnmil": lambda b: _embedding(
        b.native.cls,
        "native_mean_window_pooled_feature_for_downstream_head",
        adaptation="Downstream nonlinear head on mean pooled features differs from the native "
        "evaluation ensemble of per-window logits",
    ),
    "mhim_v1": lambda b: _mhim(b.native),
    "mhim_v2": lambda b: _mhim(b.native),
    "wikg": lambda b: _hook(b.native.fc, "native_normalized_graph_readout_before_classifier"),
    "tad_graph": lambda b: _hook(
        b.native.clf.pred_layer, "native_causal_graph_readout_before_classifier", capture_index=0
    ),
    "resgat": lambda b: _hook(b.native.fc1, "native_graph_mean_readout_before_classifier_mlp"),
    "patch_gcn": lambda b: _hook(
        b.native.classifier, "native_gated_pooled_path_rho_before_survival_classifier"
    ),
    "slidegraph_plus": lambda b: _slidegraph(b.native),
    "cs_mil": lambda b: _hook(
        b.native.fc6, "native_resolution_fusion_and_bag_attention_before_classifier"
    ),
    "zoommil": lambda b: _hook(
        b.native.classify_head, "native_sum_of_three_magnification_readouts"
    ),
    "hag_mil": lambda b: _hook(
        b.native.models[-1].fc5, "native_finest_selected_hierarchy_stage_before_classifier"
    ),
    "gfr_mil": lambda b: _hook(
        b.native.model.classfier, "native_fine_class_attention_readout_before_classifier"
    ),
    "h2_mil": lambda b: _hook(
        b.native.lin1, "native_sum_of_two_hierarchy_pool_readouts_before_classifier_mlp"
    ),
    "hipt": lambda b: _hook(
        b.native.classifier, "native_global_hierarchy_readout_before_classifier"
    ),
    "mean_mil": lambda b: _embedding(b.classifier, "native_statistical_pooling"),
    "max_mil": lambda b: _embedding(b.classifier, "native_statistical_pooling"),
    "mean_std_mil": lambda b: _embedding(b.classifier, "native_statistical_pooling"),
}

PROFILES = tuple(_BOUNDARIES)


def _profile(backbone):
    metadata = getattr(backbone, "metadata", {})
    name = metadata.get("profile", metadata.get("name", getattr(backbone, "name", None)))
    if name not in _BOUNDARIES:
        raise ValueError(
            "No representation binding for this backbone; use a supported create_model profile"
        )
    return name


def _configuration(value):
    """JSON-representable configuration; opaque values and weight paths are dropped."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _OPAQUE
    if isinstance(value, (list, tuple)):
        items = [_configuration(item) for item in value]
        return _OPAQUE if any(item is _OPAQUE for item in items) else items
    if isinstance(value, dict):
        entries = {
            key: _configuration(item)
            for key, item in value.items()
            if isinstance(key, str) and key not in _NON_CONFIGURATION_KEYS
        }
        return {key: item for key, item in entries.items() if item is not _OPAQUE}
    return _OPAQUE


def _provenance(backbone):
    metadata = getattr(backbone, "metadata", {})
    record = {
        key: metadata[key]
        for key in _PROVENANCE_KEYS
        if isinstance(metadata.get(key), (str, int, float, bool))
    }
    source = metadata.get("source")
    if isinstance(source, dict):
        record["source"] = {
            key: source[key]
            for key in ("repository", "url", "commit", "branch")
            if isinstance(source.get(key), str)
        }
    configuration = metadata.get("options", getattr(backbone, "options", None))
    if isinstance(configuration, dict):
        record["configuration"] = _configuration(configuration)
    return record


@dataclass(frozen=True)
class _Binding:
    spec: dict
    mode: str
    targets: tuple = ()
    capture_index: int = -1

    @contextmanager
    def capture(self):
        captured = []
        if self.mode == "slidegraph":
            handles = [
                module.register_forward_hook(lambda module, args, output: captured.append(output))
                for module in self.targets
            ]
        else:
            handles = [
                module.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
                for module in self.targets
            ]
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _select(self, backbone, output, captured):
        if self.mode == "embedding":
            return output.embedding
        if self.mode == "slidegraph":
            if len(captured) != len(self.targets):
                raise ValueError("SlideGraph feature stages did not match the declared binding")
            # The native adapter enforces one graph; retain its configured
            # feature pooling separately at each recorded stage.
            return torch.cat([backbone.native.pooling(stage, None) for stage in captured], dim=-1)
        if not captured:
            raise ValueError("The declared representation boundary was not executed")
        return captured[self.capture_index]

    def resolve(self, backbone, output, captured):
        value = self._select(backbone, output, captured)
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError("The bound MIL representation must be a floating-point tensor")
        dimension = self.spec["dimension"]
        if value.numel() != dimension:
            raise ValueError(
                f"Bound representation axes differ from the declared dimension {dimension}: {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("The bound MIL representation contains nonfinite values")
        # Every binding describes one slide. Class axes are flattened in their
        # native order; no class/token/node averaging is introduced here.
        return value.reshape(1, dimension)


def _binding(backbone, representation="profile"):
    if representation != "profile":
        raise ValueError(
            "representation must be 'profile'; arbitrary tensor pooling is not inferred"
        )
    name = _profile(backbone)
    boundary = _BOUNDARIES[name](backbone)
    spec = {
        "schema_version": 1,
        "profile": name,
        "dimension": int(boundary.dimension),
        "axes": boundary.axes or deepcopy(_SLIDE_AXES),
        "policy": boundary.policy,
        "adaptations": [boundary.adaptation] if boundary.adaptation else [],
        "backbone_provenance": _provenance(backbone),
    }
    if boundary.class_count is not None:
        spec["class_order"] = {
            "kind": "native_classifier_index",
            "values": list(range(boundary.class_count)),
        }
    return _Binding(spec, boundary.mode, boundary.targets, boundary.capture_index)


def get_representation_spec(backbone, representation="profile"):
    """Return a JSON-safe, inspectable binding without executing the backbone."""
    return deepcopy(_binding(backbone, representation).spec)


def representation_dimension(backbone, representation="profile"):
    return get_representation_spec(backbone, representation)["dimension"]
