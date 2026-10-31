"""Native graph and hierarchical MIL adapters; graphs/mappings are caller supplied.

No model weights, graph construction, slide data or training pipeline is fetched.
Native outputs are retained even when a standard output view is also available.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Callable

import torch
from torch import nn

from ..base import MILAdapter, MILOutput, as_bag, output_from_logits
from ..compat.loader import load_native
from ..compat import graph_repairs, h2_repairs, hipt_repairs
from ..compat.graph_ops import COOAdj


_TAD_TASKS = {"subtyping", "staging", "her2", "gleason", "survival_prediction"}
_HIPT_GLOBALS = ("global_phi", "global_transformer", "global_attn_pool", "global_rho")
_H2_OUTPUTS = ("clusters", "node_types", "scores", "trees", "coordinates", "edges")


# ---------------------------------------------------------------- input checks


def _tensor(value, name, *, ndim=None, allow_empty=False):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not allow_empty and value.numel() == 0:
        raise ValueError(f"{name} must be nonempty and finite")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _graph(inputs, dim, *, edge="edge_index", allow_empty_edges=False):
    graph = inputs.get("graph") if isinstance(inputs, dict) else inputs
    if not hasattr(graph, "x") or not hasattr(graph, edge):
        raise TypeError(f"Pass a prepared PyG graph with x and {edge}")
    as_bag(graph.x, dim)
    edges = _tensor(getattr(graph, edge), edge, ndim=2, allow_empty=allow_empty_edges)
    if edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError(f"{edge} must be a long tensor [2,E]")
    if edges.numel() and (edges.min() < 0 or edges.max() >= len(graph.x)):
        raise ValueError("Edge indices are outside the node feature array")
    if edges.device != graph.x.device:
        raise ValueError("Graph tensors must share the feature device")
    graph = graph.clone()
    if getattr(graph, "batch", None) is None:
        graph.batch = edges.new_zeros(len(graph.x))
    if (
        graph.batch.shape != (len(graph.x),)
        or graph.batch.dtype != torch.long
        or not (graph.batch == 0).all()
    ):
        raise ValueError("One-graph batch labels must be a long vector of zeros [N]")
    if graph.batch.device != graph.x.device:
        raise ValueError("Graph batch labels must share the feature device")
    return graph


def _levels(inputs, dim, *, batched=False):
    if not isinstance(inputs, dict) or "levels" not in inputs:
        raise TypeError("Pass {'levels': (low, middle, high)} in native parent-major order")
    levels = tuple(as_bag(level, dim, batched=batched) for level in inputs["levels"])
    if len(levels) != 3:
        raise ValueError("Exactly three aligned magnification levels are required")
    counts = [level.shape[-2] for level in levels]
    if counts[1] % counts[0] or counts[2] % counts[1]:
        raise ValueError("Native parent-major magnification levels require integer child ratios")
    if len({level.device for level in levels}) != 1:
        raise ValueError("All magnification levels must share a device")
    return levels


def _tad_graph(inputs, dim):
    """Validate TAD edge descriptors and add the target-major adjacency it reads."""
    graph = _graph(inputs, dim)
    attrs = _tensor(getattr(graph, "edge_attr", None), "edge_attr", ndim=2)
    if attrs.shape != (graph.edge_index.shape[1], 2) or attrs.min() < 0 or attrs.max() > 1:
        raise ValueError(
            "TAD edge_attr must contain normalized distance/angle [E,2] values in [0,1]"
        )
    n = len(graph.x)
    order = (graph.edge_index[1] * n + graph.edge_index[0]).argsort(stable=True)
    graph.adj_t = COOAdj(graph.edge_index[1, order], graph.edge_index[0, order], n)
    graph.edge_attr = attrs[order]
    return graph


def _cs_mil_inputs(inputs, dim, cluster_num):
    if not isinstance(inputs, dict) or "clusters" not in inputs:
        raise TypeError("CS-MIL requires {'clusters': list([P_i,D,3,1]), 'mask': [1,sum(P_i)]}")
    clusters = inputs["clusters"]
    if len(clusters) != cluster_num:
        raise ValueError("CS-MIL cluster count differs from cluster_num")
    for cluster in clusters:
        _tensor(cluster, "cluster", ndim=4)
        if cluster.shape[1:] != (dim, 3, 1):
            raise ValueError("CS-MIL clusters must have shape [P_i,D,3,1]")
    mask = inputs.get("mask")
    if mask is None:
        return clusters, None
    _tensor(mask, "mask", ndim=2)
    patch_count = sum(len(cluster) for cluster in clusters)
    if (
        mask.shape != (1, patch_count)
        or not ((mask == 0) | (mask == 1)).all()
        or not (mask > 0).any()
    ):
        raise ValueError("CS-MIL mask must be [1,sum(P_i)] with at least one enabled patch")
    return clusters, mask.to(clusters[0])


def _gfr_inputs(inputs, dim, high_dim):
    """Return the low-magnification bag and the keyword inputs GFR-MIL requires."""
    if not isinstance(inputs, dict):
        raise TypeError("GFR-MIL requires low/high features, coordinates and region_labels")
    low = as_bag(inputs, dim)
    high = as_bag(inputs.get("high_features"), high_dim)
    coords = _tensor(inputs.get("coords"), "coords", ndim=2)
    high_coords = _tensor(inputs.get("high_coords"), "high_coords", ndim=2)
    mapping = _tensor(inputs.get("region_labels"), "region_labels")
    if coords.shape != (len(low), 2) or high_coords.shape != (len(high), 2):
        raise ValueError("GFR coordinates must align with low/high feature rows")
    if mapping.ndim != 1 or len(mapping) != len(high) or mapping.dtype != torch.long:
        raise ValueError(
            "region_labels must be a long vector with one parent index per high feature"
        )
    if coords.min() < 0 or high_coords.min() < 0 or mapping.min() < 0 or mapping.max() >= len(low):
        raise ValueError("GFR coordinates/parent indices are outside their valid ranges")
    return low, {
        "coords": coords,
        "high_features": high,
        "high_coords": high_coords,
        "region_labels": mapping,
    }


def _h2_graph(inputs, dim):
    graph = _graph(inputs, dim, edge="edge_index_tree_8nb")
    for name in ("node_type", "node_tree", "x_y_index"):
        _tensor(getattr(graph, name, None), name)
    types, tree = graph.node_type, graph.node_tree
    if (
        types.shape != (len(graph.x),)
        or tree.shape != types.shape
        or graph.x_y_index.shape != (len(graph.x), 2)
    ):
        raise ValueError(
            "H2 node types, parent tree and normalized coordinates must align with node rows"
        )
    if (
        types.dtype != torch.long
        or tree.dtype != torch.long
        or set(types.tolist()) != {0, 1, 2}
        or (types == 0).sum() != 1
    ):
        raise ValueError("H2 requires one root and nonempty long resolution types 0/1/2")
    coarse = torch.where(types == 1)[0]
    fine = torch.where(types == 2)[0]
    ordered = torch.cat(
        (types.new_tensor([0]), types.new_ones(len(coarse)), types.new_full((len(fine),), 2))
    )
    if (
        not torch.equal(types, ordered)
        or tree[0] != -1
        or not (tree[coarse] == 0).all()
        or not torch.isin(tree[fine], coarse).all()
        or not torch.isin(coarse, tree[fine]).all()
    ):
        raise ValueError(
            "H2 requires root/coarse/fine order and at least one fine child for every coarse parent"
        )
    if graph.x_y_index.min() < 0 or graph.x_y_index.max() > 1:
        raise ValueError("H2 x_y_index must contain normalized coordinates in [0,1]")
    if getattr(graph, "data_id", None) is None:
        graph.data_id = graph.node_type.new_zeros(len(graph.x))
    return graph


def _hipt_features(inputs, variant, dim):
    value = _tensor(inputs.get("features") if isinstance(inputs, dict) else inputs, "HIPT features")
    if variant == "gp":
        return as_bag(value, 192)
    if variant == "lgp" and (value.ndim != 3 or value.shape[1:] != (256, 384)):
        raise ValueError(
            "HIPT LGP requires raster-ordered [regions,256,384] native patch embeddings"
        )
    if variant == "none" and (value.ndim != 3 or value.shape[-1] != dim):
        raise ValueError("HIPT None requires grouped [regions,patches,D] embeddings")
    return value


@contextmanager
def _capture_output(module):
    """Record ``module``'s forward outputs while the block runs."""
    captured = []
    handle = module.register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        yield captured
    finally:
        handle.remove()


# ------------------------------------------------------------- native modules


class _HAGHierarchy(nn.Module):
    """Native main.py selection cascade using three upstream IATModel modules."""

    def __init__(self, cls, dim, classes, *, select_k=8, **options):
        super().__init__()
        self.models = nn.ModuleList(
            [cls(feat_in=dim, n_classes=classes, **options) for _ in range(3)]
        )
        self.select_k = int(select_k)
        if self.select_k < 1:
            raise ValueError("select_k must be positive")

    def forward(self, levels, *, label=None, instance_eval=False):
        low, middle, high = levels
        if len(middle) != 4 * len(low) or len(high) != 4 * len(middle):
            raise ValueError("HAG-MIL requires four contiguous children per parent at each level")
        first = self.models[0](low, label=label, instance_eval=instance_eval)
        choose_low = first[3].reshape(-1).argsort(descending=True)[: self.select_k].detach()
        offsets = torch.arange(4, device=low.device)
        middle_indices = (choose_low[:, None] * 4 + offsets).reshape(-1)
        second = self.models[1](middle[middle_indices], label=label, instance_eval=instance_eval)
        choose_middle = second[3].reshape(-1).argsort(descending=True)[: self.select_k].detach()
        high_indices = (middle_indices[choose_middle, None] * 4 + offsets).reshape(-1)
        third = self.models[2](high[high_indices], label=label, instance_eval=instance_eval)
        return third, (first, second, third), (middle_indices, high_indices)


@dataclass
class _Native:
    """A constructed upstream model and what its adapter records about it."""

    module: nn.Module
    variant: str = None
    metadata: dict = field(default_factory=dict)


def _given(path):
    # Upstream configurations spell an absent checkpoint as the string 'None'.
    return path not in {None, "None"}


def _build_wikg(dim, classes, opts):
    return _Native(graph_repairs.wikg().WiKG(dim_in=dim, n_classes=classes, **opts))


def _build_resgat(dim, classes, opts):
    return _Native(
        load_native("resgat", "models.resgat").ResGATs(input_dim=dim, n_classes=classes, **opts)
    )


def _build_patch_gcn(dim, classes, opts):
    if opts.get("num_layers", 4) < 2:
        raise ValueError("Patch-GCN requires at least two concatenated feature stages")
    return _Native(
        graph_repairs.patch_gcn().PatchGCN_Surv(num_features=dim, n_classes=classes, **opts)
    )


def _build_slidegraph(dim, classes, opts):
    # Released train.py configuration, rather than the utility class's GIN default.
    opts = {
        "conv": "EdgeConv",
        "layers": [16, 16, 8],
        "pooling": "mean",
        "dropout": 0.5,
        "aggr": "max",
        **opts,
    }
    return _Native(graph_repairs.slidegraph().GNN(dim_features=dim, dim_target=classes, **opts))


def _build_tad_graph(dim, classes, opts):
    from torch_geometric.nn import MessagePassing

    opts = {"batch_size": 1, **opts}
    if opts.get("task", "subtyping") not in _TAD_TASKS:
        raise ValueError(
            "TAD task must be subtyping, staging, her2, gleason or survival_prediction"
        )
    native = graph_repairs.tad_graph().TADGraphModule(embed_size=dim, n_classes=classes, **opts)
    # The same propagate repair as compat.graph_repairs.tad_graph, per instance.
    for convolution in native.clf.convs_list:
        convolution.propagate = MethodType(MessagePassing.propagate, convolution)
    return _Native(native, metadata={"task": native.hparams.task})


def _build_zoommil(dim, classes, opts):
    return _Native(graph_repairs.zoommil().ZoomMIL(in_feat_dim=dim, n_cls=classes, **opts))


def _build_cs_mil(dim, classes, opts):
    opts = {"cluster_num": 8, **opts}
    return _Native(
        graph_repairs.cs_mil().DeepAttnMIL_Surv(input_dim=dim, num_classes=classes, **opts)
    )


def _build_hag_mil(dim, classes, opts):
    return _Native(
        _HAGHierarchy(load_native("hag_mil", "models.model").IATModel, dim, classes, **opts)
    )


def _build_gfr_mil(dim, classes, opts):
    module = load_native("gfr_mil", "gfr_mil.models", roots=("src",))
    task = opts.pop("task", "classification")
    if task == "classification":
        return _Native(module.build_model(input_dim=dim, num_classes=classes, **opts))
    if task == "survival":
        return _Native(module.build_survival_model(input_dim=dim, n_bins=classes, **opts))
    raise ValueError("GFR task must be classification or survival")


def _build_h2_mil(dim, classes, opts):
    hidden = int(opts.pop("hidden_dim", dim))
    checkpoint = opts.pop("checkpoint", None)
    if hidden < 2:
        raise ValueError("H2 hidden_dim must be at least 2")
    native = h2_repairs.module().GCN(in_feats=dim, n_hidden=hidden, out_classes=hidden, **opts)
    native.lin2 = nn.Linear(hidden // 2, classes)
    if checkpoint is None:
        return _Native(native)
    # Normalization names/widths changed deliberately; incompatible
    # historical state dictionaries require an explicit migration.
    native.load_state_dict(
        torch.load(Path(checkpoint), map_location="cpu", weights_only=True), strict=True
    )
    return _Native(native, metadata={"weights": "provided_checkpoint"})


def _hipt_input_dim(options):
    # GP consumes 192-wide region features; LGP and None consume 384-wide patch features.
    return 192 if str(options.get("variant", "lgp")).lower() == "gp" else 384


def _build_hipt(dim, classes, opts):
    variant = opts.pop("variant", "lgp").lower()
    if variant not in {"none", "gp", "lgp"}:
        raise ValueError("HIPT variant must be none, gp or lgp")
    if variant == "lgp" and dim != 384 or variant == "gp" and dim != 192:
        raise ValueError("Native HIPT LGP input width is 384; GP region width is 192")
    checkpoint = opts.pop("checkpoint", None)
    local_checkpoint = opts.pop("pretrain_4k", None)
    global_checkpoint = opts.pop("pretrain_WSI", None)
    freeze_local = opts.pop("freeze_4k", False)
    freeze_global = opts.pop("freeze_WSI", False)
    # The actual GP/LGP architecture has 192-wide global features. Upstream
    # big's 512-wide classifier is inconsistent with those native modules.
    size = opts.pop("size_arg", "small")
    if size not in {"small", "big"}:
        raise ValueError("HIPT size_arg must be small or big")
    module = hipt_repairs.module()
    cls = {"none": module.HIPT_None_FC, "gp": module.HIPT_GP_FC, "lgp": module.HIPT_LGP_FC}[variant]
    native = cls(
        path_input_dim=dim,
        n_classes=classes,
        size_arg=size if variant == "none" else "small",
        **opts,
    )
    if _given(local_checkpoint):
        if variant != "lgp":
            raise ValueError("pretrain_4k applies to HIPT LGP only")
        hipt_repairs.load_weights(native.local_vit, local_checkpoint, teacher=True)
    if _given(global_checkpoint):
        if variant == "none":
            raise ValueError("pretrain_WSI requires the native GP/LGP global transformer")
        hipt_repairs.load_weights(
            nn.ModuleDict({key: getattr(native, key) for key in _HIPT_GLOBALS}), global_checkpoint
        )
    if freeze_local:
        if not hasattr(native, "local_vit"):
            raise ValueError("freeze_4k applies to HIPT LGP only")
        native.local_vit.requires_grad_(False)
    if freeze_global:
        for key in _HIPT_GLOBALS:
            if hasattr(native, key):
                getattr(native, key).requires_grad_(False)
    if checkpoint is not None:
        hipt_repairs.load_weights(native, checkpoint)
    pretrained = any(_given(path) for path in (checkpoint, local_checkpoint, global_checkpoint))
    return _Native(
        native,
        variant=variant,
        metadata={
            "weights": "provided_checkpoint" if pretrained else "random_initialization",
            "effective_global_width": 192 if variant != "none" else None,
            "requested_size": size,
        },
    )


# ------------------------------------------------------------------- forwards


def _forward_wikg(adapter, inputs, **kwargs):
    bag = as_bag(inputs, adapter.input_dim, batched=True)
    if bag.shape[1] < adapter.native.topk:
        raise ValueError(f"WiKG requires at least topk={adapter.native.topk} instances")
    native = adapter.native(bag)
    return output_from_logits(native, native=native)


def _forward_resgat(adapter, inputs, **kwargs):
    native = adapter.native(_graph(inputs, adapter.input_dim, allow_empty_edges=True))
    return output_from_logits(native[0], aux={"log_probabilities": native[1]}, native=native)


def _forward_slidegraph(adapter, inputs, **kwargs):
    native = adapter.native(_graph(inputs, adapter.input_dim))
    return output_from_logits(
        native[0], embedding=native[2], aux={"node_scores": native[1]}, native=native
    )


def _forward_patch_gcn(adapter, inputs, **kwargs):
    edge = "edge_latent" if adapter.native.edge_agg == "latent" else "edge_index"
    graph = _graph(inputs, adapter.input_dim, edge=edge, allow_empty_edges=True)
    with _capture_output(adapter.native.classifier) as captured:
        native = adapter.native(x_path=graph)
    return MILOutput(
        logits=captured[0].reshape(1, -1),
        probabilities=None,
        attention=native[3],
        aux={"hazards": native[0], "survival": native[1], "task": "discrete_survival"},
        native_output=native,
    )


def _forward_tad_graph(adapter, inputs, **kwargs):
    graph = _tad_graph(inputs, adapter.input_dim)
    # Native TAD logs batch_size and its prior uses the current epoch.
    adapter.native.current_epoch = int(kwargs.pop("epoch", 0))
    split = kwargs.pop("split", "train" if adapter.training else "test")
    if kwargs:
        raise TypeError(f"Unknown TAD forward options: {sorted(kwargs)}")
    native = adapter.native((graph,), 0, split=split)
    aux, logits, trivial = native[0], native[1], native[2]
    if adapter.native.hparams.task != "survival_prediction":
        return output_from_logits(logits, aux={**aux, "trivial_logits": trivial}, native=native)
    hazards = logits.sigmoid()
    survival = torch.cumprod(1 - hazards, dim=1)
    return MILOutput(
        logits=logits,
        probabilities=None,
        aux={
            **aux,
            "task": "survival_prediction",
            "hazards": hazards,
            "survival": survival,
            "risk": -survival.sum(dim=1),
            "trivial_logits": trivial,
            "trivial_hazards": trivial.sigmoid(),
        },
        native_output=native,
    )


def _forward_zoommil(adapter, inputs, **kwargs):
    native = adapter.native(_levels(inputs, adapter.input_dim, batched=True))
    return MILOutput(logits=native[0], probabilities=native[2], native_output=native)


def _forward_hag_mil(adapter, inputs, **kwargs):
    native = adapter.native(_levels(inputs, adapter.input_dim), **kwargs)
    final, stages, indices = native
    return MILOutput(
        logits=final[0],
        probabilities=final[1],
        attention=final[3],
        aux={"stages": stages, "selected_indices": indices},
        native_output=native,
    )


def _forward_cs_mil(adapter, inputs, **kwargs):
    clusters, mask = _cs_mil_inputs(inputs, adapter.input_dim, adapter.native.cluster_num)
    with _capture_output(adapter.native.fc6[-2]) as captured:
        native = adapter.native(clusters, mask)
    return MILOutput(
        logits=captured[0],
        probabilities=native,
        aux={"task": "binary_or_independent_sigmoid_heads"},
        native_output=native,
    )


def _forward_gfr_mil(adapter, inputs, **kwargs):
    high_dim = adapter.options.get("high_input_dim", adapter.input_dim)
    low, regions = _gfr_inputs(inputs, adapter.input_dim, high_dim)
    with _capture_output(adapter.native.model.classfier) as captured:
        native = adapter.native(low, **regions, **kwargs)
    # Upstream survival adapter calls its hazard tensor 'logits'.
    # Capture the actual classifier output rather than propagating that label.
    return MILOutput(
        logits=captured[0].reshape(1, -1),
        probabilities=native.get("probs"),
        aux={k: v for k, v in native.items() if k not in {"logits", "probs"}},
        native_output=native,
    )


def _forward_h2_mil(adapter, inputs, **kwargs):
    graph = _h2_graph(inputs, adapter.input_dim)
    with _capture_output(adapter.native.lin2) as captured:
        native = adapter.native(graph)
    return MILOutput(
        logits=captured[0],
        probabilities=native[0],
        aux=dict(zip(_H2_OUTPUTS, native[1:])),
        native_output=native,
    )


def _forward_hipt(adapter, inputs, **kwargs):
    native = adapter.native(_hipt_features(inputs, adapter.variant, adapter.input_dim), **kwargs)
    return MILOutput(
        logits=native[0], probabilities=native[1], attention=native[3], native_output=native
    )


# ------------------------------------------------------------------- registry


@dataclass(frozen=True)
class _Model:
    build: Callable
    forward: Callable
    input_dim: int | Callable = 1024
    num_classes: int = 2

    def default_input_dim(self, options):
        return self.input_dim(options) if callable(self.input_dim) else self.input_dim


_MODELS = {
    "wikg": _Model(_build_wikg, _forward_wikg, input_dim=384),
    "tad_graph": _Model(_build_tad_graph, _forward_tad_graph, input_dim=768),
    "resgat": _Model(_build_resgat, _forward_resgat),
    "patch_gcn": _Model(_build_patch_gcn, _forward_patch_gcn, num_classes=4),
    "slidegraph_plus": _Model(_build_slidegraph, _forward_slidegraph),
    "cs_mil": _Model(_build_cs_mil, _forward_cs_mil, input_dim=2048, num_classes=1),
    "zoommil": _Model(_build_zoommil, _forward_zoommil),
    "hag_mil": _Model(_build_hag_mil, _forward_hag_mil),
    "gfr_mil": _Model(_build_gfr_mil, _forward_gfr_mil),
    "h2_mil": _Model(_build_h2_mil, _forward_h2_mil, num_classes=4),
    "hipt": _Model(_build_hipt, _forward_hipt, input_dim=_hipt_input_dim, num_classes=4),
}

NAMES = tuple(_MODELS)


class NativeGraphAdapter(MILAdapter):
    def __init__(self, name, native, dim, classes, *, variant=None, options=None):
        super().__init__()
        self.name, self.native, self.input_dim = name, native, dim
        self.num_classes, self.variant = classes, variant
        self.options = options or {}
        self.metadata = {
            "name": name,
            "implementation": "pinned_upstream_with_compatibility_repairs",
            "input_dim": dim,
            "num_classes": classes,
            "variant": variant,
            "weights": "random_initialization",
            "input_preparation": "caller_supplied",
            "repair_manifest": "compat/graph_multiscale_repairs.json",
        }

    def forward(self, inputs, **kwargs):
        return _MODELS[self.name].forward(self, inputs, **kwargs)


def build(name, input_dim=None, num_classes=None, **options):
    """Construct an actual upstream model on CPU; use .to(device) explicitly.

    Compatibility dimensions/options are model-specific. Caller-owned graphs
    and native multiscale row ordering are mandatory where the model needs them.
    """
    if name not in NAMES:
        raise ValueError(f"Unknown graph/multiscale model: {name}")
    model = _MODELS[name]
    dim = int(input_dim if input_dim is not None else model.default_input_dim(options))
    classes = int(num_classes if num_classes is not None else model.num_classes)
    if dim < 1 or classes < 1:
        raise ValueError("input_dim and num_classes must be positive")
    native = model.build(dim, classes, dict(options))
    adapter = NativeGraphAdapter(
        name, native.module, dim, classes, variant=native.variant, options=options
    )
    adapter.metadata.update(native.metadata)
    return adapter
