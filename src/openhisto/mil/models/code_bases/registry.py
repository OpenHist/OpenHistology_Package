"""Lazy model catalogue and one-bag construction API."""

from importlib import import_module
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_GROUPS = {
    "attention": {
        "abmil": "attention_deep_mil",
        "gated_abmil": "attention_deep_mil",
        "dsmil": "dsmil",
        "dtfd_mil": "dtfd_mil",
        "acmil": "acmil",
        "asmil": "asmil",
        "clam_sb": "clam",
        "clam_mb": "clam",
        "varpool": "varpool",
        "panther": "panther",
    },
    "graph_multiscale": {
        "wikg": "wikg",
        "tad_graph": "tad_graph",
        "resgat": "resgat",
        "patch_gcn": "patch_gcn",
        "slidegraph_plus": "slidegraph",
        "cs_mil": "cs_mil",
        "zoommil": "zoommil",
        "hag_mil": "hag_mil",
        "gfr_mil": "gfr_mil",
        "h2_mil": "h2_mil",
        "hipt": "hipt",
    },
    "transformers": {
        "transmil": "transmil",
        "ilra": "ilra",
        "rrt_mil": "rrt_mil",
        "dgr_mil": "dgr_mil",
        "caprmil": "caprmil",
        "lin_mil": "lin_mil",
        "mamba_mil": "mamba_mil",
        "nnmil": "nnmil",
        "mhim_v1": "mhim_mil_v1",
        "mhim_v2": "mhim_mil",
    },
    "baselines": {"mean_mil": None, "max_mil": None, "mean_std_mil": None},
}
_PROFILES = {
    name: (group, slug) for group, items in _GROUPS.items() for name, slug in items.items()
}
_ALIASES = {
    "meanmil": "mean_mil",
    "maxmil": "max_mil",
    "meanstdmil": "mean_std_mil",
    "gatedabmil": "gated_abmil",
    "dtfd": "dtfd_mil",
    "clam": "clam_sb",
    "patchgcn": "patch_gcn",
    "slidegraph": "slidegraph_plus",
    "slidegraph+": "slidegraph_plus",
    "h2mil": "h2_mil",
    "h²_mil": "h2_mil",
    "ilra_mil": "ilra",
    "r2t_mil": "rrt_mil",
    "r²t_mil": "rrt_mil",
    "rrtmil": "rrt_mil",
    "linmil": "lin_mil",
    "mambamil": "mamba_mil",
    "mhim_mil": "mhim_v1",
    "mhim_mil_v2": "mhim_v2",
}
_GRAPH = {"tad_graph", "resgat", "patch_gcn", "slidegraph_plus", "h2_mil"}
_MULTI = {"cs_mil", "zoommil", "hag_mil", "gfr_mil", "hipt"}


@lru_cache(maxsize=1)
def _hashed_implementation(signature):
    digest = hashlib.sha256()
    for name, _, _ in signature:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def implementation_hash():
    paths = list(ROOT.glob("*.py"))
    for directory in ("adapters", "compat", "tests"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend((ROOT / "compat").glob("*_repairs.json"))
    signature = tuple(
        (str(path.relative_to(ROOT)), path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(paths)
    )
    return _hashed_implementation(signature)


def canonical_name(name):
    if not isinstance(name, str):
        raise TypeError("Model name must be a string")
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in _PROFILES:
        raise ValueError("Unknown MIL profile; call list_models() for supported names")
    return normalized


def model_info(name):
    name = canonical_name(name)
    family, slug = _PROFILES[name]
    source = None
    if slug:
        record = ROOT / "compat" / "sources" / (slug + ".json")
        source = (
            json.loads(record.read_text()) if record.exists() else {"checkout": "upstream/" + slug}
        )
    coverage_path = ROOT / "coverage.json"
    report = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    coverage = dict(report.get("models", {}).get(name, {}))
    if coverage:
        coverage["verified_at_utc"] = report["verified_at_utc"]
        current_commit = source.get("commit") if source else None
        if (
            report.get("implementation_sha256") != implementation_hash()
            or coverage.get("source_commit") != current_commit
        ):
            coverage["previous_status"] = coverage["status"]
            coverage["status"] = "not_validated_for_current_implementation"
    repair_path = (
        ROOT
        / "compat"
        / (("transformer" if family == "transformers" else family) + "_repairs.json")
    )
    repairs = json.loads(repair_path.read_text()) if repair_path.exists() else {}
    return {
        "name": name,
        "family": family,
        "source": source,
        "origin": "upstream with documented adapter/repairs"
        if slug
        else "original pooling baseline",
        "source_collected": bool(source and (ROOT / source["checkout"]).is_dir()) if slug else True,
        "input_kind": "prepared_graph"
        if name in _GRAPH
        else "prepared_hierarchy"
        if name in _MULTI
        else "feature_bag",
        "batching": "one unpadded bag/graph/hierarchy per call",
        "default_device": "cpu",
        "default_dtype": "float32",
        "supported_devices": ["cpu"],
        "entrypoint": "openhisto.mil.models.code_bases.adapters." + family + ".build",
        "tasks": {
            "patch_gcn": ["discrete_survival"],
            "tad_graph": ["classification", "discrete_survival"],
            "gfr_mil": ["classification", "discrete_survival"],
            "cs_mil": ["independent_sigmoid"],
            "panther": ["representation", "classification_with_adapter_head"],
        }.get(name, ["classification"]),
        "artifacts": "explicit prototype array required"
        if name == "panther"
        else "checkpoints optional and explicit",
        "license_files": source.get("license_files", []) if source else [],
        "repairs": repairs,
        "validation": coverage or {"status": "not_yet_validated"},
        "dependencies": ["openhisto[mil]"]
        + ([] if family == "baselines" else [f"openhisto[mil-{family.replace('_', '-')}]"]),
        "training": "Native auxiliary components/hooks retained; shared training pipeline deferred",
    }


def list_models():
    """Inspect all profiles without importing Torch or upstream models."""
    return [model_info(name) for name in _PROFILES]


def create_model(name, input_dim=None, num_classes=None, device="cpu", **model_options):
    """Construct only the selected model, without downloading any artifacts."""
    name = canonical_name(name)
    import torch

    selected_device = torch.device(device)
    if selected_device.type != "cpu":
        raise ValueError(
            'This release supports device="cpu"; accelerated execution has not been validated'
        )
    for key, value in (("input_dim", input_dim), ("num_classes", num_classes)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{key} must be a positive integer or None")
    family, slug = _PROFILES[name]
    if slug and not model_info(name)["source_collected"]:
        raise FileNotFoundError(
            "Upstream sources are missing; run python -m openhisto.mil.models.code_bases.sources"
        )
    module = import_module(".adapters." + family, __package__)
    try:
        model = module.build(name, input_dim=input_dim, num_classes=num_classes, **model_options)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Missing dependency {exc.name}; install openhisto[mil-{family.replace('_', '-')}] in a separate environment"
        ) from exc
    from .base import MILAdapter

    if not isinstance(model, MILAdapter):
        raise TypeError("MIL factory did not return a MILAdapter")
    model.to(device=selected_device, dtype=torch.float32)
    model.metadata = {**model_info(name), **model.metadata}
    return model
