"""Provider dispatch, local registration, and linked manifest exports."""

from collections import defaultdict
from copy import deepcopy
import csv
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile

from .models import AcquisitionPlan, canonical_dataset, utc_now
from .state import StateStore, safe_asset_path
from . import transfer


def _provider(dataset):
    name = canonical_dataset(dataset)
    return importlib.import_module(f".{name}.provider", __package__)


def _output_path(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError(
            "Manifest outputs must stay within the dataset root without symlink aliases"
        )
    return path


def discover(dataset, selection, *, session=None) -> AcquisitionPlan:
    """Resolve an explicit cohort and associated metadata, without slide downloads."""
    if not isinstance(selection, dict) or not selection:
        raise ValueError("selection must be a nonempty dictionary")
    return _provider(dataset).discover(deepcopy(selection), session=session)


def download(plan, root, **options):
    """Transfer a saved/discovered plan and export the current slide manifest."""
    if not isinstance(plan, AcquisitionPlan):
        plan = AcquisitionPlan.load(plan)
    report = transfer.download(plan, root, **options)
    build_manifest(root)
    return report


def verify(root):
    """Recheck files against published checksums or the recorded local hashes."""
    report = transfer.verify(root)
    build_manifest(root)
    return report


def _generic_import(dataset, root, metadata):
    """TCGA/CAMELYON imports use an explicit previously discovered plan."""
    options = metadata if isinstance(metadata, dict) else {"plan": metadata}
    value = options.get("plan")
    if value is None:
        raise ValueError("Local TCGA/CAMELYON16 imports require an acquisition plan")
    plan = value if isinstance(value, AcquisitionPlan) else AcquisitionPlan.load(value)
    if plan.dataset != dataset:
        raise ValueError("Import dataset does not match the acquisition plan")
    root = Path(root).resolve()
    slides_dir = Path(options.get("slides_dir", root))
    if not slides_dir.is_absolute():
        slides_dir = root / slides_dir
    slides_dir = slides_dir.resolve()
    if not slides_dir.is_relative_to(root) or not slides_dir.is_dir():
        raise ValueError("slides_dir must be an existing directory within the dataset root")
    files = defaultdict(list)
    for path in slides_dir.rglob("*"):
        if path.is_file() and path.resolve().is_relative_to(root):
            files[path.name].append(path.resolve())
    assets = []
    for asset in plan.assets:
        expected = safe_asset_path(root, asset.relative_path)
        matches = [expected] if expected.is_file() else files.get(asset.filename, [])
        if len(matches) > 1:
            raise ValueError(
                "Local filenames are ambiguous; arrange files at the plan's explicit paths"
            )
        relative = matches[0].relative_to(root).as_posix() if matches else asset.relative_path
        assets.append(replace(asset, relative_path=relative, url=None))
    provenance = {**plan.provenance, "imported_at": utc_now(), "acquisition_mode": "local"}
    return replace(plan, assets=assets, provenance=provenance)


def _preserve_metadata_sources(root, plan):
    """Copy small source tables only; slides are always registered in place."""
    for source in plan.provenance.get("metadata_sources", []):
        path = Path(source)
        if not path.is_file() or path.suffix.lower() not in {
            ".csv",
            ".tsv",
            ".json",
            ".jsonl",
            ".xlsx",
            ".txt",
        }:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        target = safe_asset_path(
            root, f"source_metadata/imports/{digest.hexdigest()[:16]}{path.suffix.lower()}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != target.resolve() and not target.exists():
            shutil.copyfile(path, target)


def import_local(dataset, root, metadata):
    """Register approved files in place and export metadata; never copy slides.

    BRACS accepts a workbook path or options dictionary. KPMP accepts Atlas CSV
    options. TCGA/CAMELYON16 accept an AcquisitionPlan or its saved JSON path.
    """
    dataset = canonical_dataset(dataset)
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Local imports require an existing dataset root")
    if dataset in {"BRACS", "KPMP"}:
        plan = _provider(dataset).import_local(root, metadata)
    else:
        plan = _generic_import(dataset, root, metadata)
    with StateStore(root) as state:
        state.register(plan)
    plan.save(root / "acquisition_plan.json")
    _preserve_metadata_sources(root, plan)
    report = transfer.verify(root)
    build_manifest(root)
    return report


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            name = Path(stream.name)
            write(stream)
        name.replace(path)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def _jsonl(path, records):
    def write(stream):
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    _atomic_write(path, write)


def _unique_records(records):
    """Keep differing rows for the same source key, never pick the first row."""
    groups = defaultdict(dict)
    for record in records:
        value = record.to_dict()
        content = {key: item for key, item in value.items() if key != "source"}
        content["subject_ids"] = sorted(content["subject_ids"])
        signature = json.dumps(content, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if signature not in groups[record.record_id]:
            groups[record.record_id][signature] = {"value": value, "sources": set()}
        if value["source"] is not None:
            groups[record.record_id][signature]["sources"].add(value["source"])
    output, aliases, conflicts = [], {}, set()
    for record_id, rows in sorted(groups.items()):
        aliases[record_id] = []
        if len(rows) > 1:
            conflicts.add(record_id)
        for signature, entry in sorted(rows.items()):
            value = entry["value"]
            sources = sorted(entry["sources"])
            value["sources"] = sources
            value["source"] = sources[0] if len(sources) == 1 else None
            if len(rows) > 1:
                value["record_id"] = (
                    record_id + ":" + hashlib.sha256(signature.encode()).hexdigest()[:12]
                )
            aliases[record_id].append(value["record_id"])
            output.append(value)
    return output, aliases, conflicts


def build_manifest(root) -> Path:
    """Write canonical JSONL/CSV plus lossless source/clinical record exports."""
    root = Path(root).resolve()
    if (root / "acquisition_plan.json").is_symlink():
        raise ValueError("Acquisition plans cannot be symlinks")
    plan = AcquisitionPlan.load(root / "acquisition_plan.json")
    outputs = {
        name: _output_path(root, name)
        for name in (
            "source_metadata/records.jsonl",
            "clinical.jsonl",
            "manifest.jsonl",
            "slides.csv",
            "provenance.json",
        )
    }
    records = list(plan.records)
    with StateStore(root) as state:
        state.register(plan)
        states = {asset.asset_id: state.get(plan.dataset, asset.asset_id) for asset in plan.assets}
    if plan.dataset == "KPMP":
        parser = _provider("KPMP").parse_clinical
        mapping = plan.selection.get("clinical_column_mapping")
        for asset in plan.assets:
            status = states[asset.asset_id] or {}
            if asset.kind == "clinical" and status.get("status") == "complete":
                records.extend(
                    parser(safe_asset_path(root, asset.relative_path), column_mapping=mapping)
                )
    source_records, aliases, conflicts = _unique_records(records)
    clinical = [record for record in source_records if record["kind"] == "clinical"]
    clinical_ids = {record["record_id"] for record in clinical}
    clinical_by_subject = defaultdict(set)
    conflicting_ids = {item for original in conflicts for item in aliases[original]}
    for record in clinical:
        for subject in record["subject_ids"]:
            clinical_by_subject[subject].add(record["record_id"])

    slides = []
    annotations = [asset for asset in plan.assets if asset.kind == "annotation"]
    for asset in plan.assets:
        if asset.kind != "slide":
            continue
        status = states[asset.asset_id] or {}
        refs = set()
        for reference in asset.clinical_refs:
            refs.update(
                item for item in aliases.get(reference, [reference]) if item in clinical_ids
            )
        for subject in asset.subject_ids:
            refs.update(clinical_by_subject[subject])
        if refs & conflicting_ids:
            clinical_status = "conflicting"
        elif refs:
            clinical_status = "available"
        elif plan.dataset in {"BRACS", "CAMELYON16"}:
            clinical_status = "unavailable"
        else:
            clinical_status = "missing"
        slide_id = asset.slide_id or asset.asset_id
        related = [
            {
                "asset_id": annotation.asset_id,
                "local_path": annotation.relative_path,
                "status": (states[annotation.asset_id] or {}).get("status", "pending"),
                "metadata": annotation.metadata,
            }
            for annotation in annotations
            if slide_id in annotation.related_slide_ids
        ]
        slides.append(
            {
                "schema_version": plan.schema_version,
                "dataset": plan.dataset,
                "release": plan.provenance.get("release"),
                "slide_id": slide_id,
                "namespaced_slide_id": f"{plan.dataset}:{slide_id}",
                "asset_id": asset.asset_id,
                "source_filename": asset.filename,
                "subject_ids": asset.subject_ids,
                "group_ids": [f"{plan.dataset}:{subject}" for subject in asset.subject_ids],
                "sample_ids": asset.sample_ids,
                "local_path": asset.relative_path,
                "format": Path(asset.filename).suffix.lower().lstrip("."),
                "status": status.get("status", "pending"),
                "integrity": status.get("integrity"),
                "expected_size": asset.size,
                "actual_size": status.get("actual_size"),
                "source_checksums": asset.checksums,
                "local_sha256": status.get("sha256"),
                "labels": asset.labels,
                "label_status": asset.label_status,
                "split": asset.split,
                "clinical_status": clinical_status,
                "clinical_refs": sorted(refs),
                "source_mag": asset.metadata.get("source_mag"),
                "annotations": related,
                "provider_metadata": asset.metadata,
            }
        )
    _jsonl(outputs["source_metadata/records.jsonl"], source_records)
    _jsonl(outputs["clinical.jsonl"], clinical)
    _jsonl(outputs["manifest.jsonl"], slides)
    fields = [
        "dataset",
        "release",
        "slide_id",
        "asset_id",
        "subject_ids",
        "sample_ids",
        "local_path",
        "format",
        "status",
        "integrity",
        "actual_size",
        "labels",
        "label_status",
        "split",
        "clinical_status",
        "clinical_refs",
        "source_mag",
    ]

    def write_csv(stream):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for slide in slides:
            row = {field: slide[field] for field in fields}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False, allow_nan=False)
            writer.writerow(row)

    _atomic_write(outputs["slides.csv"], write_csv)
    provenance = {
        "schema_version": plan.schema_version,
        "dataset": plan.dataset,
        "selection": plan.selection,
        "source": plan.provenance,
        "manifest_built_at": utc_now(),
        "slide_count": len(slides),
        "clinical_record_count": len(clinical),
        "status_counts": {
            status: sum(slide["status"] == status for slide in slides)
            for status in sorted({slide["status"] for slide in slides})
        },
        "metadata_rules": "Provider fields retained; no survival derivation or generated splits.",
    }
    _atomic_write(
        outputs["provenance.json"],
        lambda stream: stream.write(
            json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ),
    )
    return outputs["manifest.jsonl"]
