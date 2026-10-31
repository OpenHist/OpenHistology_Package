"""Plan KPMP downloads from a user-selected Atlas CSV, without endpoint probing."""

from __future__ import annotations

import csv
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import quote

from ..models import AcquisitionPlan, Asset, MetadataRecord, utc_now


ATLAS_URL = "https://atlas.kpmp.org"
DOWNLOAD_URL = ATLAS_URL + "/api/v1/file/download"
WSI_SUFFIXES = {".svs", ".tif", ".tiff"}


def _clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def _key(value):
    return re.sub(r"[\s_]+", " ", str(value).strip()).casefold()


def _text(value):
    return "" if value is None else str(value).strip()


def _columns(headers):
    result = {}
    for header in headers:
        normalized = _key(header)
        if header is None or not normalized or normalized in result:
            raise ValueError("Metadata must have nonempty, unique column names")
        result[normalized] = header
    return result


def _read_table(path):
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("Metadata CSV has no header")
            _columns(reader.fieldnames)
            rows = list(reader)
            if any(None in row for row in rows):
                raise ValueError("Metadata CSV has rows wider than its header")
            return _clean(rows), list(reader.fieldnames)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("records") if isinstance(value, dict) else value
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".xlsx":
        try:
            import openpyxl
        except ImportError as error:
            raise ValueError("Install openpyxl to read XLSX clinical metadata") from error
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            values = iter(book.active.iter_rows(values_only=True))
            headers = next(values, None)
            if not headers:
                raise ValueError("Clinical workbook has no header")
            _columns(headers)
            rows = [
                dict(zip(headers, cells))
                for cells in values
                if any(cell is not None for cell in cells)
            ]
        finally:
            book.close()
    else:
        raise ValueError(
            "Clinical metadata must be CSV, JSON, JSONL, or XLSX; keep other source files for manual review"
        )
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Clinical JSON must contain a list of row objects (or a records list)")
    rows = _clean(rows)
    headers = list(dict.fromkeys(key for row in rows for key in row))
    if headers:
        _columns(headers)
    return rows, headers


def _stable_id(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def parse_clinical(path, *, column_mapping=None) -> list[MetadataRecord]:
    """Preserve source cells and link rows by the exported participant ID.

    ``column_mapping`` maps ``subject_id`` (or ``participant_id``) to an actual
    column. Without a mapping only Participant ID / participant_id is recognized.
    Public binned values remain exactly as supplied; no unbinning is performed.
    """
    path = Path(path).expanduser().resolve()
    rows, headers = _read_table(path)
    mapping = column_mapping or {}
    column = mapping.get("subject_id", mapping.get("participant_id", "Participant ID"))
    actual = _columns(headers).get(_key(column)) if headers else None
    if actual is None:
        raise ValueError(
            "Clinical metadata needs a Participant ID column or column_mapping['subject_id']"
        )
    records = []
    for index, row in enumerate(rows):
        participant = _text(row.get(actual))
        identity = json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False)
        records.append(
            MetadataRecord(
                record_id=f"KPMP:clinical:{index}:{_stable_id(identity)}",
                kind="clinical",
                subject_ids=[participant] if participant else [],
                data=row,
                source=str(path),
            )
        )
    return records


def discover(selection: dict, *, session=None) -> AcquisitionPlan:
    """Use only an explicit Atlas CSV export; never search private endpoints."""
    if not isinstance(selection, dict) or not selection.get("repository_csv"):
        raise ValueError(
            "KPMP requires selection['repository_csv']: export the selected public files from the Atlas repository"
        )
    repository = Path(selection["repository_csv"]).expanduser().resolve()
    if repository.suffix.lower() != ".csv":
        raise ValueError("KPMP repository_csv must be an Atlas CSV export")
    rows, headers = _read_table(repository)
    columns = _columns(headers)
    required = (
        "Access",
        "Participant ID",
        "File Name",
        "Experimental Strategy",
        "Internal Package ID",
    )
    missing = [name for name in required if _key(name) not in columns]
    if missing:
        raise ValueError("Atlas CSV is missing required columns: " + ", ".join(missing))

    def cell(row, name):
        return _text(row.get(columns.get(_key(name))))

    records = []
    assets_by_id = {}
    excluded_controlled = 0
    excluded_other = 0
    participant_filter = selection.get("participant_ids")
    file_filter = selection.get("file_ids")
    if participant_filter is not None and not isinstance(participant_filter, list):
        raise ValueError("participant_ids must be a list")
    if file_filter is not None and not isinstance(file_filter, list):
        raise ValueError("file_ids must be a list")
    for index, row in enumerate(rows):
        participant = cell(row, "Participant ID")
        record_id = f"KPMP:repository:{index}:{_stable_id(json.dumps(row, sort_keys=True))}"
        records.append(
            MetadataRecord(
                record_id,
                "source_metadata",
                [participant] if participant else [],
                row,
                str(repository),
            )
        )
        access = re.sub(r"[^a-z]", "", cell(row, "Access").lower())
        if access not in {"open", "openaccess", "public"}:
            excluded_controlled += 1
            continue
        if participant_filter is not None and participant not in participant_filter:
            continue
        file_uuid = cell(row, "File UUID")
        if file_filter is not None and file_uuid not in file_filter:
            continue
        filename = cell(row, "File Name")
        package_id = cell(row, "Internal Package ID")
        suffix = Path(filename).suffix.lower()
        descriptors = " ".join(
            cell(row, name) for name in ("Data Category", "Experimental Strategy", "Workflow Type")
        ).casefold()
        normalized = re.sub(r"[^a-z]", "", descriptors)
        microscopy = (
            "lightmicroscopy" in normalized or "lightmicroscopicwholeslideimage" in normalized
        )
        clinical = cell(
            row, "Data Category"
        ).casefold() == "clinical" or "clinicalstudydata" in re.sub(
            r"[^a-z]", "", cell(row, "Experimental Strategy").casefold()
        )
        if microscopy and suffix in WSI_SUFFIXES:
            kind = "slide"
        elif clinical and selection.get("include_clinical", True):
            kind = "clinical"
        else:
            excluded_other += 1
            continue
        if not filename or Path(filename).name != filename or "\\" in filename:
            raise ValueError(
                "Atlas File Name must be the full original basename, including its UUID prefix"
            )
        if not package_id:
            raise ValueError(
                "Selected Atlas rows require Internal Package ID; re-export the repository selection"
            )
        identity = file_uuid or _stable_id(package_id + "\0" + filename)
        safe_identity = (
            identity
            if re.fullmatch(r"[A-Za-z0-9_.-]+", identity) and identity not in {".", ".."}
            else _stable_id(identity)
        )
        asset_id = f"KPMP:{identity}"
        url = f"{DOWNLOAD_URL}/{quote(package_id, safe='')}/{quote(filename, safe='')}"
        if asset_id in assets_by_id:
            asset = assets_by_id[asset_id]
            if asset.url != url or asset.kind != kind:
                raise ValueError(
                    "Atlas export contains conflicting file locations for the same File UUID"
                )
            if participant and participant not in asset.subject_ids:
                asset.subject_ids.append(participant)
            asset.metadata["source_record_ids"].append(record_id)
            continue
        assets_by_id[asset_id] = Asset(
            asset_id=asset_id,
            dataset="KPMP",
            kind=kind,
            filename=filename,
            relative_path=f"{'slides' if kind == 'slide' else 'source_metadata'}/{safe_identity}{suffix}",
            url=url,
            size=None,
            slide_id=identity if kind == "slide" else None,
            subject_ids=[participant] if participant else [],
            metadata={
                "source_record_ids": [record_id],
                "file_uuid": file_uuid,
                "internal_package_id": package_id,
            },
        )
    sources = [str(repository)]
    if selection.get("clinical_path"):
        clinical_path = Path(selection["clinical_path"]).expanduser().resolve()
        records.extend(
            parse_clinical(clinical_path, column_mapping=selection.get("clinical_column_mapping"))
        )
        sources.append(str(clinical_path))
    assets = list(assets_by_id.values())
    clinical_records = [record for record in records if record.kind == "clinical"]
    for asset in assets:
        asset.clinical_refs = [
            record.record_id
            for record in clinical_records
            if set(record.subject_ids) & set(asset.subject_ids)
        ]
    return AcquisitionPlan(
        "KPMP",
        _clean(selection),
        assets,
        records,
        {
            "source": ATLAS_URL,
            "metadata_sources": sources,
            "retrieved_at": utc_now(),
            "discovery": "explicit Atlas repository CSV; open rows only",
            "metadata_availability": {
                "clinical": "public_binned"
                if clinical_records or any(asset.kind == "clinical" for asset in assets)
                else "not_in_repository_export",
                "controlled": "not_downloaded",
            },
            "excluded_controlled_rows": excluded_controlled,
            "excluded_other_rows": excluded_other,
            "sizes": "unknown: human-readable Atlas Size cells are retained only in source rows",
        },
    )


def import_local(root, metadata) -> AcquisitionPlan:
    """Register matching exported filenames in place, without copying slides."""
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Local KPMP root must be a directory")
    selection = dict(metadata) if isinstance(metadata, dict) else {"repository_csv": str(metadata)}
    if "path" in selection and "repository_csv" not in selection:
        selection["repository_csv"] = selection.pop("path")
    plan = discover(selection)
    by_name = {}
    selected_names = {asset.filename for asset in plan.assets}
    for path in root.rglob("*"):
        if path.name in selected_names and path.is_file():
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(
                    "Local KPMP files must remain within root, including symlink targets"
                )
            by_name.setdefault(path.name, []).append(path)
    for asset in plan.assets:
        expected = root / asset.relative_path
        candidates = [expected] if expected.is_file() else by_name.get(asset.filename, [])
        if len(candidates) > 1:
            raise ValueError(
                "Multiple local files match an Atlas File Name; use the planned relative paths"
            )
        if candidates:
            path = candidates[0].resolve()
            if not path.is_relative_to(root):
                raise ValueError("Local KPMP files must remain within root")
            asset.relative_path = path.relative_to(root).as_posix()
            asset.size = path.stat().st_size
            asset.metadata["local_status"] = "present"
        else:
            asset.metadata["local_status"] = "missing"
        asset.metadata["source_url"] = asset.url
        asset.url = None
    plan.provenance["acquisition"] = "local import; files registered in place"
    plan.provenance["imported_at"] = utc_now()
    plan.__post_init__()
    return plan
