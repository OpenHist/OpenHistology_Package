"""Discover explicitly selected open-access GDC slides; do not download images."""

from copy import deepcopy
import re
from urllib.parse import quote

from ..models import AcquisitionPlan, Asset, MetadataRecord, utc_now


API_BASE = "https://api.gdc.cancer.gov"
_PAGE_SIZE = 100
_CASE_BATCH_SIZE = 100
_TIMEOUT = 60
_FILE_FIELDS = ",".join(
    (
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "access",
        "data_category",
        "data_type",
        "data_format",
        "experimental_strategy",
        "state",
        "associated_entities.case_id",
        "associated_entities.entity_id",
        "associated_entities.entity_submitter_id",
        "associated_entities.entity_type",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.samples.sample_id",
        "cases.samples.submitter_id",
    )
)
_CLINICAL_EXPANSIONS = "demographic,diagnoses,diagnoses.treatments,follow_ups,exposures"
_SOURCES = [
    "https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/",
    "https://docs.gdc.cancer.gov/API/Users_Guide/Appendix_A_Available_Fields/",
    "https://docs.gdc.cancer.gov/API/Users_Guide/Downloading_Files/",
]


def _ids(value, name):
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list of strings")
    result = []
    for item in value:
        if not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", item):
            raise ValueError(f"{name} must contain identifiers without paths or whitespace")
        if item not in result:
            result.append(item)
    return result


def _selection(selection):
    if not isinstance(selection, dict):
        raise ValueError("TCGA selection must be a dictionary")
    unknown = set(selection) - {"projects", "case_ids", "file_ids", "diagnostic_only", "limit"}
    if unknown:
        raise ValueError(f"Unknown TCGA selection options: {', '.join(sorted(unknown))}")
    result = deepcopy(selection)
    for name in ("projects", "case_ids", "file_ids"):
        result[name] = _ids(selection.get(name), name)
    if not any(result[name] for name in ("projects", "case_ids", "file_ids")):
        raise ValueError("Choose TCGA projects, case_ids, or file_ids explicitly")
    if any(not project.startswith("TCGA-") for project in result["projects"]):
        raise ValueError("TCGA projects must use GDC project IDs beginning with 'TCGA-'")
    result["diagnostic_only"] = selection.get("diagnostic_only", True)
    if not isinstance(result["diagnostic_only"], bool):
        raise ValueError("diagnostic_only must be a boolean")
    limit = selection.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    result["limit"] = limit
    return result


def _condition(field, values):
    return {"op": "in", "content": {"field": field, "value": values}}


def _release_information(response, body, releases):
    for key in ("data_release", "data_release_version", "release"):
        value = body.get(key)
        if value is not None and isinstance(value, (str, int, float)):
            releases[key] = value
    for name, value in getattr(response, "headers", {}).items():
        if name.lower() in {"x-gdc-data-release", "x-gdc-data-release-version"}:
            releases[name.lower()] = str(value)


def _pages(session, endpoint, payload, releases, *, limit=None):
    offset = 0
    remaining = limit
    while True:
        size = min(_PAGE_SIZE, remaining) if remaining is not None else _PAGE_SIZE
        request = {**payload, "size": size, "from": offset, "format": "JSON"}
        response = session.post(f"{API_BASE}/{endpoint}", json=request, timeout=_TIMEOUT)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("data"), dict):
            raise ValueError(f"GDC /{endpoint} returned an invalid response")
        _release_information(response, body, releases)
        data = body["data"]
        hits = data.get("hits")
        if not isinstance(hits, list) or any(not isinstance(hit, dict) for hit in hits):
            raise ValueError(f"GDC /{endpoint} did not return a list of metadata objects")
        if not hits:
            return
        selected = hits[:remaining] if remaining is not None else hits
        for hit in selected:
            yield hit
        offset += len(hits)
        if remaining is not None:
            remaining -= len(selected)
            if remaining <= 0:
                return
        pagination = data.get("pagination", {})
        total = pagination.get("total") if isinstance(pagination, dict) else None
        if isinstance(total, int) and not isinstance(total, bool):
            if offset >= total:
                return
        elif len(hits) < size:
            return


def _entity_id(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"GDC metadata is missing a valid {name}")
    return value


def _objects(value, name):
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"GDC metadata field {name} must be a list of objects")
    return value


def discover(selection: dict, *, session=None) -> AcquisitionPlan:
    """Return an acquisition plan with source file metadata and nested clinical JSON.

    Projects/cases/files explicitly constrain the query. All supplied constraints
    are combined, and ``limit`` caps slide discovery before clinical retrieval.
    Tokens and controlled-access downloads are intentionally outside this adapter.
    A requests-compatible session may be injected for testing.
    """
    selection = _selection(selection)
    owned_session = session is None
    if owned_session:
        import requests

        session = requests.Session()
    releases = {}
    try:
        filters = [
            _condition("access", ["open"]),
            _condition("data_category", ["Biospecimen"]),
            _condition("data_type", ["Slide Image"]),
            _condition("data_format", ["SVS"]),
            _condition("cases.project.program.name", ["TCGA"]),
        ]
        for option, field in (
            ("projects", "cases.project.project_id"),
            ("case_ids", "cases.case_id"),
            ("file_ids", "file_id"),
        ):
            if selection[option]:
                filters.append(_condition(field, selection[option]))
        if selection["diagnostic_only"]:
            filters.append(_condition("experimental_strategy", ["Diagnostic Slide"]))
        file_payload = {
            "filters": {"op": "and", "content": filters},
            "fields": _FILE_FIELDS,
            "expand": "cases,associated_entities,cases.samples,cases.project",
            "sort": "file_id:asc",
        }
        assets = []
        records = []
        seen_files = set()
        case_ids = set()
        for original in _pages(session, "files", file_payload, releases, limit=selection["limit"]):
            file_id = _entity_id(original.get("file_id", original.get("id")), "file_id")
            if file_id in seen_files:
                raise ValueError(
                    "GDC returned a duplicate file across result pages; retry discovery"
                )
            seen_files.add(file_id)
            if original.get("access") != "open":
                raise ValueError(
                    "TCGA discovery returned a file that is not explicitly open-access"
                )
            for key, expected in (
                ("data_category", "Biospecimen"),
                ("data_type", "Slide Image"),
                ("data_format", "SVS"),
            ):
                if original.get(key, expected) != expected:
                    raise ValueError(f"GDC returned a file outside the requested {key} filter")
            cases = _objects(original.get("cases"), "cases")
            entities = _objects(original.get("associated_entities"), "associated_entities")
            subjects = {
                _entity_id(case.get("case_id", case.get("id")), "case_id") for case in cases
            }
            subjects.update(
                _entity_id(entity["case_id"], "case_id")
                for entity in entities
                if entity.get("case_id")
            )
            subjects = sorted(subjects)
            case_ids.update(subjects)
            samples = set()
            for case in cases:
                for sample in _objects(case.get("samples"), "cases.samples"):
                    if sample.get("sample_id"):
                        samples.add(_entity_id(sample["sample_id"], "sample_id"))
            slide_entities = sorted(
                {
                    entity["entity_id"]
                    for entity in entities
                    if entity.get("entity_type") == "slide" and entity.get("entity_id")
                }
            )
            checksums = {"md5": original["md5sum"]} if original.get("md5sum") else {}
            record_id = f"file:{file_id}"
            records.append(
                MetadataRecord(
                    record_id=record_id,
                    kind="metadata",
                    subject_ids=subjects,
                    data=deepcopy(original),
                    source=f"{API_BASE}/files",
                )
            )
            assets.append(
                Asset(
                    asset_id=file_id,
                    dataset="TCGA",
                    kind="slide",
                    filename=original.get("file_name") or f"{file_id}.svs",
                    relative_path=f"slides/{file_id}.svs",
                    url=f"{API_BASE}/data/{quote(file_id, safe='')}",
                    size=original.get("file_size"),
                    checksums=checksums,
                    slide_id=slide_entities[0] if len(slide_entities) == 1 else file_id,
                    subject_ids=subjects,
                    sample_ids=sorted(samples),
                    label_status="unavailable",
                    clinical_refs=[f"case:{case_id}" for case_id in subjects],
                    metadata={
                        "source_record_id": record_id,
                        "associated_entities": deepcopy(entities),
                    },
                )
            )

        found_cases = set()
        selected_cases = sorted(case_ids)
        for start in range(0, len(selected_cases), _CASE_BATCH_SIZE):
            batch = selected_cases[start : start + _CASE_BATCH_SIZE]
            case_payload = {
                "filters": _condition("case_id", batch),
                "expand": _CLINICAL_EXPANSIONS,
                "sort": "case_id:asc",
            }
            for original in _pages(session, "cases", case_payload, releases):
                case_id = _entity_id(original.get("case_id", original.get("id")), "case_id")
                if case_id not in batch:
                    raise ValueError(
                        "GDC returned clinical metadata outside the selected case batch"
                    )
                if case_id in found_cases:
                    raise ValueError("GDC returned a duplicate case across clinical result pages")
                found_cases.add(case_id)
                records.append(
                    MetadataRecord(
                        record_id=f"case:{case_id}",
                        kind="clinical",
                        subject_ids=[case_id],
                        data=deepcopy(original),
                        source=f"{API_BASE}/cases",
                    )
                )
        missing_cases = sorted(case_ids - found_cases)
        clinical_status = (
            "available"
            if found_cases and not missing_cases
            else "partial"
            if found_cases
            else "unavailable"
        )
        return AcquisitionPlan(
            dataset="TCGA",
            selection=selection,
            assets=assets,
            records=records,
            provenance={
                "provider": "NCI Genomic Data Commons",
                "api_base": API_BASE,
                "retrieved_at": utc_now(),
                "sources": _SOURCES,
                "release": releases or None,
                "access": "open",
                "clinical_availability": clinical_status,
                "clinical_case_count": len(found_cases),
                "missing_clinical_case_ids": missing_cases,
                "linkage": "GDC case UUIDs; file associations preserved without filename inference",
                "derived_labels": False,
            },
        )
    finally:
        if owned_session:
            session.close()
