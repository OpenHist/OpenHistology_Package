"""Discover anonymous native S3 objects, preserving official metadata as supplied."""

from copy import deepcopy
import csv
from io import StringIO
from pathlib import PurePosixPath
import re
from urllib.parse import quote
import xml.etree.ElementTree as ET

from ..models import AcquisitionPlan, Asset, MetadataRecord, utc_now


BUCKET = "camelyon-dataset"
REGION = "us-west-2"
BASE_URL = f"https://{BUCKET}.s3.{REGION}.amazonaws.com"
PREFIX = "CAMELYON16/"
_TIMEOUT = 60
_METADATA_LIMIT = 10 * 1024 * 1024
_METADATA_KEYS = ("README.md", "license.txt", "checksums.md5", "evaluation/reference.csv")
_SLIDE_FIELDS = {
    "image",
    "image_name",
    "slide",
    "slide_id",
    "slide_name",
    "filename",
    "file_name",
    "name",
}
_LABEL_FIELDS = {
    "label",
    "class",
    "type",
    "tumor",
    "has_tumor",
    "tumor_present",
    "diagnosis",
    "stage",
}
_SPLIT_FIELDS = {"split", "subset", "partition", "set"}
_SPLIT_ALIASES = {"train": "train", "training": "train", "test": "test", "testing": "test"}


def _slide_id(value):
    if not isinstance(value, str) or not value:
        raise ValueError("slide_ids must contain nonempty names")
    value = value.strip()
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError("CAMELYON16 slide IDs must be basenames")
    if value.lower().endswith((".tif", ".tiff")):
        value = value.rsplit(".", 1)[0]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("CAMELYON16 slide IDs contain invalid characters")
    return value


def _selection(selection):
    if not isinstance(selection, dict):
        raise ValueError("CAMELYON16 selection must be a dictionary")
    unknown = set(selection) - {"split", "splits", "slide_ids", "include_annotations", "limit"}
    if unknown:
        raise ValueError(f"Unknown CAMELYON16 selection options: {', '.join(sorted(unknown))}")
    if "split" in selection and "splits" in selection:
        raise ValueError("Choose either split or splits")
    splits = selection.get("splits", selection.get("split"))
    if isinstance(splits, str):
        splits = [splits]
    if splits is not None:
        if (
            not isinstance(splits, list)
            or not splits
            or any(not isinstance(item, str) for item in splits)
        ):
            raise ValueError("splits must contain train, test, or all")
        if any(item not in {"train", "test", "all"} for item in splits):
            raise ValueError("splits must contain train, test, or all")
        splits = ["train", "test"] if "all" in splits else sorted(set(splits))
    slide_ids = selection.get("slide_ids")
    if slide_ids is not None:
        if not isinstance(slide_ids, list) or not slide_ids:
            raise ValueError("slide_ids must be a nonempty list")
        slide_ids = list(dict.fromkeys(_slide_id(item) for item in slide_ids))
    if splits is None and slide_ids is None:
        raise ValueError("Choose CAMELYON16 split(s) or slide_ids explicitly")
    include_annotations = selection.get("include_annotations", False)
    if not isinstance(include_annotations, bool):
        raise ValueError("include_annotations must be a boolean")
    limit = selection.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    return {
        "splits": splits,
        "slide_ids": slide_ids,
        "include_annotations": include_annotations,
        "limit": limit,
    }


def _url(key):
    return f"{BASE_URL}/{quote(key, safe='/')}"


def _list_objects(session, prefix):
    token = None
    seen_tokens = set()
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token is not None:
            params["continuation-token"] = token
        response = session.get(BASE_URL, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise ValueError("CAMELYON16 S3 listing is not valid XML") from exc
        for node in root.findall("{*}Contents"):
            key = node.findtext("{*}Key")
            if not key or not key.startswith(prefix):
                raise ValueError("S3 listing returned an object outside the requested prefix")
            if ".." in PurePosixPath(key).parts or "\\" in key:
                raise ValueError("S3 listing returned an unsafe object key")
            size = node.findtext("{*}Size")
            try:
                size = int(size) if size is not None else None
            except ValueError as exc:
                raise ValueError("S3 object size is invalid") from exc
            yield {"key": key, "size": size, "etag": node.findtext("{*}ETag")}
        if root.findtext("{*}IsTruncated", "false").lower() != "true":
            return
        token = root.findtext("{*}NextContinuationToken")
        if not token or token in seen_tokens:
            raise ValueError("S3 listing did not provide a new continuation token")
        seen_tokens.add(token)


def _metadata_text(session, key):
    response = session.get(_url(key), timeout=_TIMEOUT)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    content = response.content
    if len(content) > _METADATA_LIMIT:
        raise ValueError("CAMELYON16 discovery metadata exceeds the size limit")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CAMELYON16 discovery metadata must be UTF-8 text") from exc


def _checksums(text):
    checksums = {}
    if text is None:
        return checksums
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"\s*([0-9a-fA-F]{32})\s+\*?(.+?)\s*", line)
        if not match:
            raise ValueError("CAMELYON16 checksums.md5 contains an unsupported checksum row")
        digest, key = match.groups()
        while key.startswith("./"):
            key = key[2:]
        if ".." in PurePosixPath(key).parts or key.startswith("/") or "\\" in key:
            raise ValueError("CAMELYON16 checksums contain an unsafe object path")
        if not key.startswith(PREFIX):
            key = PREFIX + key
        if key in checksums and checksums[key] != digest.lower():
            raise ValueError("CAMELYON16 checksums contain conflicting values")
        checksums[key] = digest.lower()
    return checksums


def _checksum_for(key, checksums):
    digest = checksums.get(key) or checksums.get(PREFIX + PurePosixPath(key).name)
    return {"md5": digest} if digest else {}


def _normal_header(value):
    return re.sub(r"[\s-]+", "_", value.strip().lower())


def _reference(text):
    if text is None:
        return {}, [], "unavailable"
    reader = csv.DictReader(StringIO(text))
    if not reader.fieldnames:
        return {}, [], "empty"
    names = {_normal_header(name): name for name in reader.fieldnames if name is not None}
    identifiers = [names[name] for name in names if name in _SLIDE_FIELDS]
    label_fields = [names[name] for name in names if name in _LABEL_FIELDS]
    split_fields = [names[name] for name in names if name in _SPLIT_FIELDS]
    rows = []
    mapping = {}
    for index, row in enumerate(reader):
        if None in row:
            raise ValueError("CAMELYON16 reference CSV has inconsistent row lengths")
        original = dict(row)
        rows.append((index, original))
        if len(identifiers) != 1:
            continue
        raw_id = row.get(identifiers[0])
        if not raw_id:
            continue
        slide_id = _slide_id(raw_id)
        if slide_id in mapping:
            raise ValueError("CAMELYON16 reference CSV contains duplicate slide IDs")
        labels = {
            name: row[name]
            for name in label_fields
            if row.get(name) not in {None, "", "unknown", "Unknown", "NA", "N/A"}
        }
        split_values = {
            _SPLIT_ALIASES[row[name].strip().lower()]
            for name in split_fields
            if row.get(name) and row[name].strip().lower() in _SPLIT_ALIASES
        }
        if len(split_values) > 1:
            raise ValueError("CAMELYON16 reference CSV contains conflicting splits")
        mapping[slide_id] = {
            "labels": labels,
            "split": next(iter(split_values), None),
            "row": original,
            "row_index": index,
        }
    status = "available" if len(identifiers) == 1 else "unrecognized_schema"
    return mapping, rows, status


def _native_split(slide_id):
    if slide_id.startswith(("normal_", "tumor_")):
        return "train"
    if slide_id.startswith("test_"):
        return "test"
    return None


def _reference_tumor_values(labels):
    """Compare explicit binary labels without deriving a label from disease stage."""
    values = set()
    for name, value in labels.items():
        if _normal_header(name) == "stage":
            continue
        normalized = str(value).strip().lower()
        if normalized in {"normal", "negative", "no tumor", "non-tumor", "0", "false"}:
            values.add(False)
        elif normalized in {"tumor", "positive", "1", "true"}:
            values.add(True)
    return values


def _related_slide(key, selected_ids):
    stem = PurePosixPath(key).stem
    candidates = [stem]
    for suffix in ("_mask", "_annotation", "_annotations"):
        if stem.endswith(suffix):
            candidates.append(stem[: -len(suffix)])
    return [slide_id for slide_id in candidates if slide_id in selected_ids]


def discover(selection: dict, *, session=None) -> AcquisitionPlan:
    """List original TIFF slides and read only small official discovery metadata.

    No WSI or annotation payload is fetched. Source reference label values remain
    literal strings; training ``normal_``/``tumor_`` filenames also supply the
    challenge's encoded binary label. No patient identities are inferred.
    S3 ETags are recorded as metadata, never assumed to be MD5 checksums.
    """
    selection = _selection(selection)
    owned_session = session is None
    if owned_session:
        import requests

        session = requests.Session()
    try:
        texts = {name: _metadata_text(session, PREFIX + name) for name in _METADATA_KEYS}
        checksums = _checksums(texts["checksums.md5"])
        reference, reference_rows, reference_status = _reference(texts["evaluation/reference.csv"])
        assets = []
        records = []
        for name, text in texts.items():
            if text is None:
                continue
            key = PREFIX + name
            kind = "license" if name == "license.txt" else "metadata"
            assets.append(
                Asset(
                    asset_id=f"metadata:{name}",
                    dataset="CAMELYON16",
                    kind=kind,
                    filename=PurePosixPath(name).name,
                    relative_path=f"metadata/{name}",
                    url=_url(key),
                    checksums=_checksum_for(key, checksums),
                )
            )
            if name != "evaluation/reference.csv":
                records.append(
                    MetadataRecord(
                        record_id=f"metadata:{name}",
                        kind=kind,
                        subject_ids=[],
                        data={"text": text, "filename": name},
                        source=_url(key),
                    )
                )
        requested_ids = set(selection["slide_ids"] or [])
        selected_ids = set()
        observed_ids = set()
        split_conflicts = []
        for obj in _list_objects(session, PREFIX + "images/"):
            key = obj["key"]
            filename = PurePosixPath(key).name
            if not filename.lower().endswith((".tif", ".tiff")):
                continue
            slide_id = _slide_id(filename)
            if slide_id in observed_ids:
                raise ValueError("CAMELYON16 image listing contains duplicate slide IDs")
            observed_ids.add(slide_id)
            if requested_ids and slide_id not in requested_ids:
                continue
            native_split = _native_split(slide_id)
            row = reference.get(slide_id, {})
            reference_split = row.get("split")
            split_candidates = {value for value in (native_split, reference_split) if value}
            split_conflict = len(split_candidates) > 1
            split = None if split_conflict else native_split or reference_split
            if selection["splits"] is not None and not split_candidates.intersection(
                selection["splits"]
            ):
                continue
            if split_conflict:
                split_conflicts.append(slide_id)
            labels = deepcopy(row.get("labels", {}))
            reference_tumor_values = _reference_tumor_values(labels)
            training_tumor = None
            if slide_id.startswith("normal_"):
                training_tumor = labels["training_tumor"] = False
            elif slide_id.startswith("tumor_"):
                training_tumor = labels["training_tumor"] = True
            label_conflict = len(reference_tumor_values) > 1 or (
                training_tumor is not None
                and any(value != training_tumor for value in reference_tumor_values)
            )
            label_status = (
                "conflicting"
                if label_conflict or split_conflict
                else "known"
                if labels
                else "missing"
                if row
                else "unavailable"
            )
            selected_ids.add(slide_id)
            assets.append(
                Asset(
                    asset_id=slide_id,
                    dataset="CAMELYON16",
                    kind="slide",
                    filename=filename,
                    relative_path=f"slides/{filename}",
                    url=_url(key),
                    size=obj["size"],
                    checksums=_checksum_for(key, checksums),
                    slide_id=slide_id,
                    labels=labels,
                    label_status=label_status,
                    split=split,
                    subject_ids=[],
                    clinical_refs=[],
                    metadata={
                        "s3_key": key,
                        "s3_etag": obj["etag"],
                        "label_sources": (["official reference.csv"] if row.get("labels") else [])
                        + (
                            ["official training filename prefix"] if native_split == "train" else []
                        ),
                        "reference_row_id": f"reference:{row['row_index']}" if row else None,
                        "reference_split_conflict": split_conflict,
                        "label_conflict": label_conflict,
                        "native_split": native_split,
                        "reference_split": reference_split,
                    },
                )
            )
            if selection["limit"] is not None and len(selected_ids) >= selection["limit"]:
                break
        if requested_ids - observed_ids and selection["limit"] is None:
            raise ValueError(
                "Some requested CAMELYON16 slide IDs were not present in the official image listing"
            )
        for index, row in reference_rows:
            matching = [
                slide_id
                for slide_id in selected_ids
                if reference.get(slide_id, {}).get("row_index") == index
            ]
            if matching:
                records.append(
                    MetadataRecord(
                        record_id=f"reference:{index}",
                        kind="labels",
                        subject_ids=[],
                        data=deepcopy(row),
                        source=_url(PREFIX + "evaluation/reference.csv"),
                    )
                )
        if selection["include_annotations"]:
            for folder in ("annotations", "masks"):
                for obj in _list_objects(session, f"{PREFIX}{folder}/"):
                    key = obj["key"]
                    related = _related_slide(key, selected_ids)
                    if not related:
                        continue
                    filename = PurePosixPath(key).name
                    if not filename.lower().endswith((".xml", ".tif", ".tiff", ".png")):
                        continue
                    assets.append(
                        Asset(
                            asset_id=f"annotation:{key}",
                            dataset="CAMELYON16",
                            kind="annotation",
                            filename=filename,
                            relative_path=f"{folder}/{filename}",
                            url=_url(key),
                            size=obj["size"],
                            checksums=_checksum_for(key, checksums),
                            related_slide_ids=related,
                            metadata={"s3_key": key, "s3_etag": obj["etag"]},
                        )
                    )
        return AcquisitionPlan(
            dataset="CAMELYON16",
            selection=selection,
            assets=assets,
            records=records,
            provenance={
                "provider": "CAMELYON native public S3 dataset",
                "bucket": BUCKET,
                "region": REGION,
                "prefix": PREFIX,
                "retrieved_at": utc_now(),
                "sources": [
                    "https://camelyon16.grand-challenge.org/Data/",
                    "https://registry.opendata.aws/camelyon/",
                    _url(PREFIX + "README.md"),
                    _url(PREFIX + "license.txt"),
                ],
                "metadata_unavailable": [name for name, text in texts.items() if text is None],
                "reference_availability": reference_status,
                "reference_split_conflicts": split_conflicts,
                "clinical_availability": "unavailable",
                "patient_linkage": "unavailable; no subject IDs inferred",
                "checksum_source": _url(PREFIX + "checksums.md5")
                if texts["checksums.md5"] is not None
                else None,
                "etag_is_checksum": False,
            },
        )
    finally:
        if owned_session:
            session.close()
