"""Serializable acquisition plans shared by the dataset providers."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import tempfile
from urllib.parse import parse_qsl, urlsplit


DATASETS = ("TCGA", "KPMP", "CAMELYON16", "BRACS")
SCHEMA_VERSION = 1


def canonical_dataset(value: str) -> str:
    result = value.upper()
    if result not in DATASETS:
        raise ValueError(f"Unknown dataset; choose from {', '.join(DATASETS)}")
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or path == PurePosixPath(".")
    ):
        raise ValueError("Asset paths must be nonempty root-relative paths without traversal")
    return str(path)


def public_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Assets require public HTTP(S) URLs without embedded credentials")
    secret_names = {
        "token",
        "access_token",
        "auth",
        "authorization",
        "signature",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        "awsaccesskeyid",
        "key",
        "api_key",
    }
    if any(key.lower() in secret_names for key, _ in parse_qsl(parsed.query)):
        raise ValueError("Credential-bearing URLs cannot be persisted in public acquisition plans")
    return value


def _public_selection(value):
    """Selections describe public cohorts, never an authentication configuration."""
    forbidden = {
        "password",
        "token",
        "access_token",
        "auth",
        "authorization",
        "cookie",
        "cookies",
        "credentials",
        "api_key",
        "secret",
        "client_secret",
    }
    if isinstance(value, dict):
        if any(str(key).lower().replace("-", "_") in forbidden for key in value):
            raise ValueError("Public dataset selections cannot contain credentials")
        for item in value.values():
            _public_selection(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _public_selection(item)


@dataclass
class Asset:
    asset_id: str
    dataset: str
    kind: str
    filename: str
    relative_path: str
    url: str | None = None
    size: int | None = None
    checksums: dict[str, str] = field(default_factory=dict)
    slide_id: str | None = None
    subject_ids: list[str] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    label_status: str = "unavailable"
    split: str | None = None
    clinical_refs: list[str] = field(default_factory=list)
    related_slide_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.dataset = canonical_dataset(self.dataset)
        if not isinstance(self.asset_id, str) or not self.asset_id:
            raise ValueError("asset_id must be a nonempty string")
        if self.kind not in {"slide", "clinical", "annotation", "metadata", "license"}:
            raise ValueError("Unknown asset kind")
        if not self.filename or Path(self.filename).name != self.filename or "\\" in self.filename:
            raise ValueError("filename must be a basename")
        self.relative_path = validate_relative_path(self.relative_path)
        self.url = public_url(self.url)
        if self.size is not None and (
            isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0
        ):
            raise ValueError("Asset size must be a nonnegative integer or None")
        for algorithm, digest in self.checksums.items():
            length = {"md5": 32, "sha256": 64}.get(algorithm)
            if (
                length is None
                or not isinstance(digest, str)
                or len(digest) != length
                or any(c not in "0123456789abcdefABCDEF" for c in digest)
            ):
                raise ValueError("Checksums must contain valid md5 or sha256 hex digests")
        if self.label_status not in {"known", "missing", "unavailable", "conflicting"}:
            raise ValueError("Unknown label_status")
        for name in ("subject_ids", "sample_ids", "clinical_refs", "related_slide_ids"):
            value = getattr(self, name)
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"{name} must be a list of strings")
        if not isinstance(self.labels, dict) or not isinstance(self.metadata, dict):
            raise ValueError("Asset labels and metadata must be dictionaries")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetadataRecord:
    record_id: str
    kind: str
    subject_ids: list[str]
    data: dict
    source: str | None = None

    def __post_init__(self):
        if (
            not isinstance(self.record_id, str)
            or not self.record_id
            or not isinstance(self.kind, str)
            or not self.kind
        ):
            raise ValueError("Metadata records need nonempty identifiers and kinds")
        if not isinstance(self.subject_ids, list) or any(
            not isinstance(item, str) for item in self.subject_ids
        ):
            raise ValueError("Metadata subject_ids must be a list of strings")
        if not isinstance(self.data, dict):
            raise ValueError("Metadata data must be an object")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AcquisitionPlan:
    dataset: str
    selection: dict
    assets: list[Asset] = field(default_factory=list)
    records: list[MetadataRecord] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        self.dataset = canonical_dataset(self.dataset)
        if isinstance(self.schema_version, bool) or self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported acquisition-plan schema version")
        if not isinstance(self.selection, dict) or not isinstance(self.provenance, dict):
            raise ValueError("Plan selection and provenance must be dictionaries")
        _public_selection(self.selection)
        ids = [asset.asset_id for asset in self.assets]
        paths = [asset.relative_path for asset in self.assets]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("A plan cannot contain duplicate asset IDs or local paths")
        if any(asset.dataset != self.dataset for asset in self.assets):
            raise ValueError("All assets must belong to the plan's dataset")

    @property
    def expected_bytes(self) -> int:
        return sum(asset.size for asset in self.assets if asset.size is not None)

    @property
    def unknown_size_count(self) -> int:
        return sum(asset.size is None for asset in self.assets)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        data = dict(value)
        data["assets"] = [Asset(**asset) for asset in data.get("assets", [])]
        data["records"] = [MetadataRecord(**record) for record in data.get("records", [])]
        return cls(**data)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(type(self).from_dict(self.to_dict()).to_dict(), indent=2, allow_nan=False)
            + "\n"
        )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class DownloadReport:
    root: Path
    total: int
    downloaded: int
    skipped: int
    failed: int

    def to_dict(self):
        return {**asdict(self), "root": str(self.root)}


@dataclass(frozen=True)
class VerificationReport:
    root: Path
    checked: int
    valid: int
    missing: int
    failed: int

    def to_dict(self):
        return {**asdict(self), "root": str(self.root)}
