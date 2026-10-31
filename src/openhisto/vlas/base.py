"""Dependency-light model contracts shared by the runner and isolated workers."""

from dataclasses import asdict, dataclass, field
import os
import re


class VLAError(RuntimeError):
    """An actionable inference or environment error."""

    @property
    def worker_diagnostics(self):
        """Transient worker details, including on a wrapped description failure.

        Stderr is present only when explicitly requested. It is kept separate
        from the message that may be persisted in a result's failure metadata.
        """
        error = self
        while error is not None:
            if isinstance(error, WorkerError):
                details = {"exit_status": error.exit_status}
                if error.stderr_tail is not None:
                    details["stderr_tail"] = error.stderr_tail
                return details
            error = error.__cause__
        return None


class WorkerError(VLAError):
    """A worker failure with optional, redacted diagnostics for the caller."""

    def __init__(self, message, *, exit_status=None, stderr_tail=None):
        self.exit_status = exit_status
        self.stderr_tail = stderr_tail
        if exit_status is not None:
            message += f" (worker exit status {exit_status})"
        super().__init__(message)


class AccessError(VLAError):
    pass


def error_text(error, *, limit=2000):
    """Keep runtime errors useful without persisting credential values."""
    value = str(error)
    for key, secret in os.environ.items():
        if (
            any(
                word in key.upper()
                for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY")
            )
            and len(secret) >= 6
        ):
            value = value.replace(secret, "[redacted]")
    value = re.sub(r"hf_[A-Za-z0-9]{12,}", "[redacted]", value)
    return value if limit is None else value[:limit]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    truncated: bool = False
    finish_reason: str = "stop"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ModelDescriptor:
    model_id: str
    display_name: str
    checkpoint: str | None
    checkpoint_revision: str | None
    source_url: str
    source_revision: str
    license: str
    access: str
    dependencies: tuple = ()
    preprocessing: dict = field(default_factory=dict)
    required_artifacts: tuple = ()
    status: str = "unverified"

    def to_dict(self):
        return asdict(self)


def validate_results(results, count):
    if not isinstance(results, (list, tuple)) or len(results) != count:
        raise VLAError(f"Backend must return exactly {count} ordered results")
    checked = []
    for result in results:
        if isinstance(result, dict):
            try:
                result = GenerationResult(**result)
            except (TypeError, ValueError) as exc:
                raise VLAError("Invalid backend result fields") from exc
        if (
            not isinstance(result, GenerationResult)
            or not isinstance(result.text, str)
            or not result.text.strip()
        ):
            raise VLAError("Backend returned an empty or invalid description")
        if not isinstance(result.truncated, bool) or not isinstance(result.finish_reason, str):
            raise VLAError("Backend returned invalid generation status")
        checked.append(result)
    return checked
