"""Resumable public HTTP transfers and bounded-memory local verification."""

from email.utils import parsedate_to_datetime
from copy import copy
import hashlib
import math
from pathlib import Path
import re
import time
from types import MethodType

import requests

from .models import AcquisitionPlan, DownloadReport, VerificationReport
from .state import StateStore, _control_path, safe_asset_path


class _TransferError(Exception):
    def __init__(self, code, *, discard=False, retryable=True):
        super().__init__(code)
        self.code = code
        self.discard = discard
        self.retryable = retryable


def _digest_file(path, chunk_size):
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            size += len(block)
            sha256.update(block)
            md5.update(block)
    return size, {"sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def _check_file(asset, path, *, previous_sha=None, expected_size=None, chunk_size=1048576):
    size, digests = _digest_file(path, chunk_size)
    expected = asset.size if asset.size is not None else expected_size
    if expected is not None and size != expected:
        raise _TransferError("SizeMismatch", discard=size > expected)
    if any(
        digests[algorithm].lower() != digest.lower()
        for algorithm, digest in asset.checksums.items()
    ):
        raise _TransferError("ChecksumMismatch", discard=True)
    if previous_sha is not None and digests["sha256"] != previous_sha:
        raise _TransferError("LocalHashMismatch", discard=True)
    integrity = (
        "checksum_verified"
        if asset.checksums
        else "size_verified"
        if expected is not None
        else "local_hash_only"
    )
    return size, digests["sha256"], integrity


def _baseline_size(record):
    return (
        record["actual_size"]
        if record["integrity"] in {"size_verified", "checksum_verified"}
        else None
    )


def _actual_size(path):
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def _validator(record):
    etag = record.get("etag")
    if etag and etag.startswith('"') and etag.endswith('"'):
        return etag
    modified = record.get("last_modified")
    if modified:
        try:
            if parsedate_to_datetime(modified) is not None:
                return modified
        except (ValueError, TypeError, OverflowError):
            pass
    return None


def _discard_partial(part, store, asset):
    if part.is_symlink():
        raise ValueError("Partial files cannot be symlinks")
    if part.exists():
        part.unlink()
    store.update(asset.dataset, asset.asset_id, etag=None, last_modified=None, actual_size=None)


def _stream_once(asset, part, store, session, timeout, chunk_size):
    record = store.get(asset.dataset, asset.asset_id)
    if part.is_symlink():
        raise ValueError("Partial files cannot be symlinks")
    offset = part.stat().st_size if part.exists() else 0
    validator = _validator(record)
    if offset and (validator is None or asset.size is not None and offset > asset.size):
        _discard_partial(part, store, asset)
        offset = 0
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers.update(Range=f"bytes={offset}-", **{"If-Range": validator})
    with session.get(asset.url, stream=True, timeout=timeout, headers=headers) as response:
        status = response.status_code
        if status not in {200, 206}:
            if status == 416:
                raise _TransferError("HTTP_416", discard=True)
            raise _TransferError(f"HTTP_{status}", retryable=status in {408, 429} or status >= 500)
        if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
            raise _TransferError("UnexpectedContentEncoding", discard=True)
        etag = response.headers.get("ETag")
        modified = response.headers.get("Last-Modified")
        expected_total = None
        expected_body = None
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                expected_body = int(length)
                if expected_body < 0:
                    raise ValueError
            except ValueError:
                raise _TransferError("InvalidContentLength", discard=True) from None
        if status == 206:
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if not match:
                raise _TransferError("InvalidContentRange", discard=True)
            start, end, expected_total = map(int, match.groups())
            if start != offset or end < start or end >= expected_total:
                raise _TransferError("InvalidContentRange", discard=True)
            if expected_body is not None and expected_body != end - start + 1:
                raise _TransferError("InvalidContentRange", discard=True)
            expected_body = end - start + 1
            if offset and (
                record.get("etag")
                and etag
                and record["etag"] != etag
                or record.get("last_modified")
                and modified
                and record["last_modified"] != modified
            ):
                raise _TransferError("RemoteIdentityChanged", discard=True)
        else:
            # A server may ignore Range/If-Range. This response is a complete
            # representation: overwrite the partial instead of appending it.
            offset = 0
            expected_total = expected_body
        if asset.size is not None and expected_total is not None and asset.size != expected_total:
            raise _TransferError("RemoteSizeMismatch", discard=True)
        store.update(
            asset.dataset,
            asset.asset_id,
            status="downloading",
            error=None,
            etag=etag if etag is not None else record.get("etag") if offset else None,
            last_modified=modified
            if modified is not None
            else record.get("last_modified")
            if offset
            else None,
        )
        received = 0
        with part.open("ab" if offset else "wb") as stream:
            for block in response.iter_content(chunk_size=chunk_size):
                if not block:
                    continue
                stream.write(block)
                received += len(block)
                limit = asset.size if asset.size is not None else expected_total
                if limit is not None and offset + received > limit:
                    raise _TransferError("SizeMismatch", discard=True)
        if expected_body is not None and received != expected_body:
            raise _TransferError("IncompleteTransfer")
        return expected_total


def _error_code(exc):
    if isinstance(exc, _TransferError):
        return exc.code
    if isinstance(exc, requests.Timeout):
        return "Timeout"
    if isinstance(exc, requests.RequestException):
        return "NetworkError"
    if isinstance(exc, OSError):
        return "LocalIOError"
    if isinstance(exc, ValueError):
        return "InvalidLocalPath"
    return type(exc).__name__


def _preflight(plan, root):
    plan = AcquisitionPlan.from_dict(plan.to_dict())
    root = Path(root).resolve()
    store = StateStore(root)
    store.register(plan)
    plan_path = _control_path(root, "acquisition_plan.json")
    temporary = _control_path(root, "acquisition_plan.json.tmp")
    if temporary.exists() and not temporary.is_file():
        raise ValueError("Acquisition plan temporary path must be a regular file")
    plan.save(plan_path)
    return plan, root, store


def _validate_session(session):
    if getattr(session, "auth", None):
        raise ValueError("Acquisition supports public downloads without session credentials")
    if any(
        key.lower() in {"authorization", "proxy-authorization", "cookie"}
        for key in getattr(session, "headers", {})
    ):
        raise ValueError("Acquisition sessions cannot contain credential headers")
    if getattr(session, "cookies", None):
        raise ValueError("Acquisition sessions cannot contain cookies")
    if getattr(session, "params", None):
        raise ValueError("Acquisition sessions cannot contain extra query parameters")


class _NoAuthentication(requests.auth.AuthBase):
    def __call__(self, request):
        return request


def _public_redirect_auth(self, prepared_request, response):
    # Requests otherwise consults .netrc again after every redirect. Public
    # downloads never need origin Authorization, including on the next host.
    prepared_request.headers.pop("Authorization", None)


def _download_session(session):
    """Isolate origin authentication without changing Requests transport settings."""
    result = requests.Session() if session is None else copy(session)
    if isinstance(result, requests.Session):
        # A truthy no-op auth object suppresses the initial .netrc lookup. Keep
        # trust_env so proxy, NO_PROXY and CA-bundle settings still work.
        result.auth = _NoAuthentication()
        result.rebuild_auth = MethodType(_public_redirect_auth, result)
        # Response cookies and hooks must not alter a caller's session state.
        result.cookies = result.cookies.copy()
        result.headers = result.headers.copy()
        result.hooks = {event: list(hooks) for event, hooks in result.hooks.items()}
    return result


def download(
    plan, root, *, retries=3, timeout=(10, 60), chunk_size=1048576, session=None
) -> DownloadReport:
    """Download an explicit plan; retries are additional attempts after the first.

    Same-plan completed files are rehashed before reuse. Unknown existing remote
    payloads without source checksums are downloaded again rather than trusted.
    URL-less assets are registered and verified in place, or recorded missing.
    Neither credentials nor exception messages containing URLs enter state.
    Requests proxy/CA environment settings remain active; .netrc origin
    authentication is suppressed on initial requests and redirects.
    """
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    timeouts = timeout if isinstance(timeout, (tuple, list)) else (timeout,)
    if len(timeouts) not in {1, 2} or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        or item <= 0
        for item in timeouts
    ):
        raise ValueError("timeout must contain positive finite seconds")
    if session is not None:
        _validate_session(session)
    plan, root, store = _preflight(plan, root)
    own_session = session is None
    session = _download_session(session)
    downloaded = skipped = failed = 0
    try:
        for asset in plan.assets:
            path = safe_asset_path(root, asset.relative_path)
            part = safe_asset_path(root, asset.relative_path + ".part")
            record = store.get(asset.dataset, asset.asset_id)
            previous_sha = record["sha256"]
            if path.is_file() and (asset.url is None or asset.checksums or previous_sha):
                try:
                    size, sha256, integrity = _check_file(
                        asset,
                        path,
                        previous_sha=previous_sha,
                        expected_size=_baseline_size(record),
                        chunk_size=chunk_size,
                    )
                except (_TransferError, OSError) as exc:
                    store.update(
                        asset.dataset,
                        asset.asset_id,
                        status="failed",
                        integrity="failed",
                        actual_size=_actual_size(path),
                        error=_error_code(exc),
                    )
                    if asset.url is None:
                        failed += 1
                        continue
                else:
                    store.update(
                        asset.dataset,
                        asset.asset_id,
                        status="complete",
                        integrity=integrity,
                        actual_size=size,
                        sha256=sha256,
                        error=None,
                    )
                    skipped += 1
                    continue
            if asset.url is None:
                store.update(
                    asset.dataset,
                    asset.asset_id,
                    status="missing",
                    integrity="missing",
                    error="LocalFileMissing",
                )
                failed += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(retries + 1):
                try:
                    # Recheck containment immediately before each filesystem use.
                    path = safe_asset_path(root, asset.relative_path)
                    part = safe_asset_path(root, asset.relative_path + ".part")
                    expected_size = _stream_once(asset, part, store, session, timeout, chunk_size)
                    size, sha256, integrity = _check_file(
                        asset,
                        part,
                        previous_sha=previous_sha,
                        expected_size=expected_size,
                        chunk_size=chunk_size,
                    )
                    part.replace(path)
                    store.update(
                        asset.dataset,
                        asset.asset_id,
                        status="complete",
                        integrity=integrity,
                        actual_size=size,
                        sha256=sha256,
                        error=None,
                    )
                    downloaded += 1
                    break
                except (_TransferError, requests.RequestException, OSError, ValueError) as exc:
                    if isinstance(exc, _TransferError) and exc.discard:
                        _discard_partial(part, store, asset)
                    actual_size = _actual_size(part) if not part.is_symlink() else None
                    store.update(
                        asset.dataset,
                        asset.asset_id,
                        status="failed",
                        integrity="failed",
                        actual_size=actual_size,
                        error=_error_code(exc),
                    )
                    retryable = not isinstance(exc, _TransferError) or exc.retryable
                    if attempt == retries or not retryable or isinstance(exc, ValueError):
                        failed += 1
                        break
                    time.sleep(min(0.1 * 2**attempt, 1.0))
    finally:
        if own_session:
            session.close()
    return DownloadReport(
        root=root, total=len(plan.assets), downloaded=downloaded, skipped=skipped, failed=failed
    )


def verify(root) -> VerificationReport:
    """Rehash current-plan assets without performing network requests.

    A previously recorded SHA-256 remains the baseline on failure; a changed
    payload cannot silently become a new valid local import.
    """
    root = Path(root).resolve()
    plan = AcquisitionPlan.load(_control_path(root, "acquisition_plan.json"))
    store = StateStore(root)
    store.register(plan)
    valid = missing = failed = 0
    for asset in plan.assets:
        record = store.get(asset.dataset, asset.asset_id)
        try:
            path = safe_asset_path(root, asset.relative_path)
            if not path.is_file():
                store.update(
                    asset.dataset,
                    asset.asset_id,
                    status="missing",
                    integrity="missing",
                    error="LocalFileMissing",
                )
                missing += 1
                continue
            size, sha256, integrity = _check_file(
                asset,
                path,
                previous_sha=record["sha256"],
                expected_size=_baseline_size(record),
            )
        except (_TransferError, OSError, ValueError) as exc:
            store.update(
                asset.dataset,
                asset.asset_id,
                status="failed",
                integrity="failed",
                error=_error_code(exc),
            )
            failed += 1
        else:
            store.update(
                asset.dataset,
                asset.asset_id,
                status="complete",
                integrity=integrity,
                actual_size=size,
                sha256=sha256,
                error=None,
            )
            valid += 1
    return VerificationReport(
        root=root, checked=len(plan.assets), valid=valid, missing=missing, failed=failed
    )
