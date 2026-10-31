"""Small shared helpers for bounded, credential-free cloud adapters."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import wraps
from importlib import metadata as import_metadata
from io import BytesIO
import math
import random
import threading
import time

import numpy as np

from ..base import AccessError, EncoderError, translated_errors, validate_features


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def sdk_version(module, distribution):
    version = field(module, "__version__")
    if isinstance(version, str):
        return version
    try:
        return import_metadata.version(distribution)
    except import_metadata.PackageNotFoundError:
        return None


def check_options(spec, *, allowed, dimensions=None, default_dimension):
    if spec.revision is not None or spec.checkpoint is not None:
        raise EncoderError(
            "Cloud profiles cannot pin local revisions/checkpoints; use a cache_tag to separate vendor model updates"
        )
    if set(spec.options) - set(allowed):
        raise EncoderError(
            "Unsupported cloud encoder option; use the documented task/cache_tag or AWS region/profile options"
        )
    if "cache_tag" in spec.options and not isinstance(spec.options["cache_tag"], str):
        raise EncoderError("Cloud cache_tag must be a string")
    dimension = spec.dimension or default_dimension
    if dimensions is not None and dimension not in dimensions:
        raise EncoderError("Unsupported output dimension for this cloud model")
    return dimension


def png_bytes(image):
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 1
    ):
        raise EncoderError("Cloud images must be nonempty uint8 HWC RGB arrays")
    from PIL import Image

    handle = BytesIO()
    Image.fromarray(image).save(handle, format="PNG", compress_level=6)
    return handle.getvalue()


def validated(features, count, dimension):
    with translated_errors("Cloud provider returned an invalid numeric feature array"):
        with np.errstate(over="ignore", invalid="ignore"):
            result = validate_features(features, count, dimension)
    if not np.isfinite(result).all():
        raise EncoderError("Cloud features overflowed the float32 output range")
    return result


def safe_encode(method):
    @wraps(method)
    def wrapped(self, images):
        if self._closed:
            raise EncoderError("Cloud encoder is closed")
        with translated_errors(
            "Cloud image preparation or response decoding failed; check SDK compatibility and input limits"
        ):
            return method(self, images)

    return wrapped


def _error_status(error):
    for candidate in (
        field(error, "status_code"),
        field(error, "code"),
        field(field(error, "response", {}), "status_code"),
    ):
        try:
            number = int(candidate)
        except (TypeError, ValueError):
            continue
        if 100 <= number <= 599:
            return number
    response = field(error, "response", {})
    number = field(field(response, "ResponseMetadata", {}), "HTTPStatusCode")
    try:
        return int(number) if number is not None else None
    except (TypeError, ValueError):
        return None


def _error_code(error):
    return field(field(field(error, "response", {}), "Error", {}), "Code", "")


def _retry_after(error):
    response = field(error, "response", {})
    headers = (
        field(response, "headers")
        or field(error, "headers")
        or field(field(response, "ResponseMetadata", {}), "HTTPHeaders", {})
    )
    value = None
    if hasattr(headers, "items"):
        value = next(
            (value for key, value in headers.items() if str(key).lower() == "retry-after"), None
        )
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            deadline = parsedate_to_datetime(str(value))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


class CloudBase:
    def __init__(self, spec, config, *, provider, model_name, dimension, task, sdk, sdk_name):
        if config.offline:
            raise AccessError("Cloud embeddings are unavailable in offline mode")
        self.spec = spec
        self.config = config
        self.dimension = dimension
        self._closed = False
        self._usage_lock = threading.Lock()
        self.usage = {"request_attempts": 0, "retries": 0, "successful_requests": 0, "images": 0}
        self.metadata = {
            "provider": provider,
            "profile_id": spec.model,
            "requested_model": model_name,
            "model_name": model_name,
            "resolved_revision": None,
            "dimension": dimension,
            "provider_precision": "managed_by_provider",
            "output_dtype": "float32",
            "task": task,
            "sdk": {"name": sdk_name, "version": sdk_version(sdk, sdk_name)},
            "preprocessing": {
                "local": "uint8_RGB_to_PNG_compression_6",
                "native": "managed_by_provider",
            },
        }

    def _add_usage(self, values):
        with self._usage_lock:
            for name, value in values.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                ):
                    self.usage[name] = self.usage.get(name, 0) + value

    def _request(self, operation):
        if self._closed:
            raise EncoderError("Cloud encoder is closed")
        for attempt in range(self.config.max_retries + 1):
            self._add_usage({"request_attempts": 1})
            try:
                result = operation()
            except Exception as error:
                status = _error_status(error)
                code = _error_code(error)
                name = type(error).__name__
                if (
                    status in {401, 403}
                    or code
                    in {
                        "AccessDeniedException",
                        "UnrecognizedClientException",
                        "ExpiredTokenException",
                        "InvalidSignatureException",
                    }
                    or any(
                        term in name
                        for term in (
                            "Authentication",
                            "PermissionDenied",
                            "NoCredentials",
                            "CredentialRetrieval",
                            "ProfileNotFound",
                        )
                    )
                ):
                    raise AccessError(
                        "Cloud authentication or model access failed; check the configured credential reference, permissions, and region"
                    ) from None
                transient = (
                    status in {408, 429, 500, 502, 503, 504}
                    or code
                    in {
                        "ThrottlingException",
                        "ServiceUnavailableException",
                        "ModelTimeoutException",
                        "InternalServerException",
                    }
                    or any(
                        term in name
                        for term in ("Timeout", "ConnectionError", "ConnectError", "NetworkError")
                    )
                )
                if not transient or attempt >= self.config.max_retries:
                    raise EncoderError(
                        "Cloud embedding request failed; check model access, input limits, quota, and provider service status"
                    ) from None
                delay = _retry_after(error)
                if delay is None:
                    delay = min(30.0, 2.0**attempt) * random.uniform(0.75, 1.25)
                if delay > 60.0:
                    raise EncoderError(
                        "Cloud provider requested a retry delay over 60 seconds; resume the run later"
                    ) from None
                self._add_usage({"retries": 1})
                time.sleep(delay)
            else:
                self._add_usage({"successful_requests": 1})
                return result

    def _parallel(self, operation, inputs):
        if self._closed:
            raise EncoderError("Cloud encoder is closed")
        if not inputs:
            return []
        executor = ThreadPoolExecutor(max_workers=min(self.config.cloud_concurrency, len(inputs)))
        futures = []
        try:
            futures = [executor.submit(operation, item) for item in inputs]
            return [future.result() for future in futures]
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        close = getattr(self._client, "close", None)
        try:
            if callable(close):
                close()
            elif callable(getattr(self._client, "__exit__", None)):
                self._client.__exit__(None, None, None)
        except Exception:
            raise EncoderError("Cloud SDK client could not be closed cleanly") from None
