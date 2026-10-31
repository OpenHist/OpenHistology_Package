"""Persistent SPAMS IPC without loading its OpenMP runtime in the caller.

The framed pickle channel is private to a child launched from this package;
it must never be exposed as a service or used with untrusted external input.
"""

from __future__ import annotations

import os
import pickle
import select
import struct
import subprocess
import sys
import tempfile
import threading
import time
import weakref

from openhisto._common import worker_environment


MAX_FRAME_BYTES = 256 * 1024 * 1024
_HEADER = struct.Struct("!Q")


def _read_exact(stream, count, deadline=None, *, eof_ok=False):
    data = bytearray()
    descriptor = stream.fileno()
    while len(data) < count:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
                raise TimeoutError("Timed out waiting for the SPAMS worker")
        chunk = os.read(descriptor, min(count - len(data), 65536))
        if not chunk:
            if eof_ok and not data:
                return None
            raise EOFError("SPAMS worker closed its response channel")
        data.extend(chunk)
    return bytes(data)


def read_message(stream, *, timeout=None, eof_ok=False):
    deadline = None if timeout is None else time.monotonic() + timeout
    header = _read_exact(stream, _HEADER.size, deadline, eof_ok=eof_ok)
    if header is None:
        return None
    count = _HEADER.unpack(header)[0]
    if count > MAX_FRAME_BYTES:
        raise ValueError(
            "SPAMS worker message exceeds the 256 MiB limit; use smaller RGB patches/references"
        )
    return pickle.loads(_read_exact(stream, count, deadline))


def write_message(stream, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(
            "SPAMS worker message exceeds the 256 MiB limit; use smaller RGB patches/references"
        )
    for part in (_HEADER.pack(len(payload)), payload):
        remaining = memoryview(part)
        while remaining:
            written = stream.write(remaining)
            if not written:
                raise BrokenPipeError("SPAMS worker closed its request channel")
            remaining = remaining[written:]
    stream.flush()


def _shutdown(process, diagnostics, owner_pid):
    """Finalizer callback deliberately holds no reference to its client."""
    try:
        if process.stdin is not None:
            process.stdin.close()
        if os.getpid() != owner_pid:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    finally:
        if process.stdout is not None:
            process.stdout.close()
        diagnostics.close()


class _DenseConcentrations:
    def __init__(self, array):
        self.array = array

    def toarray(self):
        return self.array


class SpamsClient:
    """Single persistent worker, with one bounded request in flight at a time."""

    execution = "subprocess"

    def __init__(self, *, timeout=300):
        self.timeout = timeout
        self._owner_pid = os.getpid()
        self._lock = threading.RLock()
        self._closed = False
        diagnostics = tempfile.TemporaryFile(mode="w+b")
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "openhisto.stain_transfer.spams_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
                env=worker_environment(),
                bufsize=0,
            )
        except BaseException:
            diagnostics.close()
            raise
        self._process, self._diagnostics = process, diagnostics
        self._finalizer = weakref.finalize(self, _shutdown, process, diagnostics, self._owner_pid)
        try:
            hello = self._request("hello", timeout=60)
            if hello != {"ready": True}:
                raise RuntimeError("Invalid SPAMS worker startup response")
        except BaseException:
            self.close()
            raise

    def _diagnostic_tail(self):
        try:
            size = os.fstat(self._diagnostics.fileno()).st_size
            return (
                os.pread(self._diagnostics.fileno(), min(size, 8192), max(0, size - 8192))
                .decode("utf-8", errors="replace")
                .strip()
            )
        except (OSError, ValueError):
            return ""

    def _request(self, operation, kwargs=None, *, timeout=None):
        with self._lock:
            if os.getpid() != self._owner_pid:
                raise RuntimeError(
                    "A SPAMS worker cannot be shared across forked processes; create a new normalizer in each process"
                )
            if self._closed:
                raise RuntimeError("SPAMS worker is closed; create a new normalizer")
            try:
                write_message(self._process.stdin, {"operation": operation, "kwargs": kwargs or {}})
                response = read_message(
                    self._process.stdout, timeout=self.timeout if timeout is None else timeout
                )
                if not isinstance(response, dict) or "ok" not in response:
                    raise ValueError("Invalid SPAMS worker response")
            except (EOFError, OSError, TimeoutError, ValueError, pickle.UnpicklingError) as exc:
                details = self._diagnostic_tail()
                self.close()
                message = f"SPAMS worker communication failed during {operation}: {exc}"
                if details:
                    message += f"; worker diagnostics: {details}"
                raise RuntimeError(message) from exc
            if not response["ok"]:
                raise RuntimeError(
                    f"SPAMS worker {operation} failed ({response.get('type', 'Error')}): {response.get('error', 'unknown worker error')}"
                )
            return response.get("result")

    def trainDL(self, **kwargs):
        return self._request("trainDL", kwargs)

    def lasso(self, **kwargs):
        return _DenseConcentrations(self._request("lasso", kwargs))

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Forked children must not terminate the original owner's worker.
            if os.getpid() == self._owner_pid:
                self._finalizer()
            else:
                self._finalizer.detach()
                self._process.stdin.close()
                self._process.stdout.close()
                self._diagnostics.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
