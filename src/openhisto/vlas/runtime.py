"""A persistent, bounded request/response connection to one isolated model."""

import base64
from collections import deque
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid

from openhisto._common import isolated_module_command
from .base import VLAError, WorkerError, error_text, validate_results
from .config import json_safe, reject_secrets


class WorkerClient:
    def __init__(self, spec, config):
        self.spec, self.config = spec, config
        self.metadata = {}
        self._closed = False
        self._messages = queue.Queue()
        self._logs = deque(maxlen=20)
        environment = os.environ.copy()
        if config.offline:
            environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if config.cache_dir:
            environment["HF_HUB_CACHE"] = str(Path(config.cache_dir).expanduser().resolve())
        try:
            self.process = subprocess.Popen(
                isolated_module_command(
                    spec.python_executable or sys.executable, "openhisto.vlas.worker", "-u"
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=environment,
            )
        except OSError as exc:
            raise VLAError(
                "Cannot start model worker; check python_executable and its environment"
            ) from exc
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self._send({"op": "prepare", "spec": spec.to_dict(), "config": config.to_dict()})
            response = self._receive(config.startup_timeout)
            if response.get("op") != "ready" or not isinstance(response.get("metadata"), dict):
                raise VLAError("Model worker did not return its ready handshake")
            reject_secrets(response["metadata"])
            self.metadata = response["metadata"]
        except BaseException:
            self.close()
            raise

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                self._messages.put(line)
        finally:
            self._messages.put(None)

    def _read_stderr(self):
        try:
            for line in self.process.stderr:
                # Redact before truncation so a token crossing the retained
                # boundary cannot leave an unrecognized partial credential.
                self._logs.append(error_text(line, limit=None)[-2000:])
        except (ValueError, OSError):
            pass

    def _worker_error(self, message):
        stderr = None
        if self.config.worker_diagnostics:
            stderr = error_text("".join(self._logs), limit=None)[-4000:]
        return WorkerError(message, exit_status=self.process.poll(), stderr_tail=stderr)

    def _send(self, message):
        try:
            self.process.stdin.write(json.dumps(json_safe(message), allow_nan=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise self._worker_error("Model worker exited while receiving a request") from exc

    def _receive(self, timeout):
        try:
            line = self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            self.close()
            raise self._worker_error(
                "Model worker timed out; increase startup_timeout or timeout after checking the selected environment"
            ) from exc
        if line is None:
            self.close()
            raise self._worker_error(
                "Model worker exited before returning a response; check its Python environment and checkpoint artifacts"
            )
        try:
            response = json.loads(line)
        except (ValueError, TypeError) as exc:
            self.close()
            raise self._worker_error(
                "Invalid worker protocol; vendor output must use stderr"
            ) from exc
        if not isinstance(response, dict):
            self.close()
            raise self._worker_error("Invalid worker protocol object")
        if response.get("op") == "error":
            self.close()
            raise self._worker_error(response.get("error", "Model worker failed"))
        return response

    def generate(self, images, prompt):
        if self._closed:
            raise VLAError("Model worker is closed")
        from PIL import Image
        import numpy as np

        payload = []
        for image in images:
            if not isinstance(image, Image.Image):
                image = np.asarray(image)
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(
                        "Worker images must be uint8 HWC RGB arrays or RGB Pillow images"
                    )
                image = Image.fromarray(image)
            if image.mode != "RGB":
                raise ValueError("Worker Pillow images must already be RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        request_id = uuid.uuid4().hex
        try:
            self._send({"op": "generate", "id": request_id, "images": payload, "prompt": prompt})
            response = self._receive(self.config.timeout)
            if response.get("op") != "result" or response.get("id") != request_id:
                raise VLAError("Worker response does not match its patch request")
            return validate_results(response.get("results"), len(images))
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = getattr(self, "process", None)
        if process is None:
            return
        try:
            if process.poll() is None:
                self._send({"op": "close"})
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
        except (OSError, VLAError):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
        finally:
            for thread in (
                getattr(self, "_stdout_thread", None),
                getattr(self, "_stderr_thread", None),
            ):
                if thread and thread is not threading.current_thread():
                    thread.join(timeout=2)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
