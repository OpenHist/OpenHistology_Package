"""Private native-SPAMS worker; launched by :mod:`openhisto.stain_transfer.spams_process`."""

import os
import sys

from .spams_process import read_message, write_message


def main():
    # Keep a dedicated protocol fd, then redirect even native/C stdout prints.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        import numpy as np
        import spams
    except Exception as exc:
        write_message(protocol, {"ok": False, "type": type(exc).__name__, "error": str(exc)[:8192]})
        return 1
    try:
        while True:
            request = read_message(sys.stdin.buffer, eof_ok=True)
            if request is None:
                return 0
            try:
                if not isinstance(request, dict) or not isinstance(request.get("kwargs"), dict):
                    raise ValueError("Invalid SPAMS worker request")
                operation, kwargs = request.get("operation"), request["kwargs"]
                if operation == "hello":
                    result = {"ready": True}
                elif operation == "trainDL":
                    result = np.asarray(spams.trainDL(**kwargs))
                elif operation == "lasso":
                    result = spams.lasso(**kwargs).toarray()
                else:
                    raise ValueError(f"Unsupported SPAMS worker operation: {operation}")
                write_message(protocol, {"ok": True, "result": result})
            except Exception as exc:
                write_message(
                    protocol, {"ok": False, "type": type(exc).__name__, "error": str(exc)[:8192]}
                )
    finally:
        protocol.close()


if __name__ == "__main__":
    raise SystemExit(main())
