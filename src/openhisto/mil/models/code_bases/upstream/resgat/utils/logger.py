"""
Training logger that writes JSON-lines to a log file under ``log_dir``.

Each line is a self-contained JSON object with a ``type`` field
(``"epoch"``, ``"test"``, or ``"info"``) so logs are easy to parse
programmatically while remaining human-readable.
"""

import json
import os
from datetime import datetime


class TrainLogger:
    """Append-only JSON-lines logger.

    Args:
        log_dir:  Directory to write the log file.
        run_name: Base name for the log file (e.g. ``"ac_fold0"``).
    """

    def __init__(self, log_dir, run_name):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{run_name}.jsonl")
        self._fh = open(self.path, "a")

    def _write(self, record: dict):
        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc,
                  **extra):
        rec = {"type": "epoch", "epoch": epoch,
               "train_loss": round(train_loss, 6),
               "train_acc": round(train_acc, 4),
               "val_loss": round(val_loss, 6),
               "val_acc": round(val_acc, 4)}
        rec.update(extra)
        self._write(rec)

    def log_test(self, **metrics):
        rec = {"type": "test"}
        rec.update({k: round(v, 6) if isinstance(v, float) else v
                    for k, v in metrics.items()})
        self._write(rec)

    def log_info(self, message, **extra):
        rec = {"type": "info", "message": message}
        rec.update(extra)
        self._write(rec)

    def close(self):
        self._fh.close()
