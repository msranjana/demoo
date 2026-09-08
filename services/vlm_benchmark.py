"""Thread-safe CSV logger for comparing VLM latency and accuracy."""

import csv
import os
import threading
from datetime import datetime

import config

_lock = threading.Lock()
_header_written = False

FIELDNAMES = [
    "timestamp",
    "detector",
    "model_id",
    "frame_number",
    "latency_ms",
    "response",
    "findings",
    "alert_type",
    "error",
]


def log_result(
    detector_name,
    model_id,
    frame_number,
    latency_ms,
    response="",
    findings=None,
    alert_type="",
    error="",
):
    if not config.VLM_BENCHMARK_ENABLED:
        return

    path = config.VLM_BENCHMARK_LOG_PATH
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "detector": detector_name,
        "model_id": model_id,
        "frame_number": frame_number,
        "latency_ms": f"{latency_ms:.1f}",
        "response": response,
        "findings": ",".join(sorted(findings or [])),
        "alert_type": alert_type or "",
        "error": error,
    }

    global _header_written
    with _lock:
        write_header = not _header_written and not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
                _header_written = True
            writer.writerow(row)
