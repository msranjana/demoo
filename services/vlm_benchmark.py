"""Thread-safe CSV logger for comparing VLM latency and accuracy."""

import csv
import os
import threading
from datetime import datetime

import config

_lock = threading.Lock()
_header_written = False
_stats = {}

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


def reset_stats():
    global _stats
    with _lock:
        _stats = {}


def _record_stat(detector_name, latency_ms, error=""):
    with _lock:
        entry = _stats.setdefault(
            detector_name,
            {"count": 0, "total_ms": 0.0, "errors": 0},
        )
        if error:
            entry["errors"] += 1
            return
        entry["count"] += 1
        entry["total_ms"] += latency_ms


def get_summary():
    with _lock:
        summary = {}
        for name, entry in _stats.items():
            count = entry["count"]
            summary[name] = {
                "frames": count,
                "errors": entry["errors"],
                "avg_latency_ms": entry["total_ms"] / count if count else 0.0,
                "total_ms": entry["total_ms"],
            }
        return summary


def print_summary():
    summary = get_summary()
    if not summary:
        return

    print("\n--- average latency ---")
    for name, stats in sorted(summary.items()):
        if stats["frames"]:
            print(
                f"{name}: {stats['avg_latency_ms']:.1f} ms avg "
                f"({stats['frames']} frames, {stats['total_ms']:.0f} ms total)"
            )
        if stats["errors"]:
            print(f"{name}: {stats['errors']} error(s)")
    print("-----------------------\n")


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
    _record_stat(detector_name, latency_ms, error=error)

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
