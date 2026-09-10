"""Load labeled frames for offline accuracy evaluation."""

import csv
import os


def load_labels(labels_path):
    """Read labels.csv and return rows with normalized ground-truth labels."""
    rows = []
    with open(labels_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = row["label"].strip().upper()
            if label not in {"NONE", "FIRE", "SMOKE"}:
                raise ValueError(f"unsupported label '{row['label']}' in {labels_path}")
            rows.append(
                {
                    "video_id": row["video_id"].strip(),
                    "frame_file": row["frame_file"].strip(),
                    "timestamp_sec": row.get("timestamp_sec", ""),
                    "label": label,
                }
            )
    return rows


def resolve_frame_path(row, frames_dir):
    """Resolve a frame path from CSV (absolute, or data/frames/<video_id>/<basename>)."""
    basename = os.path.basename(row["frame_file"])
    video_id = row["video_id"]

    candidates = [
        os.path.join(frames_dir, video_id, basename),
        os.path.join(frames_dir, basename),
        row["frame_file"],
    ]

    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        f"frame not found for {video_id}/{basename} (checked under {frames_dir})"
    )
