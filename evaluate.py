"""Offline accuracy evaluation against labeled frames.

Usage:
  python evaluate.py yolo
  python evaluate.py florence2-base moondream
  python evaluate.py --all
"""

import argparse
import csv
import os
import sys

import config
from console import say
from main import MODEL_ALIASES, normalize_model_name
from services.eval_dataset import load_labels, resolve_frame_path
from services.eval_metrics import compute_metrics, format_summary
from services.eval_runner import EVAL_ALL_MODELS, make_evaluator, run_evaluation

DEFAULT_LABELS = "data/labels.csv"
DEFAULT_FRAMES = "data/frames"
DEFAULT_OUTPUT = "logs/eval_results.csv"


def available_eval_models():
    return sorted(set(MODEL_ALIASES.keys()) - {"all"})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate smoke/fire detectors against labeled frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="models: " + ", ".join(available_eval_models()),
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="one or more model names (see --list)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="evaluate all supported models (excludes gguf unless named explicitly)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available model names and exit",
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help=f"path to labels.csv (default: {DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--frames",
        default=DEFAULT_FRAMES,
        help=f"path to frames directory (default: {DEFAULT_FRAMES})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"per-frame results CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def resolve_model_list(args):
    if args.list:
        for name in available_eval_models():
            print(name)
        return None

    if args.all:
        return list(EVAL_ALL_MODELS)

    if not args.models:
        say("pass at least one model name, or use --all (try: python evaluate.py --list)")
        return []

    keys = []
    for name in args.models:
        keys.append(normalize_model_name(name))
    return keys


def build_labeled_frames(labels_path, frames_dir):
    rows = load_labels(labels_path)
    labeled = []
    missing = []

    for row in rows:
        try:
            path = resolve_frame_path(row, frames_dir)
        except FileNotFoundError:
            missing.append(row)
            continue

        basename = os.path.basename(path)
        labeled.append(
            {
                "path": path,
                "label": row["label"],
                "meta": {
                    "video_id": row["video_id"],
                    "frame_id": basename,
                    "timestamp_sec": row["timestamp_sec"],
                },
            }
        )

    if missing:
        say(f"warning: {len(missing)} frame(s) missing under {frames_dir}")

    return labeled, missing


def write_results_csv(path, model_name, results):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    fieldnames = [
        "model",
        "video_id",
        "frame_id",
        "timestamp_sec",
        "ground_truth",
        "prediction",
        "correct",
        "latency_ms",
        "response",
        "error",
    ]

    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    "model": model_name,
                    "video_id": row["video_id"],
                    "frame_id": row["frame_id"],
                    "timestamp_sec": row["timestamp_sec"],
                    "ground_truth": row["ground_truth"],
                    "prediction": row["prediction"],
                    "correct": row["correct"],
                    "latency_ms": (
                        f"{row['latency_ms']:.1f}" if row.get("latency_ms") is not None else ""
                    ),
                    "response": row.get("response", ""),
                    "error": row.get("error", ""),
                }
            )


def main(argv=None):
    args = parse_args(argv)
    model_keys = resolve_model_list(args)
    if model_keys is None:
        return 0
    if not model_keys:
        return 1

    if not os.path.isfile(args.labels):
        say(f"labels file not found: {args.labels}")
        return 1

    labeled_frames, missing = build_labeled_frames(args.labels, args.frames)
    if not labeled_frames:
        say(f"no frames found under {args.frames} for {args.labels}")
        return 1

    say(f"evaluating {len(labeled_frames)} labeled frame(s) from {args.labels}")

    if os.path.exists(args.output):
        os.remove(args.output)

    summaries = []
    for model_key in model_keys:
        say(f"\n--- {model_key} ---")
        try:
            evaluator = make_evaluator(model_key)
        except ValueError as error:
            say(str(error))
            continue

        try:
            results = run_evaluation(evaluator, labeled_frames)
        except Exception as error:
            say(f"{model_key} failed: {error}")
            continue

        write_results_csv(args.output, evaluator.name, results)
        metrics = compute_metrics(results)
        summary = format_summary(evaluator.name, metrics)
        print(summary)
        summaries.append(summary)

    if summaries:
        say(f"per-frame results -> {args.output}")
    else:
        say("no models evaluated successfully")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
