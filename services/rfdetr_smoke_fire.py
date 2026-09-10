"""RF-DETR smoke/fire detection helpers (nano, small, ...)."""

import json
import time

import cv2
import numpy as np
import supervision as sv

import config
from services.rfdetr_models import load_rfdetr_model, resolve_checkpoint_path

SMOKE_FIRE_LABELS = {"smoke", "fire"}


def class_name_for_detection(detections, index, model):
    names = detections.data.get("class_name") if detections.data else None
    if names and index < len(names):
        return str(names[index]).lower()
    if model.class_names and index < len(detections.class_id):
        class_id = int(detections.class_id[index])
        if 0 <= class_id < len(model.class_names):
            return str(model.class_names[class_id]).lower()
    return ""


def collect_findings(detections, model):
    """Return {"fire": [confidences...], "smoke": [confidences...]} for this frame."""
    findings = {}
    confidences = detections.confidence
    if confidences is None:
        confidences = []

    for index in range(len(detections)):
        label = class_name_for_detection(detections, index, model)
        if label not in SMOKE_FIRE_LABELS:
            continue
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        findings.setdefault(label, []).append(confidence)

    return findings


def filter_smoke_fire(detections, model):
    keep = []
    for index in range(len(detections)):
        keep.append(class_name_for_detection(detections, index, model) in SMOKE_FIRE_LABELS)
    return detections[keep]


def annotate_frame(frame_bgr, detections, model):
    """Draw smoke/fire boxes on a BGR frame."""
    filtered = filter_smoke_fire(detections, model)
    labels = []
    for index in range(len(filtered)):
        name = class_name_for_detection(filtered, index, model)
        confidence = (
            float(filtered.confidence[index])
            if filtered.confidence is not None and index < len(filtered.confidence)
            else 0.0
        )
        labels.append(f"{name} {confidence:.0%}")

    annotated = sv.BoxAnnotator().annotate(frame_bgr, filtered)
    return sv.LabelAnnotator().annotate(annotated, filtered, labels)


def predict_frame(model, frame_bgr, threshold=None):
    """Run RF-DETR on one BGR frame. Returns (detections, findings, latency_ms)."""
    threshold = threshold if threshold is not None else config.RFDETR_THRESHOLD
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    started = time.perf_counter()
    detections = model.predict(frame_rgb, threshold=threshold)
    latency_ms = (time.perf_counter() - started) * 1000

    findings = collect_findings(detections, model)
    return detections, findings, latency_ms


def iter_video_frames(cap, fps, interval_seconds=None, max_frames=None):
    """Yield (source_frame_idx, bgr_frame)."""
    source_idx = 0
    processed = 0
    next_ts = 0.0

    while max_frames is None or processed < max_frames:
        if interval_seconds is None:
            success, frame = cap.read()
            if not success:
                return
            source_idx += 1
        else:
            target_idx = int(round(next_ts * fps))
            if target_idx == source_idx:
                success, frame = cap.read()
                if not success:
                    return
            else:
                while source_idx < target_idx:
                    if not cap.grab():
                        return
                    source_idx += 1
                success, frame = cap.retrieve()
                if not success:
                    return
            source_idx += 1
            next_ts += interval_seconds
        processed += 1
        yield source_idx - 1, frame


def open_video_source(source: str):
    parsed = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(parsed)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")
    return cap


def summarize_latency_ms(latencies_ms):
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "avg_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "inference_fps": float(1000.0 / np.mean(arr)),
    }


def run_video(
    source,
    threshold=None,
    device=None,
    checkpoint_path=None,
    model_size="nano",
    out_path=None,
    max_frames=None,
    every=None,
    show=False,
    results_path=None,
):
    """Process a video/RTSP source for smoke/fire with RF-DETR."""
    model = load_rfdetr_model(
        size=model_size,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    cap = open_video_source(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = None
    if out_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    latencies_ms = []
    frame_counts = {"smoke": 0, "fire": 0, "none": 0}
    processed_frames = 0

    try:
        for _, frame_bgr in iter_video_frames(cap, fps, interval_seconds=every, max_frames=max_frames):
            processed_frames += 1
            detections, findings, latency_ms = predict_frame(model, frame_bgr, threshold=threshold)
            latencies_ms.append(latency_ms)

            if findings:
                if "fire" in findings:
                    frame_counts["fire"] += 1
                elif "smoke" in findings:
                    frame_counts["smoke"] += 1
            else:
                frame_counts["none"] += 1

            annotated = annotate_frame(frame_bgr, detections, model)
            summary = ", ".join(
                f"{label} x{len(confs)} (best {max(confs):.0%})"
                for label, confs in sorted(findings.items())
            ) or "none"
            cv2.putText(
                annotated,
                summary,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("RF-DETR Smoke/Fire", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if not latencies_ms:
        raise RuntimeError("No frames processed.")

    latency_stats = summarize_latency_ms(latencies_ms)
    print(f"Processed frames: {processed_frames}")
    print(f"Frames with fire: {frame_counts['fire']}")
    print(f"Frames with smoke: {frame_counts['smoke']}")
    print(f"Frames with none: {frame_counts['none']}")
    print(f"Avg latency: {latency_stats['avg_ms']:.1f} ms")

    results = {
        "model": f"rfdetr-{model_size}-smoke-fire",
        "source": source,
        "processed_frames": processed_frames,
        "frame_counts": frame_counts,
        "latency_ms": latency_stats,
        "checkpoint": checkpoint_path or resolve_checkpoint_path(model_size) or "coco-pretrained",
    }

    if results_path:
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Results saved to {results_path}")

    return results
