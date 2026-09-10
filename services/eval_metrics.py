"""Accuracy metrics for frame-level smoke/fire evaluation."""

CLASSES = ("NONE", "FIRE", "SMOKE")


def findings_to_label(findings):
    """Map a findings set to a single ground-truth-style label."""
    if "fire" in findings:
        return "FIRE"
    if "smoke" in findings:
        return "SMOKE"
    return "NONE"


def compute_metrics(results):
    """Compute summary metrics from per-frame result dicts."""
    total = len(results)
    if total == 0:
        return {}

    correct = sum(1 for row in results if row["correct"])
    hazard_correct = sum(
        1
        for row in results
        if (row["ground_truth"] != "NONE") == (row["prediction"] != "NONE")
    )

    per_class = {}
    for label in CLASSES:
        tp = sum(
            1 for row in results if row["ground_truth"] == label and row["prediction"] == label
        )
        fp = sum(
            1 for row in results if row["ground_truth"] != label and row["prediction"] == label
        )
        fn = sum(
            1 for row in results if row["ground_truth"] == label and row["prediction"] != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        support = sum(1 for row in results if row["ground_truth"] == label)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    latencies = [row["latency_ms"] for row in results if row.get("latency_ms") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "hazard_accuracy": hazard_correct / total,
        "per_class": per_class,
        "avg_latency_ms": avg_latency,
        "errors": sum(1 for row in results if row.get("error")),
    }


def format_summary(model_name, metrics):
    """Return a human-readable summary block for one model."""
    if not metrics:
        return f"{model_name}: no frames evaluated\n"

    lines = [
        f"Model: {model_name}",
        f"  Accuracy:      {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})",
        f"  Hazard acc:    {metrics['hazard_accuracy']:.1%} (any fire/smoke vs none)",
        f"  Avg latency:   {metrics['avg_latency_ms']:.0f} ms",
    ]
    if metrics["errors"]:
        lines.append(f"  Errors:        {metrics['errors']}")

    for label in CLASSES:
        stats = metrics["per_class"][label]
        lines.append(
            f"  {label:5s}  P={stats['precision']:.2f}  R={stats['recall']:.2f}  "
            f"F1={stats['f1']:.2f}  (n={stats['support']})"
        )

    return "\n".join(lines) + "\n"
