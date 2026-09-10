"""Smoke and fire detection using RF-DETR (nano, small, ...)."""

import time

import config
from detectors.base_detector import BaseDetector
from engines.alert_engine import AlertType
from services.rfdetr_models import load_rfdetr_model, resolve_checkpoint_path
from services.rfdetr_smoke_fire import annotate_frame, predict_frame
from services.vlm_benchmark import log_result


class RfdetrSmokeFireDetector(BaseDetector):
    """Detects smoke and fire via RF-DETR (requires a fine-tuned checkpoint)."""

    def __init__(
        self,
        model_size="nano",
        model_label=None,
        threshold=None,
        device=None,
        checkpoint_path=None,
        alert_cooldown=30,
    ):
        super().__init__()
        self.model_size = model_size
        self.model_label = model_label or f"rfdetr-{model_size}"
        self.threshold = threshold if threshold is not None else config.RFDETR_THRESHOLD
        self.device = device if device is not None else config.RFDETR_DEVICE
        self.checkpoint_path = checkpoint_path or resolve_checkpoint_path(model_size)
        self.alert_cooldown = alert_cooldown
        self._model = None
        self._last_alert_at = {}

    def on_start(self):
        self._model = load_rfdetr_model(
            size=self.model_size,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
        )
        source = self.checkpoint_path or "coco-pretrained (smoke/fire unlikely without fine-tuning)"
        self.log(f"RF-DETR {self.model_size} loaded from {source}")

    def process(self, frame, frame_number=0):
        started_at = time.perf_counter()
        detections, findings, _ = predict_frame(
            self._model,
            frame,
            threshold=self.threshold,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000

        if config.VLM_BENCHMARK_ENABLED:
            finding_set = set(findings.keys())
            alert_type = ""
            if finding_set:
                alert_type = AlertType.CRITICAL if "fire" in finding_set else AlertType.WARNING
            log_result(
                self.name,
                self.checkpoint_path or self.model_label,
                frame_number,
                latency_ms,
                response=self._build_message(findings) if findings else "none",
                findings=finding_set,
                alert_type=alert_type,
            )
            self.log(
                f"benchmark {self.model_label}: {latency_ms:.0f}ms | "
                f"findings={sorted(finding_set) or ['none']}"
            )

        if not findings:
            return

        if "fire" in findings:
            alert_type = AlertType.CRITICAL
        else:
            alert_type = AlertType.WARNING

        if not self._cooldown_passed(alert_type):
            return

        if config.VLM_BENCHMARK_ENABLED and not config.VLM_BENCHMARK_RAISE_ALERTS:
            return

        annotated = annotate_frame(frame, detections, self._model)
        self.alert(
            self.to_base64(annotated),
            alert_type,
            self._build_message(findings),
        )

    def _build_message(self, findings):
        parts = [
            f"{label} x{len(confidences)} (best {max(confidences):.0%})"
            for label, confidences in sorted(findings.items())
        ]
        return "detected " + ", ".join(parts)

    def _cooldown_passed(self, alert_type):
        now = time.time()
        if now - self._last_alert_at.get(alert_type, 0) < self.alert_cooldown:
            return False
        self._last_alert_at[alert_type] = now
        return True
