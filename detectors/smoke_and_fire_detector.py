"""Smoke and fire detection using a YOLOv8n model fine tuned on the D-Fire dataset.

The model file lives in models/ and knows two classes: "smoke" and "fire".
"""

import time

from ultralytics import YOLO

import config
from detectors.base_detector import BaseDetector
from engines.alert_engine import AlertType
from services.vlm_benchmark import log_result


class SmokeAndFireDetector(BaseDetector):
    """Alerts when the model sees smoke or fire in a frame.

    Fire is raised as CRITICAL and smoke on its own as WARNING. After an alert the
    same kind stays quiet for `alert_cooldown` seconds, otherwise a burning frame
    would send one alert per processed frame.
    """

    def __init__(self, model_path=None, confidence=0.4, alert_cooldown=30):
        super().__init__()
        self.model_path = model_path or config.SMOKE_FIRE_MODEL_PATH
        self.confidence = confidence
        self.alert_cooldown = alert_cooldown

        self._model = None
        self._last_alert_at = {}

    def on_start(self):
        self._model = YOLO(self.model_path)
        self.log(f"model loaded from {self.model_path}")

    def process(self, frame, frame_number=0):
        started_at = time.perf_counter()
        result = self._model.predict(frame, conf=self.confidence, verbose=False)[0]
        latency_ms = (time.perf_counter() - started_at) * 1000
        findings = self._collect_findings(result)

        if config.VLM_BENCHMARK_ENABLED:
            finding_set = set(findings.keys())
            alert_type = ""
            if finding_set:
                alert_type = AlertType.CRITICAL if "fire" in finding_set else AlertType.WARNING
            log_result(
                self.name,
                self.model_path,
                frame_number,
                latency_ms,
                response=self._build_message(findings) if findings else "none",
                findings=finding_set,
                alert_type=alert_type,
            )
            self.log(
                f"benchmark yolo: {latency_ms:.0f}ms | "
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

        self.alert(self.to_base64(result.plot()), alert_type, self._build_message(findings))

    def _collect_findings(self, result):
        """Return {"fire": [confidences...], "smoke": [confidences...]} for this frame."""
        findings = {}
        for box in result.boxes:
            label = result.names[int(box.cls)]
            findings.setdefault(label, []).append(float(box.conf))
        return findings

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
