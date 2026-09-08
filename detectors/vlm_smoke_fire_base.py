"""Shared smoke/fire logic for vision-language model detectors."""

import re
import time
from abc import abstractmethod

import config
from detectors.base_detector import BaseDetector
from engines.alert_engine import AlertType
from services.vlm_benchmark import log_result

DEFAULT_PROMPT = (
    "You are analyzing a security camera frame. "
    "Reply with exactly two tokens in this format: fire:yes or fire:no, smoke:yes or smoke:no"
)


class VlmSmokeFireDetectorBase(BaseDetector):
    """Base class for VLM detectors that answer a structured smoke/fire prompt."""

    def __init__(
        self,
        model_id,
        label=None,
        prompt=None,
        alert_cooldown=None,
        raise_alerts=None,
    ):
        super().__init__()
        self.model_id = model_id
        self.label = label or model_id
        self.prompt = prompt or config.VLM_PROMPT
        self.alert_cooldown = alert_cooldown or config.VLM_ALERT_COOLDOWN
        self.raise_alerts = (
            raise_alerts if raise_alerts is not None else config.VLM_BENCHMARK_RAISE_ALERTS
        )
        self._last_alert_at = {}

    @abstractmethod
    def _ask(self, frame):
        """Run model inference and return the raw text response."""

    def process(self, frame, frame_number=0):
        started_at = time.perf_counter()
        response = ""
        error = ""

        try:
            response = self._ask(frame)
        except Exception as exc:
            error = str(exc)
            latency_ms = (time.perf_counter() - started_at) * 1000
            log_result(
                self.name,
                self.model_id,
                frame_number,
                latency_ms,
                response=response,
                error=error,
            )
            self.log(f"inference failed: {error}", level="ERROR")
            return

        latency_ms = (time.perf_counter() - started_at) * 1000
        findings = self._classify_findings(response)
        alert_type = self._resolve_alert_type(findings) if findings else ""

        log_result(
            self.name,
            self.model_id,
            frame_number,
            latency_ms,
            response=response,
            findings=findings,
            alert_type=alert_type,
        )

        summary = (
            f"benchmark {self.label}: {latency_ms:.0f}ms | {response} | "
            f"findings={sorted(findings) or ['none']}"
        )
        self.log(summary)

        if not findings:
            return

        if not self._cooldown_passed(alert_type):
            return

        if not self.raise_alerts:
            return

        self.alert(
            self.to_base64(frame),
            alert_type,
            self._build_message(findings, response),
        )

    def _classify_findings(self, response):
        text = response.strip().lower()
        if not text:
            return set()

        structured = self._parse_structured_response(text)
        if structured is not None:
            return structured

        if " or both" in text or re.search(r"none.*smoke.*fire", text):
            return set()

        first = re.split(r"[\s,.:;]+", text)[0]
        single_word = {
            "none": set(),
            "no": set(),
            "smoke": {"smoke"},
            "fire": {"fire"},
            "both": {"smoke", "fire"},
        }
        if first in single_word:
            return single_word[first]

        findings = set()
        words = set(re.findall(r"[a-z]+", text))
        if "both" in words:
            findings.update({"smoke", "fire"})
        else:
            if "fire" in words or "flame" in words or "burning" in words:
                findings.add("fire")
            if "smoke" in words:
                findings.add("smoke")

        return findings

    def _parse_structured_response(self, text):
        fire_match = re.search(r"fire\s*:\s*(yes|no)", text)
        smoke_match = re.search(r"smoke\s*:\s*(yes|no)", text)
        if not fire_match and not smoke_match:
            return None

        findings = set()
        if fire_match and fire_match.group(1) == "yes":
            findings.add("fire")
        if smoke_match and smoke_match.group(1) == "yes":
            findings.add("smoke")
        return findings

    def _resolve_alert_type(self, findings):
        if "fire" in findings:
            return AlertType.CRITICAL
        return AlertType.WARNING

    def _build_message(self, findings, response):
        labels = ", ".join(sorted(findings))
        return f"{self.label} detected {labels} ({response})"

    def _cooldown_passed(self, alert_type):
        now = time.time()
        if now - self._last_alert_at.get(alert_type, 0) < self.alert_cooldown:
            return False
        self._last_alert_at[alert_type] = now
        return True
