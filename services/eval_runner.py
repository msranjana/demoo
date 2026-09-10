"""Run each detector on labeled frames for offline evaluation."""

import time

import cv2

import config
from detectors.florence2_smoke_fire_detector import Florence2SmokeFireDetector
from detectors.gguf_vlm_smoke_fire_detector import GgufVlmSmokeFireDetector
from detectors.moondream_onnx_smoke_fire_detector import MoondreamOnnxSmokeFireDetector
from detectors.smoke_and_fire_detector import SmokeAndFireDetector
from detectors.transformers_vlm_smoke_fire_detector import TransformersVlmSmokeFireDetector
from services.eval_metrics import findings_to_label

EVAL_ALL_MODELS = [
    "yolo",
    "florence2_base",
    "moondream_onnx",
    "smolvlm2_500m",
    "smolvlm2_256m",
    "internvl3_1b",
    "qwen35_0_8b",
]


class FrameEvaluator:
    """Base class for single-frame prediction during evaluation."""

    name = "evaluator"

    def setup(self):
        """Load model weights."""

    def teardown(self):
        """Optional cleanup."""

    def predict(self, frame):
        """Return (label, raw_response, latency_ms)."""
        raise NotImplementedError


class YoloEvaluator(FrameEvaluator):
    def __init__(self, detector=None):
        self.detector = detector or SmokeAndFireDetector()
        self.name = "yolo"

    def setup(self):
        self.detector.on_start()

    def predict(self, frame):
        started = time.perf_counter()
        result = self.detector._model.predict(frame, conf=self.detector.confidence, verbose=False)[0]
        latency_ms = (time.perf_counter() - started) * 1000
        findings = self.detector._collect_findings(result)
        label = findings_to_label(set(findings.keys()))
        response = self.detector._build_message(findings) if findings else "none"
        return label, response, latency_ms


class VlmEvaluator(FrameEvaluator):
    def __init__(self, detector, name):
        self.detector = detector
        self.name = name

    def setup(self):
        self.detector.on_start()

    def predict(self, frame):
        started = time.perf_counter()
        response = self.detector._ask(frame)
        latency_ms = (time.perf_counter() - started) * 1000
        findings = self.detector._classify_findings(response)
        label = findings_to_label(findings)
        return label, response, latency_ms


def make_evaluator(model_key):
    if model_key == "yolo":
        return YoloEvaluator()

    if model_key == "florence2_base":
        return VlmEvaluator(Florence2SmokeFireDetector(), "florence2_base")

    if model_key == "moondream_onnx":
        return VlmEvaluator(MoondreamOnnxSmokeFireDetector(), "moondream_onnx")

    if model_key == "smolvlm2_500m":
        return VlmEvaluator(
            TransformersVlmSmokeFireDetector(
                model_id=config.SMOLVLM2_500M_MODEL,
                label="smolvlm2-500m",
            ),
            "smolvlm2_500m",
        )

    if model_key == "smolvlm2_256m":
        return VlmEvaluator(
            TransformersVlmSmokeFireDetector(
                model_id=config.SMOLVLM2_256M_MODEL,
                label="smolvlm2-256m",
            ),
            "smolvlm2_256m",
        )

    if model_key == "internvl3_1b":
        return VlmEvaluator(
            TransformersVlmSmokeFireDetector(
                model_id=config.INTERNVL3_MODEL,
                label="internvl3-1b",
            ),
            "internvl3_1b",
        )

    if model_key == "qwen35_0_8b":
        return VlmEvaluator(
            TransformersVlmSmokeFireDetector(
                model_id=config.QWEN35_MODEL,
                label="qwen3.5-0.8b",
                trust_remote_code=True,
            ),
            "qwen35_0_8b",
        )

    if model_key == "gguf_smolvlm2_256m":
        return VlmEvaluator(
            GgufVlmSmokeFireDetector(
                model_id=config.GGUF_MODEL_ID,
                label="gguf-smolvlm2-256m",
            ),
            "gguf_smolvlm2_256m",
        )

    raise ValueError(f"no evaluator for model '{model_key}'")


def run_evaluation(evaluator, labeled_frames):
    """Run evaluator over labeled frames. labeled_frames: list of dicts with path, label, meta."""
    evaluator.setup()
    results = []

    try:
        for index, item in enumerate(labeled_frames, start=1):
            frame = cv2.imread(item["path"])
            if frame is None:
                results.append(
                    {
                        "ground_truth": item["label"],
                        "prediction": "",
                        "correct": False,
                        "response": "",
                        "latency_ms": None,
                        "error": f"failed to read {item['path']}",
                        **item["meta"],
                    }
                )
                continue

            try:
                prediction, response, latency_ms = evaluator.predict(frame)
                results.append(
                    {
                        "ground_truth": item["label"],
                        "prediction": prediction,
                        "correct": prediction == item["label"],
                        "response": response,
                        "latency_ms": latency_ms,
                        "error": "",
                        **item["meta"],
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "ground_truth": item["label"],
                        "prediction": "",
                        "correct": False,
                        "response": "",
                        "latency_ms": None,
                        "error": str(exc),
                        **item["meta"],
                    }
                )

            print(f"  [{evaluator.name}] {index}/{len(labeled_frames)} {item['meta']['frame_id']}")
    finally:
        evaluator.teardown()

    return results
