"""Backward-compatible alias for the shared transformers VLM detector."""

from detectors.transformers_vlm_smoke_fire_detector import TransformersVlmSmokeFireDetector

# Kept for existing imports; prefer TransformersVlmSmokeFireDetector with an explicit model_id.
SmolVLM2SmokeFireDetector = TransformersVlmSmokeFireDetector
