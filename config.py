"""All settings are read from the .env file so nothing is hardcoded in the code."""

import os

from dotenv import load_dotenv

load_dotenv()


def _enabled(name, default="false"):
    return os.getenv(name, default).lower() in ("1", "true", "yes")


RTSP_URL = os.getenv("RTSP_URL", "")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/events.log")
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "5"))

SMOKE_FIRE_MODEL_PATH = os.getenv("SMOKE_FIRE_MODEL_PATH", "models/fire_smoke_yolov8n.pt")

# Shared VLM smoke/fire prompt and benchmark settings
VLM_PROMPT = os.getenv(
    "VLM_PROMPT",
    "You are analyzing a security camera frame. "
    "Reply with exactly two tokens in this format: fire:yes or fire:no, smoke:yes or smoke:no",
)
VLM_ALERT_COOLDOWN = int(os.getenv("VLM_ALERT_COOLDOWN", "30"))
VLM_TEST_FPS = float(os.getenv("VLM_TEST_FPS", "0.2"))

VLM_BENCHMARK_ENABLED = _enabled("VLM_BENCHMARK_ENABLED", "true")
VLM_BENCHMARK_LOG_PATH = os.getenv("VLM_BENCHMARK_LOG_PATH", "logs/vlm_benchmark.csv")
VLM_BENCHMARK_RAISE_ALERTS = _enabled("VLM_BENCHMARK_RAISE_ALERTS", "false")

YOLO_ENABLED = _enabled("YOLO_ENABLED", "true")
YOLO_FPS = float(os.getenv("YOLO_FPS", "2"))

# RF-DETR Nano — smoke/fire detection (requires fine-tuned checkpoint)
RFDETR_ENABLED = _enabled("RFDETR_ENABLED", "false")
RFDETR_MODEL_PATH = os.getenv("RFDETR_MODEL_PATH", "")
RFDETR_THRESHOLD = float(os.getenv("RFDETR_THRESHOLD", "0.5"))
RFDETR_DEVICE = os.getenv("RFDETR_DEVICE", "cpu")
RFDETR_FPS = float(os.getenv("RFDETR_FPS", "5"))

# Per-model toggles for parallel latency/accuracy testing
SMOLVLM2_500M_ENABLED = _enabled("SMOLVLM2_500M_ENABLED", "true")
SMOLVLM2_500M_MODEL = os.getenv(
    "SMOLVLM2_500M_MODEL", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
)

SMOLVLM2_256M_ENABLED = _enabled("SMOLVLM2_256M_ENABLED", "true")
SMOLVLM2_256M_MODEL = os.getenv(
    "SMOLVLM2_256M_MODEL", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
)

INTERNVL3_ENABLED = _enabled("INTERNVL3_ENABLED", "true")
INTERNVL3_MODEL = os.getenv("INTERNVL3_MODEL", "OpenGVLab/InternVL3-1B-hf")

GGUF_ENABLED = _enabled("GGUF_ENABLED", "false")
GGUF_MODEL_ID = os.getenv(
    "GGUF_MODEL_ID", "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF"
)
GGUF_SERVER_URL = os.getenv("GGUF_SERVER_URL", "http://127.0.0.1:8080")
GGUF_TIMEOUT = int(os.getenv("GGUF_TIMEOUT", "120"))

MOONDREAM_ENABLED = _enabled("MOONDREAM_ENABLED", "false")
MOONDREAM_REPO_ID = os.getenv("MOONDREAM_REPO_ID", "vikhyatk/moondream2")
MOONDREAM_REPO_REVISION = os.getenv("MOONDREAM_REPO_REVISION", "onnx")
MOONDREAM_MODEL_FILE = os.getenv(
    "MOONDREAM_MODEL_FILE", "moondream-0_5b-int4.mf.gz"
)
MOONDREAM_MODEL_PATH = os.getenv(
    "MOONDREAM_MODEL_PATH", "models/moondream/moondream-0_5b-int4.mf.gz"
)

QWEN35_ENABLED = _enabled("QWEN35_ENABLED", "false")
QWEN35_MODEL = os.getenv("QWEN35_MODEL", "Qwen/Qwen3.5-0.8B")

FLORENCE2_ENABLED = _enabled("FLORENCE2_ENABLED", "false")
FLORENCE2_MODEL = os.getenv("FLORENCE2_MODEL", "onnx-community/Florence-2-base")
FLORENCE2_PROMPT = os.getenv(
    "FLORENCE2_PROMPT",
    "Detect all smoke and fire in the image and return their locations in the form of "
    'coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], '
    '"label": "smoke" # or "fire"}.',
)
FLORENCE2_OV_CATEGORIES = os.getenv("FLORENCE2_OV_CATEGORIES", "smoke,fire")
FLORENCE2_ONNX_VARIANT = os.getenv("FLORENCE2_ONNX_VARIANT", "")
FLORENCE2_MAX_NEW_TOKENS = int(os.getenv("FLORENCE2_MAX_NEW_TOKENS", "256"))

# Legacy single-model settings (still read by older docs)
SMOLVLM2_ENABLED = _enabled("SMOLVLM2_ENABLED", "true")
SMOLVLM2_MODEL_PATH = os.getenv("SMOLVLM2_MODEL_PATH", SMOLVLM2_500M_MODEL)
SMOLVLM2_PROMPT = VLM_PROMPT
SMOLVLM2_FPS = VLM_TEST_FPS
SMOLVLM2_ALERT_COOLDOWN = VLM_ALERT_COOLDOWN

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_FROM_EMAIL = os.getenv("ALERT_FROM_EMAIL", "")

ALERT_TO_EMAILS = [
    email.strip() for email in os.getenv("ALERT_TO_EMAILS", "").split(",") if email.strip()
]
