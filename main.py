"""Entry point: reads the camera once and feeds every attached detection in parallel.

Usage:
  python main.py                    # run all models enabled in .env
  python main.py smolvlm2-500m      # run one model only
  python main.py --list             # show available model names
"""

import argparse
import sys
import time

import config
from console import say
from detectors.gguf_vlm_smoke_fire_detector import GgufVlmSmokeFireDetector
from detectors.moondream_onnx_smoke_fire_detector import MoondreamOnnxSmokeFireDetector
from detectors.smoke_and_fire_detector import SmokeAndFireDetector
from detectors.transformers_vlm_smoke_fire_detector import TransformersVlmSmokeFireDetector
from engines.alert_engine import AlertEngine
from engines.event_engine import EventEngine
from services.DetectionWrapper import DetectionWrapper
from services.RTSPService import RTSPService
from services.vlm_benchmark import print_summary, reset_stats

MODEL_ALIASES = {
    "yolo": "yolo",
    "smolvlm2-500m": "smolvlm2_500m",
    "smolvlm2_500m": "smolvlm2_500m",
    "smolvlm2-256m": "smolvlm2_256m",
    "smolvlm2_256m": "smolvlm2_256m",
    "internvl3": "internvl3_1b",
    "internvl3-1b": "internvl3_1b",
    "internvl3_1b": "internvl3_1b",
    "gguf": "gguf_smolvlm2_256m",
    "gguf-smolvlm2-256m": "gguf_smolvlm2_256m",
    "gguf_smolvlm2_256m": "gguf_smolvlm2_256m",
    "moondream": "moondream_onnx",
    "moondream2": "moondream_onnx",
    "moondream-onnx": "moondream_onnx",
    "moondream_onnx": "moondream_onnx",
    "qwen3.5-0.8b": "qwen35_0_8b",
    "qwen35": "qwen35_0_8b",
    "qwen35-0.8b": "qwen35_0_8b",
    "qwen35_0_8b": "qwen35_0_8b",
    "all": "all",
}


def available_models():
    return sorted(set(MODEL_ALIASES.keys()) - {"all"})


def normalize_model_name(name):
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        options = ", ".join(available_models())
        raise ValueError(f"unknown model '{name}'. choose one of: {options}, all")
    return MODEL_ALIASES[key]


def make_detection(model_key):
    fps = config.VLM_TEST_FPS

    if model_key == "yolo":
        return DetectionWrapper("yolo_smoke_fire", config.YOLO_FPS, SmokeAndFireDetector())

    if model_key == "smolvlm2_500m":
        return DetectionWrapper(
            "smolvlm2_500m",
            fps,
            TransformersVlmSmokeFireDetector(
                model_id=config.SMOLVLM2_500M_MODEL,
                label="smolvlm2-500m",
            ),
        )

    if model_key == "smolvlm2_256m":
        return DetectionWrapper(
            "smolvlm2_256m",
            fps,
            TransformersVlmSmokeFireDetector(
                model_id=config.SMOLVLM2_256M_MODEL,
                label="smolvlm2-256m",
            ),
        )

    if model_key == "internvl3_1b":
        return DetectionWrapper(
            "internvl3_1b",
            fps,
            TransformersVlmSmokeFireDetector(
                model_id=config.INTERNVL3_MODEL,
                label="internvl3-1b",
            ),
        )

    if model_key == "gguf_smolvlm2_256m":
        return DetectionWrapper(
            "gguf_smolvlm2_256m",
            fps,
            GgufVlmSmokeFireDetector(
                model_id=config.GGUF_MODEL_ID,
                label="gguf-smolvlm2-256m",
            ),
        )

    if model_key == "moondream_onnx":
        return DetectionWrapper(
            "moondream_onnx",
            fps,
            MoondreamOnnxSmokeFireDetector(),
        )

    if model_key == "qwen35_0_8b":
        return DetectionWrapper(
            "qwen35_0_8b",
            fps,
            TransformersVlmSmokeFireDetector(
                model_id=config.QWEN35_MODEL,
                label="qwen3.5-0.8b",
                trust_remote_code=True,
            ),
        )

    raise ValueError(f"no factory for model '{model_key}'")


def build_detections_from_env():
    """Return detections based on .env ENABLED flags."""
    detections = []

    if config.YOLO_ENABLED:
        detections.append(make_detection("yolo"))
    if config.SMOLVLM2_500M_ENABLED:
        detections.append(make_detection("smolvlm2_500m"))
    if config.SMOLVLM2_256M_ENABLED:
        detections.append(make_detection("smolvlm2_256m"))
    if config.INTERNVL3_ENABLED:
        detections.append(make_detection("internvl3_1b"))
    if config.GGUF_ENABLED:
        detections.append(make_detection("gguf_smolvlm2_256m"))
    if config.MOONDREAM_ENABLED:
        detections.append(make_detection("moondream_onnx"))
    if config.QWEN35_ENABLED:
        detections.append(make_detection("qwen35_0_8b"))

    return detections


def build_detections(model_name=None):
    if model_name is None or model_name == "all":
        return build_detections_from_env()
    return [make_detection(model_name)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CCTV smoke/fire detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="models: " + ", ".join(available_models()),
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="run a single model (default: all enabled in .env)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available model names and exit",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.list:
        for name in available_models():
            print(name)
        return 0

    model_name = None
    if args.model:
        try:
            model_name = normalize_model_name(args.model)
        except ValueError as error:
            say(str(error))
            return 1

    if not config.RTSP_URL:
        say("RTSP_URL is missing, set it in the .env file")
        return 1

    reset_stats()

    alert_engine = AlertEngine(
        smtp_config={
            "host": config.SMTP_HOST,
            "port": config.SMTP_PORT,
            "username": config.SMTP_USERNAME,
            "password": config.SMTP_PASSWORD,
            "from_email": config.ALERT_FROM_EMAIL,
            "to_emails": config.ALERT_TO_EMAILS,
        }
    )
    event_engine = EventEngine(config.LOG_FILE_PATH, alert_engine=alert_engine)
    reader = RTSPService(config.RTSP_URL, reconnect_delay=config.RECONNECT_DELAY)

    try:
        detections = build_detections(model_name)
    except ValueError as error:
        say(str(error))
        return 1

    if not detections:
        say("No detection attached. Pass a model name or enable one in .env")
        return 1

    if model_name and model_name != "all":
        say(f"running single model: {args.model}")
    elif config.VLM_BENCHMARK_ENABLED:
        say(f"benchmark mode ON -> {config.VLM_BENCHMARK_LOG_PATH}")

    if config.VLM_BENCHMARK_ENABLED and not config.VLM_BENCHMARK_RAISE_ALERTS:
        say("alerts disabled during benchmark (VLM_BENCHMARK_RAISE_ALERTS=true to enable)")

    alert_engine.start()
    event_engine.start()
    reader.start()

    for detection in detections:
        detection.start(reader=reader, event_engine=event_engine)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        say("\nshutting down...")
    finally:
        for detection in detections:
            detection.stop()
        reader.stop()
        event_engine.stop()
        alert_engine.stop()
        print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
