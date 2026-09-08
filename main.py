"""Entry point: reads the camera once and feeds every attached detection in parallel.

To add a detection:
  1. create a class in detectors/ that extends BaseDetector
  2. add one DetectionWrapper line to `build_detections()` below
"""

import time

import config
from console import say
from detectors.gguf_vlm_smoke_fire_detector import GgufVlmSmokeFireDetector
from detectors.smoke_and_fire_detector import SmokeAndFireDetector
from detectors.transformers_vlm_smoke_fire_detector import TransformersVlmSmokeFireDetector
from engines.alert_engine import AlertEngine
from engines.event_engine import EventEngine
from services.DetectionWrapper import DetectionWrapper
from services.RTSPService import RTSPService


def build_detections():
    """Return the detections to run. Each VLM runs in its own thread for benchmarking."""
    detections = []
    fps = config.VLM_TEST_FPS

    if config.YOLO_ENABLED:
        detections.append(
            DetectionWrapper(
                "yolo_smoke_fire",
                config.YOLO_FPS,
                SmokeAndFireDetector(),
            )
        )

    if config.SMOLVLM2_500M_ENABLED:
        detections.append(
            DetectionWrapper(
                "smolvlm2_500m",
                fps,
                TransformersVlmSmokeFireDetector(
                    model_id=config.SMOLVLM2_500M_MODEL,
                    label="smolvlm2-500m",
                ),
            )
        )

    if config.SMOLVLM2_256M_ENABLED:
        detections.append(
            DetectionWrapper(
                "smolvlm2_256m",
                fps,
                TransformersVlmSmokeFireDetector(
                    model_id=config.SMOLVLM2_256M_MODEL,
                    label="smolvlm2-256m",
                ),
            )
        )

    if config.INTERNVL3_ENABLED:
        detections.append(
            DetectionWrapper(
                "internvl3_1b",
                fps,
                TransformersVlmSmokeFireDetector(
                    model_id=config.INTERNVL3_MODEL,
                    label="internvl3-1b",
                ),
            )
        )

    if config.GGUF_ENABLED:
        detections.append(
            DetectionWrapper(
                "gguf_smolvlm2_256m",
                fps,
                GgufVlmSmokeFireDetector(
                    model_id=config.GGUF_MODEL_ID,
                    label="gguf-smolvlm2-256m",
                ),
            )
        )

    return detections


def main():
    if not config.RTSP_URL:
        say("RTSP_URL is missing, set it in the .env file")
        return

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

    detections = build_detections()
    if not detections:
        say("No detection attached yet, enable at least one model in .env")
        return

    if config.VLM_BENCHMARK_ENABLED:
        say(f"VLM benchmark mode ON -> {config.VLM_BENCHMARK_LOG_PATH}")
        if not config.VLM_BENCHMARK_RAISE_ALERTS:
            say("Alerts disabled during benchmark (set VLM_BENCHMARK_RAISE_ALERTS=true to enable)")

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


if __name__ == "__main__":
    main()
