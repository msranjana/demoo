# CCTV Detection Pipeline

One RTSP camera is read once, and every detection you attach receives those frames in
parallel at its own frames-per-second. Detections send their findings to the event
engine, which writes the log file and forwards alerts to the alert engine.

```
RTSPService  ->  DetectionWrapper(name, fps, detector)  ->  EventEngine  ->  logs/events.log
   (frames)          (one thread per detection)                |
                                                               +-------->  AlertEngine (queue)
```

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` uses minimum versions rather than exact pins on purpose. An exact
pin like `numpy==1.26.4` has no prebuilt wheel for newer Python versions, so pip tries
to compile it from source and fails unless a C compiler is installed.

Then fill in `.env`:

| Variable | Meaning |
| --- | --- |
| `RTSP_URL` | camera stream url |
| `LOG_FILE_PATH` | where the event engine writes log lines |
| `RECONNECT_DELAY` | seconds to wait before reconnecting to the camera |
| `SMOKE_FIRE_MODEL_PATH` | path to the smoke and fire model file |
| `SMOLVLM2_ENABLED` | set to `false` to disable the VLM detector |
| `SMOLVLM2_MODEL_PATH` | HuggingFace model id or local path |
| `SMOLVLM2_FPS` | frames per second for the VLM detector (default `0.2`) |
| `SMOLVLM2_ALERT_COOLDOWN` | seconds between repeated VLM alerts |
| `SMTP_*`, `ALERT_*` | alert engine email settings, can stay empty for now |

Run it with `python main.py`.

## Smoke and fire detection

`detectors/smoke_and_fire_detector.py` uses `models/fire_smoke_yolov8n.pt`, a YOLOv8n
model fine tuned on the D-Fire dataset with two classes, `smoke` and `fire`.

- fire in the frame raises a `CRITICAL` alert, smoke on its own raises `WARNING`
- the alert image is the frame with the detection boxes drawn on it
- after an alert the same severity stays quiet for `alert_cooldown` seconds (30 by
  default), so a burning frame does not send one alert per frame

Tuning options: `SmokeAndFireDetector(confidence=0.4, alert_cooldown=30)`. Raise
`confidence` for fewer false alarms, lower it to catch more.

If the model file is ever missing, download it again with:

```bash
curl -L -o models/fire_smoke_yolov8n.pt \
  https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt
```

## SmolVLM2 smoke and fire detection

`detectors/smolvlm2_smoke_fire_detector.py` runs a second smoke/fire check in parallel
using [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).
The model is loaded locally via HuggingFace `transformers` and runs on CPU.

- asks the model: "Is there smoke or fire visible in this image? Answer only yes or no."
- raises a `WARNING` alert when the model answers yes
- runs at `SMOLVLM2_FPS` (default `0.2`, one frame every 5 seconds) because CPU inference is slow
- after an alert the detector stays quiet for `SMOLVLM2_ALERT_COOLDOWN` seconds (30 by default)

On first run, HuggingFace downloads the model weights (~500 MB). Set `SMOLVLM2_ENABLED=false`
in `.env` to disable without removing the code.

CPU inference typically takes several seconds per frame. Lower `SMOLVLM2_FPS` further if
the detector cannot keep up.

## Adding a detection

Create a class in `detectors/` that extends `BaseDetector` and implements `process`:

```python
from detectors.base_detector import BaseDetector
from engines.alert_engine import AlertType


class SmokeDetector(BaseDetector):
    def on_start(self):
        # optional: load a model here
        pass

    def process(self, frame):
        if not self._looks_like_smoke(frame):
            self.log("no smoke")
            return

        self.alert(self.to_base64(frame), AlertType.CRITICAL, "smoke detected")
```

Attach it in `build_detections()` in `main.py`:

```python
return [
    DetectionWrapper("smoke_and_fire_detection", 2, SmokeAndFireDetector()),
    DetectionWrapper("person_detection", 10, PersonDetector()),
]
```

`10` is the frames per second that detection gets. Frames it cannot keep up with are
skipped, so a slow detection never blocks the camera or the other detections.

## What a detector can call

| Call | Result |
| --- | --- |
| `self.log(message, level="INFO")` | one line in the log file |
| `self.alert(image_base64, alert_type, message)` | log line plus an entry in the alert queue |
| `self.to_base64(frame)` | frame as a base64 JPEG string for the alert |

Alert types are `AlertType.NORMAL`, `AlertType.WARNING` and `AlertType.CRITICAL`.

## Alert engine

`AlertEngine` consumes the alert queue in its own thread and currently just prints each
alert. When the mail account is ready, add the credentials to `.env` and implement
`_send_email` in `engines/alert_engine.py`; nothing else has to change.
