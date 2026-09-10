"""Smoke and fire detection using RF-DETR Small."""

import config
from detectors.rfdetr_smoke_fire_detector import RfdetrSmokeFireDetector


class RfdetrSmokeFireSmallDetector(RfdetrSmokeFireDetector):
    """Detects smoke and fire via RF-DETR Small (requires a fine-tuned checkpoint)."""

    def __init__(
        self,
        threshold=None,
        device=None,
        checkpoint_path=None,
        alert_cooldown=30,
    ):
        super().__init__(
            model_size="small",
            model_label="rfdetr-small",
            threshold=threshold,
            device=device,
            checkpoint_path=checkpoint_path or config.RFDETR_SMALL_MODEL_PATH,
            alert_cooldown=alert_cooldown,
        )
