"""Smoke/fire detection using Moondream2 ONNX (int4 .mf.gz bundle)."""

import os

import cv2
from PIL import Image

import config
from detectors.vlm_smoke_fire_base import VlmSmokeFireDetectorBase


def resolve_model_path():
    if config.MOONDREAM_MODEL_PATH and os.path.isfile(config.MOONDREAM_MODEL_PATH):
        return config.MOONDREAM_MODEL_PATH

    folder = os.path.dirname(config.MOONDREAM_MODEL_PATH) or "models/moondream"
    os.makedirs(folder, exist_ok=True)

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=config.MOONDREAM_REPO_ID,
        filename=config.MOONDREAM_MODEL_FILE,
        revision=config.MOONDREAM_REPO_REVISION,
        local_dir=folder,
    )
    return downloaded


class MoondreamOnnxSmokeFireDetector(VlmSmokeFireDetectorBase):
    """Runs the quantized Moondream2 ONNX bundle via moondream 0.0.5."""

    def __init__(self, model_path=None, max_new_tokens=32, **kwargs):
        super().__init__(
            model_id=config.MOONDREAM_MODEL_FILE,
            label="moondream2-0.5b-int4",
            **kwargs,
        )
        self.model_path = model_path or config.MOONDREAM_MODEL_PATH
        self.max_new_tokens = max_new_tokens
        self._model = None

    def on_start(self):
        import moondream as md

        path = self.model_path
        if not os.path.isfile(path):
            self.log(f"downloading Moondream ONNX from {config.MOONDREAM_REPO_ID}...")
            path = resolve_model_path()

        self.log(f"loading Moondream ONNX from {path}...")
        self._model = md.vl(model=path)
        self.log("Moondream ONNX model loaded")

    def _ask(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Moondream query re-encodes the image each call; keep API simple for now.
        result = self._model.query(
            pil_image,
            self.prompt,
            settings={"max_tokens": self.max_new_tokens},
        )
        return (result.get("answer") or "").strip()
