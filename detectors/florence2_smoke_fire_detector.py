"""Smoke and fire detection using Microsoft Florence-2 open-vocabulary detection."""

import cv2
import torch
from PIL import Image

import config
from detectors.vlm_smoke_fire_base import VlmSmokeFireDetectorBase

# Only look for smoke and fire — not general <OD> object detection.
SMOKE_FIRE_TASK = "<OPEN_VOCABULARY_DETECTION>"


def _load_florence(model_id):
    """Load native HF Florence-2 for community models, legacy remote code for microsoft/*."""
    if model_id.startswith("microsoft/"):
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        return processor, model

    from transformers import Florence2ForConditionalGeneration, Florence2Processor

    processor = Florence2Processor.from_pretrained(model_id)
    model = Florence2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    return processor, model


class Florence2SmokeFireDetector(VlmSmokeFireDetectorBase):
    """Detects only smoke and fire via Florence-2 open-vocabulary detection."""

    def __init__(
        self,
        model_id=None,
        max_new_tokens=None,
        num_beams=3,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id or config.FLORENCE2_MODEL,
            label="florence-2-smoke-fire",
            **kwargs,
        )
        self.task = SMOKE_FIRE_TASK
        self.categories = config.FLORENCE2_OV_CATEGORIES
        self.max_new_tokens = max_new_tokens or config.FLORENCE2_MAX_NEW_TOKENS
        self.num_beams = num_beams
        self._processor = None
        self._model = None

    def on_start(self):
        self.log(
            f"loading {self.label} from {self.model_id} "
            f"(categories={self.categories})..."
        )
        self._processor, self._model = _load_florence(self.model_id)
        self._model.eval()
        self.log(f"model loaded: {self.label}")

    def _build_task_prompt(self):
        return self.task + self.categories

    def _ask(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        prompt = self._build_task_prompt()

        inputs = self._processor(text=prompt, images=pil_image, return_tensors="pt")

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=self.num_beams,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        try:
            parsed = self._processor.post_process_generation(
                generated_text,
                task=self.task,
                image_size=(pil_image.width, pil_image.height),
            )
            return self._format_response(parsed, generated_text)
        except Exception:
            return generated_text.strip()

    def _extract_labels(self, parsed):
        task_data = parsed.get(self.task) if isinstance(parsed, dict) else None
        if not isinstance(task_data, dict):
            return []

        labels = task_data.get("labels") or []
        normalized = []
        for label in labels:
            text = str(label).strip().lower()
            if text in ("smoke", "fire"):
                normalized.append(text)
        return normalized

    def _format_response(self, parsed, generated_text):
        labels = self._extract_labels(parsed)
        fire = "yes" if "fire" in labels else "no"
        smoke = "yes" if "smoke" in labels else "no"
        return f"fire:{fire}, smoke:{smoke}"
