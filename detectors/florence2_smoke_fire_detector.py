"""Smoke and fire detection using Microsoft Florence-2."""

import json

import cv2
import torch
from PIL import Image

import config
from detectors.vlm_smoke_fire_base import VlmSmokeFireDetectorBase


def _load_florence(model_id):
    try:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Florence2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        return processor, model
    except (ImportError, OSError, ValueError):
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        return processor, model


class Florence2SmokeFireDetector(VlmSmokeFireDetectorBase):
    """Uses Florence-2 task prompts to detect or describe smoke/fire in a frame."""

    def __init__(
        self,
        model_id=None,
        task=None,
        max_new_tokens=None,
        num_beams=3,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id or config.FLORENCE2_MODEL,
            label="florence-2-base",
            **kwargs,
        )
        self.task = task or config.FLORENCE2_TASK
        self.max_new_tokens = max_new_tokens or config.FLORENCE2_MAX_NEW_TOKENS
        self.num_beams = num_beams
        self._processor = None
        self._model = None

    def on_start(self):
        self.log(f"loading {self.label} from {self.model_id} (CPU)...")
        self._processor, self._model = _load_florence(self.model_id)
        self._model.eval()
        self.log(f"model loaded: {self.label} (task={self.task})")

    def _build_task_prompt(self):
        if self.task == "<OPEN_VOCABULARY_DETECTION>":
            return self.task + config.FLORENCE2_OV_CATEGORIES
        return self.task

    def _ask(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        prompt = self._build_task_prompt()

        inputs = self._processor(text=prompt, images=pil_image, return_tensors="pt")

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
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

    def _format_response(self, parsed, generated_text):
        task_data = parsed.get(self.task) if isinstance(parsed, dict) else None

        if isinstance(task_data, dict) and task_data.get("labels"):
            labels = task_data["labels"]
            if isinstance(labels, list):
                return ", ".join(str(label) for label in labels)

        if isinstance(task_data, str):
            return task_data.strip()

        if isinstance(parsed, dict):
            return json.dumps(parsed)

        return generated_text.strip()
