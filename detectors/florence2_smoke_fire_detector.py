"""Smoke and fire detection using Florence-2 open-vocabulary detection."""

import cv2
import torch
from PIL import Image

import config
from detectors.florence2_onnx_runner import Florence2OnnxRunner
from detectors.vlm_smoke_fire_base import VlmSmokeFireDetectorBase

SMOKE_FIRE_TASK = "<OPEN_VOCABULARY_DETECTION>"

DEFAULT_FLORENCE2_PROMPT = (
    "Detect all smoke and fire in the image and return their locations in the form of "
    'coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], '
    '"label": "smoke" # or "fire"}.'
)


def _load_florence(model_id):
    """Load ONNX weights for onnx-community/*, native HF otherwise."""
    if model_id.startswith("onnx-community/"):
        runner = Florence2OnnxRunner(
            model_id,
            variant=config.FLORENCE2_ONNX_VARIANT,
        )
        return runner.processor, runner

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
    """Detects smoke and fire via Florence-2 open-vocabulary detection."""

    def __init__(
        self,
        model_id=None,
        prompt=None,
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
        self.prompt_text = prompt or config.FLORENCE2_PROMPT
        self.max_new_tokens = max_new_tokens or config.FLORENCE2_MAX_NEW_TOKENS
        self.num_beams = num_beams
        self._processor = None
        self._model = None
        self._use_onnx = False

    def on_start(self):
        self.log(
            f"loading {self.label} from {self.model_id} "
            f"(prompt={self.prompt_text[:80]}...)..."
        )
        self._processor, self._model = _load_florence(self.model_id)
        self._use_onnx = isinstance(self._model, Florence2OnnxRunner)
        if not self._use_onnx:
            self._model.eval()
        self.log(f"model loaded: {self.label} ({'onnx' if self._use_onnx else 'pytorch'})")

    def _build_task_prompt(self):
        return self.task + self.prompt_text

    def _ask(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        prompt = self._build_task_prompt()

        if self._use_onnx:
            generated_text = self._model.generate(
                pil_image,
                prompt,
                max_new_tokens=self.max_new_tokens,
            )
        else:
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

        labels = task_data.get("bboxes_labels") or task_data.get("labels") or []
        normalized = []
        for label in labels:
            text = str(label).strip().lower()
            if "fire" in text:
                normalized.append("fire")
            elif "smoke" in text:
                normalized.append("smoke")
        return normalized

    def _format_response(self, parsed, generated_text):
        labels = self._extract_labels(parsed)
        fire = "yes" if "fire" in labels else "no"
        smoke = "yes" if "smoke" in labels else "no"
        return f"fire:{fire}, smoke:{smoke}"
