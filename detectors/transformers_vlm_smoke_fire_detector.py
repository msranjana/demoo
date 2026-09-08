"""Smoke/fire detection for HuggingFace transformers VLMs (SmolVLM2, InternVL3, etc.)."""

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from detectors.vlm_smoke_fire_base import VlmSmokeFireDetectorBase


class TransformersVlmSmokeFireDetector(VlmSmokeFireDetectorBase):
    """Runs any AutoModelForImageTextToText VLM with the shared smoke/fire prompt."""

    def __init__(self, model_id, label=None, max_new_tokens=16, torch_dtype=None, **kwargs):
        super().__init__(model_id=model_id, label=label, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype or torch.float32
        self._processor = None
        self._model = None

    def on_start(self):
        self.log(f"loading {self.label} from {self.model_id} (CPU)...")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            _attn_implementation="eager",
        )
        self._model.eval()
        self.log(f"model loaded: {self.label}")

    def _ask(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )

        input_ids = inputs["input_ids"]
        new_tokens = generated_ids[:, input_ids.shape[1] :]
        decoded = self._processor.batch_decode(new_tokens, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""
