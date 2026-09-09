"""ONNX Runtime inference for onnx-community/Florence-2-* checkpoints."""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import Florence2Processor

# File sets exported by onnx-community Florence-2 repos.
_ONNX_VARIANTS = {
    "": (
        "vision_encoder.onnx",
        "embed_tokens.onnx",
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_model_merged.onnx",
    ),
    "fp16": (
        "vision_encoder_fp16.onnx",
        "embed_tokens_fp16.onnx",
        "encoder_model_fp16.onnx",
        "decoder_model_fp16.onnx",
        "decoder_model_merged_fp16.onnx",
    ),
    "q4f16": (
        "vision_encoder_q4f16.onnx",
        "embed_tokens_q4f16.onnx",
        "encoder_model_q4f16.onnx",
        "decoder_model_q4f16.onnx",
        "decoder_model_merged_q4.onnx",
    ),
}


def _resolve_onnx_files(variant):
    files = _ONNX_VARIANTS.get(variant)
    if files is None:
        supported = ", ".join(sorted(_ONNX_VARIANTS))
        raise ValueError(f"Unknown FLORENCE2_ONNX_VARIANT '{variant}'. Supported: {supported}")
    return files


def _download_onnx_file(model_id, filename):
    return hf_hub_download(repo_id=model_id, filename=f"onnx/{filename}")


class Florence2OnnxRunner:
    """Runs Florence-2 from the five ONNX sessions in onnx-community repos."""

    def __init__(self, model_id, variant="", providers=None):
        vision_name, embed_name, encoder_name, decoder_name, merged_name = _resolve_onnx_files(
            variant
        )

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.model_id = model_id
        self.processor = Florence2Processor.from_pretrained(model_id)
        self.vision_encoder = ort.InferenceSession(
            _download_onnx_file(model_id, vision_name),
            providers=providers,
        )
        self.text_embed = ort.InferenceSession(
            _download_onnx_file(model_id, embed_name),
            providers=providers,
        )
        self.encoder = ort.InferenceSession(
            _download_onnx_file(model_id, encoder_name),
            providers=providers,
        )
        self.decoder_prefill = ort.InferenceSession(
            _download_onnx_file(model_id, decoder_name),
            providers=providers,
        )
        self.decoder_decode = ort.InferenceSession(
            _download_onnx_file(model_id, merged_name),
            providers=providers,
        )

    def generate(self, pil_image, prompt, max_new_tokens=256):
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="np",
            do_resize=True,
        )

        image_features = self.vision_encoder.run(
            None, {"pixel_values": inputs["pixel_values"]}
        )[0]

        inputs_embeds = self.text_embed.run(None, {"input_ids": inputs["input_ids"]})[0]

        batch_size, image_token_length = image_features.shape[:-1]
        image_attention_mask = np.ones((batch_size, image_token_length), dtype=np.int64)
        task_prefix_embeds = inputs_embeds
        task_prefix_attention_mask = np.ones(
            (batch_size, task_prefix_embeds.shape[1]), dtype=np.int64
        )

        if task_prefix_attention_mask.ndim == 3:
            task_prefix_attention_mask = task_prefix_attention_mask[:, 0]

        inputs_embeds = np.concatenate([image_features, task_prefix_embeds], axis=1)
        attention_mask = np.concatenate([image_attention_mask, task_prefix_attention_mask], axis=1)

        encoder_hidden_states = self.encoder.run(
            None,
            {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask},
        )[0]

        decoder_outs = self.decoder_prefill.run(
            None,
            {
                "inputs_embeds": inputs_embeds[:, -1:],
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": attention_mask,
            },
        )
        encoder_kv = decoder_outs[1:]

        generated_tokens = []
        eos_token_id = self.processor.tokenizer.eos_token_id or 2

        while len(generated_tokens) < max_new_tokens:
            logits = decoder_outs[0]
            decoder_kv = decoder_outs[1:]

            next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            generated_tokens.append(next_token)

            if next_token == eos_token_id:
                break

            next_input_embeds = self.text_embed.run(
                None,
                {"input_ids": np.array([[next_token]], dtype=np.int64)},
            )[0]

            decode_inputs = {
                "use_cache_branch": np.array([True], dtype=np.bool_),
                "inputs_embeds": next_input_embeds,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": attention_mask,
            }

            for layer_idx in range(len(decoder_kv) // 4):
                base = layer_idx * 4
                decode_inputs[f"past_key_values.{layer_idx}.decoder.key"] = decoder_kv[base]
                decode_inputs[f"past_key_values.{layer_idx}.decoder.value"] = decoder_kv[base + 1]
                decode_inputs[f"past_key_values.{layer_idx}.encoder.key"] = encoder_kv[base + 2]
                decode_inputs[f"past_key_values.{layer_idx}.encoder.value"] = encoder_kv[base + 3]

            decoder_outs = self.decoder_decode.run(None, decode_inputs)

        return self.processor.batch_decode([generated_tokens], skip_special_tokens=False)[0]
