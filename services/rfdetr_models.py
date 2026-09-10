"""RF-DETR model size registry and loader."""

from pathlib import Path

from rfdetr import RFDETRNano, RFDETRSmall

import config

RFDETR_CLASSES = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
}

DEFAULT_CHECKPOINTS = {
    "nano": lambda: config.RFDETR_MODEL_PATH,
    "small": lambda: config.RFDETR_SMALL_MODEL_PATH,
}


def resolve_checkpoint_path(size, checkpoint_path=None):
    if checkpoint_path:
        return checkpoint_path
    resolver = DEFAULT_CHECKPOINTS.get(size)
    return resolver() if resolver else ""


def load_rfdetr_model(size="nano", device=None, checkpoint_path=None):
    """Load RF-DETR by size (nano, small), optionally from a fine-tuned checkpoint."""
    if size not in RFDETR_CLASSES:
        supported = ", ".join(sorted(RFDETR_CLASSES))
        raise ValueError(f"Unknown RF-DETR size '{size}'. Supported: {supported}")

    device = device or config.RFDETR_DEVICE
    checkpoint_path = resolve_checkpoint_path(size, checkpoint_path)
    model_cls = RFDETR_CLASSES[size]

    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"RF-DETR {size} checkpoint not found: {checkpoint_path}. "
                f"Fine-tune RFDETR{size.title()} on smoke/fire and set the model path in .env."
            )
        return model_cls.from_checkpoint(str(path), device=device)

    return model_cls(device=device)
