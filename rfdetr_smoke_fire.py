"""RF-DETR Nano smoke/fire detection CLI.

Requires a fine-tuned checkpoint with smoke and fire classes (COCO weights alone
will not detect smoke/fire). Set RFDETR_MODEL_PATH in .env after training.

Install: pip install rfdetr supervision

Usage:
  python rfdetr_smoke_fire.py --source traffic.mp4
  python rfdetr_smoke_fire.py --source rtsp://...
  python rfdetr_smoke_fire.py --source video.mp4 --checkpoint models/rfdetr/checkpoint_best_total.pth
  python rfdetr_smoke_fire.py --source video.mp4 --out outputs/rfdetr_annotated.mp4
"""

import argparse
import sys

from services.rfdetr_smoke_fire import run_video


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RF-DETR Nano smoke/fire detection")
    parser.add_argument("--source", required=True, help="video path / webcam index / rtsp url")
    parser.add_argument("--checkpoint", default=None, help="fine-tuned RF-DETR checkpoint path")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--out", default=None, help="output annotated video path")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--every", type=float, default=None, help="sample one frame every N seconds")
    parser.add_argument("--results", default=None, help="save summary JSON")
    parser.add_argument("--show", action="store_true", help="open preview window (omit on SSH)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_video(
        source=args.source,
        threshold=args.threshold,
        device=args.device,
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        max_frames=args.max_frames,
        every=args.every,
        show=args.show,
        results_path=args.results,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
