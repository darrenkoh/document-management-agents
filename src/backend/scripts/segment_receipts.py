#!/usr/bin/env python3
"""Standalone CLI for receipt segmentation (SAM3).

Examples:
  python src/backend/scripts/segment_receipts.py --input /path/to/image.jpg
  python src/backend/scripts/segment_receipts.py --config src/backend/config/config.yaml --input ./test.jpg

This script only performs segmentation + writes PNG crops; it does not ingest into the DB.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path when run as a script
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.backend.services.receipt_segmentation import SegmentationConfig, Sam3ReceiptSegmenter
from src.backend.utils.config import Config


def _iter_images(path: Path):
    if path.is_file():
        yield path
        return

    exts = {'.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp'}
    for p in path.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(description='Segment receipt images into individual PNGs using SAM3')
    parser.add_argument('--config', type=str, default='src/backend/config/config.yaml', help='Path to config.yaml')
    parser.add_argument('--input', required=True, type=str, help='Path to input image file or directory')

    # Optional overrides
    parser.add_argument('--output-dir', type=str, default=None, help='Override segmentation.output_dir')
    parser.add_argument('--device', type=str, default=None, choices=['auto', 'mps', 'cpu'], help='Override segmentation.device')
    parser.add_argument('--checkpoint', type=str, default=None, help='Override segmentation.checkpoint_path')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    cfg = Config(args.config)

    seg_cfg = SegmentationConfig(
        enable=True,
        output_dir=Path(args.output_dir or cfg.segmentation_output_dir),
        device=args.device or cfg.segmentation_device,
        checkpoint_path=Path(args.checkpoint or cfg.segmentation_checkpoint_path) if (args.checkpoint or cfg.segmentation_checkpoint_path) else None,
        text_prompt=cfg.segmentation_text_prompt,
        confidence_threshold=cfg.segmentation_confidence_threshold,
        max_masks=cfg.segmentation_max_masks,
        max_segments=cfg.segmentation_max_segments,
        min_area_ratio=cfg.segmentation_min_area_ratio,
        min_width_px=cfg.segmentation_min_width_px,
        min_height_px=cfg.segmentation_min_height_px,
        min_fill_ratio=cfg.segmentation_min_fill_ratio,
        iou_dedup_threshold=cfg.segmentation_iou_dedup_threshold,
        bbox_padding_px=cfg.segmentation_bbox_padding_px,
    )

    if not seg_cfg.checkpoint_path:
        print(
            "Error: SAM3 checkpoint path is not set.\n\n"
            "Provide it via either:\n"
            "  1) config.yaml: segmentation.checkpoint_path: \"/path/to/checkpoint\"\n"
            "  2) CLI flag:   --checkpoint /path/to/checkpoint\n",
            file=sys.stderr,
        )
        return 2

    segmenter = Sam3ReceiptSegmenter(seg_cfg)

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    total_created = 0
    for img_path in _iter_images(inp):
        created = segmenter.segment_receipts(img_path)
        total_created += len(created)
        print(f"{img_path} -> {len(created)} segments")
        for p in created:
            print(f"  - {p}")

    print(f"\nDone. Wrote {total_created} PNG(s) to {seg_cfg.output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
