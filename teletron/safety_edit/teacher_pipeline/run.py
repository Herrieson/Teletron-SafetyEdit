from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import TeacherPipeline, discover_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate safety-edit teacher pseudo labels.")
    parser.add_argument("--config", required=True, help="YAML/JSON config for local teacher adapters.")
    parser.add_argument("--input", required=True, help="Image directory, image file, txt file, or JSONL manifest.")
    parser.add_argument("--output-dir", default=None, help="Override writer.output_dir in config.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of new samples to process.")
    parser.add_argument("--resume", action="store_true", help="Skip sample IDs already present in manifest.jsonl.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    image_paths = discover_images(Path(args.input))
    pipeline = TeacherPipeline.from_config(args.config, output_dir=args.output_dir)
    rows = pipeline.run(image_paths, limit=args.limit, resume=args.resume)
    logging.info("Wrote %d manifest rows.", len(rows))


if __name__ == "__main__":
    main()

