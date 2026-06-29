from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logger = logging.getLogger(__name__)


@dataclass
class SourceRow:
    id: str
    image_path: str
    source_dataset: str
    source_label: str | None = None
    risk_type: str | None = None
    safe_flag: bool | None = None
    source_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "image_path": self.image_path,
            "source_dataset": self.source_dataset,
            "source_label": self.source_label,
            "risk_type": self.risk_type,
            "safe_flag": self.safe_flag,
            "source_metadata": self.source_metadata or {},
        }
        return {key: value for key, value in data.items() if value is not None}


PRESETS = {
    "unsafe_bench": {
        "dataset": "yiting/UnsafeBench",
        "image_field": "image",
        "label_candidates": ["label", "safety_label", "safe_label", "unsafe_label", "category"],
        "risk_candidates": ["category", "risk_type", "unsafe_category", "class"],
    },
    "t2i_safety": {
        "dataset": "OpenSafetyLab/t2i_safety_dataset",
        "image_field": "image",
        "label_candidates": ["label", "safety_label", "safe_label", "unsafe", "is_safe"],
        "risk_candidates": ["category", "risk_type", "task", "class"],
    },
    "coco_caption2017": {
        "dataset": "lmms-lab/COCO-Caption2017",
        "image_field": "image",
        "label_value": "safe",
        "risk_value": "none",
        "safe_flag": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare safety-edit source_manifest.jsonl.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("local-dir", help="Create a source manifest from a local image directory.")
    local.add_argument("--input-dir", required=True)
    local.add_argument("--output-dir", required=True)
    local.add_argument("--source-dataset", default="local")
    local.add_argument("--source-label", default=None)
    local.add_argument("--risk-type", default=None)
    local.add_argument("--safe-flag", choices=["true", "false", "auto"], default="auto")
    local.add_argument("--copy-images", action="store_true")
    local.add_argument("--limit", type=int, default=None)

    hf = subparsers.add_parser("hf", help="Create a source manifest from a Hugging Face dataset.")
    hf.add_argument("--dataset", default=None, help="HF dataset name. Optional when --preset is set.")
    hf.add_argument("--preset", choices=sorted(PRESETS), default=None)
    hf.add_argument("--split", default="train")
    hf.add_argument("--output-dir", required=True)
    hf.add_argument("--image-field", default=None)
    hf.add_argument("--id-field", default=None)
    hf.add_argument("--label-field", default=None)
    hf.add_argument("--risk-field", default=None)
    hf.add_argument("--safe-field", default=None)
    hf.add_argument("--streaming", action="store_true")
    hf.add_argument("--limit", type=int, default=1000)
    hf.add_argument("--trust-remote-code", action="store_true")

    inspect = subparsers.add_parser("inspect-hf", help="Print first rows/fields from a Hugging Face dataset.")
    inspect.add_argument("--dataset", required=True)
    inspect.add_argument("--split", default="train")
    inspect.add_argument("--streaming", action="store_true")
    inspect.add_argument("--limit", type=int, default=3)
    inspect.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    if args.command == "local-dir":
        rows = prepare_local_dir(args)
        logger.info("Wrote %d local source rows.", rows)
    elif args.command == "hf":
        rows = prepare_hf(args)
        logger.info("Wrote %d HF source rows.", rows)
    elif args.command == "inspect-hf":
        inspect_hf(args)


def prepare_local_dir(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, image_path in enumerate(discover_local_images(input_dir)):
        if args.limit is not None and idx >= args.limit:
            break
        sample_id = f"{args.source_dataset}_{idx:08d}"
        if args.copy_images:
            dst = images_dir / f"{sample_id}{image_path.suffix.lower()}"
            shutil.copy2(image_path, dst)
            manifest_image_path = str(dst.relative_to(output_dir))
        else:
            manifest_image_path = str(image_path)

        row = SourceRow(
            id=sample_id,
            image_path=manifest_image_path,
            source_dataset=args.source_dataset,
            source_label=args.source_label,
            risk_type=args.risk_type,
            safe_flag=parse_optional_bool(args.safe_flag),
            source_metadata={"original_path": str(image_path)},
        )
        rows.append(row.to_dict())

    write_manifest(output_dir / "source_manifest.jsonl", rows)
    return len(rows)


def prepare_hf(args: argparse.Namespace) -> int:
    preset = PRESETS.get(args.preset or "", {})
    dataset_name = args.dataset or preset.get("dataset")
    if dataset_name is None:
        raise ValueError("--dataset is required when --preset is not set.")

    datasets = require_datasets()
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    ds = datasets.load_dataset(dataset_name, **load_kwargs)

    image_field = args.image_field or preset.get("image_field") or "image"
    rows = []
    for idx, sample in enumerate(ds):
        if args.limit is not None and idx >= args.limit:
            break
        sample_id = str(sample.get(args.id_field)) if args.id_field and args.id_field in sample else f"{slug(dataset_name)}_{idx:08d}"
        image = extract_image(sample, image_field)
        suffix = ".jpg"
        image_rel_path = Path("images") / f"{sample_id}{suffix}"
        image_save_path = output_dir / image_rel_path
        image_save_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(image_save_path)

        source_label = (
            sample.get(args.label_field)
            if args.label_field
            else preset.get("label_value", first_existing(sample, preset.get("label_candidates", [])))
        )
        risk_type = (
            sample.get(args.risk_field)
            if args.risk_field
            else preset.get("risk_value", first_existing(sample, preset.get("risk_candidates", [])))
        )
        safe_flag = (
            bool(sample.get(args.safe_field))
            if args.safe_field and args.safe_field in sample
            else preset.get("safe_flag", infer_safe_flag(source_label, risk_type))
        )

        row = SourceRow(
            id=sample_id,
            image_path=str(image_rel_path),
            source_dataset=dataset_name,
            source_label=str(source_label) if source_label is not None else None,
            risk_type=str(risk_type) if risk_type is not None else None,
            safe_flag=safe_flag,
            source_metadata=summarize_sample(sample, exclude={image_field}),
        )
        rows.append(row.to_dict())

    write_manifest(output_dir / "source_manifest.jsonl", rows)
    return len(rows)


def inspect_hf(args: argparse.Namespace) -> None:
    datasets = require_datasets()
    load_kwargs = {
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    ds = datasets.load_dataset(args.dataset, **load_kwargs)
    for idx, sample in enumerate(ds):
        if idx >= args.limit:
            break
        print(f"\n## sample {idx}")
        for key, value in sample.items():
            print(f"{key}: {type(value).__name__}: {str(value)[:240]}")


def discover_local_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_image(sample: dict[str, Any], image_field: str) -> Image.Image:
    value = sample[image_field]
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    if isinstance(value, dict):
        for key in ("path", "bytes"):
            if key not in value:
                continue
            if key == "path" and value[key]:
                return Image.open(value[key]).convert("RGB")
            if key == "bytes" and value[key]:
                import io

                return Image.open(io.BytesIO(value[key])).convert("RGB")
    raise TypeError(f"Unsupported image field type for {image_field}: {type(value)}")


def first_existing(sample: dict[str, Any], candidates: Iterable[str]) -> Any | None:
    for key in candidates:
        if key in sample:
            return sample[key]
    return None


def infer_safe_flag(source_label: Any | None, risk_type: Any | None) -> bool | None:
    text = f"{source_label} {risk_type}".lower()
    if any(token in text for token in ["unsafe", "harm", "risk", "violent", "weapon", "sexual", "hate"]):
        return False
    if any(token in text for token in ["safe", "benign", "normal", "none"]):
        return True
    return None


def summarize_sample(sample: dict[str, Any], exclude: set[str]) -> dict[str, Any]:
    metadata = {}
    for key, value in sample.items():
        if key in exclude:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        elif isinstance(value, (list, tuple)):
            metadata[key] = value[:20]
        elif isinstance(value, dict):
            metadata[key] = {
                sub_key: sub_value
                for sub_key, sub_value in value.items()
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None
            }
        else:
            metadata[key] = str(type(value).__name__)
    return metadata


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_optional_bool(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "true"


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def require_datasets() -> Any:
    try:
        import datasets
    except ModuleNotFoundError as exc:
        raise ImportError("HF source preparation requires the datasets package.") from exc
    return datasets


if __name__ == "__main__":
    main()

