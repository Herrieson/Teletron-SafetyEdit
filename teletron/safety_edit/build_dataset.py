from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)


ASSET_KEYS = (
    "image_path",
    "teacher_condition_path",
    "teacher_output_path",
    "teacher_mask_path",
    "vlm_hidden_path",
)

STAGE_REQUIRED_KEYS = {
    "condition": ("vlm_hidden_path", "teacher_condition_path"),
    "image": ("image_path", "teacher_output_path"),
    "full": ("image_path", "teacher_condition_path", "teacher_output_path", "vlm_hidden_path"),
}


@dataclass
class BuildResult:
    rows: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    stats: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build safety-edit training manifests from teacher outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Merge, validate, filter, and split teacher manifests.")
    build.add_argument("--input", action="append", required=True, help="Teacher manifest, teacher run dir, or parent dir.")
    build.add_argument("--output-dir", required=True)
    build.add_argument("--stage", choices=sorted(STAGE_REQUIRED_KEYS), default="condition")
    build.add_argument("--include-rejected", action="store_true", help="Keep rows where accepted is false.")
    build.add_argument("--allow-missing-assets", action="store_true")
    build.add_argument("--copy-assets", action="store_true", help="Copy assets into output-dir/assets and store relative paths.")
    build.add_argument("--dedupe-key", choices=["id", "image_path", "none"], default="id")
    build.add_argument("--max-samples", type=int, default=None)
    build.add_argument("--max-safe", type=int, default=None)
    build.add_argument("--max-unsafe", type=int, default=None)
    build.add_argument("--val-ratio", type=float, default=0.05)
    build.add_argument("--test-ratio", type=float, default=0.0)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--inspect-tensors", action="store_true")
    build.add_argument("--log-rejected", action="store_true")

    stats = subparsers.add_parser("stats", help="Print stats for teacher or built manifests.")
    stats.add_argument("--input", action="append", required=True)
    stats.add_argument("--inspect-tensors", action="store_true")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "build":
        result = build_dataset(
            inputs=[Path(path) for path in args.input],
            output_dir=Path(args.output_dir),
            stage=args.stage,
            include_rejected=args.include_rejected,
            allow_missing_assets=args.allow_missing_assets,
            copy_assets=args.copy_assets,
            dedupe_key=args.dedupe_key,
            max_samples=args.max_samples,
            max_safe=args.max_safe,
            max_unsafe=args.max_unsafe,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            inspect_tensors=args.inspect_tensors,
            log_rejected=args.log_rejected,
        )
        print_summary(result.stats)
    elif args.command == "stats":
        rows, rejected = load_and_validate(
            inputs=[Path(path) for path in args.input],
            stage="condition",
            include_rejected=True,
            allow_missing_assets=True,
            dedupe_key="none",
            inspect_tensors=args.inspect_tensors,
        )
        print_summary(summarize_rows(rows, rejected))


def build_dataset(
    inputs: list[Path],
    output_dir: Path,
    stage: str = "condition",
    include_rejected: bool = False,
    allow_missing_assets: bool = False,
    copy_assets: bool = False,
    dedupe_key: str = "id",
    max_samples: int | None = None,
    max_safe: int | None = None,
    max_unsafe: int | None = None,
    val_ratio: float = 0.05,
    test_ratio: float = 0.0,
    seed: int = 0,
    inspect_tensors: bool = False,
    log_rejected: bool = False,
) -> BuildResult:
    rows, rejected = load_and_validate(
        inputs=inputs,
        stage=stage,
        include_rejected=include_rejected,
        allow_missing_assets=allow_missing_assets,
        dedupe_key=dedupe_key,
        inspect_tensors=inspect_tensors,
    )
    rows = apply_limits(rows, max_samples=max_samples, max_safe=max_safe, max_unsafe=max_unsafe)

    output_dir.mkdir(parents=True, exist_ok=True)
    if copy_assets:
        rows = [copy_row_assets(row, output_dir=output_dir) for row in rows]

    splits = split_rows(rows, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    stats = summarize_rows(rows, rejected)
    stats["inputs"] = [str(path) for path in inputs]
    stats["output_dir"] = str(output_dir)
    stats["stage"] = stage
    stats["copy_assets"] = copy_assets
    stats["splits"] = {name: len(split_rows_) for name, split_rows_ in splits.items()}

    write_jsonl(output_dir / "manifest.jsonl", rows)
    split_dir = output_dir / "splits"
    for name, split_rows_ in splits.items():
        write_jsonl(split_dir / f"{name}.jsonl", split_rows_)
    write_json(output_dir / "stats.json", stats)
    if log_rejected:
        write_jsonl(output_dir / "rejected.jsonl", rejected)

    logger.info("Wrote %d rows to %s", len(rows), output_dir / "manifest.jsonl")
    return BuildResult(rows=rows, rejected=rejected, stats=stats)


def load_and_validate(
    inputs: list[Path],
    stage: str,
    include_rejected: bool,
    allow_missing_assets: bool,
    dedupe_key: str,
    inspect_tensors: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required_keys = STAGE_REQUIRED_KEYS[stage]
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for manifest_path in discover_manifests(inputs):
        for row_idx, raw_row in enumerate(read_jsonl(manifest_path), start=1):
            row = normalize_row(raw_row, manifest_path=manifest_path, row_idx=row_idx)
            reasons = validation_reasons(row, required_keys=required_keys, allow_missing_assets=allow_missing_assets)

            if not include_rejected and row.get("accepted") is False:
                reasons.append("not_accepted")

            duplicate_key = make_dedupe_key(row, dedupe_key)
            if duplicate_key is not None and duplicate_key in seen:
                reasons.append(f"duplicate_{dedupe_key}")
            if reasons:
                rejected.append({"id": row.get("id"), "manifest_path": str(manifest_path), "reasons": reasons})
                continue

            if duplicate_key is not None:
                seen.add(duplicate_key)
            if inspect_tensors:
                row.update(inspect_row_tensors(row))
            rows.append(row)

    return rows, rejected


def discover_manifests(inputs: Iterable[Path]) -> list[Path]:
    manifests: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            manifests.append(input_path)
            continue
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        direct_manifest = input_path / "manifest.jsonl"
        if direct_manifest.exists():
            manifests.append(direct_manifest)
            continue
        manifests.extend(sorted(input_path.rglob("manifest.jsonl")))
    if not manifests:
        raise FileNotFoundError(f"No manifest.jsonl found in inputs: {[str(path) for path in inputs]}")
    return sorted(dict.fromkeys(manifests))


def normalize_row(raw_row: dict[str, Any], manifest_path: Path, row_idx: int) -> dict[str, Any]:
    manifest_root = manifest_path.parent
    row = dict(raw_row)
    row.setdefault("id", f"{manifest_root.name}_{row_idx:08d}")
    for key in ASSET_KEYS:
        value = row.get(key)
        if value:
            row[key] = str(resolve_asset_path(value, manifest_root))
    row["teacher_manifest_path"] = str(manifest_path)
    row["teacher_manifest_root"] = str(manifest_root)
    row["teacher_row_index"] = row_idx
    row["edit_needed"] = None if row.get("safe_flag") is None else not bool(row.get("safe_flag"))
    return row


def resolve_asset_path(value: str, manifest_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_root / path


def validation_reasons(row: dict[str, Any], required_keys: tuple[str, ...], allow_missing_assets: bool) -> list[str]:
    reasons: list[str] = []
    if not row.get("id"):
        reasons.append("missing_id")
    if not isinstance(row.get("safe_flag"), bool):
        reasons.append("safe_flag_not_bool")
    if not row.get("teacher_prompt"):
        reasons.append("missing_teacher_prompt")
    for key in required_keys:
        path = row.get(key)
        if not path:
            reasons.append(f"missing_{key}")
            continue
        if not allow_missing_assets and not Path(path).exists():
            reasons.append(f"missing_asset_{key}")
    if row.get("safe_flag") is True and row.get("teacher_prompt") != "no edit needed":
        reasons.append("safe_prompt_not_noop")
    return reasons


def make_dedupe_key(row: dict[str, Any], dedupe_key: str) -> str | None:
    if dedupe_key == "none":
        return None
    value = row.get(dedupe_key)
    return None if value is None else str(value)


def inspect_row_tensors(row: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for name, key in (("teacher_condition", "teacher_condition_path"), ("vlm_hidden", "vlm_hidden_path")):
        path = row.get(key)
        if not path or not Path(path).exists():
            continue
        try:
            value = torch_load(path)
            info[f"{name}_shape"] = tensor_like_shape(value)
            info[f"{name}_type"] = type(value).__name__
        except Exception as exc:
            info[f"{name}_inspect_error"] = str(exc)
    return info


def torch_load(path: str | Path) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError("Tensor inspection requires torch.") from exc
    return torch.load(path, map_location="cpu")


def tensor_like_shape(value: Any) -> Any:
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: tensor_like_shape(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_like_shape(sub_value) for sub_value in value[:8]]
    return None


def apply_limits(
    rows: list[dict[str, Any]],
    max_samples: int | None,
    max_safe: int | None,
    max_unsafe: int | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    safe_count = 0
    unsafe_count = 0
    for row in rows:
        is_safe = row.get("safe_flag") is True
        if is_safe:
            if max_safe is not None and safe_count >= max_safe:
                continue
            safe_count += 1
        else:
            if max_unsafe is not None and unsafe_count >= max_unsafe:
                continue
            unsafe_count += 1
        output.append(row)
        if max_samples is not None and len(output) >= max_samples:
            break
    return output


def copy_row_assets(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    copied = dict(row)
    sample_id = str(row["id"])
    asset_dir = output_dir / "assets" / sample_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    for key in ASSET_KEYS:
        path = row.get(key)
        if not path:
            continue
        src = Path(path)
        if not src.exists():
            continue
        dst = asset_dir / f"{key}{src.suffix}"
        shutil.copy2(src, dst)
        copied[key] = str(dst.relative_to(output_dir))
    return copied


def split_rows(rows: list[dict[str, Any]], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("--val-ratio and --test-ratio must be non-negative and sum to less than 1.")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    if total >= 10 and val_ratio > 0 and val_count == 0:
        val_count = 1
    if total >= 20 and test_ratio > 0 and test_count == 0:
        test_count = 1
    test_rows = shuffled[:test_count]
    val_rows = shuffled[test_count : test_count + val_count]
    train_rows = shuffled[test_count + val_count :]
    return {"train": train_rows, "val": val_rows, "test": test_rows}


def summarize_rows(rows: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    safe = Counter(str(row.get("safe_flag")) for row in rows)
    accepted = Counter(str(row.get("accepted")) for row in rows)
    risk = Counter(str(row.get("risk_type")) for row in rows)
    source_dataset = Counter(
        str(((row.get("metadata") or {}).get("source") or {}).get("source_dataset"))
        for row in rows
    )
    rejected_reasons: Counter[str] = Counter()
    for item in rejected:
        rejected_reasons.update(item.get("reasons", []))

    shape_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for key in ("teacher_condition_shape", "vlm_hidden_shape"):
        counts = Counter(json.dumps(row.get(key), sort_keys=True) for row in rows if key in row)
        if counts:
            shape_counts[key] = dict(counts.most_common())

    return {
        "rows": len(rows),
        "rejected_rows": len(rejected),
        "safe_flag": dict(safe),
        "accepted": dict(accepted),
        "risk_type": dict(risk.most_common()),
        "source_dataset": dict(source_dataset.most_common()),
        "rejected_reasons": dict(rejected_reasons.most_common()),
        "tensor_shapes": shape_counts,
    }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_summary(stats: dict[str, Any]) -> None:
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
