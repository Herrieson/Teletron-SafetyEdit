from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset


class SafetyEditDataset(Dataset):
    """Dataset for safety-edit teacher distillation manifests.

    This dataset intentionally avoids Teletron generic video-data transform
    imports so bridge-distillation jobs can read saved tensors independently.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        load_tensors: bool = True,
        load_images: bool = False,
        load_masks: bool = False,
        map_location: str = "cpu",
        filter_cfg: dict | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else self.manifest_path.parent
        self.transform = transform
        self.load_tensors = load_tensors
        self.load_images = load_images
        self.load_masks = load_masks
        self.map_location = map_location
        self.filter_cfg = filter_cfg or {}
        self.rows = self.filter_rows(self.load_rows())

    def load_rows(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        rows: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row = resolve_paths(row, self.data_root)
                row["edit_needed"] = None if row.get("safe_flag") is None else not bool(row["safe_flag"])
                rows.append(row)
        return rows

    def filter_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cfg = self.filter_cfg
        if cfg.get("accepted_only", False):
            rows = [row for row in rows if row.get("accepted") is True]
        if "safe_flags" in cfg:
            allowed = set(cfg["safe_flags"])
            rows = [row for row in rows if row.get("safe_flag") in allowed]
        if "risk_types" in cfg:
            allowed = set(cfg["risk_types"])
            rows = [row for row in rows if row.get("risk_type") in allowed]
        if cfg.get("require_condition", False):
            rows = [row for row in rows if row.get("teacher_condition_path") and Path(row["teacher_condition_path"]).exists()]
        if cfg.get("require_vlm_hidden", False):
            rows = [row for row in rows if row.get("vlm_hidden_path") and Path(row["vlm_hidden_path"]).exists()]
        if cfg.get("require_output", False):
            rows = [row for row in rows if row.get("teacher_output_path") and Path(row["teacher_output_path"]).exists()]
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data = dict(self.rows[idx])
        if self.load_tensors:
            if data.get("vlm_hidden_path"):
                data["vlm_hidden"] = load_tensor(data["vlm_hidden_path"], self.map_location)
            if data.get("teacher_condition_path"):
                data["teacher_condition"] = load_tensor(data["teacher_condition_path"], self.map_location)
        if self.load_images:
            if data.get("image_path"):
                data["image"] = Image.open(data["image_path"]).convert("RGB")
            if data.get("teacher_output_path"):
                data["teacher_output"] = Image.open(data["teacher_output_path"]).convert("RGB")
        if self.load_masks and data.get("teacher_mask_path"):
            data["teacher_mask"] = Image.open(data["teacher_mask_path"])
        if self.transform is not None:
            data = self.transform(data)
        return data


def resolve_paths(row: dict[str, Any], manifest_root: Path) -> dict[str, Any]:
    resolved = dict(row)
    for key in (
        "image_path",
        "teacher_condition_path",
        "teacher_output_path",
        "teacher_mask_path",
        "vlm_hidden_path",
    ):
        value = resolved.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = manifest_root / path
        resolved[key] = str(path)
    return resolved


def load_tensor(path: str, map_location: str) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError("SafetyEditDataset tensor loading requires torch.") from exc
    return torch.load(path, map_location=map_location)
