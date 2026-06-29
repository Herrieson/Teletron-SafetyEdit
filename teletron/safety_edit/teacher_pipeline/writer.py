from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .schemas import TeacherSample

try:
    import torch
except ModuleNotFoundError:
    torch = None


class TeacherDataWriter:
    """Persist teacher pseudo labels as assets plus a JSONL manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        copy_images: bool = True,
        include_rejected: bool = True,
        image_format: str = "jpg",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.copy_images = copy_images
        self.include_rejected = include_rejected
        self.image_format = image_format.lstrip(".")

        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.images_dir = self.output_dir / "images"
        self.outputs_dir = self.output_dir / "outputs"
        self.conditions_dir = self.output_dir / "conditions"
        self.vlm_hidden_dir = self.output_dir / "vlm_hidden"
        self.masks_dir = self.output_dir / "masks"
        self.logs_dir = self.output_dir / "logs"

        for path in [
            self.images_dir,
            self.outputs_dir,
            self.conditions_dir,
            self.vlm_hidden_dir,
            self.masks_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def write(self, sample: TeacherSample) -> dict[str, Any] | None:
        verifier = sample.verifier_result
        if not verifier.accepted and not self.include_rejected:
            return None

        sample_id = sample.sample_id
        image_path = self._write_input_image(sample_id, sample.image_path)
        condition_path = self._write_tensor_like(
            self.conditions_dir / f"{sample_id}.pt",
            sample.editor_result.teacher_condition,
        )
        vlm_hidden_path = self._write_tensor_like(
            self.vlm_hidden_dir / f"{sample_id}.pt",
            sample.plan.vlm_hidden,
        )
        output_path = self._write_image_like(
            self.outputs_dir / f"{sample_id}.{self.image_format}",
            sample.editor_result.teacher_output,
        )
        mask_path = self._write_image_like(
            self.masks_dir / f"{sample_id}.png",
            sample.editor_result.teacher_mask,
        )

        row = {
            "id": sample_id,
            "image_path": self._rel(image_path) if image_path else str(sample.image_path),
            "teacher_prompt": sample.plan.teacher_prompt,
            "teacher_condition_path": self._rel(condition_path),
            "teacher_output_path": self._rel(output_path),
            "teacher_mask_path": self._rel(mask_path),
            "vlm_hidden_path": self._rel(vlm_hidden_path),
            "safe_flag": sample.plan.safe_flag,
            "risk_type": sample.plan.risk_type,
            "risk_description": sample.plan.risk_description,
            "edit_region": sample.plan.edit_region,
            "no_edit_reason": sample.plan.no_edit_reason,
            "verifier_score": verifier.verifier_score,
            "accepted": verifier.accepted,
            "reject_reasons": verifier.reject_reasons,
            "metadata": {
                **sample.metadata,
                "vlm": sample.plan.metadata,
                "editor": sample.editor_result.metadata,
                "verifier": verifier.metadata,
            },
        }
        self._append_manifest(row)
        return row

    def _write_input_image(self, sample_id: str, image_path: Path) -> Path | None:
        if not self.copy_images:
            return None
        suffix = image_path.suffix or f".{self.image_format}"
        dst = self.images_dir / f"{sample_id}{suffix}"
        if image_path.resolve() != dst.resolve():
            shutil.copy2(image_path, dst)
        return dst

    def _write_tensor_like(self, path: Path, value: Any) -> Path | None:
        if value is None:
            return None
        if torch is not None:
            torch.save(value, path)
        else:
            with path.open("w", encoding="utf-8") as f:
                json.dump(value, f)
        return path

    def _write_image_like(self, path: Path, value: Any) -> Path | None:
        if value is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, Image.Image):
            image = value
        elif isinstance(value, np.ndarray):
            if value.ndim == 2:
                image = Image.fromarray(value)
            else:
                image = Image.fromarray(value.astype(np.uint8)).convert("RGB")
        elif torch is not None and isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
            if array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
            image = Image.fromarray(array.squeeze())
        elif isinstance(value, (str, Path)):
            shutil.copy2(value, path)
            return path
        else:
            raise TypeError(f"Unsupported image-like value: {type(value)}")

        if path.suffix.lower() in {".jpg", ".jpeg"}:
            image = image.convert("RGB")
        image.save(path)
        return path

    def _append_manifest(self, row: dict[str, Any]) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _rel(self, path: Path | None) -> str | None:
        if path is None:
            return None
        return str(path.relative_to(self.output_dir))

    def existing_ids(self) -> set[str]:
        if not self.manifest_path.exists():
            return set()
        ids = set()
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(json.loads(line)["id"])
        return ids
