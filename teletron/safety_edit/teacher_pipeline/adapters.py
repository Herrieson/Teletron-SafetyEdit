from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

from .schemas import EditorResult, TeacherPlan, VerifierResult

try:
    import torch
except ModuleNotFoundError:
    torch = None


class StaticVLMTeacher:
    """Deterministic VLM stand-in for smoke tests and wiring checks.

    Real local VLM adapters should expose the same ``plan(image_path, image)``
    method and return ``TeacherPlan``.
    """

    def __init__(
        self,
        teacher_prompt: str = "no edit needed",
        safe_flag: bool = True,
        risk_type: str = "none",
        risk_description: str = "",
        hidden_shape: tuple[int, int] = (16, 64),
        hidden_value: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.teacher_prompt = teacher_prompt
        self.safe_flag = safe_flag
        self.risk_type = risk_type
        self.risk_description = risk_description
        self.hidden_shape = tuple(hidden_shape)
        self.hidden_value = hidden_value
        self.metadata = metadata or {"adapter": self.__class__.__name__}

    def plan(self, image_path: Path, image: Image.Image) -> TeacherPlan:
        if torch is not None:
            hidden = torch.full(self.hidden_shape, self.hidden_value, dtype=torch.float32)
        else:
            hidden = nested_constant(self.hidden_shape, self.hidden_value)
        return TeacherPlan(
            teacher_prompt=self.teacher_prompt,
            safe_flag=self.safe_flag,
            risk_type=self.risk_type,
            risk_description=self.risk_description,
            no_edit_reason="static safe sample" if self.safe_flag else None,
            vlm_hidden=hidden,
            raw_response={
                "image_path": str(image_path),
                "teacher_prompt": self.teacher_prompt,
                "safe_flag": self.safe_flag,
            },
            metadata=self.metadata.copy(),
        )


class JsonPlanVLMTeacher:
    """Read precomputed VLM plans from a JSON/JSONL file keyed by image path or stem."""

    def __init__(self, plan_path: str, key_field: str = "image_path", default_safe: bool = True) -> None:
        self.plan_path = Path(plan_path)
        self.key_field = key_field
        self.default_safe = default_safe
        self.plans = self._load_plans(self.plan_path)

    def _load_plans(self, plan_path: Path) -> dict[str, dict[str, Any]]:
        if plan_path.suffix == ".jsonl":
            rows = [json.loads(line) for line in plan_path.read_text().splitlines() if line.strip()]
        else:
            data = json.loads(plan_path.read_text())
            rows = data if isinstance(data, list) else data.get("samples", [])

        plans = {}
        for row in rows:
            key = str(row.get(self.key_field) or row.get("id") or Path(row["image_path"]).stem)
            plans[key] = row
            if "image_path" in row:
                plans[str(Path(row["image_path"]))] = row
                plans[Path(row["image_path"]).stem] = row
        return plans

    def plan(self, image_path: Path, image: Image.Image) -> TeacherPlan:
        row = self.plans.get(str(image_path)) or self.plans.get(image_path.stem)
        if row is None:
            return TeacherPlan(
                teacher_prompt="no edit needed",
                safe_flag=self.default_safe,
                risk_type="none",
                no_edit_reason="missing precomputed plan",
                metadata={"adapter": self.__class__.__name__, "missing_plan": True},
            )
        return TeacherPlan(
            teacher_prompt=row.get("teacher_prompt") or row.get("edit_instruction") or "no edit needed",
            safe_flag=bool(row.get("safe_flag", False)),
            risk_type=row.get("risk_type", "unknown"),
            risk_description=row.get("risk_description", ""),
            edit_region=row.get("edit_region"),
            no_edit_reason=row.get("no_edit_reason"),
            raw_response=row,
            metadata={"adapter": self.__class__.__name__},
        )


class CopyEditorTeacher:
    """Editor stand-in that copies the input image and emits a stable condition tensor."""

    def __init__(
        self,
        condition_shape: tuple[int, int] = (16, 64),
        condition_value: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.condition_shape = tuple(condition_shape)
        self.condition_value = condition_value
        self.metadata = metadata or {"adapter": self.__class__.__name__}

    def edit(self, image_path: Path, image: Image.Image, plan: TeacherPlan) -> EditorResult:
        if torch is not None:
            condition = torch.full(self.condition_shape, self.condition_value, dtype=torch.float32)
        else:
            condition = nested_constant(self.condition_shape, self.condition_value)
        return EditorResult(
            teacher_condition=condition,
            teacher_output=image.copy(),
            teacher_mask=None,
            metadata=self.metadata.copy(),
        )


class AcceptAllVerifier:
    """Verifier stand-in that accepts every sample."""

    def __init__(self, score: float = 1.0) -> None:
        self.score = score

    def verify(
        self,
        image_path: Path,
        image: Image.Image,
        plan: TeacherPlan,
        editor_result: EditorResult,
    ) -> VerifierResult:
        return VerifierResult(
            accepted=True,
            verifier_score=self.score,
            reject_reasons=[],
            metadata={"adapter": self.__class__.__name__},
        )


class PixelDiffVerifier:
    """Simple no-op verifier based on pixel difference for local smoke tests.

    If a sample is marked safe, the edited image must remain close to the input.
    For unsafe samples this verifier only checks that an output image exists.
    """

    def __init__(self, safe_mean_abs_diff_max: float = 2.0, unsafe_requires_output: bool = True) -> None:
        self.safe_mean_abs_diff_max = safe_mean_abs_diff_max
        self.unsafe_requires_output = unsafe_requires_output

    def verify(
        self,
        image_path: Path,
        image: Image.Image,
        plan: TeacherPlan,
        editor_result: EditorResult,
    ) -> VerifierResult:
        if editor_result.teacher_output is None:
            accepted = not self.unsafe_requires_output
            reasons = [] if accepted else ["missing_teacher_output"]
            return VerifierResult(accepted=accepted, verifier_score=0.0, reject_reasons=reasons)

        output = editor_result.teacher_output.convert("RGB")
        original = image.convert("RGB")
        if output.size != original.size:
            output = output.resize(original.size)

        diff = ImageChops.difference(original, output)
        mean_abs_diff = sum(ImageStat.Stat(diff).mean) / 3.0
        if plan.safe_flag and mean_abs_diff > self.safe_mean_abs_diff_max:
            return VerifierResult(
                accepted=False,
                verifier_score=max(0.0, 1.0 - mean_abs_diff / 255.0),
                reject_reasons=["safe_image_changed_too_much"],
                metadata={"mean_abs_diff": mean_abs_diff, "adapter": self.__class__.__name__},
            )
        return VerifierResult(
            accepted=True,
            verifier_score=max(0.0, 1.0 - mean_abs_diff / 255.0),
            reject_reasons=[],
            metadata={"mean_abs_diff": mean_abs_diff, "adapter": self.__class__.__name__},
        )


def nested_constant(shape: tuple[int, ...], value: float) -> Any:
    if len(shape) == 0:
        return value
    return [nested_constant(shape[1:], value) for _ in range(shape[0])]
