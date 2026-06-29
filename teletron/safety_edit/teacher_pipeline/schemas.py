from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TeacherPlan:
    """Structured output from the local VLM safety/edit planner."""

    teacher_prompt: str
    safe_flag: bool
    risk_type: str = "unknown"
    risk_description: str = ""
    edit_region: dict[str, Any] | None = None
    no_edit_reason: str | None = None
    vlm_hidden: Any | None = None
    raw_response: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditorResult:
    """Outputs from the local prompt-based editor teacher."""

    teacher_condition: Any | None = None
    teacher_output: Any | None = None
    teacher_mask: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierResult:
    """Verifier decision for a generated teacher sample."""

    accepted: bool = True
    verifier_score: float | None = None
    reject_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherSample:
    """One fully processed teacher sample before manifest serialization."""

    sample_id: str
    image_path: Path
    plan: TeacherPlan
    editor_result: EditorResult
    verifier_result: VerifierResult
    metadata: dict[str, Any] = field(default_factory=dict)

