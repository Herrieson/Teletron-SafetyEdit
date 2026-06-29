from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .adapters import AcceptAllVerifier, CopyEditorTeacher, StaticVLMTeacher
from .loader import build_component, load_teacher_config
from .schemas import EditorResult, TeacherPlan, TeacherSample, VerifierResult
from .writer import TeacherDataWriter

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TeacherPipeline:
    """Generate safety-edit teacher pseudo labels with local model adapters."""

    def __init__(
        self,
        vlm: Any,
        editor: Any,
        verifier: Any,
        writer: TeacherDataWriter,
        prompt_normalizer: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.vlm = vlm
        self.editor = editor
        self.verifier = verifier
        self.writer = writer
        self.prompt_normalizer = prompt_normalizer
        self.metadata = metadata or {}

    @classmethod
    def from_config(cls, config_path: str | Path | dict[str, Any], output_dir: str | Path | None = None) -> "TeacherPipeline":
        config = load_teacher_config(config_path)
        writer_config = config.get("writer", {})
        if output_dir is not None:
            writer_config["output_dir"] = str(output_dir)
        if "output_dir" not in writer_config:
            raise ValueError("writer.output_dir or --output-dir is required.")

        vlm = build_component(
            config.get("vlm"),
            default_target="teletron.safety_edit.teacher_pipeline.adapters:StaticVLMTeacher",
        )
        editor = build_component(
            config.get("editor"),
            default_target="teletron.safety_edit.teacher_pipeline.adapters:CopyEditorTeacher",
        )
        verifier = build_component(
            config.get("verifier"),
            default_target="teletron.safety_edit.teacher_pipeline.adapters:AcceptAllVerifier",
        )
        prompt_normalizer = build_component(config.get("prompt_normalizer"), default_target=None)
        writer = TeacherDataWriter(**writer_config)
        return cls(
            vlm=vlm,
            editor=editor,
            verifier=verifier,
            writer=writer,
            prompt_normalizer=prompt_normalizer,
            metadata=config.get("metadata", {}),
        )

    def run(
        self,
        image_paths: Iterable[str | Path | dict[str, Any]],
        limit: int | None = None,
        resume: bool = False,
    ) -> list[dict[str, Any]]:
        existing_ids = self.writer.existing_ids() if resume else set()
        rows = []
        processed = 0
        for item in image_paths:
            image_path, source_metadata = normalize_input_item(item)
            sample_id = source_metadata.get("id") or make_sample_id(image_path)
            if sample_id in existing_ids:
                logger.info("Skip existing sample %s (%s)", sample_id, image_path)
                continue

            try:
                row = self.process_one(image_path, sample_id=sample_id, source_metadata=source_metadata)
            except Exception:
                logger.exception("Failed to process image: %s", image_path)
                raise

            if row is not None:
                rows.append(row)
            processed += 1
            if limit is not None and processed >= limit:
                break
        return rows

    def process_one(
        self,
        image_path: str | Path,
        sample_id: str | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        image_path = Path(image_path)
        sample_id = sample_id or make_sample_id(image_path)
        source_metadata = source_metadata or {}
        image = Image.open(image_path).convert("RGB")

        plan = self._run_vlm(image_path, image)
        if self.prompt_normalizer is not None:
            plan.teacher_prompt = self.prompt_normalizer.normalize(plan.teacher_prompt, plan=plan, image_path=image_path)

        editor_result = self._run_editor(image_path, image, plan)
        verifier_result = self._run_verifier(image_path, image, plan, editor_result)

        sample = TeacherSample(
            sample_id=sample_id,
            image_path=image_path,
            plan=plan,
            editor_result=editor_result,
            verifier_result=verifier_result,
            metadata={**self.metadata.copy(), "source": source_metadata},
        )
        return self.writer.write(sample)

    def _run_vlm(self, image_path: Path, image: Image.Image) -> TeacherPlan:
        if not hasattr(self.vlm, "plan"):
            raise TypeError(f"VLM adapter must implement plan(image_path, image), got {type(self.vlm)}")
        plan = self.vlm.plan(image_path=image_path, image=image)
        if not isinstance(plan, TeacherPlan):
            raise TypeError(f"VLM adapter must return TeacherPlan, got {type(plan)}")
        if plan.safe_flag and not plan.teacher_prompt:
            plan.teacher_prompt = "no edit needed"
        return plan

    def _run_editor(self, image_path: Path, image: Image.Image, plan: TeacherPlan) -> EditorResult:
        if not hasattr(self.editor, "edit"):
            raise TypeError(f"Editor adapter must implement edit(image_path, image, plan), got {type(self.editor)}")
        result = self.editor.edit(image_path=image_path, image=image, plan=plan)
        if not isinstance(result, EditorResult):
            raise TypeError(f"Editor adapter must return EditorResult, got {type(result)}")
        return result

    def _run_verifier(
        self,
        image_path: Path,
        image: Image.Image,
        plan: TeacherPlan,
        editor_result: EditorResult,
    ) -> VerifierResult:
        if self.verifier is None:
            return AcceptAllVerifier().verify(image_path, image, plan, editor_result)
        if not hasattr(self.verifier, "verify"):
            raise TypeError(f"Verifier adapter must implement verify(...), got {type(self.verifier)}")
        result = self.verifier.verify(
            image_path=image_path,
            image=image,
            plan=plan,
            editor_result=editor_result,
        )
        if not isinstance(result, VerifierResult):
            raise TypeError(f"Verifier adapter must return VerifierResult, got {type(result)}")
        return result


def discover_images(input_path: str | Path, extensions: set[str] | None = None) -> list[Path | dict[str, Any]]:
    input_path = Path(input_path)
    extensions = extensions or IMAGE_EXTENSIONS
    if input_path.is_file():
        if input_path.suffix == ".jsonl":
            return _read_jsonl_image_paths(input_path)
        if input_path.suffix.lower() in extensions:
            return [input_path]
        return [Path(line.strip()) for line in input_path.read_text().splitlines() if line.strip()]

    paths = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(paths)


def _read_jsonl_image_paths(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "image_path" not in row:
                raise ValueError(f"Missing image_path in {path}: {row}")
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            row = {**row, "image_path": str(image_path)}
            rows.append(row)
    return rows


def normalize_input_item(item: str | Path | dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if isinstance(item, dict):
        metadata = dict(item)
        image_path = Path(metadata["image_path"])
        return image_path, metadata
    image_path = Path(item)
    return image_path, {"image_path": str(image_path)}


def make_sample_id(image_path: Path) -> str:
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    return f"{image_path.stem}_{digest}"
