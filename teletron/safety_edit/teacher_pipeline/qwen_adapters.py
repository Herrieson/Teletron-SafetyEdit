from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .schemas import EditorResult, TeacherPlan


DEFAULT_SAFETY_EDIT_PROMPT = """You are a safety-aware image editing planner.
Inspect the input image and decide whether it contains visible content that should be safely edited.

Return only a JSON object with these fields:
{
  "safe_flag": true or false,
  "risk_type": "none | weapon | blood | nudity | hate | illegal_activity | self_harm | other",
  "risk_description": "short description",
  "teacher_prompt": "image editing instruction for an image editor",
  "edit_region": null or {"type": "bbox", "bbox": [x1, y1, x2, y2]},
  "no_edit_reason": null or "why no edit is needed"
}

Rules:
- If no unsafe content is visible, set safe_flag=true and teacher_prompt="no edit needed".
- If editing is needed, teacher_prompt must ask the editor to change only unsafe content.
- Preserve safe regions, identity-neutral attributes, pose, background, lighting, composition, and image style.
- Do not introduce new unsafe content.
"""


class LocalQwen36VLMTeacher:
    """Local Qwen3.6 VLM planner with optional hidden-state extraction.

    This adapter intentionally imports heavy dependencies inside ``__init__`` so
    the rest of the teacher pipeline can be used without a full model runtime.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3.6-27B",
        device: str | None = None,
        device_map: str | dict[str, Any] | None = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 768,
        planner_prompt: str = DEFAULT_SAFETY_EDIT_PROMPT,
        trust_remote_code: bool = True,
        extract_hidden: bool = True,
        hidden_layer: int = -1,
        hidden_strategy: str = "all",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.device_map = device_map
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.planner_prompt = planner_prompt
        self.trust_remote_code = trust_remote_code
        self.extract_hidden = extract_hidden
        self.hidden_layer = hidden_layer
        self.hidden_strategy = hidden_strategy

        self.torch = _require_torch()
        transformers = _require_transformers()
        self.processor = transformers.AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            model_cls = transformers.AutoModelForCausalLM

        kwargs = {
            "trust_remote_code": trust_remote_code,
            "dtype": _resolve_torch_dtype(self.torch, dtype),
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        self.model = model_cls.from_pretrained(model_path, **kwargs)
        self.model.eval()
        if device is not None and device_map is None:
            self.model.to(device)

    def plan(self, image_path: Path, image: Image.Image) -> TeacherPlan:
        messages = self._build_messages(image)
        inputs = self._build_inputs(messages)

        vlm_hidden = None
        if self.extract_hidden:
            vlm_hidden = self._extract_hidden(inputs)

        response_text = self._generate(inputs)
        parsed, parse_error = parse_json_object(response_text)
        if parsed is None:
            parsed = {
                "safe_flag": False,
                "risk_type": "other",
                "risk_description": "Failed to parse structured VLM output.",
                "teacher_prompt": "Replace or remove visible unsafe content while preserving the rest of the image.",
                "edit_region": None,
                "no_edit_reason": None,
            }

        safe_flag = bool(parsed.get("safe_flag", False))
        teacher_prompt = parsed.get("teacher_prompt") or parsed.get("edit_instruction") or ""
        if safe_flag:
            teacher_prompt = "no edit needed"

        return TeacherPlan(
            teacher_prompt=teacher_prompt,
            safe_flag=safe_flag,
            risk_type=parsed.get("risk_type", "unknown"),
            risk_description=parsed.get("risk_description", ""),
            edit_region=parsed.get("edit_region"),
            no_edit_reason=parsed.get("no_edit_reason"),
            vlm_hidden=vlm_hidden,
            raw_response={
                "text": response_text,
                "parsed": parsed,
                "parse_error": parse_error,
            },
            metadata={
                "adapter": self.__class__.__name__,
                "model_path": self.model_path,
                "hidden_layer": self.hidden_layer,
                "hidden_strategy": self.hidden_strategy,
            },
        )

    def _build_messages(self, image: Image.Image) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.planner_prompt},
                ],
            }
        ]

    def _build_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            inputs = self._apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            text = self._apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(text=[text], images=[messages[0]["content"][0]["image"]], return_tensors="pt")

        target_device = self._model_device()
        return {key: value.to(target_device) if hasattr(value, "to") else value for key, value in inputs.items()}

    def _apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        try:
            return self.processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.processor.apply_chat_template(messages, **kwargs)

    def _extract_hidden(self, inputs: dict[str, Any]) -> Any:
        with self.torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            return None
        hidden = hidden_states[self.hidden_layer].detach().cpu()
        if self.hidden_strategy == "mean":
            return hidden.mean(dim=1)
        if self.hidden_strategy == "last":
            return hidden[:, -1]
        return hidden.squeeze(0)

    def _generate(self, inputs: dict[str, Any]) -> str:
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        generated = outputs[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True)

    def _model_device(self) -> Any:
        if self.device is not None:
            return self.torch.device(self.device)
        return next(self.model.parameters()).device


class LocalQwenImageEditTeacher:
    """Local Qwen-Image-Edit teacher via Diffusers.

    The adapter returns the edited image and best-effort prompt condition
    tensors from the pipeline's ``encode_prompt`` method when available.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen-Image-Edit",
        device: str = "cuda",
        device_map: str | dict[str, Any] | None = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
        seed: int | None = 0,
        skip_safe_editor: bool = True,
        extract_condition: bool = True,
        pipeline_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.device_map = device_map
        self.dtype_name = dtype
        self.trust_remote_code = trust_remote_code
        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.skip_safe_editor = skip_safe_editor
        self.extract_condition = extract_condition
        self.pipeline_kwargs = pipeline_kwargs or {}

        self.torch = _require_torch()
        self.pipeline = self._load_pipeline()

    def edit(self, image_path: Path, image: Image.Image, plan: TeacherPlan) -> EditorResult:
        condition = self._extract_condition(plan.teacher_prompt) if self.extract_condition else None
        if plan.safe_flag and self.skip_safe_editor:
            return EditorResult(
                teacher_condition=condition,
                teacher_output=image.copy(),
                teacher_mask=None,
                metadata={
                    "adapter": self.__class__.__name__,
                    "model_path": self.model_path,
                    "skipped_editor": True,
                },
            )

        generator = None
        if self.seed is not None:
            generator = self.torch.Generator(device=self.device).manual_seed(self.seed)

        inputs = {
            "image": image,
            "prompt": plan.teacher_prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "generator": generator,
            **self.pipeline_kwargs,
        }
        if self.true_cfg_scale is not None:
            inputs["true_cfg_scale"] = self.true_cfg_scale

        with self.torch.inference_mode():
            output = self.pipeline(**inputs)
        output_image = output.images[0]
        return EditorResult(
            teacher_condition=condition,
            teacher_output=output_image,
            teacher_mask=None,
            metadata={
                "adapter": self.__class__.__name__,
                "model_path": self.model_path,
                "skipped_editor": False,
                "num_inference_steps": self.num_inference_steps,
                "true_cfg_scale": self.true_cfg_scale,
            },
        )

    def _load_pipeline(self) -> Any:
        diffusers = _require_diffusers()
        dtype = _resolve_torch_dtype(self.torch, self.dtype_name)
        pipe_cls = getattr(diffusers, "QwenImageEditPipeline", None)
        if pipe_cls is None:
            pipe_cls = diffusers.DiffusionPipeline

        kwargs = {"trust_remote_code": self.trust_remote_code}
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        try:
            pipe = pipe_cls.from_pretrained(self.model_path, torch_dtype=dtype, **kwargs)
        except TypeError:
            pipe = pipe_cls.from_pretrained(self.model_path, dtype=dtype, **kwargs)

        if self.device_map is None:
            pipe.to(dtype)
            pipe.to(self.device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=None)
        return pipe

    def _extract_condition(self, prompt: str) -> Any:
        if not hasattr(self.pipeline, "encode_prompt"):
            return {"prompt": prompt, "warning": "pipeline_has_no_encode_prompt"}

        encode_prompt = self.pipeline.encode_prompt
        attempts = [
            {
                "prompt": prompt,
                "device": self.device,
                "num_images_per_prompt": 1,
                "do_classifier_free_guidance": True,
                "negative_prompt": self.negative_prompt,
            },
            {
                "prompt": prompt,
                "device": self.device,
                "num_images_per_prompt": 1,
                "do_classifier_free_guidance": False,
            },
            {"prompt": prompt, "device": self.device},
            {"prompt": prompt},
        ]
        last_error = None
        for kwargs in attempts:
            try:
                filtered_kwargs = filter_kwargs(encode_prompt, kwargs)
                with self.torch.inference_mode():
                    condition = encode_prompt(**filtered_kwargs)
                return detach_to_cpu(named_condition(condition))
            except Exception as exc:  # best-effort across diffusers versions
                last_error = repr(exc)
        return {
            "prompt": prompt,
            "warning": "encode_prompt_failed",
            "error": last_error,
        }


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as exc:
            last_error = str(exc)
    return None, last_error


def named_condition(condition: Any) -> Any:
    if isinstance(condition, dict):
        return condition
    if isinstance(condition, tuple):
        names = [
            "prompt_embeds",
            "negative_prompt_embeds",
            "pooled_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "prompt_attention_mask",
            "negative_prompt_attention_mask",
        ]
        return {names[idx] if idx < len(names) else f"value_{idx}": value for idx, value in enumerate(condition)}
    return {"prompt_embeds": condition}


def detach_to_cpu(value: Any) -> Any:
    torch = _maybe_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_to_cpu(item) for item in value)
    return value


def filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _resolve_torch_dtype(torch_module: Any, dtype: str) -> Any:
    if dtype in {"auto", "none", None}:
        return None
    if not hasattr(torch_module, dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return getattr(torch_module, dtype)


def _maybe_torch() -> Any | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def _require_torch() -> Any:
    torch = _maybe_torch()
    if torch is None:
        raise ImportError("Local Qwen adapters require torch.")
    return torch


def _require_transformers() -> Any:
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ImportError("LocalQwen36VLMTeacher requires transformers.") from exc
    return transformers


def _require_diffusers() -> Any:
    try:
        import diffusers
    except ModuleNotFoundError as exc:
        raise ImportError("LocalQwenImageEditTeacher requires diffusers.") from exc
    return diffusers
