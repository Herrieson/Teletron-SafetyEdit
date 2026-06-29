from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Small nested dict with attribute access for standalone teacher tools."""

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        data = dict(data or {})
        data.update(kwargs)
        super().__init__()
        for key, value in data.items():
            self[key] = self._convert(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._convert(value)

    def _convert(self, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, Config):
            return Config(value)
        if isinstance(value, list):
            return [self._convert(item) for item in value]
        return value


def load_teacher_config(config_path: str | Path | dict[str, Any] | Config) -> Config:
    if isinstance(config_path, Config):
        return config_path
    if isinstance(config_path, dict):
        return Config(config_path)

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported teacher config suffix: {path.suffix}")
    return Config(data or {})


def import_target(target: str) -> Any:
    """Import a class/function from ``package.module:attr`` or ``package.module.attr``."""

    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    else:
        module_name, attr_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def build_component(component_config: dict[str, Any] | Config | None, default_target: str | None = None) -> Any:
    if component_config is None:
        if default_target is None:
            return None
        component_config = {"target": default_target}

    config = Config(component_config)
    target = config.get("target", default_target)
    if target is None:
        raise ValueError("Component config must define 'target'.")

    params = config.get("params", {})
    component_cls_or_fn = import_target(target)
    return component_cls_or_fn(**params)
