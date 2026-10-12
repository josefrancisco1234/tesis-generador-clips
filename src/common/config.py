"""Carga de configuración YAML con interpolación simple ${clave} y semillas globales."""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

_VAR = re.compile(r"\$\{([a-zA-Z0-9_]+)\}")


class Config(dict):
    """dict con acceso por atributo (cfg.paths.raw) y rutas ya resueltas."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def path(self, key: str) -> Path:
        """Ruta absoluta para paths.<key>; relativas al repo si no empiezan con unidad/raíz."""
        p = Path(self["paths"][key])
        return p if p.is_absolute() else PROJECT_ROOT / p


def _interpolate(paths: dict[str, str]) -> dict[str, str]:
    out = dict(paths)
    for _ in range(5):
        for k, v in out.items():
            if isinstance(v, str):
                out[k] = _VAR.sub(lambda m: str(out.get(m.group(1), m.group(0))), v)
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    path = Path(path or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    base_name = raw.pop("extends", None)  # perfil que hereda de otro (p. ej. configs/gpu.yaml)
    if base_name:
        base_path = Path(base_name)
        if not base_path.is_absolute():
            base_path = PROJECT_ROOT / base_path
        with open(base_path, encoding="utf-8") as fh:
            base = yaml.safe_load(fh)
        base.pop("extends", None)
        _deep_update(base, raw)
        raw = base
    if overrides:
        _deep_update(raw, overrides)
    raw["paths"] = _interpolate(raw["paths"])
    if "logging" in raw and isinstance(raw["logging"].get("file"), str):
        raw["logging"]["file"] = _VAR.sub(lambda m: raw["paths"].get(m.group(1), m.group(0)), raw["logging"]["file"])
    return Config(raw)


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # torch es opcional para partes del pipeline
        pass
