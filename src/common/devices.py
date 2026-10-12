"""Selección de dispositivo (CPU/GPU) y tipo de cómputo, compartida por whisper, PANNs y transformers.

`device: auto` usa CUDA si hay una GPU NVIDIA visible. `compute_type: auto` elige:
  - CPU                     → int8
  - GTX 16xx (Turing sin tensor cores) → float32: medido en la 1660 SUPER, fp16 es ~2× más lento (D-036)
  - GPU con CC ≥ 7.0        → float16 (RTX 20xx en adelante, con tensor cores)
  - GPU con CC 6.x (Pascal) → int8_float32 (la GTX 10xx no acelera fp16)
"""
from __future__ import annotations

import re
from functools import lru_cache

from src.common.logging_utils import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def cuda_info() -> tuple[bool, str, tuple[int, int] | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "", None
        return True, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
    except Exception as exc:  # torch sin CUDA, driver ausente, etc.
        log.debug("CUDA no disponible: %s", exc)
        return False, "", None


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    has, name, cc = cuda_info()
    if has:
        log.info("GPU detectada: %s (compute capability %s)", name, cc)
        return "cuda"
    return "cpu"


def resolve_compute_type(requested: str = "auto", device: str | None = None) -> str:
    device = device or resolve_device()
    if requested != "auto":
        return requested
    if device != "cuda":
        return "int8"
    _, name, cc = cuda_info()
    if re.search(r"GTX 16", name or ""):
        return "float32"
    return "float16" if cc and cc[0] >= 7 else "int8_float32"
