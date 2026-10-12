"""Logging uniforme: consola + archivo rotativo."""
from __future__ import annotations

import logging
import os
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", file: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    for noisy in ("urllib3", "httpx", "huggingface_hub", "filelock", "numba", "matplotlib", "PIL", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
