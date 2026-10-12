"""Transcripción con faster-whisper (CPU int8) con timestamps por palabra y caché en JSON."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from src.common.io import read_json, write_json
from src.common.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Transcript:
    language: str
    duration: float
    words: list[Word] = field(default_factory=list)
    model: str = ""

    @property
    def text(self) -> str:
        return "".join(w.text for w in self.words).strip()

    def to_dict(self) -> dict:
        return {"language": self.language, "duration": self.duration, "model": self.model,
                "words": [[w.text, round(w.start, 3), round(w.end, 3), round(w.prob, 3)] for w in self.words]}

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(language=d["language"], duration=d["duration"], model=d.get("model", ""),
                   words=[Word(*w) for w in d["words"]])

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        return cls.from_dict(read_json(path))


@lru_cache(maxsize=2)
def _model(name: str, compute_type: str, cpu_threads: int, device: str = "auto"):
    from faster_whisper import WhisperModel

    from src.common.devices import resolve_compute_type, resolve_device

    dev = resolve_device(device)
    if dev == "cuda":
        # CTranslate2 busca cublas64_12.dll/cudnn en el PATH; PyTorch las trae en torch\lib y las registra
        # al importarse. Sin este import, con device/compute_type explícitos falla al codificar (D-036).
        import torch  # noqa: F401
    ct = resolve_compute_type(compute_type, dev)
    log.info("cargando faster-whisper '%s' (%s, %s%s)", name, dev, ct,
             f", {cpu_threads} hilos" if dev == "cpu" else "")
    return WhisperModel(name, device=dev, compute_type=ct, cpu_threads=cpu_threads)


def transcribe(audio: Path, cache_path: Path, model_name: str, compute_type: str = "auto",
               cpu_threads: int = 8, beam_size: int = 1, vad_filter: bool = True,
               language: str | None = None, device: str = "auto") -> Transcript:
    if cache_path.exists():
        return Transcript.load(cache_path)
    model = _model(model_name, compute_type, cpu_threads, device)
    t0 = time.time()
    segments, info = model.transcribe(str(audio), beam_size=beam_size, word_timestamps=True,
                                      vad_filter=vad_filter, language=language)
    words = [Word(w.word, float(w.start), float(w.end), float(w.probability))
             for seg in segments for w in (seg.words or [])]
    tr = Transcript(language=info.language, duration=float(info.duration), words=words, model=model_name)
    elapsed = time.time() - t0
    log.info("transcrito %s: %.0fs de audio en %.0fs (RTF %.3f), idioma=%s, %d palabras",
             audio.name, info.duration, elapsed, elapsed / max(info.duration, 1), info.language, len(words))
    tr.save(cache_path)
    return tr
