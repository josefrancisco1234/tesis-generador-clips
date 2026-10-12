"""Segmentación de la transcripción en oraciones con timestamps (base de candidatos, P2 PodReels)."""
from __future__ import annotations

from dataclasses import dataclass

from src.transcribe.whisper import Transcript

TERMINAL = (".", "?", "!", "…", "。", "？", "！")


@dataclass
class Sentence:
    idx: int
    start: float
    end: float
    text: str
    w0: int  # índice de la primera palabra (inclusive)
    w1: int  # índice de la última palabra (exclusive)

    @property
    def duration(self) -> float:
        return self.end - self.start


def split_sentences(tr: Transcript, max_words: int = 40, pause_split_s: float = 0.9) -> list[Sentence]:
    words = tr.words
    out: list[Sentence] = []
    start = 0
    for i, w in enumerate(words):
        last = i == len(words) - 1
        ends = w.text.strip().endswith(TERMINAL)
        long_pause = not last and (words[i + 1].start - w.end) >= pause_split_s
        too_long = (i - start + 1) >= max_words
        if last or ends or long_pause or too_long:
            chunk = words[start:i + 1]
            text = "".join(x.text for x in chunk).strip()
            if text:
                out.append(Sentence(len(out), chunk[0].start, chunk[-1].end, text, start, i + 1))
            start = i + 1
    return out


def is_complete(text: str) -> bool:
    return text.strip().endswith(TERMINAL)
