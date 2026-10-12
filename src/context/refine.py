"""Contexto sobre el top-N (P1 Lotus, P2 PodReels) y selección top-K por LLM (P4 Minimal Clips)."""
from __future__ import annotations

import re

from src.common.logging_utils import get_logger
from src.context.llm import LLMClient, parse_json

log = get_logger(__name__)

STYLES = ("informativo", "curiosidad", "gracioso", "emotivo")

REFINE_SYSTEM = (
    "Eres editor de video corto (TikTok/Reels/Shorts). Recibes la transcripción de un clip candidato extraído "
    "de un video largo. Responde SOLO un JSON con: "
    '{"title": str (≤60 caracteres, en el idioma del clip), "hook": str (frase de apertura en pantalla, ≤80 caracteres), '
    '"style": uno de ["informativo","curiosidad","gracioso","emotivo"], '
    '"self_contained": bool (¿se entiende sin haber visto el resto del video?), '
    '"trim_start_sentences": int 0-2 (oraciones iniciales a quitar si arrancan a mitad de idea), '
    '"reason": str (≤30 palabras: por qué este momento retiene al espectador)}. No inventes hechos.'
)

TOPK_SYSTEM = (
    "You analyze a long video represented as a script of numbered 20-second segments. "
    "Pick the {k} segments most likely to be replayed or shared as short clips. "
    'Answer ONLY a JSON list of segment numbers, best first, e.g. [12, 3, 40].'
)


def _cut(text: str, n: int) -> str:
    """Recorta en borde de palabra y agrega '…' si hace falta."""
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def template_context(clip: dict, lang: str) -> dict:
    """Plantilla determinista usada cuando el LLM está desactivado."""
    text = clip["text"].strip()
    first = re.split(r"(?<=[.?!¿¡])\s+", text)[0]
    concept = clip.get("concepts", [None])[0] or (clip["keywords"][0]["term"] if clip.get("keywords") else "")
    f = clip.get("features", {})
    if f.get("s_laughs", 0) > 0:
        style = "gracioso"
    elif f.get("h_question", 0) > 0:
        style = "curiosidad"
    elif f.get("s_intensity_mean", 0) > 0.6:
        style = "emotivo"
    else:
        style = "informativo"
    return {
        "title": _cut(concept[:1].upper() + concept[1:] if concept else first, 60),
        "hook": _cut(first, 80),
        "style": style,
        "self_contained": bool(f.get("f_ends_complete", 1)) and not bool(f.get("f_starts_dangling", 0)),
        "trim_start_sentences": 1 if f.get("f_starts_dangling", 0) else 0,
        "reason": "Plantilla (sin LLM): score alto del modelo en hook/flow/value.",
        "source": "template",
    }


def refine(clips: list[dict], llm: LLMClient, lang: str) -> list[dict]:
    for c in clips:
        ctx = None
        if llm.enabled:
            user = (f"Idioma: {lang}\nKeywords: {', '.join(k['term'] for k in c.get('keywords', []))}\n"
                    f"Transcripción del clip ({c['end'] - c['start']:.0f} s):\n{c['text'][:4000]}")
            data = parse_json(llm.complete(REFINE_SYSTEM, user))
            if isinstance(data, dict) and "title" in data:
                ctx = {k: data.get(k) for k in ("title", "hook", "style", "self_contained", "trim_start_sentences", "reason")}
                ctx["source"] = f"{llm.provider}:{llm.model}"
        c["context"] = ctx or template_context(c, lang)
    return clips


def llm_select_topk(segment_texts: list[str], llm: LLMClient, k: int) -> list[int] | None:
    """Etiqueta secundaria / baseline P4: el LLM elige top-k ventanas de 20 s. None si no hay LLM."""
    if not llm.enabled:
        return None
    script = "\n".join(f"[{i}] {t[:400]}" for i, t in enumerate(segment_texts))
    data = parse_json(llm.complete(TOPK_SYSTEM.format(k=k), script[:120_000]))
    if not isinstance(data, list):
        return None
    return [int(x) for x in data if isinstance(x, (int, float)) and 0 <= int(x) < len(segment_texts)][:k]
