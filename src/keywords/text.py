"""Normalización de texto, stopwords y léxicos ES/EN usados por keywords y features."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?|\d+(?:[.,]\d+)?", re.UNICODE)

FILLERS = {
    "es": {"eh", "em", "este", "osea", "o sea", "digamos", "tipo", "bueno", "pues", "mmm", "ehh"},
    "en": {"um", "uh", "erm", "hmm", "like", "you know", "i mean", "kinda", "sorta", "uhh"},
}
# Marcadores de hook (P2 PodReels + rúbricas públicas de Opus/Klap): contraste, absolutos, secretos.
HOOK_WORDS = {
    "es": {"nunca", "nadie", "siempre", "secreto", "verdad", "error", "problema", "increíble", "importante",
           "peor", "mejor", "jamás", "imagina", "sabías", "cuidado", "mentira", "realmente", "clave", "todos"},
    "en": {"never", "nobody", "always", "secret", "truth", "mistake", "problem", "incredible", "important",
           "worst", "best", "imagine", "crazy", "insane", "actually", "honestly", "key", "everyone", "wrong"},
}
CONTRAST = {"es": {"pero", "sin embargo", "aunque", "en cambio"}, "en": {"but", "however", "although", "yet"}}
SECOND_PERSON = {"es": {"tú", "tu", "te", "ti", "usted", "ustedes", "vos", "tus"}, "en": {"you", "your", "you're", "yours"}}
FIRST_PERSON = {"es": {"yo", "me", "mi", "mí", "nosotros", "mis"}, "en": {"i", "me", "my", "we", "our", "i'm"}}
# Arranques que suelen depender de contexto previo (anáfora colgante / conectores).
DANGLING_START = {
    "es": {"y", "entonces", "eso", "esto", "ese", "esa", "él", "ella", "ellos", "porque", "o", "pero", "además", "también", "así"},
    "en": {"and", "so", "that", "this", "it", "he", "she", "they", "because", "or", "but", "also", "which", "then"},
}
INTENSIFIERS = {"es": {"muy", "súper", "tan", "demasiado", "totalmente", "absolutamente", "literalmente"},
                "en": {"very", "super", "so", "really", "totally", "absolutely", "literally", "extremely"}}
LAUGH_RE = re.compile(r"\b(ja(ja)+|je(je)+|ha(ha)+|lol|\[risas\]|\(risas\)|\[laughter\])\b", re.IGNORECASE)


def lang_key(lang: str | None) -> str:
    return "es" if (lang or "").startswith("es") else "en"


@lru_cache(maxsize=4)
def stopwords(lang: str) -> frozenset[str]:
    if lang_key(lang) == "es":
        from spacy.lang.es.stop_words import STOP_WORDS
    else:
        from spacy.lang.en.stop_words import STOP_WORDS
    return frozenset(STOP_WORDS) | frozenset(FILLERS[lang_key(lang)])


def normalize(token: str) -> str:
    token = token.lower().strip("'’")
    return unicodedata.normalize("NFC", token)


def tokenize(text: str) -> list[str]:
    return [normalize(t) for t in TOKEN_RE.findall(text)]


def content_token(word: str, lang: str) -> str | None:
    """Token de contenido para TF-IDF o None (stopword, número corto, muy corto)."""
    toks = tokenize(word)
    if not toks:
        return None
    t = toks[0]
    if "'" in t or "’" in t:  # contracciones (i'll, don't): casi siempre palabras funcionales
        return None
    if len(t) < 3 or t in stopwords(lang) or t.isdigit():
        return None
    return t
