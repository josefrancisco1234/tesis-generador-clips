"""Construcción de la tabla de features por ventana.

Grupos (prefijo de columna):
  k_  keywords: TF-IDF KVSUM + KeyBERT + frases nominales       (técnicas 5, 6, 1)
  s_  sentimiento y energía léxica                              (técnica 7)
  a_  audio: RMS, pitch, pausas, velocidad de habla             (técnica 8)
  h_  hook: primeros segundos del clip                          (Opus "Hook", P2)
  f_  flow: completitud, coherencia, duración                   (Opus "Flow", P2)
  p_  posición relativa en el video (sólo ablación / baseline)
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger
from src.features.audio import AudioFrames
from src.features.extra import THIRD_PERSON
from src.features.sentiment import sentence_sentiment
from src.keywords.extractor import KeywordIndex, build_index
from src.keywords.text import (CONTRAST, DANGLING_START, FILLERS, FIRST_PERSON, HOOK_WORDS, INTENSIFIERS, LAUGH_RE,
                               SECOND_PERSON, lang_key, tokenize)
from src.segment.candidates import Window
from src.segment.sentences import Sentence, is_complete
from src.transcribe.whisper import Transcript

log = get_logger(__name__)

NOVELTY_WINDOW_S = 300.0
GROUP_PREFIX = {"K": "k_", "S": "s_", "A": "a_", "H": "h_", "F": "f_", "P": "p_", "E": "e_", "L": "l_",
                "U": "u_"}  # U: embeddings de habla (sólo con GPU, ver configs/gpu.yaml)
# Dimensiones tipo Opus para el desglose del score (D-006)
DIMENSIONS = {"hook": ("h_",), "flow": ("f_",), "value": ("k_", "s_", "a_", "e_", "l_", "u_")}


def feature_columns(df: pd.DataFrame, groups: list[str]) -> list[str]:
    prefixes = tuple(GROUP_PREFIX[g] for g in groups)
    return [c for c in df.columns if c.startswith(prefixes)]


@dataclass
class VideoContext:
    tr: Transcript
    sents: list[Sentence]
    kw: KeywordIndex
    sentiment: np.ndarray            # (n_sent, 3)
    audio: AudioFrames | None
    lang: str
    word_start: np.ndarray
    word_end: np.ndarray
    word_tok: list[list[str]]        # tokens normalizados por palabra
    seen_before: np.ndarray          # 1 si el token de contenido apareció en los NOVELTY_WINDOW_S previos
    coh_next: np.ndarray             # coseno entre oración i e i+1
    events: pd.DataFrame | None = None   # probabilidades PANNs por segundo (grupo E); None = las añade extra.py


def build_context(tr: Transcript, sents: list[Sentence], cfg: dict, audio: AudioFrames | None,
                  cache_dir: Path | None, key: str | None) -> VideoContext:
    kw = build_index(tr, sents, cfg["keywords"], cache_dir / "emb" if cache_dir else None, key)
    sent = sentence_sentiment([s.text for s in sents], cfg["features"]["sentiment_model"],
                              cfg["features"]["sentiment_batch"],
                              cache_dir / "sentiment" / f"{key}.npy" if cache_dir and key else None,
                              cfg["features"].get("device", "auto"))
    ws = np.array([w.start for w in tr.words], dtype=np.float64)
    we = np.array([w.end for w in tr.words], dtype=np.float64)
    seen_before = local_seen(ws, kw.word_tokens, NOVELTY_WINDOW_S)
    e = kw.sent_emb
    coh = (e[:-1] * e[1:]).sum(axis=1) if len(e) > 1 else np.zeros(0)
    return VideoContext(tr, sents, kw, sent, audio, lang_key(tr.language), ws, we,
                        [tokenize(w.text) for w in tr.words], seen_before, coh)


def local_seen(starts: np.ndarray, tokens: list[str | None], window_s: float) -> np.ndarray:
    """1 si el término apareció en los `window_s` segundos previos (D-018).

    Una novedad global (desde el inicio del video) correlacionaba −0.6 con la posición y
    filtraba el sesgo de inicio; la versión local mide información nueva respecto del contexto reciente.
    """
    last: dict[str, float] = {}
    out = np.zeros(len(tokens), dtype=np.int8)
    for i, t in enumerate(tokens):
        if t:
            out[i] = starts[i] - last.get(t, -1e9) <= window_s
            last[t] = starts[i]
    return out


def _rate(n: float, minutes: float) -> float:
    return n / minutes if minutes > 0 else 0.0


def window_features(ctx: VideoContext, w: Window, hook_s: float, hook_words: int) -> dict[str, float]:
    lang = ctx.lang
    i0 = int(np.searchsorted(ctx.word_start, w.start, side="left"))
    i1 = int(np.searchsorted(ctx.word_start, w.end, side="left"))
    n_words = max(0, i1 - i0)
    dur = max(w.end - w.start, 1e-3)
    minutes = dur / 60.0
    toks = [t for ts in ctx.word_tok[i0:i1] for t in ts]
    ctoks = [t for t in ctx.kw.word_tokens[i0:i1] if t]
    s0, s1 = w.s0, w.s1
    sents = ctx.sents[s0:s1] if s0 >= 0 else []
    text = " ".join(s.text for s in sents)

    f: dict[str, float] = {}
    # ---------------- K: keywords ----------------
    tw = ctx.kw.term_weight
    f["k_kvsum_mean"] = float(np.mean([tw.get(t, 0.0) for t in ctoks])) if ctoks else 0.0
    f["k_kvsum_max"] = float(max((tw.get(t, 0.0) for t in ctoks), default=0.0))
    local = 0.0
    for wi in range(i0, i1):
        t = ctx.kw.word_tokens[wi]
        if t:
            p = min(int(ctx.word_start[wi] // ctx.kw.passage_s), len(ctx.kw.passage_weights) - 1)
            local += ctx.kw.passage_weights[p].get(t, 0.0)
    f["k_kvsum_local"] = local / max(n_words, 1)
    top = set(ctx.kw.top_terms)
    f["k_top_terms_pm"] = _rate(sum(t in top for t in ctoks), minutes)
    f["k_content_ratio"] = len(ctoks) / max(n_words, 1)
    f["k_novelty"] = float(1 - ctx.seen_before[i0:i1][[t is not None for t in ctx.kw.word_tokens[i0:i1]]].mean()) if ctoks else 0.0
    if sents:
        emb = ctx.kw.sent_emb[s0:s1].mean(axis=0)
        emb = emb / (np.linalg.norm(emb) or 1.0)
        f["k_keybert_sim"] = float(emb @ ctx.kw.doc_emb)
    else:
        f["k_keybert_sim"] = np.nan
    low = text.lower()
    f["k_keyphrase_pm"] = _rate(sum(low.count(p) for p, _ in ctx.kw.keyphrases), minutes)
    f["k_noun_chunks_ps"] = (float(np.mean([len(ctx.kw.noun_chunks[i]) for i in range(s0, s1)]))
                             if sents and ctx.kw.noun_chunks else np.nan)

    # ---------------- S: sentimiento / energía léxica ----------------
    if sents:
        sp = ctx.sentiment[s0:s1]
        pol = sp[:, 0] - sp[:, 2]
        inten = np.abs(pol)
        f.update(s_pos_mean=float(sp[:, 0].mean()), s_neg_mean=float(sp[:, 2].mean()),
                 s_polarity_mean=float(pol.mean()), s_polarity_std=float(pol.std()),
                 s_intensity_mean=float(inten.mean()), s_intensity_max=float(inten.max()))
    else:
        f.update({k: np.nan for k in ["s_pos_mean", "s_neg_mean", "s_polarity_mean", "s_polarity_std",
                                      "s_intensity_mean", "s_intensity_max"]})
    f["s_questions_pm"] = _rate(text.count("?") + text.count("¿"), minutes)
    f["s_exclaim_pm"] = _rate(text.count("!") + text.count("¡"), minutes)
    ntok = max(len(toks), 1)
    f["s_second_person"] = sum(t in SECOND_PERSON[lang] for t in toks) / ntok
    f["s_first_person"] = sum(t in FIRST_PERSON[lang] for t in toks) / ntok
    f["s_numbers"] = sum(t[0].isdigit() for t in toks) / ntok
    f["s_intensifiers"] = sum(t in INTENSIFIERS[lang] for t in toks) / ntok
    f["s_hook_words"] = sum(t in HOOK_WORDS[lang] for t in toks) / ntok
    f["s_laughs"] = float(len(LAUGH_RE.findall(text)))

    # ---------------- A: audio y prosodia ----------------
    # (antes estas 7 variables vivian en audio.py:window_audio; ahora estan aqui, en el orden de la pagina web)
    fr = ctx.audio
    if fr is not None:
        sl = fr.slice(w.start, w.end)
        rz, f0, vc = fr.rms_z[sl], fr.f0_st[sl], fr.voiced[sl]          # volumen z, tono en semitonos, frames con voz
    sin_audio = fr is None or len(rz) == 0
    if sin_audio:
        for k in ("a_rms_mean", "a_rms_p95", "a_rms_std", "a_pitch_std", "a_pitch_range", "a_voiced_ratio", "a_energy_delta"):
            f[k] = np.nan
    else:
        f0v = f0[~np.isnan(f0)]
        prev = fr.rms_z[fr.slice(max(0.0, w.start - 30.0), w.start)]   # 30 s anteriores
        f["a_rms_mean"] = float(rz.mean())                              # volumen medio (z respecto del video)
        f["a_rms_p95"] = float(np.percentile(rz, 95))                   # picos de volumen
        f["a_rms_std"] = float(rz.std())                                # dinamica del volumen (susurro vs enfasis)
        f["a_pitch_std"] = float(f0v.std()) if len(f0v) > 10 else np.nan                      # variacion de entonacion
        f["a_pitch_range"] = float(np.percentile(f0v, 95) - np.percentile(f0v, 5)) if len(f0v) > 10 else np.nan   # rango de tono
        f["a_voiced_ratio"] = float(vc.mean())                          # fraccion de frames con voz
        f["a_energy_delta"] = float(rz.mean() - prev.mean()) if len(prev) else 0.0   # subida de energia vs 30 s previos
    f["a_speech_rate"] = n_words / dur                                  # palabras por segundo
    if n_words > 1:
        gaps = ctx.word_start[i0 + 1:i1] - ctx.word_end[i0:i1 - 1]      # silencio entre palabras consecutivas
        f["a_pause_ratio"] = float(np.clip(gaps, 0, None)[gaps > 0.3].sum() / dur)   # tiempo en pausas > 0.3 s / dur
        f["a_long_pauses_pm"] = _rate(float((gaps > 1.0).sum()), minutes)           # pausas > 1 s por minuto
    else:
        f["a_pause_ratio"], f["a_long_pauses_pm"] = 1.0, 0.0

    # ---------------- H: hook ----------------
    onset = fr.rms_z[fr.slice(w.start, w.start + hook_s)] if fr is not None else np.empty(0)
    f["h_rms_onset"] = np.nan if (sin_audio or len(onset) == 0) else float(onset.mean())   # volumen de los primeros 3 s
    j = min(i1, i0 + hook_words)
    hook_toks = [t for ts in ctx.word_tok[i0:j] for t in ts]
    hook_text = "".join(x.text for x in ctx.tr.words[i0:j])
    first = sents[0] if sents else None
    f["h_question"] = float("?" in hook_text or (first is not None and first.text.strip().endswith("?")))
    f["h_number"] = float(any(t[0].isdigit() for t in hook_toks))
    f["h_second_person"] = float(any(t in SECOND_PERSON[lang] for t in hook_toks))
    f["h_hook_word"] = float(any(t in HOOK_WORDS[lang] for t in hook_toks))
    f["h_contrast"] = float(any(t in CONTRAST[lang] for t in hook_toks))
    f["h_first_sentence_s"] = first.duration if first else np.nan
    f["h_first_intensity"] = float(abs(ctx.sentiment[s0, 0] - ctx.sentiment[s0, 2])) if sents else np.nan
    f["h_first_keybert"] = float(ctx.kw.sent_emb[s0] @ ctx.kw.doc_emb) if sents else np.nan

    # ---------------- F: flow ----------------
    first_tok = hook_toks[0] if hook_toks else ""
    f["f_starts_dangling"] = float(first_tok in DANGLING_START[lang])
    f["f_ends_complete"] = float(is_complete(sents[-1].text)) if sents else 0.0
    c = ctx.coh_next[s0:s1 - 1] if (sents and s1 - s0 > 1) else np.empty(0)
    half = (s0 + s1) // 2
    if c.size > 0 and half > s0 and s1 > half:
        f["f_coherence_mean"], f["f_coherence_min"] = float(c.mean()), float(c.min())
        a, b = ctx.kw.sent_emb[s0:half].mean(axis=0), ctx.kw.sent_emb[half:s1].mean(axis=0)
        f["f_topic_drift"] = float(1 - (a @ b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))
    else:
        # candidato con muy pocas oraciones (o el punto medio coincide con un
        # borde): no hay suficientes datos para medir coherencia/deriva de tema.
        # Se deja como NaN en vez de reventar (bug real encontrado en 2026-10-01
        # al correr el pipeline sobre un video con oraciones muy cortas).
        f["f_coherence_mean"] = f["f_coherence_min"] = f["f_topic_drift"] = np.nan
    f["f_n_sentences"] = float(max(s1 - s0, 0))
    f["f_duration"] = dur
    fill = FILLERS[lang]
    f["f_filler_rate"] = sum(t in fill for t in toks) / ntok
    f["f_word_conf"] = float(np.mean([x.prob for x in ctx.tr.words[i0:i1]])) if n_words else 0.0

    # ---------------- L: linguistica (misma formula que extra.py:linguistic_features) ----------------
    if len(toks) == 0:
        for k in ("l_entropy", "l_type_token", "l_pron_12_vs_3", "l_fillers", "l_silence", "l_word_len"):
            f[k] = np.nan
    else:
        cnt = Counter(toks)
        nt = len(toks)
        f["l_entropy"] = -sum((v / nt) * math.log2(v / nt) for v in cnt.values())      # Shannon: variedad lexica
        f["l_type_token"] = len(cnt) / nt                                              # tokens distintos / tokens
        p12 = sum(cnt[t] for t in FIRST_PERSON[lang] | SECOND_PERSON[lang])
        p3 = sum(cnt[t] for t in THIRD_PERSON[lang])
        f["l_pron_12_vs_3"] = (p12 + 1) / (p3 + 1)                                     # dialogo directo vs terceros
        f["l_fillers"] = sum(cnt[t] for t in FILLERS[lang] if " " not in t) / nt       # muletillas / tokens
        hablado = sum(x.end - x.start for x in ctx.tr.words[i0:i1])
        f["l_silence"] = 1 - hablado / max(w.end - w.start, 1e-3)                      # 1 - tiempo hablado / dur
        f["l_word_len"] = float(np.mean([len(t) for t in toks]))                       # letras por palabra

    # ---------------- E: eventos de audio (misma formula que extra.py:event_features) ----------------
    # Solo si el contexto trae las probabilidades PANNs; si no, el pipeline las agrega despues con extra.py:augment_table.
    if ctx.events is not None:
        ev = ctx.events
        t_ev = ev["t"].to_numpy()
        m = (t_ev >= w.start) & (t_ev < w.end)
        despues = (t_ev >= w.end) & (t_ev < w.end + 6)           # reaccion en los 6 s posteriores
        cola = (t_ev >= w.end - 6) & (t_ev < w.end)              # 6 s finales (remate)
        for k in ("laugh", "applause", "cheering", "music", "speech"):
            vm = ev[k].to_numpy()[m]
            f[f"e_{k}_max"] = float(vm.max()) if vm.size else np.nan
            f[f"e_{k}_mean"] = float(vm.mean()) if vm.size else np.nan
        lv = ev["laugh"].to_numpy()
        f["e_laugh_frac"] = float((lv[m] > 0.2).mean()) if m.any() else np.nan
        f["e_laugh_tail"] = float(lv[cola].max()) if cola.any() else np.nan
        f["e_laugh_after"] = float(lv[despues].max()) if despues.any() else np.nan
        f["e_applause_after"] = float(ev["applause"].to_numpy()[despues].max()) if despues.any() else np.nan

    # ---------------- P: posición (sólo ablación) ----------------
    rel = ((w.start + w.end) / 2) / max(ctx.tr.duration, 1.0)
    f["p_rel_pos"], f["p_rel_pos_sq"] = rel, rel * rel
    return f


def windows_frame(ctx: VideoContext, windows: list[Window], hook_s: float, hook_words: int,
                  extra: dict | None = None) -> pd.DataFrame:
    rows = []
    for k, w in enumerate(windows):
        r = {"win_id": k, "start": w.start, "end": w.end, "s0": w.s0, "s1": w.s1, "kind": w.kind}
        r.update(window_features(ctx, w, hook_s, hook_words))
        if extra:
            r.update(extra)
        rows.append(r)
    return pd.DataFrame(rows)
