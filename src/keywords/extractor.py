"""Palabras de mayor peso y su contexto (técnicas 1, 5 y 6).

- KVSUM (TF-IDF por pasaje, Ec. 1): w_ij = log(tf_ij+1)·log(N/PF_i) / sqrt(Σ_k [log(tf_kj+1)·log(N/PF_k)]²)
  La normalización se hace sobre los términos del pasaje j (coseno de Salton). Ver D-010: si se
  normalizara sobre j para un término fijo, el factor IDF se cancelaría.
  Peso global: W(a_i) = Σ_j w_ij.
- KeyBERT (Grootendorst, 2020): n-gramas candidatos → similitud coseno con el embedding del video
  (centroide de oraciones, para no truncar documentos largos) → MMR para diversidad.
- Conceptos tipo Lotus: frases nominales de spaCy (noun_chunks) por oración.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.common.logging_utils import get_logger
from src.keywords.text import content_token, lang_key, stopwords, tokenize
from src.segment.sentences import Sentence
from src.transcribe.whisper import Transcript

log = get_logger(__name__)


@lru_cache(maxsize=1)
def embedding_model(name: str):
    from sentence_transformers import SentenceTransformer

    log.info("cargando modelo de embeddings %s", name)
    return SentenceTransformer(name, device="cpu")


@lru_cache(maxsize=2)
def spacy_model(lang: str):
    import spacy

    name = "es_core_news_sm" if lang_key(lang) == "es" else "en_core_web_sm"
    return spacy.load(name, disable=["ner", "lemmatizer"])


@dataclass
class KeywordIndex:
    lang: str
    word_tokens: list[str | None]                 # token de contenido por palabra (alineado a Transcript.words)
    term_weight: dict[str, float]                 # W(a_i) de KVSUM
    passage_weights: list[dict[str, float]]       # w_ij por pasaje
    passage_s: float
    top_terms: list[str]
    keyphrases: list[tuple[str, float]]           # KeyBERT (frase, score)
    sent_emb: np.ndarray                          # (n_sent, d) normalizados
    doc_emb: np.ndarray                           # (d,)
    noun_chunks: list[list[str]] = field(default_factory=list)


def kvsum_weights(tokens_by_passage: list[list[str]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    n = len(tokens_by_passage)
    tfs = [Counter(toks) for toks in tokens_by_passage]
    pf = Counter(t for tf in tfs for t in tf)
    passage_w: list[dict[str, float]] = []
    total: dict[str, float] = defaultdict(float)
    for tf in tfs:
        raw = {t: math.log(c + 1) * math.log(n / pf[t]) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
        w = {t: v / norm for t, v in raw.items()}
        passage_w.append(w)
        for t, v in w.items():
            total[t] += v
    return passage_w, dict(total)


def keybert_phrases(sentences: list[str], sent_emb: np.ndarray, doc_emb: np.ndarray, lang: str, model_name: str,
                    top_n: int = 20, ngram_range=(1, 2), diversity: float = 0.5, max_candidates: int = 300) -> list[tuple[str, float]]:
    from sklearn.feature_extraction.text import CountVectorizer

    if not sentences:
        return []
    try:
        cv = CountVectorizer(ngram_range=tuple(ngram_range), stop_words=list(stopwords(lang)), min_df=2,
                             token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
        counts = cv.fit_transform(sentences)
    except ValueError:  # vocabulario vacío (transcripciones muy cortas)
        return []
    vocab = np.array(cv.get_feature_names_out())
    freq = np.asarray(counts.sum(axis=0)).ravel()
    cand = vocab[np.argsort(-freq)[:max_candidates]]
    cand_emb = embedding_model(model_name).encode(list(cand), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sim_doc = cand_emb @ doc_emb
    # MMR (Carbonell & Goldstein, 1998), como en KeyBERT
    selected = [int(np.argmax(sim_doc))]
    while len(selected) < min(top_n, len(cand)):
        rest = [i for i in range(len(cand)) if i not in selected]
        sim_sel = cand_emb[rest] @ cand_emb[selected].T
        mmr = (1 - diversity) * sim_doc[rest] - diversity * sim_sel.max(axis=1)
        selected.append(rest[int(np.argmax(mmr))])
    return [(str(cand[i]), float(sim_doc[i])) for i in selected]


def build_index(tr: Transcript, sents: list[Sentence], cfg_kw: dict, cache_dir: Path | None = None,
                cache_key: str | None = None, with_noun_chunks: bool = True) -> KeywordIndex:
    lang = lang_key(tr.language)
    word_tokens = [content_token(w.text, lang) for w in tr.words]
    passage_s = float(cfg_kw["passage_s"])
    n_pass = max(1, int(math.ceil(max(tr.duration, 1.0) / passage_s)))
    by_passage: list[list[str]] = [[] for _ in range(n_pass)]
    for w, t in zip(tr.words, word_tokens):
        if t:
            by_passage[min(int(w.start // passage_s), n_pass - 1)].append(t)
    passage_w, term_w = kvsum_weights(by_passage)
    top_terms = [t for t, _ in sorted(term_w.items(), key=lambda kv: -kv[1])[: cfg_kw["top_n_video"]]]

    texts = [s.text for s in sents]
    emb_path = cache_dir / f"{cache_key}.npy" if cache_dir and cache_key else None
    sent_emb = None
    if emb_path and emb_path.exists():
        cached = np.load(emb_path)
        # El cache queda obsoleto si la transcripcion/segmentacion de una
        # corrida anterior tenia OTRO numero de oraciones (p.ej. correr una
        # vez desde el .wav y otra vez desde el .mp4 del mismo video_id puede
        # dar una VAD/segmentacion levemente distinta). Usar un cache de
        # distinto largo desalinea sent_emb[s0:s1] con las oraciones reales y
        # produce slices vacios mas adelante (bug real encontrado 2026-10-01).
        if len(cached) == len(texts):
            sent_emb = cached
    if sent_emb is None:
        sent_emb = embedding_model(cfg_kw["embedding_model"]).encode(texts, batch_size=64, normalize_embeddings=True,
                                                                     show_progress_bar=False) if texts else np.zeros((0, 384))
        if emb_path:
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(emb_path, sent_emb)
    doc = sent_emb.mean(axis=0) if len(sent_emb) else np.zeros(sent_emb.shape[1] if sent_emb.ndim == 2 else 384)
    doc_emb = doc / (np.linalg.norm(doc) or 1.0)
    phrases = keybert_phrases(texts, sent_emb, doc_emb, lang, cfg_kw["embedding_model"], cfg_kw["top_n_video"],
                              cfg_kw["keybert_ngram_range"])
    chunks: list[list[str]] = []
    if with_noun_chunks and texts:
        nlp = spacy_model(lang)
        for doc_ in nlp.pipe(texts, batch_size=64):
            try:
                chunks.append([c.text.lower() for c in doc_.noun_chunks if len(c.text) > 3])
            except (NotImplementedError, ValueError):
                chunks.append([])
    return KeywordIndex(lang, word_tokens, term_w, passage_w, passage_s, top_terms, phrases, sent_emb, doc_emb, chunks)


def window_keywords(idx: KeywordIndex, tr: Transcript, sents: list[Sentence], start: float, end: float,
                    k: int = 5) -> list[dict]:
    """Top-k palabras de mayor peso dentro de la ventana, con la oración donde aparecen (contexto)."""
    scores: dict[str, float] = defaultdict(float)
    first_pos: dict[str, int] = {}
    for i, (w, t) in enumerate(zip(tr.words, idx.word_tokens)):
        if t and start <= w.start < end:
            p = min(int(w.start // idx.passage_s), len(idx.passage_weights) - 1)
            scores[t] += idx.passage_weights[p].get(t, 0.0)
            first_pos.setdefault(t, i)
    kp = {p for p, _ in idx.keyphrases}
    for t in scores:  # bonificación semántica: el término forma parte de una keyphrase KeyBERT del video
        if any(t in p.split() for p in kp):
            scores[t] *= 1.5
    out = []
    for t, s in sorted(scores.items(), key=lambda kv: -kv[1])[:k]:
        wi = first_pos[t]
        ctx = next((x.text for x in sents if x.w0 <= wi < x.w1), "")
        out.append({"term": t, "weight": round(float(s), 4), "context": ctx})
    return out


def highlight_terms(idx: KeywordIndex, tr: Transcript, start: float, end: float, max_ratio: float) -> set[str]:
    """Términos a resaltar en subtítulos: los de mayor peso local + keyphrases, limitado a max_ratio de palabras."""
    toks = [(w, t) for w, t in zip(tr.words, idx.word_tokens) if start <= w.start < end]
    n_words = max(1, len(toks))
    budget = max(1, int(n_words * max_ratio))
    ranked = window_keywords(idx, tr, [], start, end, k=budget)
    terms = {r["term"] for r in ranked}
    return terms | {t for t in (x for p, _ in idx.keyphrases[:5] for x in tokenize(p)) if any(t == tt for _, tt in toks)}
