"""Subtítulos ASS palabra por palabra con keywords resaltadas (técnica 10).

Color primario: palabras de mayor peso del clip (KVSUM + KeyBERT).
Color secundario: números y marcadores emocionales/hook.
La palabra que se está pronunciando se agranda (estilo "pop" de Opus Clip).
"""
from __future__ import annotations

from pathlib import Path

from src.keywords.text import HOOK_WORDS, INTENSIFIERS, lang_key, tokenize
from src.segment.sentences import TERMINAL
from src.transcribe.whisper import Word


def _ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _esc(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def word_class(word: str, keywords: set[str], lang: str) -> str:
    toks = tokenize(word)
    if not toks:
        return "base"
    t = toks[0]
    if t in keywords:
        return "primary"
    if t[0].isdigit() or t in HOOK_WORDS[lang_key(lang)] or t in INTENSIFIERS[lang_key(lang)]:
        return "secondary"
    return "base"


def build_ass(words: list[Word], clip_start: float, clip_end: float, keywords: set[str], lang: str, cfg: dict,
              width: int, height: int, hook_text: str | None = None) -> str:
    c = cfg
    colors = {"primary": c["primary_color"], "secondary": c["secondary_color"], "base": c["base_color"]}
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{c['font']},{c['font_size']},{c['base_color']},&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,6,2,2,60,60,{int(height * 0.28)},1
Style: Hook,{c['font']},{int(c['font_size'] * 0.8)},&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,1,0,0,0,100,100,0,0,3,4,0,8,60,60,{int(height * 0.08)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ws = [w for w in words if clip_start <= w.start < clip_end]
    lines: list[list[Word]] = []
    cur: list[Word] = []
    for w in ws:
        cur.append(w)
        if len(cur) >= c["words_per_line"] or w.text.strip().endswith(TERMINAL):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    events = []
    if hook_text:
        events.append(f"Dialogue: 1,{_ts(0)},{_ts(min(3.0, clip_end - clip_start))},Hook,,0,0,0,,{_esc(hook_text)}")
    for li, line in enumerate(lines):
        line_end = lines[li + 1][0].start if li + 1 < len(lines) else min(line[-1].end + 0.4, clip_end)
        for wi, w in enumerate(line):
            t0 = w.start - clip_start
            t1 = (line[wi + 1].start if wi + 1 < len(line) else line_end) - clip_start
            parts = []
            for wj, x in enumerate(line):
                txt = _esc(x.text.strip())
                if c.get("uppercase"):
                    txt = txt.upper()
                col = colors[word_class(x.text, keywords, lang)]
                if wj == wi:
                    parts.append(f"{{\\c{col}\\fscx118\\fscy118}}{txt}{{\\fscx100\\fscy100}}")
                else:
                    parts.append(f"{{\\c{col}}}{txt}")
            events.append(f"Dialogue: 0,{_ts(t0)},{_ts(max(t1, t0 + 0.05))},Cap,,0,0,0,,{' '.join(parts)}")
    return header + "\n".join(events) + "\n"


def write_ass(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
