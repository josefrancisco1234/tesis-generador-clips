"""3 ejemplos explicados (SOLO contenido): selecciona 3 videos de los videos NO usados para fijar la formula 1 donde el clip #1 toca un pico real,
y para cada uno explica por que se eligio cada clip con todas las variables. Pagina HTML en entregables/demo_3_ejemplos."""
import html
from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from scripts.reverse_engineering_formula import heat_intervals, top_k_nms  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.data.youtube_dataset import transcript_path  # noqa: E402
from src.ranking.labels import corrected_graph  # noqa: E402
from src.ranking.retention import calibrate, formula_raw, load_formula  # noqa: E402
from src.transcribe.whisper import Transcript  # noqa: E402

cfg = load_config("configs/gpu.yaml")
F = load_formula(ROOT / "models")
W, TAB = F["weights"], F["calibrator"]
dev = set(json.loads((ROOT / "reports" / "conjunto_desarrollo_D038.json").read_text(encoding="utf-8"))["video_ids"])
ids = sorted(p.stem for p in (cfg.path("processed") / "features" / "bin").glob("*.parquet") if p.stem not in dev)
DESC = {"a_pause_ratio": "proporcion de la ventana en pausas > 0.3 s", "l_silence": "1 - tiempo hablado / duracion",
        "a_long_pauses_pm": "pausas > 1 s por minuto", "a_rms_std": "variacion del volumen",
        "a_pitch_range": "rango de tono (p95 - p5, semitonos)", "a_voiced_ratio": "fraccion de frames con voz",
        "k_keybert_sim": "parecido al tema central del video", "e_speech_mean": "probabilidad media de habla pura (PANNs)",
        "k_noun_chunks_ps": "sintagmas nominales por oracion", "a_speech_rate": "palabras por segundo",
        "a_pitch_std": "desvio del tono", "e_music_max": "maxima probabilidad de musica", "a_rms_mean": "volumen medio",
        "e_music_mean": "probabilidad media de musica", "f_n_sentences": "numero de oraciones",
        "s_questions_pm": "preguntas por minuto", "s_second_person": "fraccion de palabras en 2da persona (tu/you)"}
GRP = {"k": "K palabras clave", "s": "S sentimiento", "a": "A audio/prosodia", "h": "H gancho", "f": "F flujo",
       "l": "L linguistica", "e": "E eventos de audio"}


def fmt(s):
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


cands = []
for v in ids:
    hp = cfg.path("interim") / "heatmaps" / f"{v}.json"
    bp = cfg.path("processed") / "features" / "bin" / f"{v}.parquet"
    tp = transcript_path(cfg, v, "base")
    if not (hp.exists() and bp.exists() and tp.exists()):
        continue
    m = json.loads(hp.read_text(encoding="utf-8"))
    df = pd.read_parquet(bp)
    st, en, vals = corrected_graph(m["heatmap"])
    S, E = df.start.to_numpy(), df.end.to_numpy()
    hot = heat_intervals((st, en, vals))
    raw = formula_raw(df, W)
    ok = (E - S) <= 45
    top = top_k_nms(S, E, np.where(ok, raw, raw.min() - 10), k=5)
    touch = [any(min(E[i], b) - max(S[i], a) > 0 for a, b in hot) for i in top]
    cands.append(dict(v=v, m=m, df=df, st=st, en=en, vals=vals, S=S, E=E, raw=raw, top=top, touch=touch))
n = len(cands)
hit1 = [c for c in cands if c["touch"][0]]
hit5 = sum(any(c["touch"]) for c in cands)
print(f"cohorte fresca 1 con datos: {n} | V1 sin curva: hit@5 {hit5}/{n} = {hit5 / n:.0%} | clip #1 toca pico: {len(hit1)}/{n} = {len(hit1) / n:.0%}")


def lang(c):
    return "es" if (c["m"].get("language") or "en").startswith("es") else "en"


def cat(c):
    return (c["m"].get("categories") or ["?"])[0]


from collections import Counter
print("distribucion de clips que tocan pico (de 5):", dict(sorted(Counter(sum(c["touch"]) for c in cands).items())))
for c in sorted(cands, key=lambda c: -sum(c["touch"]))[:8]:
    print(sum(c["touch"]), c["touch"], c["v"], lang(c), cat(c), c["m"]["title"][:50])
pool = sorted(cands, key=lambda c: (-sum(c["touch"]), c["v"]))
byid = {c["v"]: c for c in cands}
pick = [byid[v] for v in ("l_-igAkbVII", "UIC10AA34YU", "3KfZ30O2fEQ")]   # los de mas aciertos (4/5, 4/5, 5/5); 2 EN + 1 ES


def card(c):
    m, df, raw = c["m"], c["df"], c["raw"]
    tr = Transcript.load(transcript_path(cfg, c["v"], "base"))
    ws = np.array([w.start for w in tr.words])
    rk = df[list(W)].rank(pct=True).fillna(0.5)
    sc = calibrate(raw, TAB)
    pk = int(np.argmax(c["vals"]))
    out = [f'<section><h2>{html.escape(m["title"])}</h2><p class="meta">{html.escape(m["channel"])} · {fmt(m["duration"])} · '
           f'idioma {lang(c)} · categoria {html.escape(cat(c))} · '
           f'<a href="https://youtu.be/{c["v"]}" target="_blank">https://youtu.be/{c["v"]}</a></p>']
    out.append(f'<p>Pico real de YouTube: <b>{fmt(c["st"][pk])}-{fmt(c["en"][pk])}</b> (calor {c["vals"][pk]:.2f}). '
               f'Calor medio del video {c["vals"].mean():.2f}. Clips que tocan un pico: <b>{sum(c["touch"])}/5</b>. '
               'La curva NO se uso para elegir; solo para calificar despues.</p>')
    for r, i in enumerate(c["top"], 1):
        txt = "".join(w.text for w in tr.words[np.searchsorted(ws, c["S"][i]):np.searchsorted(ws, c["E"][i])]).strip()
        heat = c["vals"][(c["en"] > c["S"][i]) & (c["st"] < c["E"][i])].mean()
        ct = (rk.iloc[i] * pd.Series(W)).sort_values(key=abs, ascending=False)
        rows = "".join(
            f'<tr><td>{k}</td><td>{DESC.get(k, "")}</td><td>{df.iloc[i][k]:.3g}</td><td>{rk.iloc[i][k]:.0%}</td>'
            f'<td>{W[k]:+.3f}</td><td class="{"p" if ct[k] > 0 else "n"}">{ct[k]:+.3f}</td></tr>' for k in ct.index)
        allf = [k for k in df.columns if k[:2] in tuple(g + "_" for g in GRP) and k not in W
                and pd.api.types.is_numeric_dtype(df[k])]
        rk2 = df[allf].rank(pct=True).fillna(0.5)
        rows2 = "".join(f'<tr><td>{k}</td><td>{GRP[k[0]]}</td><td>{df.iloc[i][k]:.3g}</td><td>{rk2.iloc[i][k]:.0%}</td></tr>'
                        for k in allf)
        tag = "TOCA el pico real" if c["touch"][r - 1] else "fuera del pico"
        out.append(
            f'<div class="clip"><h3>Clip #{r} · {fmt(c["S"][i])}-{fmt(c["E"][i])} ({c["E"][i] - c["S"][i]:.0f} s) · '
            f'score {sc[i]}/99 · <span class="{"ok" if c["touch"][r - 1] else "no"}">{tag}</span> · calor real {heat:.2f}</h3>'
            f'<p><a href="https://youtu.be/{c["v"]}?t={int(c["S"][i])}" target="_blank">ver este momento en YouTube</a></p>'
            f'<p class="tx">“{html.escape(txt[:420])}…”</p>'
            f'<p><b>Por que se eligio:</b> su puntaje crudo es {raw[i]:.3f}, posicion {int((raw > raw[i]).sum()) + 1} de '
            f'{len(raw)} ventanas del video. Pesa lo que esta ventana tiene <i>de mas</i> respecto a las otras del mismo video '
            f'(percentil dentro del video).</p>'
            f'<table><tr><th>variable de la formula</th><th>que mide</th><th>valor</th><th>percentil en el video</th>'
            f'<th>peso</th><th>aporte</th></tr>{rows}</table>'
            f'<details><summary>Las demas {len(allf)} variables analizadas (no entran a la formula V0)</summary>'
            f'<table><tr><th>variable</th><th>grupo</th><th>valor</th><th>percentil en el video</th></tr>{rows2}</table>'
            f'</details></div>')
    out.append("</section>")
    return "".join(out)


css = """body{font:14px/1.5 Segoe UI,system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px;color:#1c1f26}
h1{font-size:22px}h2{font-size:18px;margin-bottom:2px}section{border:1px solid #d9dde4;border-radius:12px;padding:16px;margin:18px 0}
.meta{color:#667}.clip{border-top:1px solid #e6e9ef;margin-top:14px;padding-top:8px}.ok{color:#117a3a;font-weight:700}.no{color:#a33}
table{border-collapse:collapse;font-size:12.5px;margin:6px 0;width:100%}th,td{border:1px solid #e0e3e9;padding:3px 7px;text-align:left}
th{background:#f3f5f8}.p{color:#117a3a;font-weight:600}.n{color:#a33;font-weight:600}.tx{background:#f6f7fa;padding:8px;border-radius:8px}
.note{background:#fff7e0;border:1px solid #f0d98a;border-radius:10px;padding:10px 12px}"""
head = (
    f'<h1>3 ejemplos explicados: como el sistema elige clips SOLO con el contenido</h1>'
    f'<div class="note"><b>Aviso honesto:</b> estos 3 videos se escogieron <i>a proposito</i> entre los {n} de los videos NO usados para fijar la formula '
    f'(videos que el modelo nunca vio al entrenar) porque son de los que mas clips acierta (4 o 5 de 5). Distribucion real de aciertos por video (de 5 clips): '
    f'{dict(sorted(Counter(sum(c["touch"]) for c in cands).items()))}. Solo ~1 % de los videos llega a 4/5 o mas. En toda la cohorte, el metodo sin ver '
    f'la curva toca un pico con al menos uno de sus 5 clips en <b>{hit5}/{n} = {hit5 / n:.0%}</b> de los videos (azar ~30 %) y su '
    f'clip #1 en <b>{len(hit1)}/{n} = {len(hit1) / n:.0%}</b>. Metodo V1: formula de 17 pesos + solo clips &le; 45 s. '
    f'Score = percentil del puntaje crudo respecto a los top-10 de los videos de desarrollo.</div>'
    f'<p><b>Como se calcula:</b> cada variable se convierte en percentil <i>dentro del mismo video</i> (0 = la ventana mas baja, '
    f'100 % = la mas alta); puntaje = suma de (peso &times; percentil). Los pesos salen de la correlacion parcial de cada variable '
    f'con el calor real de YouTube en 121 videos de desarrollo (D-045). Las variables con peso negativo restan.</p>')
outdir = Path(r"D:\tesis\entregables\demo_3_ejemplos")
outdir.mkdir(parents=True, exist_ok=True)
page = (f'<!doctype html><html lang="es"><meta charset="utf-8"><title>3 ejemplos</title><style>{css}</style><body>'
        f'{head}{"".join(card(c) for c in pick)}</body></html>')
(outdir / "index.html").write_text(page, encoding="utf-8")
for c in pick:
    print(c["v"], "https://youtu.be/" + c["v"], c["m"]["title"][:60], "| toca", sum(c["touch"]), "/5")
print(outdir / "index.html")
