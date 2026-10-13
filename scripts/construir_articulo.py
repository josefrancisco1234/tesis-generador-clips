# -*- coding: utf-8 -*-
"""Rellena la plantilla A.3 del articulo (hasta EXPERIMENTACION) con el contenido real del proyecto."""
import copy
import sys
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC = Path(r"D:\tesis\entrada\A.3. Plantilla_Artículo de investigación-CIS_2026_RE EDITAR.docx")
OUT = Path(r"D:\tesis\entregables\Articulo_Investigacion_CIS2026_v9.docx")
FIG = Path(r"D:\tesis\entrada\fig_pipeline.png")

TITLE = "Selección automática de clips verticales con alto potencial de retención en videos conversacionales a partir de señales de audio y texto"
AUTOR = "José Francisco Núñez del Prado Vega"
ASESOR = "Angela Gabriela Maywa"

d = docx.Document(str(SRC))


class _Live:
    def __getitem__(self, i):
        return d.paragraphs[i]

    def __len__(self):
        return len(d.paragraphs)


P = _Live()


def find(prefix, start=0):
    ps = d.paragraphs
    for i in range(start, len(ps)):
        if ps[i].text.strip().startswith(prefix):
            return i
    raise KeyError(prefix)


def set_text(p, text, bold=None, italic=None, size=None):
    runs = p.runs
    if runs:
        runs[0].text = text
        for r in runs[1:]:
            r.text = ""
        r0 = runs[0]
    else:
        r0 = p.add_run(text)
    if bold is not None:
        r0.bold = bold
    if italic is not None:
        r0.italic = italic
    if size:
        r0.font.size = Pt(size)
    r0.font.color.rgb = docx.shared.RGBColor(0, 0, 0)
    return r0


def set_all_wt(p, first, rest=""):
    ts = p._p.xpath(".//w:t")
    for k, t in enumerate(ts):
        t.text = first if k == 0 else rest


# ----------------------------------------------------------------------------------------------- portada y encabezado
i = find("TÍTULO DEL TRABAJO")
set_text(P[i], TITLE.upper(), bold=True)
set_text(P[find("[Nombres y apellidos del alumno]")], AUTOR)
set_text(P[find("Código [Insertar")], "Código 20221730")
j = find("Asesor")
set_text(P[j + 1], ASESOR)
k = find("Haga clic aquí")
set_all_wt(P[k], "2026")

i = find("Título")
set_text(P[i], TITLE, bold=True, size=16)
j = find("El título debe ser")
set_text(P[j], "Un sistema abierto y auditable que elige y puntúa los momentos con mayor retención de un video largo", italic=True)
a1 = find("nombres y apellidos")
set_text(P[a1], AUTOR)
set_text(P[a1 + 1], "20220553@aloe.ulima.edu.pe")
a2 = find("nombres y apellidos", a1 + 1)
set_text(P[a2], f"{ASESOR} (asesora)")
set_text(P[a2 + 1], "")

# ----------------------------------------------------------------------------------------------- utilidades de insercion
HEADS = ["Resumen", "Palabras clave", "Abstract:", "Keywords:", "INTRODUCCIÓN", "TRABAJOS RELACIONADOS", "MARCO TEÓRICO",
         "METODOLOGÍA", "EXPERIMENTACIÓN", "RESULTADOS", "DISCUSIÓN", "CONCLUSIONES", "TRABAJOS FUTUROS", "REFERENCIAS"]


def body_template():
    j = find("Considerar 1 cara")
    return P[j]


def clear_between(head_text, next_text):
    d2 = docx.Document  # noqa
    ps = d.paragraphs
    h = next(n for n, p in enumerate(ps) if p.text.strip().startswith(head_text) and p.text.strip().upper() == p.text.strip())
    e = next(n for n, p in enumerate(ps) if n > h and p.text.strip().startswith(next_text))
    for p in ps[h + 1:e]:
        p._p.getparent().remove(p._p)
    return d.paragraphs[h], d.paragraphs[h + 1]


def new_par(before, text="", bold=False, italic=False, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True, space_after=6):
    p = before.insert_paragraph_before()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    if indent and align == WD_ALIGN_PARAGRAPH.JUSTIFY:
        pf.first_line_indent = Inches(0.3)
    if text:
        add_runs(p, text, bold, italic, size)
    return p


def add_runs(p, text, bold=False, italic=False, size=12):
    """Soporta **negrita** y _cursiva_ simples con marcadores ** y __."""
    import re
    parts = re.split(r"(\*\*.+?\*\*|__.+?__)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("__"):
            r = p.add_run(part[2:-2])
            r.italic = True
            r.bold = bold
        else:
            r = p.add_run(part)
            r.bold = bold
            r.italic = italic
        r.font.size = Pt(size)
        r.font.color.rgb = docx.shared.RGBColor(0, 0, 0)


def sub(before, text):
    return new_par(before, text, bold=True, size=12, indent=False, space_after=4)


def table_before(before, rows, widths, caption, header=True, size=9):
    cap = new_par(before, caption, bold=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=3)
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        t.style = d.styles["Table Grid"]
    except KeyError:
        pass
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.width = Inches(widths[c])
            cell.text = ""
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            run = para.add_run(str(val))
            run.font.size = Pt(size)
            run.bold = header and r == 0
    # borde fino por si no existe el estilo
    tbl = t._tbl
    tblPr = tbl.tblPr
    borders = tblPr.find(qn("w:tblBorders"))
    if borders is None:
        from docx.oxml import OxmlElement
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{side}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:color"), "808080")
            borders.append(el)
        tblPr.append(borders)
    before._p.addprevious(tbl)
    spacer = new_par(before, "", indent=False, space_after=4)
    return t


def code_before(before, caption, lines):
    cap = new_par(before, caption, bold=True, size=10, align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, space_after=2)
    for ln in lines:
        p = before.insert_paragraph_before()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Inches(0.2)
        r = p.add_run(ln)
        r.font.name = "Consolas"
        r.font.size = Pt(9)
    new_par(before, "", indent=False, space_after=4)


def fig_before(before, path, caption, width=6.3):
    p = before.insert_paragraph_before()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.page_break_before = True
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    new_par(before, caption, italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=8)


# ----------------------------------------------------------------------------------------------- figuras de flujo (tamano pagina)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG2 = Path(r"D:\tesis\entrada\fig_clips.png")
GRIS, SAL = "#EDEDED", "#FFD9CC"


def flow_big(path, steps, ncols, nrows, legend_l, legend_r, head, fig_w=6.3, fig_h=8.4, fs_t=11.5, fs_b=9.4, fs_o=8.6):
    """steps: lista ordenada (titulo, cuerpo, salida, color). Recorrido en serpiente; cajas grandes para leerse a tamano de hoja."""
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=230)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    m, top, bottom = 0.15, 0.7, 0.62
    gapx, gapy = 0.34, 0.5
    bw = (fig_w - 2 * m - (ncols - 1) * gapx) / ncols
    bh = (fig_h - top - bottom - (nrows - 1) * gapy) / nrows
    ax.text(fig_w / 2, fig_h - 0.3, head, fontsize=11, fontweight="bold", va="center", ha="center")
    pos = []
    for i in range(len(steps)):
        r, c = divmod(i, ncols)
        if r % 2 == 1:
            c = ncols - 1 - c
        pos.append((m + c * (bw + gapx), fig_h - top - (r + 1) * bh - r * gapy))
    for (t, body, out, col), (x, y) in zip(steps, pos):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.015,rounding_size=0.09", fc=col, ec="#444444", lw=1.1))
        ax.text(x + bw / 2, y + bh - 0.32, t, ha="center", va="center", fontsize=fs_t, fontweight="bold")
        ax.text(x + bw / 2, y + bh * 0.55, body, ha="center", va="center", fontsize=fs_b, color="#1a1a1a", linespacing=1.4)
        ax.plot([x + 0.14, x + bw - 0.14], [y + 0.60, y + 0.60], color="#999999", lw=0.7)
        ax.text(x + bw / 2, y + 0.30, out, ha="center", va="center", fontsize=fs_o, color="#6a2400", style="italic", linespacing=1.2)
    for i in range(len(steps) - 1):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if abs(y0 - y1) < 1e-6:
            a, b = ((x0 + bw, y0 + bh / 2), (x1, y1 + bh / 2)) if x1 > x0 else ((x0, y0 + bh / 2), (x1 + bw, y1 + bh / 2))
        else:
            a, b = (x0 + bw / 2, y0), (x1 + bw / 2, y1 + bh)
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.6, mutation_scale=16))
    ax.add_patch(FancyBboxPatch((m, 0.2), 0.26, 0.2, boxstyle="round,pad=0.0", fc=GRIS, ec="#444444", lw=0.8))
    ax.text(m + 0.36, 0.3, legend_l, fontsize=8.8, va="center")
    ax.add_patch(FancyBboxPatch((fig_w / 2 + 0.05, 0.2), 0.26, 0.2, boxstyle="round,pad=0.0", fc=SAL, ec="#444444", lw=0.8))
    ax.text(fig_w / 2 + 0.41, 0.3, legend_r, fontsize=8.8, va="center")
    fig.savefig(path)
    plt.close(fig)


ETL = [
    ("1 · Búsqueda", "50 consultas (26 ES, 24 EN).\nSolo videos con curva de\nre-reproducción;\nmáx. 2 por canal.", "manifest CSV · ≈ 770 videos", GRIS),
    ("2 · Curva", "yt-dlp descarga el heatmap:\n100 tramos con valor 0 a 1.\nSe corrige el primer tramo.", "un JSON por video", GRIS),
    ("3 · Audio", "Se descarga solo el audio\n(WAV), no el video\ncompleto.", "≈ 765 archivos · 66 GB", GRIS),
    ("4 · Transcripción", "Whisper local en GPU:\ncada palabra con inicio,\nfin y confianza; detecta\nel idioma.", "765 transcripciones", GRIS),
    ("5 · Variables", "71 variables por ventana\n(audio, texto y eventos);\ncada una pasa a percentil\ndentro del video.", "tabla por video · 678 videos", GRIS),
    ("6 · Etiquetas", "Calor medio del tramo,\ngrado 0 a 4 por cuantiles\ny marca de momento\ncaliente (> 0.5).", "solo para entrenar y calificar", SAL),
]
flow_big(FIG, ETL, 2, 3, "gris: dato o variable de entrada", "salmón: etiqueta, no entra al sistema",
         "ETL del conjunto de datos", fs_t=12.5, fs_b=10.4, fs_o=9.4)

PIPE = [
    ("1 · Video crudo", "URL o archivo.\nSe extrae el audio\ny se transcribe\ncon Whisper.", "palabras con tiempo", GRIS),
    ("2 · Oraciones", "Corte por puntuación,\npausas > 0.9 s o\n40 palabras.", "239 en un video\nde 50 min", GRIS),
    ("3 · Candidatos", "Ventanas solapadas\nde 15 a 90 s,\nalineadas a\noraciones.", "708 candidatos", GRIS),
    ("4 · Variables", "71 variables por\ncandidato; cada una\nse vuelve percentil\ndentro del video.", "tabla de variables", GRIS),
    ("5 · Puntaje", "Fórmula de 17 pesos:\nsuma de peso ×\npercentil.\nScore de 0 a 99.", "345 candidatos ≤ 45 s", GRIS),
    ("6 · Elegir 5", "Se ordena por\npuntaje; se descarta\nlo que solape\nmás de 30 %.", "5 clips", SAL),
    ("7 · Ajustes", "Corte de escena si\ncae en una pausa;\nmargen de 0.15 s.", "inicio y fin finales", SAL),
    ("8 · Recorte 9:16", "Reencuadre al\nhablante y subtítulos\nsincronizados\n(ffmpeg).", "mp4 vertical", SAL),
    ("9 · Explicación", "Aporte de cada\nvariable y desglose\nen Hook, Flow\ny Value.", "tabla por clip", SAL),
]
flow_big(FIG2, PIPE, 3, 3, "gris: todos los candidatos", "salmón: solo los clips elegidos",
         "De un video crudo a cinco clips verticales", fs_t=10.2, fs_b=8.5, fs_o=7.8)

# ----------------------------------------------------------------------------------------------- RESUMEN
RES = ("La producción de clips cortos a partir de grabaciones largas depende de la edición manual y de herramientas comerciales que no "
       "documentan cómo estiman el potencial de un segmento. Este trabajo propone un sistema abierto y auditable que, a partir del audio y la "
       "transcripción de un video conversacional, elige y puntúa de 0 a 99 los momentos con mayor probabilidad de retención. La referencia de "
       "entrenamiento y evaluación es la curva Most Replayed de YouTube, usada únicamente para calificar y nunca como entrada. Se construyó un "
       "conjunto propio de unos 770 videos en español e inglés, se descubrió una fórmula de 17 variables ponderadas (pausas, silencio, "
       "variación de tono y de volumen, cercanía al tema central) y se la comparó con modelos de aprendizaje bajo un protocolo de "
       "desarrollo, confirmación y cohortes frescas con hipótesis previa. En una cohorte fresca de 93 videos, el sistema acertó al menos un "
       "pico de retención con sus cinco clips en 47.3 % (fórmula), 54.8 % (clips de hasta 45 s) y 62.4 % (modelo de aprendizaje), frente a "
       "26.9 % del azar; en una segunda cohorte de 63 videos el modelo de aprendizaje alcanzó 49.2 % frente a 30.0 %. Las variantes de "
       "aprendizaje profundo, texto semántico, señal visual y ensamble no mejoraron de forma confirmada a la fórmula. El techo observado con "
       "solo contenido es de 50 a 69 %, mientras que usar la curva real llega a 85–95 %, pero esta no existe en un video aún no publicado.")
ABS = ("The production of short clips from long recordings relies on manual editing and on commercial tools that do not disclose how they "
       "estimate the potential of a segment. This work proposes an open, auditable system that, from the audio and transcript of a "
       "conversational video, selects and scores from 0 to 99 the moments most likely to retain the audience. The YouTube Most Replayed curve "
       "is used only as the training and evaluation reference and never as an input. A dataset of about 770 Spanish and English videos was "
       "built, a formula of 17 weighted variables was discovered (pauses, silence, pitch and loudness variation, closeness to the central "
       "topic), and it was compared with learning models under a development, confirmation and fresh-cohort protocol with prior "
       "hypotheses. On a fresh cohort of 93 videos, the five selected clips hit at least one retention peak in 47.3 % (formula), 54.8 % "
       "(clips up to 45 s) and 62.4 % (learning model) of the videos, against 26.9 % for chance; on a second cohort of 63 videos the "
       "learning model reached 49.2 % against 30.0 %. Deep sequence models, semantic text, visual signals and an ensemble did not improve "
       "on the formula in a confirmed way. The ceiling observed with content alone is 50 to 69 %, while using the real curve reaches 85–95 %, "
       "but that curve does not exist for an unpublished video.")
i = find("Resumen")
set_text(P[i + 1], RES, bold=False)
P[i + 1].paragraph_format.first_line_indent = None
for r in P[i + 1].runs[1:]:
    r.text = ""
# eliminar segundo parrafo guia si existe
nxt = d.paragraphs[i + 2]
if nxt.text.strip().startswith("Debe permitir"):
    nxt._p.getparent().remove(nxt._p)

i = find("Palabras clave")
set_text(d.paragraphs[i + 1], "detección de momentos destacados; retención de audiencia; aprendizaje de ranking; extracción de características de audio y texto; "
         "generación automática de clips; contenido conversacional; validación con hipótesis previa", bold=False)

i = find("Abstract:")
p = d.paragraphs[i]
for r in p.runs:
    r.text = ""
r = p.runs[0] if p.runs else p.add_run()
r.text = "Abstract: "
r.bold = True
r2 = p.add_run(ABS)
r2.bold = False
r2.font.size = Pt(12)
i = find("Keywords:")
p = d.paragraphs[i]
for r in p.runs:
    r.text = ""
r = p.runs[0] if p.runs else p.add_run()
r.text = "Keywords: "
r.bold = True
r2 = p.add_run("highlight detection; audience retention; learning-to-rank; audio and text feature extraction; automatic clip generation; "
               "conversational content; preregistered validation")
r2.bold = False
r2.font.size = Pt(12)

# ----------------------------------------------------------------------------------------------- INTRODUCCION
h, anchor = clear_between("INTRODUCCIÓN", "TRABAJOS RELACIONADOS")
anchor = [p for p in d.paragraphs if p.text.strip().startswith("TRABAJOS RELACIONADOS")][0]
INTRO = [
    "La creación de contenido audiovisual de formato corto (Shorts, Reels, TikTok) a partir de grabaciones extensas, como podcasts, entrevistas y "
    "transmisiones, se ha vuelto una práctica central de la distribución de contenido digital. Identificar y recortar los momentos con mayor "
    "potencial de retención se hace, en la práctica, de forma manual, lo que demanda tiempo y depende del criterio subjetivo de quien edita "
    "(Wang et al., 2024). Herramientas comerciales como Opus Clip y Vizard automatizan parte de este flujo, pero no publican la arquitectura ni "
    "los criterios con los que estiman el potencial de un segmento (Opus.pro, 2024; Vizard.ai, 2024). Esta opacidad impide auditar, replicar y "
    "mejorar el criterio de selección. El presente trabajo aborda ese problema desde el contenido de los videos conversacionales.",

    "El problema afecta a creadores independientes, medios, instituciones educativas y equipos de marketing que producen contenido hablado de "
    "larga duración y necesitan difundirlo en plataformas de video corto. Para ellos, el costo no es solo el tiempo de edición: una elección "
    "pobre de momentos reduce la retención y el alcance del contenido (Li et al., 2024). Además, las herramientas cerradas no permiten saber "
    "por qué un clip recibe un puntaje alto, lo que limita la confianza y el aprendizaje del equipo editorial. Los trabajos que asisten a la "
    "edición muestran que la intervención humana sigue siendo el cuello de botella del proceso (Barua et al., 2025; Wang et al., 2024).",

    "La literatura académica ha estudiado la detección de momentos destacados sobre todo con señales visuales y en dominios como deportes y "
    "video en primera persona (Badamdorj et al., 2021; Kwak et al., 2025). Los conjuntos Mr. HiSum y Rhapsody usan la curva de "
    "re-reproducción de YouTube como etiqueta débil de interés, el primero con videos generalistas y el segundo con episodios de podcast "
    "(Sul et al., 2023; Park et al., 2025). Rhapsody evidencia que el habla y su transcripción son la fuente principal en contenido "
    "conversacional, pero su propuesta se centra en un conjunto de datos y en modelos pesados. La limitación común es que ninguno entrega un "
    "sistema completo, explicable y reproducible que vaya del video al clip con su puntaje.",

    "Un segundo grupo de soluciones usa modelos de lenguaje para elegir o narrar segmentos. Lotus combina resumen extractivo y abstractivo, "
    "PodReels estudia los ingredientes de un buen teaser de podcast y HIVE propone un flujo de edición con comprensión narrativa multimodal "
    "(Barua et al., 2025; Wang et al., 2024; Wang et al., 2025). Estos enfoques mejoran la evaluación humana de los clips, pero dependen de "
    "modelos de lenguaje de gran tamaño, evalúan con juicios subjetivos y no contrastan su elección con una medida objetiva de retención "
    "de la audiencia. Por eso no responden a la pregunta de qué señales del contenido anticipan los momentos que la gente vuelve a ver.",

    "La brecha que se aborda es doble: falta un sistema abierto que elija y puntúe momentos con señales de audio y texto auditables, y falta "
    "evidencia validada, con protocolo estricto, de qué tan lejos se puede llegar solo con el contenido. La propuesta de este trabajo "
    "contribuye con: (i) un pipeline completo de video a clips verticales 9:16 con puntaje de 0 a 99; (ii) una fórmula abierta de 17 "
    "variables ponderadas, explicable por clip; (iii) un protocolo de desarrollo, confirmación y cohortes frescas con hipótesis previa; y (iv) "
    "un resultado negativo documentado: modelos más complejos no mejoraron de forma confirmada a la fórmula sencilla.",

    "El artículo se organiza así: la sección de trabajos relacionados discute la literatura sobre conjuntos de datos con re-reproducción, "
    "señales multimodales, generación de clips y predicción de engagement. La sección de marco teórico presenta los fundamentos del ranking, "
    "de la ponderación léxica y de la prosodia. La metodología describe el pipeline, el conjunto de datos, las variables y la fórmula. "
    "La sección de experimentación detalla el entorno, la configuración técnica, la partición de los datos y el protocolo de validación. "
    "Finalmente se presentan resultados, discusión, conclusiones y trabajos futuros.",
]
for t in INTRO:
    new_par(anchor, t)

# ----------------------------------------------------------------------------------------------- TRABAJOS RELACIONADOS
h, _ = clear_between("TRABAJOS RELACIONADOS", "MARCO TEÓRICO")
anchor = [p for p in d.paragraphs if p.text.strip().startswith("MARCO TEÓRICO")][0]
REL = [
    "Una primera línea de trabajos aborda la detección de momentos destacados con señales visuales, y evidencia que sumar audio mejora la "
    "detección. Badamdorj et al. (2021) aprenden de forma conjunta señales visuales y de audio y reportan una mejora consistente sobre "
    "modelos puramente visuales. Liu et al. (2022) llegan a una conclusión parecida con UMT: fusionar audio y video, sin consulta textual, "
    "supera a los modelos visuales. Kwak et al. (2025) proponen SPOT, que regresa la curva Most Replayed con una rama convolucional y un "
    "TimeSformer, y reconocen como limitación no usar audio ni texto. En conjunto, estos trabajos justifican incorporar el audio, pero no "
    "tratan el habla como señal principal ni producen un puntaje explicable para el editor.",

    "Una segunda línea construye conjuntos de datos a partir de la curva de re-reproducción que YouTube expone. Sul et al. (2023) publican "
    "Mr. HiSum con 31 892 videos generalistas, y Park et al. (2025) presentan Rhapsody con 13 364 episodios de podcast; este último "
    "informa que un embedding de habla autosupervisado (HuBERT) es la mejora de audio más relevante en su comparación. Lee et al. (2026) "
    "extienden el enfoque a videos deportivos de duración extrema. Estos trabajos coinciden en usar la re-reproducción como proxy de interés, "
    "decisión que esta investigación adopta y limita a su uso como regla de evaluación. A diferencia de ellos, aquí el conjunto es propio, "
    "bilingüe y se acompaña de un protocolo que separa desarrollo de confirmación.",

    "Una tercera línea usa modelos de lenguaje para seleccionar y narrar. Barua et al. (2025) combinan resumen extractivo y abstractivo para "
    "producir videos cortos, Wang et al. (2024) estudian con PodReels qué hace bueno a un teaser de podcast (duración, gancho, contexto) y "
    "Pennec et al. (2026) proponen segmentar en ventanas fijas de 20 s y pedirle a un modelo de lenguaje los mejores segmentos. Wang et al. "
    "(2025) amplían la idea con HIVE, que integra comprensión narrativa multimodal. La limitación compartida es que estas soluciones se "
    "validan con juicios humanos o con modelos cerrados, y no contrastan la elección contra una medida objetiva de retención de audiencia. "
    "Por ello, aportan criterios de edición, pero no una fórmula verificable de qué señales del contenido anticipan lo que la gente repite.",

    "Una cuarta línea estudia el engagement y la popularidad de los videos cortos. Li et al. (2024) definen el compromiso como retención, "
    "con las métricas NAWP y ECR, definición que este trabajo adopta para nombrar su puntaje como indicador de retención y no de viralidad. "
    "Cho et al. (2024) y Javed et al. (2025) predicen popularidad con atención multimodal y con calidad técnica antes de la publicación, y "
    "Li et al. (2025) proponen un modelo no supervisado que aprende representaciones de highlight sin etiquetas. En este conjunto, el "
    "habla vuelve a quedar como señal secundaria. La síntesis de los vacíos es que la literatura rara vez trata el habla como eje, no "
    "publica un sistema auditable de extremo a extremo y no reporta cuánto se logra con el contenido solo frente al azar y frente a la curva real.",
]
for t in REL:
    new_par(anchor, t)

# ----------------------------------------------------------------------------------------------- MARCO TEORICO
h, _ = clear_between("MARCO TEÓRICO", "METODOLOGÍA")
anchor = [p for p in d.paragraphs if p.text.strip().startswith("METODOLOGÍA")][0]
MT = [
    "**Aprendizaje de ranking.** En lugar de predecir un valor absoluto por segmento, el aprendizaje de ranking por listas optimiza el orden "
    "de los candidatos dentro de una misma consulta. LambdaRank optimiza directamente una métrica de orden, la ganancia acumulada descontada "
    "normalizada (NDCG), mediante gradientes que dependen del cambio en esa métrica al intercambiar dos elementos (Burges, 2010). Esto es "
    "adecuado aquí porque importa el orden relativo de las ventanas dentro de un video y no su valor entre videos de popularidad distinta. "
    "LightGBM implementa el objetivo sobre árboles de decisión con histogramas, lo que lo hace eficiente en una computadora personal (Ke et al., 2017).",

    "**Ponderación léxica y similitud temática.** El peso de un término dentro de un pasaje se adapta de la familia TF-IDF, con el logaritmo "
    "de la frecuencia del término y el de su rareza entre pasajes, normalizado por la norma del vector. KeyBERT complementa esa ponderación "
    "midiendo la similitud semántica entre un n-grama candidato y el tema global del documento mediante embeddings de oraciones "
    "(Grootendorst, 2020). Estas medidas dan las variables del grupo K, que indican cuánto se acerca una ventana al tema central del video.",

    "**Prosodia como proxy de énfasis.** La energía, la variación del tono y la estructura de pausas se han asociado con momentos de mayor "
    "carga comunicativa en la detección de momentos destacados (Badamdorj et al., 2021; Liu et al., 2022). Estas variables son baratas de "
    "calcular, no dependen de un modelo de lenguaje y se estandarizan respecto del propio video, para capturar énfasis relativo y no volumen "
    "absoluto. Los eventos de audio (risa, aplauso, música) se estiman con redes preentrenadas sobre AudioSet (Kong et al., 2020).",

    "**Transcripción y representaciones del habla.** La transcripción con marcas de tiempo por palabra se obtiene con Whisper, un modelo de "
    "reconocimiento de voz entrenado con supervisión débil a gran escala (Radford et al., 2023). Como alternativa de representación, "
    "HuBERT aprende representaciones del habla de forma autosupervisada (Hsu et al., 2021) y se evaluó como variable adicional. La "
    "explicabilidad del puntaje se apoya en contribuciones aditivas de tipo SHAP, que reparten la predicción entre las variables "
    "(Lundberg & Lee, 2017).",

    "**Validación estadística.** Para comparar métodos sobre los mismos videos se emplean pruebas pareadas: la prueba de rangos con signo "
    "de Wilcoxon (1945) para métricas continuas por video y la prueba de McNemar (1947) para el acierto binario. Las comparaciones "
    "múltiples se controlan con el umbral de Bonferroni, y la evidencia entre cohortes se combina con el método de Stouffer de forma "
    "descriptiva. Este aparato permite afirmar una mejora solo si supera el azar y el ruido entre videos.",
]
for t in MT:
    new_par(anchor, t)

# ----------------------------------------------------------------------------------------------- METODOLOGIA
h, _ = clear_between("METODOLOGÍA", "EXPERIMENTACIÓN")
anchor = [p for p in d.paragraphs if p.text.strip().startswith("EXPERIMENTACIÓN")][0]

sub(anchor, "A. Enfoque y diseño del estudio")
new_par(anchor, "La investigación es de enfoque cuantitativo y de tipo aplicado, con un diseño experimental que compara el sistema propuesto "
        "frente al azar y frente a modelos de aprendizaje más complejos. La población son videos de YouTube de contenido conversacional "
        "(podcasts, entrevistas, charlas) en español e inglés con duraciones de 15 a 75 minutos. Los datos provienen de fuentes primarias "
        "obtenidas con búsqueda dirigida en YouTube, e incluyen la curva de re-reproducción que la plataforma publica para cada video "
        "(Sul et al., 2023; Park et al., 2025). El sistema recibe el audio y su transcripción y no usa la curva para elegir; la curva se "
        "emplea únicamente para entrenar y calificar.")

sub(anchor, "B. Flujos de la metodología: dataset y generación de clips")
new_par(anchor, "La metodología tiene dos flujos. El primero es el ETL del conjunto de datos (Fig. 1): se buscan videos con curva de "
        "re-reproducción, se descarga la curva y el audio, se transcribe con marcas de tiempo por palabra, se calculan las variables por "
        "ventana y, por último, se pega a cada ventana su etiqueta de calor. Cada etapa guarda su resultado en disco y es reanudable, "
        "de modo que el conjunto puede crecer sin repetir trabajo.")
fig_before(anchor, FIG, "Fig. 1. ETL del conjunto de datos: de la búsqueda de videos a las ventanas con variables y etiquetas.")
new_par(anchor, "El segundo flujo es la generación de clips a partir de un video crudo (Fig. 2). El video se transcribe y se agrupa en oraciones; "
        "a partir de ellas se generan muchas ventanas candidatas solapadas, a cada una de las cuales se le calculan 71 variables. Una fórmula "
        "de 17 variables ponderadas puntúa a todos los candidatos, se eligen los cinco mejores sin solapar más de 30 % y solo esos pasan al "
        "ajuste de bordes, al recorte vertical con subtítulos y a la explicación de su puntaje. En un video de 50 minutos, por ejemplo, 239 "
        "oraciones producen 708 candidatos, de los cuales 345 cumplen el límite de 45 s y la condición de ser autocontenidos.")
fig_before(anchor, FIG2, "Fig. 2. De un video crudo a cinco clips verticales con puntaje y explicación.")

sub(anchor, "C. Conjunto de datos propio (YT-MR)")
new_par(anchor, "Siguiendo la receta de Rhapsody (Park et al., 2025), se construyó un conjunto propio de videos con curva de re-reproducción. "
        "Se usaron 50 consultas (26 en español y 24 en inglés) que cubren géneros como podcast, entrevista, conferencia, humor, deporte y "
        "ciencia; se filtró la duración, se conservaron solo los videos con curva disponible y se limitó a dos videos por canal para que un "
        "presentador no domine el conjunto. Posteriormente se amplió con cohortes nuevas descargadas con consultas distintas, que se "
        "describen en la sección de experimentación. La Tabla I resume la composición actual.")
table_before(anchor, [
    ["Elemento", "Cantidad"],
    ["Videos con curva y audio", "≈ 770"],
    ["Transcripciones (Whisper base)", "765"],
    ["Videos con las 71 variables calculadas", "678"],
    ["Idiomas", "Español e inglés"],
    ["Curva de re-reproducción", "100 tramos por video"],
    ["Máximo de videos por canal", "2"],
], [3.6, 2.3], "TABLA I. COMPOSICIÓN DEL CONJUNTO DE DATOS")

sub(anchor, "D. Variables")
new_par(anchor, "Por cada ventana se calculan 71 variables agrupadas en siete familias (Tabla II). Las variables de texto y de audio se calculan "
        "para todos los candidatos; los pasos costosos (recorte y explicación) se reservan para los clips elegidos. Cada variable se "
        "transforma en su percentil dentro del propio video, de modo que el sistema compara cada ventana contra sus hermanas y no contra un "
        "valor absoluto. La posición relativa en el video se excluye del modelo para evitar que aprenda que los finales siempre son calientes.")
import json as _json
sys.path.insert(0, "D:/tesis/desarrollo/scripts")
import glosario_variables as GL
_pesos = _json.loads(Path("D:/tesis/desarrollo/models/formula_calor_D045.json").read_text(encoding="utf-8"))["weights"]
_rows = [["Variable", "Familia", "Fórmula", "Peso"]]
for _k in GL.ORDER:
    _g, _f, _i = GL.G[_k]
    _rows.append([_k, _g.split(" ")[0], _f, f"{_pesos[_k]:+.2f}" if _k in _pesos else "—"])
table_before(anchor, _rows, [1.35, 0.5, 3.6, 0.5], "TABLA II. LAS 71 VARIABLES, SU FÓRMULA Y SU PESO EN LA FÓRMULA DE PUNTUACIÓN (— = no entra a la fórmula)", size=7)

sub(anchor, "E. Segmentación en candidatos")
new_par(anchor, "La transcripción se divide en oraciones mediante puntuación terminal, pausas mayores a 0.9 s o un máximo de 40 palabras. Sobre esa "
        "secuencia se generan candidatos alineados a los límites de oración, con duraciones objetivo de 20, 40 y 60 s y restringidos a un rango de 15 a "
        "90 s (Algoritmo 1). Esta decisión, frente a ventanas de duración fija (Pennec et al., 2026), garantiza que ningún clip comience o "
        "termine a mitad de una frase. En un video de 50 minutos con 239 oraciones se generan 708 candidatos solapados.")
code_before(anchor, "Algoritmo 1. Generación de candidatos", [
    "para cada oración i:",
    "  para cada duración objetivo t en {20, 40, 60} s:",
    "    j ← primera oración tal que fin(j) − inicio(i) ≥ t",
    "    si fin(j) − inicio(i) > 90 s: retroceder j hasta que ≤ 90 s",
    "    si 15 s ≤ duración ≤ 90 s: agregar ventana (inicio(i), fin(j))",
    "devolver ventanas únicas ordenadas por inicio",
])

sub(anchor, "F. Fórmula de puntuación")
new_par(anchor, "La fórmula combina 17 variables con pesos fijos que provienen de la correlación parcial de Spearman de cada variable con "
        "el calor real, controlando la posición en el video, tomada como mediana entre 121 videos de desarrollo. El puntaje crudo es la "
        "suma de peso por percentil de cada variable (Ecuación 1) y se calibra a una escala de 0 a 99 frente a los diez mejores clips de cada "
        "video de desarrollo. La Tabla III muestra las variables de mayor peso: pesan las pausas, el silencio y la variación de tono y "
        "volumen, lo que sugiere que los picos de retención se parecen a fronteras de segmento.")
new_par(anchor, "puntaje(w) = Σᵢ  pesoᵢ × percentilᵢ(w)          (1)", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
table_before(anchor, [
    ["Variable", "Peso", "Qué mide"],
    ["a_pause_ratio", "+0.18", "Tiempo en pausas de más de 0.3 s"],
    ["l_silence", "+0.16", "Silencio dentro de la ventana"],
    ["a_long_pauses_pm", "+0.15", "Pausas de más de 1 s por minuto"],
    ["a_rms_std", "+0.12", "Variación del volumen"],
    ["a_pitch_range", "+0.11", "Rango de tono de la voz"],
    ["a_voiced_ratio", "−0.11", "Fracción de habla continua"],
    ["k_keybert_sim", "+0.10", "Cercanía al tema central del video"],
    ["e_speech_mean", "−0.10", "Probabilidad media de habla pura"],
    ["Otras 9 variables", "±0.06 a ±0.10", "Tema, ritmo, música, preguntas, 2.ª persona"],
], [1.8, 1.1, 3.0], "TABLA III. VARIABLES DE LA FÓRMULA Y SU PESO")

sub(anchor, "G. Selección de clips y generación")
new_par(anchor, "Los candidatos se ordenan por puntaje y se descartan los que solapan más de 30 % con uno ya elegido (supresión de no máximos). "
        "En la variante V1 solo compiten ventanas de hasta 45 s y que sean autocontenidas, es decir, que no comiencen con una palabra colgada ni "
        "terminen a mitad de frase. Los bordes de cada clip se ajustan a un corte de escena cuando este cae en una pausa entre palabras, y el "
        "recorte vertical 9:16 sigue al hablante con subtítulos sincronizados. Cada clip se acompaña del aporte de sus variables y de una "
        "descomposición en gancho, fluidez y valor mediante contribuciones aditivas (Lundberg & Lee, 2017).")

sub(anchor, "H. Etiquetas y métricas")
new_par(anchor, "La curva de YouTube se corrige igualando el primer tramo al segundo, porque todos los usuarios ven el inicio. Se definen tres "
        "etiquetas: el calor medio de una ventana, un grado de 0 a 4 por cuantiles dentro del video y una marca de momento caliente para los "
        "tramos con valor mayor a 0.5 (máximo 20 por video). Las métricas son el acierto en los cinco clips (al menos uno toca un pico), el "
        "F1 temporal entre el tiempo de los clips y el de los picos, y el cociente entre el calor del clip y el calor medio del video. Todas se "
        "comparan con el azar, calculado por video según el número de tramos calientes, y se reporta además la matriz de confusión por ventana.")

# ----------------------------------------------------------------------------------------------- EXPERIMENTACION
h, _ = clear_between("EXPERIMENTACIÓN", "RESULTADOS")
anchor = [p for p in d.paragraphs if p.text.strip().startswith("RESULTADOS")][0]

sub(anchor, "A. Entorno experimental")
new_par(anchor, "Todos los experimentos se ejecutaron en una computadora personal con procesador Intel Core i5-10400 (2.90 GHz), 15.9 GB de memoria "
        "RAM y una GPU NVIDIA GeForce GTX 1660 SUPER de 6 GB, bajo Windows 10 y Python 3.11.9. La transcripción se ejecutó en la GPU con "
        "faster-whisper (modelo base en precisión completa para el conjunto de datos y modelo medium en int8 para inferencia). Los embeddings "
        "HuBERT se calcularon en precisión completa porque la mitad de precisión devolvía valores no numéricos. El proceso de adquisición y de "
        "cálculo de variables es reanudable y guarda cada etapa en disco, lo que permitió ampliar el conjunto sin repetir trabajo.")

sub(anchor, "B. Recolección del conjunto de datos y preprocesamiento")
new_par(anchor, "Los videos se obtuvieron con yt-dlp, que entrega el audio y la curva de re-reproducción de 100 tramos en un archivo por video. "
        "Se validó la coherencia entre la duración del audio, la transcripción y la tabla de variables, y se descartaron videos cuyo idioma no "
        "coincidía con el soportado (8 de 101 en la primera cohorte fresca). Las variables de texto se calcularon por ventana con modelos "
        "multilingües (sentimiento con DistilBERT multilingüe, similitud con embeddings de oraciones MiniLM y sintagmas nominales con spaCy). "
        "Las de audio se calcularon con librosa sobre cuadros de 32 ms, normalizadas con puntaje z respecto del video, y las de eventos con "
        "PANNs CNN14 a una muestra por segundo.")

sub(anchor, "C. Configuración técnica de los modelos")
new_par(anchor, "El modelo de aprendizaje de referencia (R0) es LightGBM con objetivo LambdaRank, truncamiento de NDCG en 20, tasa de "
        "aprendizaje de 0.05, 15 hojas, mínimo de 50 muestras por hoja, fracción de variables y de muestras de 0.7, regularización L2 de 10, "
        "300 iteraciones y semilla 42, con el video como grupo de consulta y la relevancia graduada de 0 a 4 como etiqueta. Las variables "
        "entran como percentil dentro del video. Se evaluaron además variantes de representación con valores absolutos, estadísticos del "
        "video y vecinos temporales (R4), la adición de componentes HuBERT reducidos con PCA a 32 dimensiones congelado, y un ensamble (E3).")

sub(anchor, "D. Partición del conjunto de datos y protocolo de validación")
new_par(anchor, "La partición separa desarrollo, confirmación y cohortes frescas (Tabla IV). Los pesos de la fórmula se fijaron solo con 121 "
        "videos de desarrollo y se congelaron antes de evaluar en 352 videos de confirmación, que se usaron una única vez. Las cohortes "
        "frescas (93 y 63 videos) se descargaron después, con consultas y géneros distintos, y cada prueba se corrió una sola vez con "
        "modelos y pesos congelados. Antes de cada prueba se escribió la hipótesis, la métrica primaria, el umbral y la prueba estadística. "
        "Para los modelos entrenados se usó validación cruzada de 10 particiones agrupada por canal, de modo que ningún canal aparece a la "
        "vez en entrenamiento y prueba.")
table_before(anchor, [
    ["Conjunto", "Videos", "Uso"],
    ["Desarrollo", "121", "descubrir variables y fijar los 17 pesos"],
    ["Desarrollo ampliado", "444", "entrenar los modelos de aprendizaje (validación por canal)"],
    ["Confirmación", "352", "prueba única de la fórmula con pesos congelados"],
    ["Cohorte fresca 1", "93", "réplica de la fórmula y de los modelos; géneros nuevos"],
    ["Cohorte fresca 2", "63", "confirmación de la familia R4 y del ensamble E3"],
], [1.8, 0.9, 3.2], "TABLA IV. PARTICIÓN DEL CONJUNTO DE DATOS")

sub(anchor, "E. Estadística y control de la exploración múltiple")
new_par(anchor, "Las diferencias entre métodos sobre el mismo video se contrastaron con la prueba de Wilcoxon unilateral para el F1 (Wilcoxon, 1945) "
        "y con la prueba exacta de McNemar para el acierto (McNemar, 1947). Cuando una familia de variantes se probó a la vez se aplicó el "
        "umbral de Bonferroni sobre las secundarias (0.0125 con cuatro comparaciones), y la evidencia de dos cohortes se combinó con el "
        "método de Stouffer de forma descriptiva. Las pruebas con menos de 40 videos se tratan como descriptivas. Este protocolo busca "
        "evitar que la exploración de muchas variantes sobreestime el desempeño.")

sub(anchor, "F. Experimentos realizados")
new_par(anchor, "Además de la fórmula y de LightGBM, se evaluaron familias de señal adicionales: una batería de nueve modelos (regresión, "
        "clasificación binaria, ridge, logística y ensambles), modelos de secuencia (BiGRU, TCN y Transformer entrenados con recortes de 40 "
        "tramos), embeddings de texto por tramo, la señal visual gruesa de las miniaturas de YouTube, un juez basado en un modelo de "
        "lenguaje, un léxico de palabras aprendido dentro de cada partición, la categoría del video y varias etiquetas alternativas. Todas se "
        "contrastaron contra la fórmula o contra R0 con la misma partición por canal, y sus resultados se reportan en la sección siguiente.")

sub(anchor, "G. Reproducibilidad")
new_par(anchor, "El código, la fórmula con sus pesos, la lista de videos y los documentos de decisión están en el repositorio de GitHub "
        "https://github.com/josefrancisco1234/tesis-generador-clips (privado; acceso por invitación). Cada decisión experimental se registró con fecha, hipótesis y resultado en un bitácora de 73 entradas, y "
        "las semillas, las listas de videos de cada cohorte y la huella del PCA congelado se guardaron para poder repetir las pruebas.")

sub(anchor, "H. Matriz de confusión por ventana")
new_par(anchor, "Para describir el comportamiento de la selección como clasificador de ventanas se construyó la matriz de confusión sobre "
        "los 557 videos que no se usaron para fijar la fórmula, con la variante V1 y solo contenido. Cada video se divide en 100 ventanas; el "
        "sistema elige cinco y la curva de re-reproducción define cuáles son calientes (valor mayor a 0.5, máximo 20 por video). Una ventana "
        "elegida y caliente es un verdadero positivo. De las 55 700 ventanas, 468 fueron elegidas y calientes, 2 317 elegidas y no calientes, "
        "4 039 calientes no elegidas y 48 876 ni elegidas ni calientes (Tabla V).")
table_before(anchor, [
    ["", "Caliente (pico real)", "No caliente"],
    ["Elegida (top 5)", "468", "2 317"],
    ["No elegida", "4 039", "48 876"],
], [1.8, 1.7, 1.5], "TABLA V. MATRIZ DE CONFUSIÓN POR VENTANA (557 VIDEOS, SOLO CONTENIDO)", size=10)
_pf = anchor.insert_paragraph_before()
_pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
_pf.add_run().add_picture("D:/tesis/entrada/fig_matriz.png", width=Inches(4.2))
new_par(anchor, "Fig. 3. Matriz de confusión por ventana de los 557 videos de evaluación (solo contenido).", italic=True, size=10,
        align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, space_after=8)
new_par(anchor, "A partir de esa matriz se calculan la precisión de la selección (16.8 %), que se compara con la tasa base de ventanas calientes "
        "(8.1 %), y la cobertura (10.4 %). Por la fuerte asimetría de clases, ya que el 92 % de las ventanas no son calientes, la métrica "
        "principal del protocolo es el acierto por video, es decir, que al menos una de las cinco ventanas elegidas caiga sobre un pico "
        "(310 de 557 videos, 55.7 %), y la matriz se usa como complemento.")

# ----------------------------------------------------------------------------------------------- REFERENCIAS (APA 7)
h, _ = clear_between("REFERENCIAS", "ZZZZ_NO_EXISTE") if False else (None, None)
ps = d.paragraphs
ri = next(n for n, p in enumerate(ps) if p.text.strip().startswith("REFERENCIAS"))
for p in ps[ri + 1:]:
    if p.text.strip() and not p.text.strip().startswith("["):
        break
    if p.text.strip().startswith("["):
        p._p.getparent().remove(p._p)
refs = [
    "Badamdorj, T., Rochan, M., Wang, Y., & Cheng, L. (2021). Joint visual and audio learning for video highlight detection. En Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 8127–8137).",
    "Barua, A., Benharrak, K., Chen, M., Huh, M., & Pavel, A. (2025). Lotus: Creating short videos from long videos with abstractive and extractive summarization. En Proceedings of the 30th International Conference on Intelligent User Interfaces (pp. 967–981). https://doi.org/10.1145/3708359.3712090",
    "Burges, C. J. C. (2010). From RankNet to LambdaRank to LambdaMART: An overview (Informe técnico MSR-TR-2010-82). Microsoft Research.",
    "Cho, M., Jeong, D., & Park, E. (2024). AMPS: Predicting popularity of short-form videos using multi-modal attention mechanisms in social media marketing environments. Journal of Retailing and Consumer Services, 78, 103778. https://doi.org/10.1016/j.jretconser.2024.103778",
    "Grootendorst, M. (2020). KeyBERT: Minimal keyword extraction with BERT [Software]. Zenodo. https://doi.org/10.5281/zenodo.4461265",
    "Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29, 3451–3460.",
    "Javed, A., Abid, N., Shoaib, M., Shahzad, M. F., Sabah, F., & Sarwar, R. (2025). A framework to predict the quality of a video for popularity on social media. Engineering Reports. https://doi.org/10.1002/eng2.70250",
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.",
    "Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28, 2880–2894.",
    "Kwak, S., Lee, J., & Park, S. (2025). The effective highlight-detection model for video clips using spatial–perceptual. Electronics, 14(18), 3640. https://doi.org/10.3390/electronics14183640",
    "Lee, D., Ki, Y., Kang, J., & Kim, T. (2026). SVHighlights: Towards extremely long sport video highlight detection. arXiv:2606.06926.",
    "Li, D., Li, W., Lu, B., Li, H., Ma, S., Krishnan, G., & Wang, J. (2024). Delving deep into engagement prediction of short videos. En Computer Vision – ECCV 2024 (pp. 289–306). Springer. https://doi.org/10.1007/978-3-031-72949-2_17",
    "Li, T., Sun, Z., & Xiao, X. (2024). Unsupervised modality-transferable video highlight detection with representation activation sequence learning. IEEE Transactions on Image Processing, 33, 1911–1922. https://doi.org/10.1109/TIP.2024.3372469",
    "Liu, Y., Li, S., Wu, Y., Chen, C. W., Shan, Y., & Qie, X. (2022). UMT: Unified multi-modal transformers for joint video moment retrieval and highlight detection. En Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 3032–3041). https://doi.org/10.1109/CVPR52688.2022.00305",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.",
    "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2), 153–157.",
    "Opus.pro. (2024). Opus Clip: AI video clipping tool. https://www.opus.pro",
    "Park, Y., Diwan, A., Harwath, D., & Choi, E. (2025). Rhapsody: A dataset for highlight detection in podcasts. Conference on Language Modeling (COLM).",
    "Pennec, G., Liu, Z., Asher, N., Muller, P., & Chen, N. F. (2026). Minimal clips, maximum salience: Long video summarization via key moment extraction. En Proceedings of the 16th International Workshop on Spoken Dialogue System Technology (pp. 205–221).",
    "Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. En Proceedings of the 40th International Conference on Machine Learning (ICML).",
    "Sul, J., Han, J., & Lee, J. (2023). Mr. HiSum: A large-scale dataset for video highlight detection and summarization. Advances in Neural Information Processing Systems, 36, Datasets and Benchmarks Track.",
    "Vizard.ai. (2024). Vizard: AI-powered video clipping. https://vizard.ai",
    "Wang, S., Ning, Z., Truong, A., Dontcheva, M., Li, D., & Chilton, L. B. (2024). PodReels: Human-AI co-creation of video podcast teasers. En Proceedings of the ACM Designing Interactive Systems Conference (pp. 958–974). https://doi.org/10.1145/3643834.3661591",
    "Wang, X., Li, X., Wei, Y., Song, Y., et al. (2025). From long videos to engaging clips: A human-inspired video editing framework with multimodal narrative understanding. En Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track (pp. 2764–2781). https://doi.org/10.18653/v1/2025.emnlp-industry.185",
    "Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80–83.",
    "Wu, Y., et al. (2025). Repurpose-10K: A dataset for long video repurposing. Proceedings of the AAAI Conference on Artificial Intelligence.",
]
ps = d.paragraphs
ri = next(n for n, p in enumerate(ps) if p.text.strip().startswith("REFERENCIAS"))
last = ps[-1] if ri + 1 >= len(ps) else None
tail = ps[ri + 1] if ri + 1 < len(ps) else None
for t in refs:
    if tail is not None:
        p = new_par(tail, t, size=10, indent=False, space_after=4)
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
    else:
        p = d.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        add_runs(p, t, size=10)

# --- encabezados (paginas pares e impares) y titulo de portada mas compacto
SHORT = "Selección automática de clips con alto potencial de retención"
for sec in d.sections:
    for name in ("header", "even_page_header", "first_page_header"):
        h = getattr(sec, name)
        for p in h.paragraphs:
            for t in p._p.xpath(".//w:t"):
                if "Nombre del Trabajo" in t.text:
                    t.text = t.text.replace("Nombre del Trabajo [páginas impares]", SHORT)
                if "X. Xxxxxx" in t.text:
                    t.text = t.text.replace("X. Xxxxxx [Inicial del nombre y apellido – Páginas pares]", "J. F. Núñez del Prado Vega")
cover = next(p for p in d.paragraphs if p.text.strip().startswith("SELECCIÓN AUTOMÁTICA DE CLIPS"))
for r in cover.runs:
    r.font.size = Pt(18)

OUT.parent.mkdir(parents=True, exist_ok=True)
d.save(str(OUT))
print("guardado", OUT)
