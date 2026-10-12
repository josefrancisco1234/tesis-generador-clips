"""
PASO 3 — ETIQUETAR: convertir la curva Most Replayed en la "respuesta correcta"
===================================================================================
Codigo real de: src/ranking/labels.py

Copia COMENTADA, no se ejecuta. El codigo real vive en src/ranking/labels.py.

Entrada:  el heatmap.json del video (100 bins con start_time/end_time/value) +
          la lista de candidatos (Window) del paso 1.
Salida:   4 columnas nuevas en la tabla: y_mr, y_gt_overlap, y_relevant, y_grade.
          y_grade es la que realmente usa LightGBM para entrenar (ver paso 4).
"""
import numpy as np
import pandas as pd


def corrected_graph(heatmap: list[dict]):
    # heatmap: la lista de 100 dicts tal cual viene de yt-dlp, p.ej.
    #   {"start_time": 3172.2, "end_time": 3207.41, "value": 1.0}
    starts = np.array([b["start_time"] for b in heatmap], dtype=float)
    ends = np.array([b["end_time"] for b in heatmap], dtype=float)
    vals = np.array([b["value"] for b in heatmap], dtype=float)

    if len(vals) > 1:
        vals = vals.copy()
        # CORRECCION DEL SESGO DEL PRIMER BIN (igual que el paper Rhapsody):
        # el bin 0 (el inicio del video) casi siempre sale artificialmente alto
        # porque TODO el mundo pasa por ahi al empezar a ver el video, sin que
        # signifique que el contenido sea interesante. Se "tapa" ese valor con
        # el del bin 1, que es mas representativo.
        vals[0] = vals[1]

    return starts, ends, vals
    # En nuestro video de ejemplo: vals[0] paso de 0.8270 a 0.2316 (el valor de vals[1]).


def gt_bins(values, max_n=20, threshold=0.5):
    # "bins highlight" segun la definicion de Rhapsody: cualquier bin con
    # value > 0.5 (threshold), como maximo 20 por video (max_n), ordenados
    # de mayor a menor. Esto se usa para la etiqueta BINARIA (y_relevant),
    # no para la graduada (y_grade, que es la que importa para entrenar).
    idx = np.nonzero(values > threshold)[0]
    return idx[np.argsort(values[idx])[::-1]][:max_n].tolist()


def _overlaps(w, starts, ends):
    # Cuantos segundos se solapan el candidato "w" con cada uno de los 100 bins.
    # clip(..., 0, None) evita solapamientos negativos (cuando no hay overlap).
    return np.clip(np.minimum(ends, w.end) - np.maximum(starts, w.start), 0, None)


def window_mr(w, starts, ends, vals):
    # Promedio del heatmap PONDERADO por cuantos segundos cubre el candidato
    # de cada bin. Si el candidato cae justo en el pico (bin con value=1.0),
    # su y_mr sale cerca de 1.0. Si cae en una zona plana, sale bajo.
    ov = _overlaps(w, starts, ends)
    return float((ov * vals).sum() / ov.sum()) if ov.sum() > 0 else np.nan


def add_labels(df, windows, heatmap, max_n, threshold, n_grades, relevant_overlap):
    # df:       la tabla de features ya armada (una fila por candidato, paso 2)
    # windows:  la misma lista de Window del paso 1 (en el mismo orden que df)
    # heatmap:  los 100 bins crudos de ESTE video

    starts, ends, vals = corrected_graph(heatmap)   # curva ya corregida
    gt = gt_bins(vals, max_n, threshold)             # bins "highlight" (para y_relevant)

    df = df.copy()

    # y_mr: el numero crudo (0 a 1) de que tan "replayed" es cada candidato
    df["y_mr"] = [window_mr(w, starts, ends, vals) for w in windows]

    # y_gt_overlap: que fraccion del TIEMPO del candidato cae dentro de un bin
    # "highlight" (usado para comparar contra clips elegidos por humanos, D-023)
    df["y_gt_overlap"] = [window_gt_overlap(w, starts, ends, gt) for w in windows]

    # y_relevant: version binaria (1 si al menos el 50% del candidato -ver
    # relevant_overlap- cae en zona highlight, 0 si no)
    df["y_relevant"] = (df["y_gt_overlap"] >= relevant_overlap).astype(int)

    # y_grade: LA ETIQUETA QUE REALMENTE ENTRENA AL MODELO (ver grade_within abajo)
    df["y_grade"] = grade_within(df["y_mr"], n_grades)

    return df


def grade_within(y: pd.Series, n_grades: int) -> pd.Series:
    """Relevancia graduada 0..n-1 por cuantiles DENTRO de un mismo video."""
    # ESTO ES CLAVE: no se compara y_mr entre videos distintos (un video viral
    # tiene todo mas "alto" de forma pareja, eso no serviria). Se compara SOLO
    # contra los otros ~400 candidatos del MISMO video.

    # .rank(method="first") le da a cada candidato su posicion de menor a mayor
    # y_mr dentro del video (1, 2, 3, ..., 400). Restamos 1 y dividimos entre el
    # total -> queda un numero entre 0 (el peor) y casi 1 (el mejor) del video.
    r = (y.rank(method="first") - 1) / max(y.notna().sum(), 1)

    # Se multiplica por n_grades (5 por defecto, osea grados 0,1,2,3,4) y se
    # trunca a entero. El candidato con el y_mr mas alto del video cae en el
    # grado 4 (el quintil superior); el mas bajo, en el grado 0.
    return np.minimum((r * n_grades).fillna(0).astype(int), n_grades - 1).astype(int)


# ------------------------------------------------------------------------------
# EJEMPLO REAL (mismo candidato del minuto 53 de -Gzz2LZGowU):
#   y_mr       = 0.982   (esta casi exacto sobre el pico de la curva)
#   y_relevant = 1       (si, cae en zona "highlight")
#   y_grade    = 4       (el grado mas alto posible: es de los mejores 20% de
#                         los 400 candidatos de ESE video, no de todo el dataset)
# ------------------------------------------------------------------------------
