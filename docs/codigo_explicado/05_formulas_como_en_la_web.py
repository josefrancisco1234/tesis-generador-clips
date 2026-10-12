"""
05 - LAS FORMULAS, ARMADAS EN EL MISMO ORDEN QUE LA PAGINA WEB
===============================================================

La pagina muestra, por cada clip, tres cosas en este orden. Este archivo las sigue igual:

    PARTE A.  "¿Por que se eligio?"   -> score global con 17 variables           (esto DECIDE los clips)
    PARTE B.  "Desglose por variable" -> Hook, Flow y Value                      (esto EXPLICA el clip)
    PARTE C.  "Glosario"              -> formula de cada variable, por grupo     (K, S, A, H, F, L, E)

Cada funcion es la formula real del sistema, escrita de forma autocontenida (con datos simples) para poder leerla de
corrido. Donde vive el codigo de produccion queda indicado como  >> archivo:linea.

Para ver el resultado con numeros reales (video 1 de la demo):
    python docs/codigo_explicado/05_formulas_como_en_la_web.py
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]


# =====================================================================================================================
# PARTE A.  SCORE GLOBAL  (lo que decide que clips salen)
# =====================================================================================================================

# --- A1. Percentil dentro del video ------------------------------------------------------------------------------
def percentil_en_video(valores: pd.Series) -> np.ndarray:
    """A cada ventana le da su posicion (0..1) entre TODAS las ventanas del mismo video.

    Por que: cada video tiene su linea base (un podcast tranquilo vs un stand-up ruidoso). Lo que importa es cuanto
    destaca una ventana frente a sus hermanas. NaN se trata como 0.5 (neutro).
    >> src/ranking/retention.py:29  formula_raw
    """
    return valores.rank(pct=True).fillna(0.5).to_numpy()


# --- A2. Los 17 pesos ---------------------------------------------------------------------------------------------
# Salen de la correlacion parcial de Spearman de cada variable con el calor real de YouTube, controlando la posicion
# en el video, mediana entre 121 videos de desarrollo.  >> scripts/reverse_engineering_formula.py:118 y :342
PESOS_17 = json.loads((RAIZ / "models" / "formula_calor_D045.json").read_text(encoding="utf-8"))["weights"]
TABLA_CALIBRACION = json.loads((RAIZ / "models" / "formula_calor_D045.json").read_text(encoding="utf-8"))["calibrator"]


def puntaje_crudo(ventanas: pd.DataFrame, pesos: dict[str, float] = PESOS_17) -> np.ndarray:
    """raw(w) = suma_i  peso_i * percentil_i(w)      (17 terminos)

    Positivo suma, negativo resta. Cada termino es peso * percentil, y es lo que la web llama "aporte al score".
    >> src/ranking/retention.py:35
    """
    columnas = []
    for variable in pesos:                                              # las 17 variables de la tabla
        serie = ventanas[variable] if variable in ventanas else pd.Series(np.nan, index=ventanas.index)
        columnas.append(percentil_en_video(serie))
    return np.column_stack(columnas) @ np.array(list(pesos.values()))   # suma ponderada


def aporte_por_variable(ventanas: pd.DataFrame, i: int, pesos: dict[str, float] = PESOS_17) -> pd.DataFrame:
    """Lo que muestra la tabla "¿Por que se eligio?": valor, puntaje 0-99, peso y aporte de cada variable.

    >> src/ranking/retention.py:99  explain
    """
    filas = []
    for variable, peso in pesos.items():
        pct = float(percentil_en_video(ventanas[variable])[i])
        filas.append({"variable": variable, "valor": ventanas[variable].iloc[i], "puntaje_0_99": round(pct * 99),
                      "peso": peso, "aporte": peso * pct})
    return pd.DataFrame(filas)


# --- A3. Score 0-99 -----------------------------------------------------------------------------------------------
def score_0_99(raw: np.ndarray, tabla: list[float] = TABLA_CALIBRACION) -> np.ndarray:
    """Posicion del puntaje crudo dentro de 100 cuantiles de los 10 mejores clips de cada video de desarrollo.
    Un 90 significa "esta en el 10 % superior de los mejores clips conocidos".  (percentil, no probabilidad)
    >> src/ranking/retention.py:43
    """
    return np.clip(np.searchsorted(np.asarray(tabla), raw, side="right") - 1, 0, 99).astype(int)


# --- A4. Seleccion de los 5 clips ---------------------------------------------------------------------------------
def seleccionar_clips(inicio: np.ndarray, fin: np.ndarray, raw: np.ndarray, k: int = 5, max_clip_s: float = 45.0,
                      solape_max: float = 0.30) -> list[int]:
    """V1: ordena por raw, descarta lo que dure mas de 45 s y lo que solape > 30 % con un clip ya elegido.

    Por que: sin esto los 5 mejores serian la misma ventana repetida; los clips <= 45 s aciertan mas picos.
    >> src/ranking/retention.py:88  (limite de 45 s)   y   src/segment/candidates.py:87  nms
    """
    orden = np.argsort(-raw, kind="stable")
    elegidos: list[int] = []
    for i in orden:
        if fin[i] - inicio[i] > max_clip_s:
            continue
        solapa = False
        for j in elegidos:
            inter = max(0.0, min(fin[i], fin[j]) - max(inicio[i], inicio[j]))
            if inter / max(min(fin[i] - inicio[i], fin[j] - inicio[j]), 1e-9) > solape_max:
                solapa = True
                break
        if not solapa:
            elegidos.append(int(i))
        if len(elegidos) == k:
            break
    return elegidos


# --- A5. Validacion (aqui si se usa la curva de YouTube, y solo para calificar) -----------------------------------
def toca_pico(clip_ini: float, clip_fin: float, bins_ini: np.ndarray, bins_fin: np.ndarray, calor: np.ndarray) -> bool:
    """Un clip "toca el pico" si se solapa con algun bin con calor > 0.5 (maximo los 20 mas altos).
    El "4/5" de la pantalla es cuantos de los 5 clips cumplen esto.
    >> src/ranking/labels.py:21  gt_bins   y   scripts/web_clips.py  build_result
    """
    calor = calor.copy()
    calor[0] = calor[1]                                   # correccion del primer bin (como Rhapsody)
    calientes = np.nonzero(calor > 0.5)[0]
    calientes = calientes[np.argsort(calor[calientes])[::-1]][:20]
    return any(min(clip_fin, bins_fin[b]) - max(clip_ini, bins_ini[b]) > 0 for b in calientes)


# =====================================================================================================================
# PARTE B.  HOOK / FLOW / VALUE  (explican el clip; NO deciden la seleccion)
# =====================================================================================================================
# Cada dimension es la suma de los aportes SHAP de sus variables, convertida a 0-99.
#   Hook  = variables h_*                      Flow = variables f_*
#   Value = k_*, s_*, a_*, e_*, l_*, u_*
# >> src/features/builder.py:35 DIMENSIONS ; src/ranking/model.py:141 dimension_contributions ; src/pipeline.py:230 _dim_vars
DIMENSIONES = {"hook": ("h_",), "flow": ("f_",), "value": ("k_", "s_", "a_", "e_", "l_", "u_")}


def dimension_de(variable: str) -> str:
    for nombre, prefijos in DIMENSIONES.items():
        if variable.startswith(prefijos):
            return nombre
    return "otra"


def puntaje_dimension(aportes_shap: dict[str, float], dimension: str) -> float:
    """aporte de la dimension = suma de los aportes de sus variables (luego se calibra a 0-99)."""
    return sum(v for k, v in aportes_shap.items() if dimension_de(k) == dimension)


# =====================================================================================================================
# PARTE C.  GLOSARIO: LA FORMULA DE CADA VARIABLE, EN EL ORDEN DEL CODIGO (K, S, A, H, F, L, E)
# =====================================================================================================================
# Convenciones: `palabras` = lista de (inicio, fin, texto) de la ventana; `dur` = duracion en s; `minutos = dur/60`.
# >> src/features/builder.py  window_features: AQUI estan todas (K, S, A, H, F, L, E) en este mismo orden.
#    El numero de linea exacto de cada variable esta en scripts/var_loc.json y en la tabla de la web.

def tasa_por_minuto(n: float, minutos: float) -> float:
    return n / minutos if minutos > 0 else 0.0


# ------------------------------------------------------------------ K  Palabras clave  (builder.py:107-131)
def k_kvsum_mean(pesos_kvsum: list[float]) -> float:           # media del peso KVSUM de las palabras de contenido
    return float(np.mean(pesos_kvsum)) if pesos_kvsum else 0.0
# KVSUM:  w = log(tf+1) * log(N/PF) / norma     (por pasaje de 300 s)   >> src/keywords/extractor.py:59

def k_kvsum_max(pesos_kvsum: list[float]) -> float:            # el termino mas central de la ventana
    return float(max(pesos_kvsum, default=0.0))

def k_top_terms_pm(n_terminos_top: int, minutos: float) -> float:   # cuanto se habla del tema principal
    return tasa_por_minuto(n_terminos_top, minutos)

def k_content_ratio(n_contenido: int, n_palabras: int) -> float:    # densidad informativa
    return n_contenido / max(n_palabras, 1)

def k_novelty(ya_vistas: list[bool]) -> float:                 # 1 - fraccion de palabras ya dichas en los 300 s previos
    return float(1 - np.mean(ya_vistas)) if ya_vistas else 0.0

def k_keybert_sim(emb_ventana: np.ndarray, emb_video: np.ndarray) -> float:   # coseno con el tema central
    return float(emb_ventana @ emb_video / ((np.linalg.norm(emb_ventana) * np.linalg.norm(emb_video)) or 1.0))

def k_keyphrase_pm(n_frases_clave: int, minutos: float) -> float:   # frases-resumen KeyBERT por minuto
    return tasa_por_minuto(n_frases_clave, minutos)

def k_noun_chunks_ps(sintagmas_por_oracion: list[int]) -> float:    # densidad explicativa (spaCy)
    return float(np.mean(sintagmas_por_oracion)) if sintagmas_por_oracion else float("nan")


# ------------------------------------------------------------------ S  Sentimiento  (builder.py:133-152)
def s_polaridad(p_pos: np.ndarray, p_neg: np.ndarray) -> dict[str, float]:
    pol = p_pos - p_neg                                        # por oracion (DistilBERT multilingue)
    inten = np.abs(pol)
    return {"s_pos_mean": float(p_pos.mean()), "s_neg_mean": float(p_neg.mean()),
            "s_polarity_mean": float(pol.mean()), "s_polarity_std": float(pol.std()),
            "s_intensity_mean": float(inten.mean()), "s_intensity_max": float(inten.max())}

def s_questions_pm(texto: str, minutos: float) -> float:       # ( ? + ¿ ) por minuto
    return tasa_por_minuto(texto.count("?") + texto.count("¿"), minutos)

def fraccion_de(tokens: list[str], conjunto: set[str]) -> float:    # base de s_second_person, s_first_person, etc.
    return sum(t in conjunto for t in tokens) / max(len(tokens), 1)
# s_second_person = fraccion_de(tokens, {"tu","ti","you",...});  s_first_person = fraccion_de(tokens, {"yo","I",...})
# s_numbers = tokens que empiezan con digito / tokens;  s_intensifiers, s_hook_words: igual con su lista


# ------------------------------------------------------------------ A  Audio y prosodia  (builder.py, seccion A)
def a_rms(rms_z: np.ndarray, rms_z_previo_30s: np.ndarray) -> dict[str, float]:
    # rms_z = (log(volumen) - promedio del video) / desvio del video
    return {"a_rms_mean": float(rms_z.mean()), "a_rms_p95": float(np.percentile(rms_z, 95)),
            "a_rms_std": float(rms_z.std()),                                           # dinamica del volumen
            "a_energy_delta": float(rz_mean(rms_z) - rz_mean(rms_z_previo_30s))}       # subida vs los 30 s previos

def rz_mean(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else 0.0

def a_pitch(f0_semitonos: np.ndarray, voiced: np.ndarray) -> dict[str, float]:
    f0 = f0_semitonos[~np.isnan(f0_semitonos)]
    ok = len(f0) > 10
    return {"a_pitch_std": float(f0.std()) if ok else float("nan"),
            "a_pitch_range": float(np.percentile(f0, 95) - np.percentile(f0, 5)) if ok else float("nan"),
            "a_voiced_ratio": float(voiced.mean())}                                    # frames con voz / frames

def a_ritmo_y_pausas(inicios: np.ndarray, fines: np.ndarray, dur: float) -> dict[str, float]:
    minutos = dur / 60
    huecos = inicios[1:] - fines[:-1]                       # silencio entre palabras consecutivas
    return {"a_speech_rate": len(inicios) / dur,                                        # palabras por segundo
            "a_pause_ratio": float(np.clip(huecos, 0, None)[huecos > 0.3].sum() / dur),  # tiempo en pausas > 0.3 s
            "a_long_pauses_pm": tasa_por_minuto(float((huecos > 1.0).sum()), minutos)}   # pausas > 1 s por minuto


# ------------------------------------------------------------------ H  Gancho  (builder.py, seccion H, primeras 15 palabras)
def h_gancho(tokens_gancho: list[str], texto_gancho: str, primera_oracion: dict) -> dict[str, float]:
    return {"h_question": float("?" in texto_gancho),                          # abre con pregunta
            "h_number": float(any(t[:1].isdigit() for t in tokens_gancho)),    # abre con un dato
            "h_first_sentence_s": primera_oracion["duracion"],                 # aperturas cortas retienen mas
            "h_first_intensity": abs(primera_oracion["p_pos"] - primera_oracion["p_neg"])}   # emocion inicial
# h_second_person / h_hook_word / h_contrast: 1 si el arranque trae interpelacion / palabra gancho / contraste
# h_rms_onset: media del RMS de los primeros 3 s ;  h_first_keybert: coseno de la 1.ª oracion con el tema


# ------------------------------------------------------------------ F  Flujo  (builder.py:178-198)
def f_flujo(primera_palabra: str, ultima_oracion_completa: bool, coherencias: np.ndarray, emb_a: np.ndarray,
            emb_b: np.ndarray, n_oraciones: int, dur: float, muletillas: int, n_tokens: int) -> dict[str, float]:
    colgadas = {"eso", "el", "ella", "and", "but"}          # (lista corta de ejemplo; la real es por idioma)
    cos = emb_a @ emb_b / ((np.linalg.norm(emb_a) * np.linalg.norm(emb_b)) or 1.0)
    return {"f_starts_dangling": float(primera_palabra in colgadas),            # empieza a mitad de idea
            "f_ends_complete": float(ultima_oracion_completa),                  # termina en frase completa
            "f_coherence_mean": float(coherencias.mean()), "f_coherence_min": float(coherencias.min()),
            "f_topic_drift": float(1 - cos),                                    # cambio de tema dentro del clip
            "f_n_sentences": float(n_oraciones), "f_duration": dur,
            "f_filler_rate": muletillas / max(n_tokens, 1)}                     # fluidez


# ------------------------------------------------------------------ L  Linguistica  (builder.py, seccion L)
def l_linguistica(tokens: list[str], tiempo_hablado: float, dur: float) -> dict[str, float]:
    c = Counter(tokens)
    n = len(tokens)
    return {"l_entropy": -sum((v / n) * math.log2(v / n) for v in c.values()),   # Shannon: variedad lexica
            "l_type_token": len(c) / n,                                          # riqueza vs repeticion
            "l_silence": 1 - tiempo_hablado / max(dur, 1e-3),                    # silencio total del tramo
            "l_word_len": float(np.mean([len(t) for t in tokens]))}              # registro tecnico vs coloquial


# ------------------------------------------------------------------ E  Eventos de audio  (builder.py, seccion E)
def e_eventos(p_evento: np.ndarray, p_tras_el_clip: np.ndarray) -> dict[str, float]:
    # p_evento = probabilidad por segundo (PANNs CNN14) dentro de la ventana, para: laugh, applause, cheering, music, speech
    return {"max": float(p_evento.max()), "mean": float(p_evento.mean()),                 # e_<evento>_max / _mean
            "tras_el_clip": float(p_tras_el_clip.max()) if len(p_tras_el_clip) else float("nan")}   # e_laugh_after, ...


# =====================================================================================================================
# DEMO con datos reales: reproduce la tabla "¿Por que se eligio?" del video 1 de la demo
# =====================================================================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    DATOS = Path("D:/tesis_data/data")
    DEMO = ["l_-igAkbVII", "UIC10AA34YU", "3KfZ30O2fEQ"]            # los 3 videos de la demo
    for vid in (sys.argv[1:] or DEMO):
        ventanas = pd.read_parquet(DATOS / "processed" / "features" / "bin" / f"{vid}.parquet")
        raw = puntaje_crudo(ventanas)                                          # A2: suma de peso x percentil
        score = score_0_99(raw)                                                # A3: pasa a 0-99
        top = seleccionar_clips(ventanas["start"].to_numpy(), ventanas["end"].to_numpy(), raw)   # A4: 5 clips
        calor = json.loads((DATOS / "interim" / "heatmaps" / f"{vid}.json").read_text(encoding="utf-8"))["heatmap"]
        b_ini = np.array([b["start_time"] for b in calor])
        b_fin = np.array([b["end_time"] for b in calor])
        b_val = np.array([b["value"] for b in calor], float)
        print("=" * 100)
        print(f"VIDEO {vid}: {len(ventanas)} ventanas -> 5 clips elegidos SOLO con contenido")
        aciertos = 0
        for r, i in enumerate(top, 1):
            ok = toca_pico(ventanas.start.iloc[i], ventanas.end.iloc[i], b_ini, b_fin, b_val)      # A5 (solo para calificar)
            aciertos += ok
            marca = "TOCA PICO" if ok else "fuera"
            print(f"  Clip #{r}  {ventanas.start.iloc[i] / 60:5.1f}-{ventanas.end.iloc[i] / 60:5.1f} min   "
                  f"score {score[i]:2d}/99   raw {raw[i]:.3f}   {marca}")
        print(f"  -> {aciertos}/5 clips sobre un pico real de YouTube")
        print()
        i = top[0]
        t = aporte_por_variable(ventanas, i)
        t = t.assign(valor=t.valor.round(3), peso=t.peso.round(3), aporte=t.aporte.round(3))
        print("  Por que se eligio el clip #1 (17 variables):")
        for linea in t.to_string(index=False).splitlines():
            print("    " + linea)
        print(f"  suma de aportes = raw = {t.aporte.sum():.3f}")
        print()
