# Mapa: dónde está cada fórmula en el código (y el flujo para sustentarlo)

Cómo usarlo: sigue el flujo de arriba hacia abajo. Cada bloque dice **qué fórmula es**, **en qué archivo y línea vive**
(clic en el enlace abre el archivo en esa línea) y trae el código real con comentarios `//` para leerlo en voz alta.

---

## 0. Flujo completo en una pantalla

```
URL / video
  │  src/pipeline.py  run()                       <- orquesta todo
  ├─ 1 descargar        src/ingest/youtube.py     download_video
  ├─ 2 transcribir      src/transcribe/whisper.py transcribe          (palabras con tiempos)
  ├─ 3 segmentar        src/segment/sentences.py + candidates.py      (candidatos de 15-45 s, alineados a oración)
  ├─ 4 variables        src/features/builder.py   window_features     (68 variables por candidato)
  ├─ 5 puntuar          src/ranking/retention.py  apply               (F2 + F4: fórmula de 17 pesos y score 0-99)
  ├─ 6 elegir 5         src/segment/candidates.py nms                 (F5: sin solapar > 30 %)
  ├─ 7 explicar         src/ranking/retention.py  explain             (aporte de las 17 variables)
  │                     src/pipeline.py          _dim_vars            (Hook/Flow/Value con SHAP)
  └─ 8 recortar 9:16    src/render/export.py      export_clip         (ffmpeg + subtítulos)

Offline (cómo se obtuvieron los pesos):  scripts/reverse_engineering_formula.py  -> F3 (rho parcial)
                                         scripts/build_formula_model.py          -> models/formula_calor_D045.json
Validación (usa la curva SOLO para calificar): src/ranking/labels.py + scripts/reverse_engineering_formula.py -> F6
```

---

## F1. Percentil dentro del video

**Archivo:** [src/ranking/retention.py:29](../src/ranking/retention.py#L29) · `formula_raw`

```python
def formula_raw(df, weights):
    cols = []
    for c in weights:                                   # // recorre las 17 variables de la fórmula
        s = df[c] if c in df else pd.Series(np.nan, index=df.index)   # // si falta la columna, queda en NaN
        cols.append(s.rank(pct=True).fillna(0.5).to_numpy())          # // F1: percentil DENTRO del video (0..1); NaN -> 0.5 (neutro)
    return np.column_stack(cols) @ np.array(list(weights.values()))   # // F2: suma ponderada (ver abajo)
```

**Qué decir:** "cada variable se compara contra las otras ventanas del *mismo* video; así un podcast tranquilo y un stand-up ruidoso
quedan en la misma escala".

---

## F2. Puntaje crudo (fórmula V0, 17 pesos)

**Archivo:** [src/ranking/retention.py:35](../src/ranking/retention.py#L35) (la última línea de `formula_raw`)
**Pesos:** [models/formula_calor_D045.json](../models/formula_calor_D045.json)

```
raw(w) = Σ peso_i · percentil_i(w)       // 17 términos; los pesos son los del JSON
```

**Qué decir:** "es una combinación lineal auditable: cada variable aporta peso × percentil, y se puede leer cuánto sumó cada una".

---

## F3. De dónde salen los pesos (rho parcial de Spearman)

**Archivos:**
- [scripts/reverse_engineering_formula.py:118](../scripts/reverse_engineering_formula.py#L118) · `partial_spearman`
- [scripts/reverse_engineering_formula.py:226](../scripts/reverse_engineering_formula.py#L226) · se calcula por variable y por video
- [scripts/reverse_engineering_formula.py:266](../scripts/reverse_engineering_formula.py#L266) · mediana entre videos
- [scripts/reverse_engineering_formula.py:342](../scripts/reverse_engineering_formula.py#L342) · `formula_calor` (los pesos finales)
- [scripts/build_formula_model.py](../scripts/build_formula_model.py) · guarda `models/formula_calor_D045.json`

```python
rho[t].loc[f, v] = partial_spearman(x, y, data[v]["pos"])   # // para cada variable f y video v: correlación con el calor y,
                                                           # // quitando el efecto de la posición en el video
r[f"{t}_rho_parcial_mediano"] = float(vals.median())        # // mediana entre videos: robusta a videos raros
"formula_calor": {k: float(x) for k, x in w_heat...}        # // esos números son los pesos de F2
```

**Qué decir:** "controlo la posición porque los inicios y finales son calientes por sí solos; sin eso premiaría solo estar al principio.
Los pesos se fijaron con 121 videos y se congelaron antes de probar en videos nuevos".

---

## F4. Score 0–99 (calibración)

**Archivo:** [src/ranking/retention.py:43](../src/ranking/retention.py#L43) · `calibrate`

```python
def calibrate(raw, table):
    # // table = 100 cuantiles de los top-10 clips de cada video de desarrollo
    return np.clip(np.searchsorted(np.asarray(table), raw, side="right") - 1, 0, 99).astype(int)
    # // posición de raw dentro de esa tabla = score. 90 => está en el 10 % superior de los mejores clips conocidos
```

---

## F5. Elección de los 5 clips (NMS) y límite de 45 s (V1)

**Archivos:**
- [src/ranking/retention.py:63](../src/ranking/retention.py#L63) · `apply` (aplica la fórmula y el límite `max_clip_s`)
- [src/ranking/retention.py:88](../src/ranking/retention.py#L88) · en `apply`: `ok = (duración <= max_clip_s)` (V1)
- [src/segment/candidates.py:87](../src/segment/candidates.py#L87) · `nms`
- [src/pipeline.py:164](../src/pipeline.py#L164) · llamada a `nms` con `output_overlap`

```python
ok = ((df["end"] - df["start"]).to_numpy() <= float(max_clip_s))   # // V1: solo clips cortos (<= 45 s)
df["raw"] = np.where(ok, base, float(base.min()) - 10.0)           # // los que no cumplen quedan al final
keep = nms(wins, df["raw"].to_numpy(), overlap, top_n, mode="iomin")   # // ordena por raw; descarta lo que solape >30 % con uno mejor
```

---

## F6. Métricas y validación (aquí SÍ se usa la curva, solo para calificar)

**Archivos:**
- [src/ranking/labels.py:10](../src/ranking/labels.py#L10) · `corrected_graph` (bin 0 = bin 1, como Rhapsody)
- [src/ranking/labels.py:21](../src/ranking/labels.py#L21) · `gt_bins` (picos: valor > 0.5, máximo 20)
- [src/ranking/labels.py:31](../src/ranking/labels.py#L31) · `window_mr` (calor medio de una ventana)
- [scripts/reverse_engineering_formula.py:85](../scripts/reverse_engineering_formula.py#L85) · `top_k_nms`
- [scripts/reverse_engineering_formula.py:77](../scripts/reverse_engineering_formula.py#L77) · `f1_temporal`
- [scripts/reverse_engineering_formula.py:176](../scripts/reverse_engineering_formula.py#L176) · `heat_intervals`
- [scripts/web_clips.py](../scripts/web_clips.py) · `build_result`: marca cada clip como "toca pico" y cuenta los 4/5

```python
idx = np.nonzero(values > 0.5)[0]                      # // bins "calientes" del mapa de YouTube
touch = any(min(clip.end, b) - max(clip.start, a) > 0 for a, b in hot)   # // el clip "toca" un pico si se solapa con alguno
hits = sum(touch for clip in clips)                    # // el "4/5" de la pantalla
```

---

## Variables: dónde se calcula cada una

Todas se calculan por ventana en [src/features/builder.py:97](../src/features/builder.py#L97) · `window_features`
(**todas las variables, K, S, A, H, F, L y E, se calculan ahora en este mismo archivo**; audio.py y extra.py solo aportan los cuadros de audio y los eventos PANNs).

| Variable (en el ORDEN en que se calcula en builder.py) | Línea | Peso | Cálculo |
|---|---|---|---|
| `k_keybert_sim` | [builder.py:129](../src/features/builder.py#L129) | +0.102 | coseno(emb. de la ventana, emb. del video) |
| `k_noun_chunks_ps` | [builder.py:134](../src/features/builder.py#L134) | -0.098 | sintagmas nominales por oración |
| `s_questions_pm` | [builder.py:148](../src/features/builder.py#L148) | +0.058 | (? + ¿) / min |
| `s_second_person` | [builder.py:151](../src/features/builder.py#L151) | +0.056 | tokens en 2.ª persona / tokens |
| `a_rms_mean` | [builder.py:171](../src/features/builder.py#L171) | -0.082 | media del RMS (z) |
| `a_rms_std` | [builder.py:173](../src/features/builder.py#L173) | +0.122 | desvío del volumen RMS (z) |
| `a_pitch_std` | [builder.py:174](../src/features/builder.py#L174) | +0.089 | desvío de f0 |
| `a_pitch_range` | [builder.py:175](../src/features/builder.py#L175) | +0.112 | p95(f0) − p5(f0) |
| `a_voiced_ratio` | [builder.py:176](../src/features/builder.py#L176) | -0.107 | frames con voz / frames |
| `a_speech_rate` | [builder.py:178](../src/features/builder.py#L178) | -0.098 | n_palabras / dur |
| `a_pause_ratio` | [builder.py:181](../src/features/builder.py#L181) | +0.180 | suma(huecos entre palabras > 0.3 s) / dur |
| `a_long_pauses_pm` | [builder.py:182](../src/features/builder.py#L182) | +0.148 | n(huecos > 1 s) / min |
| `f_n_sentences` | [builder.py:218](../src/features/builder.py#L218) | +0.060 | número de oraciones |
| `l_silence` | [builder.py:238](../src/features/builder.py#L238) | +0.163 | 1 − tiempo_hablado / dur |
| `e_speech_mean` | [builder.py:251](../src/features/builder.py#L251) | -0.101 | prob. media de «habla» (PANNs) |
| `e_music_max` | [builder.py:251](../src/features/builder.py#L251) | +0.083 | máx. prob. de música |
| `e_music_mean` | [builder.py:251](../src/features/builder.py#L251) | +0.067 | prob. media de música |

Otras fórmulas con su archivo:

| Fórmula | Archivo |
|---|---|
| KVSUM `w = log(tf+1)·log(N/PF) / norma` | [src/keywords/extractor.py:59](../src/keywords/extractor.py#L59) |
| Novedad local `k_novelty` | [builder.py:78](../src/features/builder.py#L78) · `local_seen` y [builder.py:125](../src/features/builder.py#L125) |
| Sentimiento por oración | [src/features/sentiment.py:26](../src/features/sentiment.py#L26) |
| Entropía léxica `l_entropy = −Σ p log2 p` | [builder.py:231](../src/features/builder.py#L231) |
| Deriva de tema `f_topic_drift` | [builder.py:211](../src/features/builder.py#L211) |
| Hook (preguntas, números, contraste) | [builder.py:193](../src/features/builder.py#L193) |

---

## Hook / Flow / Value (explican el clip; NO deciden la selección)

**Archivos:**
- [src/features/builder.py:38](../src/features/builder.py#L38) · `DIMENSIONS` (qué prefijo va a qué dimensión)
- [src/ranking/model.py:141](../src/ranking/model.py#L141) · `dimension_contributions` (suma SHAP por dimensión)
- [src/ranking/model.py:123](../src/ranking/model.py#L123) · `ScoreCalibrator` (pasa a 0–99)
- [src/pipeline.py:230](../src/pipeline.py#L230) · `_dim_vars` (aporte y puntaje de cada variable, lo que ves en la web)

```python
DIMENSIONS = {"hook": ("h_",), "flow": ("f_",), "value": ("k_","s_","a_","e_","l_","u_")}
# // Hook = suma de aportes SHAP de las variables h_*; Flow = f_*; Value = el resto.
contrib = booster.predict(df[features], pred_contrib=True)[:, :-1]   # // aporte de CADA variable a la predicción (SHAP)
out[dim] = contrib[:, cols_de_la_dimension].sum(axis=1)               # // se suman por dimensión
```

**Qué decir:** "Hook, Flow y Value son una explicación; la decisión sale de las 17 variables. Son dos cosas distintas y la
página las separa a propósito".

---

## Orden recomendado para la sustentación (5 minutos)

1. Flujo completo (sección 0).
2. [pipeline.py:81](../src/pipeline.py#L81) `run` — "este es el director de orquesta".
3. [builder.py:97](../src/features/builder.py#L97) `window_features` — "aquí nacen las variables".
4. [retention.py:29](../src/ranking/retention.py#L29) `formula_raw` — "aquí se convierten en un puntaje".
5. [candidates.py:87](../src/segment/candidates.py#L87) `nms` — "aquí se eligen los 5".
6. [labels.py:10](../src/ranking/labels.py#L10) y [web_clips.py](../scripts/web_clips.py) — "aquí se valida contra la curva".
