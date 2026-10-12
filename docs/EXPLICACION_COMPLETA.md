# Explicacion completa: del video de YouTube al clip con score 0-99

Autor: Jose Francisco Nunez del Prado Vega - Universidad de Lima.
Este documento responde, en orden, a seis preguntas:

1. [ETL: como se arma el dataset](#1-etl-como-se-arma-el-dataset)
2. [Como se entrena](#2-como-se-entrena)
3. [Todas las formulas y todas las variables](#3-todas-las-formulas-y-todas-las-variables)
4. [Como se mide: azar vs nuestro metodo (F1 y acierto)](#4-como-se-mide-azar-vs-nuestro-metodo)
5. [Como llegamos a ese porcentaje (el proceso)](#5-como-llegamos-a-ese-porcentaje)
6. [Como mejora el metodo: se baja la prueba, se corrige, siguiente video](#6-el-ciclo-de-mejora)

> Fuente de verdad de cada decision: `docs/DECISIONES.md` (D-001 a D-070).
> Resultado en una linea: **solo con el contenido del video** se acierta en ~50-65 % de los videos
> (azar ~27-32 %). **Usando la curva de calor de YouTube como validador** se llega a ~85 %.
> 100 % no es alcanzable solo con contenido (ver seccion 5).

---

## 1. ETL: como se arma el dataset

ETL = Extraer, Transformar, Cargar. Aqui cada video pasa por 5 etapas; cada una guarda su
resultado en disco (si se corta, se reanuda sin repetir).

```
 Buscar videos        Extraer              Transformar              Transformar            Cargar
 (yt-dlp search) ->  heatmap + audio  ->  transcripcion Whisper -> ventanas candidatas -> features por
 manifest CSV        (JSON + WAV)         (palabras con tiempos)   + features             ventana (parquet)
```

| # | Etapa | Que hace | Donde queda | Codigo |
|---|---|---|---|---|
| E1 | **Buscar** | Consultas de podcasts/entrevistas largas (ES y EN). Se queda con videos de duracion valida, maximo N por canal, **que tengan heatmap**. | `data/manifests/youtube_mr.csv` | `src/data/youtube_dataset.py::build_manifest` |
| E2 | **Extraer calor** | yt-dlp baja la curva "Most Replayed": 100 bins, cada uno con `start_time, end_time, value` (0-1). | `interim/heatmaps/<id>.json` | `src/ingest/youtube.py` |
| E3 | **Extraer audio** | Audio WAV del video. | `raw/wav/<id>.wav` | `acquire()` |
| E4 | **Transcribir** | Whisper `base` (GPU, float32): palabras con inicio, fin y confianza; detecta idioma. | `interim/transcripts/base/<id>.json` | `src/transcribe/whisper.py` |
| E5 | **Featurizar** | Para cada ventana (100 bins del video) calcula ~68 variables (seccion 3). Grupos K,S,A,H,F,E,L,P,U. | `processed/features/bin/<id>.parquet` | `src/features/builder.py` |
| E6 | **Etiquetar** | Pega a cada ventana el calor real de YouTube (`y_mr`), un grado 0-4 dentro del video (`y_grade`) y una bandera de "momento caliente" (`y_relevant`). | mismas columnas del parquet | `src/ranking/labels.py` |

**Reglas de calidad del ETL (para que sea auditable):**

- Se filtra por idioma (los de idioma no soportado se descartan; 8 de 101 en la cohorte fresca 1).
- Se corrige el sesgo del primer bin (el calor del bin 0 se iguala al del bin 1, como en Rhapsody).
- Cache con validacion de longitud (bug real de caches desalineados, ya corregido).
- Los videos se agrupan **por canal** al validar: ningun canal esta en entrenamiento y prueba a la vez.
- Cada cohorte nueva se congela como lista de `video_id` **antes** de mirar resultados
  (`reports/conjunto_confirmacion_fresca_D050.json`, `..._D065.json`).

Tamano actual: 444 videos de desarrollo + 93 de cohorte fresca 1 (confirmada) + cohorte fresca 2
(en proceso, D-065) + ampliacion de entrenamiento D-071 en cola.

### Las etiquetas (las tres "respuestas correctas" que el modelo intenta adivinar)

| Etiqueta | Definicion | Para que se usa |
|---|---|---|
| `y_mr` | Calor medio de la curva de YouTube dentro de la ventana, ponderado por solapamiento: `sum(overlap_i * value_i) / sum(overlap_i)` | Objetivo continuo |
| `y_grade` | `y_mr` partido en 5 cuantiles **dentro del mismo video** (0 = frio ... 4 = lo mas caliente) | Lo que consume LambdaRank |
| `y_relevant` | La ventana se solapa con los "bins highlight": valor > 0.5, maximo 20 por video | Medir acierto (hit) y F1 |

---

## 2. Como se entrena

### 2.1 El problema

Dado **un video**, ordenar sus ventanas de la mas a la menos probable de estar "caliente" en YouTube.
No se predice un numero absoluto: se predice un **orden dentro del video** (learning-to-rank).

### 2.2 El modelo de produccion y el de referencia

- **LightGBM LambdaRank** (`objective=lambdarank`): arboles de decision que optimizan directamente el
  ranking (NDCG). `group = video`, o sea, el modelo solo compara ventanas del mismo video.
  Hiperparametros fijos a priori (D-020): 300 arboles, `learning_rate=0.05`, `num_leaves=15`,
  `min_data_in_leaf=50`, `feature_fraction=0.7`, `bagging_fraction=0.7`, `lambda_l2=10`.
- **Formula congelada V0** (la "formula secreta" simple): 17 pesos lineales sobre rangos percentiles.
  Es la que usa el one-click (`scripts/generar_clips.py`).

### 2.3 Que entra al modelo

Cada variable se convierte en **rango percentil dentro del video** (0 = la ventana mas baja de ese
video, 1 = la mas alta). Asi el modelo no depende de si el video es ruidoso o tranquilo en general:
compara ventanas contra sus hermanas. Estas son las `__rk` (D-022).

**Nunca entra la posicion** (`p_rel_pos`): daria fuga ("el final siempre es caliente"). Regla D-018.

### 2.4 Validacion (para no enganarse)

1. **GroupKFold(10) por canal**: en cada vuelta, 9/10 de los canales entrenan y 1/10 se evalua.
2. Todo lo que se **elige** (grupos de features, pesos, hiperparametros) se elige **dentro** del fold.
3. **Desarrollo / confirmacion separados** (D-021): lo que se descubre en videos de desarrollo se
   congela y se prueba **una sola vez** en videos nuevos con hipotesis escrita antes (preregistro).
4. Pruebas pareadas por video: Wilcoxon (F1), McNemar (acierto), correccion Bonferroni cuando hay varias.

### 2.5 Del score al clip

```
ventanas candidatas  ->  score del modelo  ->  NMS (quita solapados >30 %)  ->  top-5
        |                       |                                                |
 oraciones completas     calibrado a 0-99                           cortar con ffmpeg a 9:16 -> mp4
```

El score 0-99 es el **percentil** del score crudo frente a la tabla de calibracion `calibrator`
(100 cuantiles de los top-10 por video de desarrollo, `models/formula_calor_D045.json`).

---

## 3. Todas las formulas y todas las variables

### 3.1 La formula V0 (17 pesos)

Para cada ventana `w` de un video:

```
raw(w) = sum_i  peso_i * rango_percentil_i(w)         # rango dentro del video; NaN -> 0.5
score(w) = clip( posicion_en_calibrador(raw(w)) , 0 , 99 )
```

Los pesos salen del **rho parcial de Spearman** mediano (feature vs calor, controlando la posicion
del video) sobre 121 videos / 101 canales (D-045).

| Variable | Peso | Lectura |
|---|---|---|
| `a_pause_ratio` | **+0.180** | Mas pausas (>0.3 s) -> mas calor: los picos son fronteras entre segmentos |
| `l_silence` | **+0.163** | Mas silencio |
| `a_long_pauses_pm` | **+0.148** | Mas pausas largas (>1 s) por minuto |
| `a_rms_std` | **+0.122** | Mas variacion de volumen |
| `a_pitch_range` | **+0.112** | Mas rango de tono (p95-p5) |
| `a_voiced_ratio` | -0.107 | Menos voz continua |
| `k_keybert_sim` | +0.102 | Se parece al tema central del video |
| `e_speech_mean` | -0.101 | Menos "habla pura" (hay otro evento: risa, musica) |
| `k_noun_chunks_ps` | -0.098 | Menos sustantivos por oracion (menos explicativo) |
| `a_speech_rate` | -0.098 | Menos palabras por segundo |
| `a_pitch_std` | +0.089 | Mas variacion de tono |
| `e_music_max` | +0.083 | Hay musica |
| `a_rms_mean` | -0.082 | Menos volumen medio |
| `e_music_mean` | +0.067 | Musica media |
| `f_n_sentences` | +0.060 | Mas oraciones |
| `s_questions_pm` | +0.058 | Mas preguntas |
| `s_second_person` | +0.056 | Mas "tu / you" |

**Lectura integrada:** el pico de retencion se parece a un *cambio de segmento*: silencio, pausa,
tono mas variable, menos explicacion corrida y menos conectores ("entonces", "pero").

### 3.2 V1

V0 pero solo candidatos de **<= 45 s**. Sube el acierto (47 -> 55 %) y baja un poco el F1 (compromiso, D-049).

### 3.3 Las 68 variables, una por una

Ventana = trozo del video (1/100 del video o ventana de oraciones). `dur` = duracion en segundos,
`min` = dur/60. "rk" = se usa como rango percentil.

**K - Palabras clave (9)**

| Variable | Formula / que mide |
|---|---|
| `k_kvsum_mean` | Media del peso KVSUM de las palabras de contenido de la ventana |
| `k_kvsum_max` | Maximo peso KVSUM |
| `k_kvsum_local` | Peso KVSUM segun el pasaje (300 s) donde cae cada palabra / n palabras |
| `k_top_terms_pm` | Terminos "top del video" por minuto |
| `k_content_ratio` | Palabras de contenido / palabras totales |
| `k_novelty` | 1 - fraccion de palabras ya vistas en los 300 s anteriores (novedad local, D-018) |
| `k_keybert_sim` | Coseno entre el embedding medio de las oraciones y el embedding del documento |
| `k_keyphrase_pm` | Frases clave KeyBERT por minuto |
| `k_noun_chunks_ps` | Sintagmas nominales por oracion (spaCy) |

**S - Sentimiento y energia lexica (14)**

| Variable | Que mide |
|---|---|
| `s_pos_mean`, `s_neg_mean` | Probabilidad media positiva / negativa por oracion (DistilBERT multilingue) |
| `s_polarity_mean`, `s_polarity_std` | Media y desvio de `pos - neg` |
| `s_intensity_mean`, `s_intensity_max` | Media y maximo de `abs(pos - neg)` |
| `s_questions_pm` | `?` y `¿` por minuto |
| `s_second_person`, `s_first_person` | Fraccion de tokens en 2da / 1ra persona |
| `s_numbers` | Fraccion de tokens que empiezan con digito |
| `s_intensifiers` | Fraccion de intensificadores ("muy", "super", "really") |
| `s_hook_words` | Fraccion de palabras gancho ("secreto", "nunca", "never") |

**A - Audio y prosodia (12; se calculan con librosa)**

| Variable | Formula |
|---|---|
| `a_rms_mean`, `a_rms_p95`, `a_rms_std` | Media, percentil 95 y desvio del volumen RMS normalizado (z) |
| `a_pitch_std` | Desvio del tono f0 en semitonos |
| `a_pitch_range` | `p95(f0) - p5(f0)` |
| `a_voiced_ratio` | Fraccion de frames con voz |
| `a_energy_delta` | `RMS medio de la ventana - RMS medio de los 30 s previos` |
| `a_speech_rate` | `n_palabras / dur` |
| `a_pause_ratio` | `suma(huecos entre palabras > 0.3 s) / dur` |
| `a_long_pauses_pm` | `n(huecos > 1 s) / min` |
| `h_rms_onset` | RMS medio de los primeros 3 s |

**H - Gancho / hook (8, primeros 15 palabras / 3 s)**

`h_question`, `h_number`, `h_second_person`, `h_hook_word`, `h_contrast` (0/1: aparece en el gancho),
`h_first_sentence_s` (duracion de la primera oracion), `h_first_intensity` (intensidad emocional de
la primera oracion), `h_first_keybert` (similitud tematica de la primera oracion).

**F - Flujo / coherencia (8)**

| Variable | Que mide |
|---|---|
| `f_starts_dangling` | 1 si empieza colgado ("eso", "el", "and"...) |
| `f_ends_complete` | 1 si la ultima oracion esta completa |
| `f_coherence_mean`, `f_coherence_min` | Media y minimo de similitud entre oraciones consecutivas |
| `f_topic_drift` | `1 - coseno(primera mitad, segunda mitad)` |
| `f_n_sentences`, `f_duration` | Numero de oraciones y duracion |
| `f_filler_rate` | Muletillas / tokens |
| `f_word_conf` | Confianza media de Whisper |

**L - Linguistica de engagement (6)**

| Variable | Formula |
|---|---|
| `l_entropy` | Entropia de Shannon de los tokens: `-sum p log2 p` |
| `l_type_token` | Tokens distintos / tokens |
| `l_pron_12_vs_3` | `(pron 1a+2a + 1) / (pron 3a + 1)` |
| `l_fillers` | Muletillas / tokens |
| `l_silence` | `1 - tiempo_hablado / dur` |
| `l_word_len` | Largo medio de palabra |

**E - Eventos de audio (13; PANNs/AudioSet, 1 muestra por segundo)**

Para `laugh, applause, cheering, music, speech`: `e_<ev>_max` y `e_<ev>_mean` dentro de la ventana
(10 variables) mas `e_laugh_tail` (max risa en los 6 s finales), `e_laugh_after` (max risa en los 6 s
posteriores), `e_applause_after` (max aplauso en los 6 s posteriores).

**U - HuBERT (32 componentes PCA, solo en las pruebas con U)**

Embedding de habla (capa 10, trozos de 10 s) reducido con PCA congelado (sha256 `87a80b42f96a4565`).

**P - Posicion (solo ablacion, NUNCA al modelo)**

`p_rel_pos = centro / duracion`, `p_rel_pos_sq`.

### 3.4 Formulas de las metricas

| Metrica | Definicion |
|---|---|
| **hit@5** | 1 si **al menos uno** de los 5 clips elegidos (tras NMS 30 %) toca un bin caliente (`y_relevant`). Se promedia por video. |
| **F1 temporal@5** | `P = tiempo(clips ∩ picos) / tiempo(clips)`, `R = tiempo(clips ∩ picos) / tiempo(picos)`, `F1 = 2PR/(P+R)` con los picos = bins con valor > 0.5 (max 20). |
| **Lift** | `calor medio del clip / calor medio del video`. 1.0 = igual que elegir al azar. |
| **Azar (hit)** | `1 - C(100 - r, 5) / C(100, 5)`, con r = bins relevantes: probabilidad de pegarle a uno con 5 clips al azar (depende del video). |
| **Exceso** | `hit - azar` por video: cuanto aporta el metodo por encima de tirar dados. |

---

## 4. Como se mide: azar vs nuestro metodo

Cohorte fresca 1: 93 videos nuevos, 86 canales, 32 consultas nuevas, ES y EN. Una sola corrida,
pesos congelados (D-052, D-064).

| Metodo | hit@5 | F1 temporal@5 | Lift | Usa la curva de YouTube |
|---|---|---|---|---|
| **Azar** | 26.9 % | 0.072 | 1.00x | no |
| Azar restringido a <= 45 s | 30.6 % | 0.063 | 1.00x | no |
| **V0** (formula, 17 pesos) | 47.3 % | **0.149** | 1.35x | no |
| **V1** (V0 + <= 45 s) | **54.8 %** | 0.141 | 1.43x | no |
| LightGBM por bin (R0) | 62.4 % | 0.142 | 1.52x | no |
| Ensamble E3 | 68.8 % | 0.166 | 1.57x | no |
| BiGRU (deep learning) | 53.8 % | 0.129 | 1.37x | no |
| Valida con 20 % superior del calor y ordena por BiGRU | **84.9 %** | 0.327 | 2.45x | **si (valida)** |
| Calor directo | 94.6 % | 0.555 | 3.25x | **si (solo curva)** |

**Cuanto mejora vs azar (F1):** V0 duplica el F1 del azar (0.149 vs 0.072); V0 > azar con
p = 3e-9. En 352 videos adicionales (D-047): F1 0.131 vs 0.070 (+0.060, p = 4e-17).

**Por idioma (V1):** EN 52.3 % (azar 22.9 %), ES 57.1 % (azar 30.4 %).

**Honestidad:** E3 y R4 (los mejores sin curva) **no** pasaron la confirmacion preregistrada
(R4 p = 0.12; E3 p = 0.025 pero Bonferroni exigia 0.0125). Por eso **lo adoptado en el producto
sigue siendo V0/V1**; la cohorte 2 decide E3.

---

## 5. Como llegamos a ese porcentaje

1. **Se descubrio de donde sale el calor.** `Most Replayed` mide donde la gente **vuelve a ver** un
   tramo. Es una senal de comportamiento, no de calidad editorial: los clips que elige un editor
   humano **no** coinciden con los picos (D-028, reconfirmado en todos los analisis).
2. **Se busco que acompana al calor** (D-044/D-045): 68 variables, rho parcial controlando posicion.
   Resultado: pausas, silencio y variacion de tono (fronteras de segmento), no "emocion" ni palabras gancho.
3. **Se congelo una formula lineal simple** (17 pesos) y se probo en videos nuevos (D-046/D-047).
4. **Se compitieron 9 modelos contra la formula** (D-050/D-051): LightGBM, regresion, logistica,
   ridge, ensambles. Ninguno la supero de forma confiable -> el techo de contenido pareca ~50-55 %.
5. **Se probaron 7 familias de senal nueva** (D-053 a D-064). Ninguna supero la formula en confirmacion:
   modelo profundo de secuencia, juez LLM, senal visual gruesa, texto semantico, HuBERT, ingenieria
   de etiqueta, representacion R4. R4/E3 mejoran en direccion (+0.015 a +0.024 F1) pero sin potencia
   suficiente con n = 93.
6. **Techo real.** Los picos dependen tambien de factores que no estan en el video (de que se
   comparte, que se recomienda, que se vuelve a ver por social). Por eso solo-contenido satura en
   ~50-69 % y con la curva se llega a ~85-95 %.

**Por que no 100 %:** con solo contenido, el techo medido es ~69 % de hit@5 (E3, aun sin confirmar).
La unica forma de pasar 85 % es **validar con la curva real** de YouTube.

---

## 6. El ciclo de mejora

Regla: cada mejora se escribe como hipotesis **antes** de correr; se prueba **una vez** en videos
no vistos; si falla, se registra como negativa y se corrige con el siguiente lote de videos.

| Paso | Idea | Prueba | Resultado | Correccion siguiente |
|---|---|---|---|---|
| D-044 | Que features acompanan al calor (10 videos) | Correlaciones | Pausas y variacion | Subir a 121 videos |
| D-045 | Formula de 17 pesos | 121 videos | Candidata | Confirmarla en videos nuevos |
| D-046/47 | **Confirmar** formula | 352 videos nuevos | **Confirmada**: F1 0.131 vs azar 0.070 | Mejorar el acierto por video (hit) |
| D-048/49 | V1: solo clips <= 45 s | 352 videos | **Confirmada**: hit 49.4 -> 53.4 %, F1 baja -0.013 | Probar modelos mas ricos |
| D-050/51 | 9 modelos (LightGBM, etc.) | 473 videos, CV por canal | **Ninguno supera la formula** | Probar senales nuevas |
| D-053/57 | Visual (storyboards) | 121 videos | Correlaciona, no aporta sobre audio+texto | Probar secuencia |
| D-054/55 | Deep learning de secuencia | CV + cohorte fresca | **No supera LightGBM**; fusion curva+modelo 81-85 % | Texto semantico |
| D-056/58 | Embeddings de texto, HuBERT | CV | No mejoran | Etiqueta |
| D-058/60 | Ingenieria de etiqueta | CV | Ninguna mejora | Representacion |
| D-059/62 | R4 (rango + valor + vecinos) | CV | Mejora robusta en desarrollo | Confirmar en fresca |
| D-064 | R4 y E3 en 93 videos frescos | Una corrida | **No confirmada** (R4 p = 0.12; E3 p = 0.025 no pasa Bonferroni) | Segunda cohorte (n mayor) |
| D-065 | Cohorte 2 | **En curso** | -- | Se corre una sola vez a las 10:50 |
| D-066 | Ingenieria inversa | 444 videos | Picos = fronteras de segmento | Lexico, genero |
| D-068 | Lexico por fold | CV | **Negativo** (F1 0.130 -> 0.128) | -- |
| D-069 | Por genero | 444 videos | Sin diferencia significativa (p = 0.24) | Categoria como contexto |
| D-070 | Categoria como contexto | CV | **Negativo** (+0.0013, p = 0.45) | -- |
| D-071 | Mas videos para entrenar | **En cola** | -- | Entrenar con ~200 videos extra |

**Regla de adopcion:** ninguna variante reemplaza V0/V1 hasta que su hipotesis preregistrada se
confirme en videos nuevos. Si falla, V0/V1 siguen vigentes.

### Como se corrige "con el siguiente video"

El sistema no se reentrena solo con cada video. Hay tres lazos:

1. **Lazo de investigacion** (el de la tabla): se prueba, se registra, se ajusta con nuevos videos.
2. **Lazo de produccion con curva**: `ranking.method: auto` usa la curva real de YouTube si el
   video la tiene (validacion), y la formula V0/V1 si no. Ver `src/ranking/retention.py`.
3. **Lazo de datos**: cada video nuevo con heatmap entra al manifest, se featuriza y se vuelve
   material de entrenamiento (D-071) sin tocar los conjuntos de confirmacion.

---

## Anexo: donde esta cada cosa

| Quiero ver | Archivo |
|---|---|
| Formula y pesos | `models/formula_calor_D045.json`, `src/ranking/retention.py` |
| Todas las features | `src/features/builder.py`, `audio.py`, `extra.py` |
| Etiquetas | `src/ranking/labels.py` |
| Metricas (hit, F1, NMS) | `scripts/reverse_engineering_formula.py` |
| Todas las decisiones | `docs/DECISIONES.md` |
| One-click URL -> mp4 | `scripts/generar_clips.py` (+ `.bat`) |
| Resultados por experimento | `reports/<experimento>/` |
