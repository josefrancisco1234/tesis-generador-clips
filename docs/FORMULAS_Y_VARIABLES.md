# Fórmulas y variables, una por una (con su sustentación básica)

Cómo leer este documento: primero las **fórmulas del sistema** (F1 a F6), después las **68 variables** agrupadas por tipo
(K, S, A, H, F, L, E, P). Para cada variable: qué mide, cómo se calcula y **por qué se espera que se relacione con la retención**.
La sustentación dice la intuición; la evidencia (el peso) viene de los datos (D-045, 121 videos de desarrollo).

Convenciones: *ventana* = tramo de video candidato (1/100 del video en la evaluación; 15–45 s en el producto).
`dur` = duración en segundos, `min = dur/60`, `n_palabras` = palabras con inicio dentro de la ventana.
*Percentil en el video* = posición de la ventana entre todas las del mismo video (0 = la más baja, 1 = la más alta).

---

## Parte 1. Las fórmulas del sistema

### F1. Percentil dentro del video

```
rk_i(w) = rango_percentil( x_i(w) )   entre todas las ventanas del mismo video      (NaN -> 0.5)
```

**Sustentación:** cada video tiene su propia "línea base" (un podcast tranquilo vs. un stand-up ruidoso). Lo que importa para
elegir clips es qué ventana destaca **respecto a las demás del mismo video**, no su valor absoluto. Además evita que el
modelo aprenda "este canal habla fuerte".

### F2. Puntaje crudo V0 (la fórmula de 17 pesos)

```
raw(w) = sum_{i=1..17}  peso_i * rk_i(w)
```

**Sustentación:** combinación lineal simple de las variables que más acompañan al calor de YouTube. Se eligió lineal porque, en
la batería de 9 modelos (D-050/D-051), ningún modelo no lineal la superó de forma confiable: con ~450 videos la señal es
débil y los modelos complejos sobreajustan. Es auditable: cada aporte se puede leer.

### F3. Peso de cada variable (cómo se obtuvo)

```
peso_i = mediana_sobre_videos( rho_parcial_spearman( x_i , calor_real | posición ) )
```

**Sustentación:** `rho parcial` mide cuánto se mueve la variable con el calor **quitando el efecto de la posición** en el
video (los inicios y finales tienen más calor por sí solos; sin controlarlo, "premiaríamos" simplemente estar al principio). La
mediana entre videos evita que un video raro domine. Se calculó en 121 videos de desarrollo y se congeló antes de probar en
videos nuevos (352 + 93 + 63).

### F4. Score 0–99 (calibración)

```
score(w) = clip( posición_de_raw(w)_en_la_tabla_de_100_cuantiles , 0 , 99 )
```

**Sustentación:** el crudo no tiene unidad interpretable. La tabla se construyó con los 10 mejores clips por video de
desarrollo; un score de 90 significa "este clip está en el 10 % superior de los mejores clips que vimos". Es un percentil,
no una probabilidad de viralidad.

### F5. Selección de los 5 clips (NMS)

```
ordenar por raw descendente; quedarse con el siguiente solo si solapa < 30 % con los ya elegidos; parar en 5
V1: solo candidatos de duración <= 45 s
```

**Sustentación:** sin NMS, los 5 mejores serían ventanas casi iguales (vecinas) y cubrirían un único momento. Los clips ≤ 45 s
aumentan el acierto (47 → 55 %) porque un clip corto "toca" un pico con menos dilución.

### F6. Métricas de evaluación

```
hit@5  = 1 si al menos uno de los 5 clips toca un bin caliente (valor > 0.5, máximo 20 por video)
F1     = 2PR/(P+R),  P = tiempo(clips ∩ picos)/tiempo(clips),  R = tiempo(clips ∩ picos)/tiempo(picos)
lift   = calor_medio(clip) / calor_medio(video)
azar   = 1 - C(100-r, 5)/C(100, 5)          r = bins calientes del video
```

**Sustentación:** hit@5 responde "¿al menos encontré un momento bueno?" (lo que le importa a un usuario que revisa 5
propuestas). F1 mide cuánto del tiempo caliente real cubrí. Lift compara con elegir al azar. Todas se comparan contra azar
y se prueban en videos que el modelo nunca vio.

---

## Parte 2. Las 17 variables de la fórmula (con su peso)

| # | Variable | Peso | Qué mide | Sustentación básica |
|---|---|---|---|---|
| 1 | `a_pause_ratio` | **+0.180** | Proporción de la ventana en pausas > 0.3 s | Los picos son *fronteras de segmento*: la gente repite el momento donde algo nuevo empieza; ahí el hablante hace pausa |
| 2 | `l_silence` | **+0.163** | 1 − tiempo hablado / duración | Mismo mecanismo: más silencio = transición entre ideas |
| 3 | `a_long_pauses_pm` | **+0.148** | Pausas > 1 s por minuto | Pausas largas = cambio de tema o remate, no habla corrida |
| 4 | `a_rms_std` | **+0.122** | Variación del volumen | Dinámica (susurro y énfasis) acompaña al momento destacado; el volumen plano es explicación monótona |
| 5 | `a_pitch_range` | **+0.112** | Rango de tono (p95 − p5, semitonos) | Prosodia expresiva = emoción, anécdota, énfasis |
| 6 | `a_voiced_ratio` | **−0.107** | Fracción de frames con voz | Más voz continua = explicación corrida; menos voz = pausas, risas, música |
| 7 | `k_keybert_sim` | +0.102 | Parecido al tema central del video | El tramo que "resume" el tema es el que la gente busca de nuevo |
| 8 | `e_speech_mean` | **−0.101** | Prob. media de "habla pura" (PANNs) | Si hay otro evento (risa, aplauso, música) además del habla, suele ser el pasaje reaccionado |
| 9 | `k_noun_chunks_ps` | −0.098 | Sintagmas nominales por oración | Más sustantivos = explicación técnica densa; menos = narración/anécdota |
| 10 | `a_speech_rate` | −0.098 | Palabras por segundo | Habla muy rápida = relleno; el momento destacado suele ir más pausado |
| 11 | `a_pitch_std` | +0.089 | Desvío del tono | Igual que el rango: entonación variada |
| 12 | `e_music_max` | +0.083 | Máx. prob. de música | Música = intro/transición/remate editado |
| 13 | `a_rms_mean` | −0.082 | Volumen medio | Volumen alto sostenido es ruido ambiente o discusión; lo notable es la dinámica (#4), no el nivel |
| 14 | `e_music_mean` | +0.067 | Prob. media de música | Igual que #12 |
| 15 | `f_n_sentences` | +0.060 | Número de oraciones | Más oraciones en la ventana = más contenido por clip |
| 16 | `s_questions_pm` | +0.058 | Preguntas por minuto | Preguntas abren curiosidad y marcan cambio de turno/tema |
| 17 | `s_second_person` | +0.056 | Fracción en 2.ª persona (tú/you) | Interpelar al espectador engancha |

**Lectura integrada:** el pico de retención se parece a **un cambio de segmento** (silencio, pausa, tono variable, menos
explicación continua), no a "la frase más emocional". Esto fue la hipótesis de D-066 y es consistente con los pesos.

---

## Parte 3. Las 68 variables (todas las analizadas)

### Grupo K — Palabras clave (9)
Calculadas sobre las palabras de contenido (sin stopwords) con KVSUM por pasaje de 300 s y KeyBERT.

| Variable | Fórmula | Sustentación |
|---|---|---|
| `k_kvsum_mean` | media del peso KVSUM de las palabras de contenido de la ventana | Densidad de términos importantes del video |
| `k_kvsum_max` | máximo de ese peso | Un solo término muy central ya puede definir el tramo |
| `k_kvsum_local` | suma de pesos KVSUM según el pasaje (300 s) donde cae cada palabra / n_palabras | Importancia relativa al contexto local |
| `k_top_terms_pm` | términos "top del video" por minuto | ¿Se habla del tema principal? |
| `k_content_ratio` | palabras de contenido / palabras totales | Densidad informativa vs. relleno |
| `k_novelty` | 1 − fracción de palabras ya vistas en los 300 s previos | Novedad local (corregida para no filtrar el futuro, D-018) |
| `k_keybert_sim` | coseno(embedding medio de oraciones, embedding del documento) | Cercanía al tema central |
| `k_keyphrase_pm` | frases clave KeyBERT por minuto | Frases-resumen presentes |
| `k_noun_chunks_ps` | sintagmas nominales por oración (spaCy) | Densidad explicativa |

### Grupo S — Sentimiento y energía léxica (11 + 3)
Modelo DistilBERT multilingüe por oración: `(pos, neu, neg)`; `pol = pos − neg`.

| Variable | Fórmula | Sustentación |
|---|---|---|
| `s_pos_mean`, `s_neg_mean` | media de P(pos) / P(neg) | Carga emocional |
| `s_polarity_mean`, `s_polarity_std` | media y desvío de `pol` | Tono y cambios de tono |
| `s_intensity_mean`, `s_intensity_max` | media y máx. de `abs(pol)` | Intensidad emocional, independiente del signo |
| `s_questions_pm` | (`?` + `¿`) por minuto | Interpelación |
| `s_second_person`, `s_first_person` | fracción de tokens en 2.ª / 1.ª persona | Hablarle al espectador / hablar de uno |
| `s_numbers` | fracción de tokens que inician con dígito | Datos concretos ("3 razones…") |
| `s_intensifiers` | fracción de intensificadores (muy, super, really) | Énfasis léxico |
| `s_hook_words` | fracción de palabras gancho (secreto, nunca, never) | Lenguaje de gancho |
| `s_exclaim_pm`, `s_laughs` | exclamaciones por minuto; risas ("jaja") en el texto | Expresividad escrita |

### Grupo A — Audio y prosodia (12)
Frames con librosa (`hop=512`), RMS normalizado en z, f0 en semitonos.

| Variable | Fórmula | Sustentación |
|---|---|---|
| `a_rms_mean`, `a_rms_p95`, `a_rms_std` | media, percentil 95 y desvío del RMS (z) | Nivel y dinámica de volumen |
| `a_pitch_std`, `a_pitch_range` | desvío de f0; `p95(f0) − p5(f0)` | Expresividad |
| `a_voiced_ratio` | frames con voz / frames | Continuidad de la voz |
| `a_energy_delta` | `RMS(ventana) − RMS(30 s previos)` | Subida de energía respecto al momento anterior |
| `a_speech_rate` | `n_palabras / dur` | Ritmo |
| `a_pause_ratio` | `suma(huecos entre palabras > 0.3 s) / dur` | Pausas |
| `a_long_pauses_pm` | `n(huecos > 1 s) / min` | Pausas largas |
| `h_rms_onset` | RMS medio de los primeros 3 s | Arranque fuerte |

### Grupo H — Gancho / hook (8, primeras 15 palabras o 3 s)

| Variable | Fórmula | Sustentación |
|---|---|---|
| `h_question`, `h_number`, `h_second_person`, `h_hook_word`, `h_contrast` | 0/1: ¿el arranque trae pregunta / número / "tú" / palabra gancho / contraste (pero, sin embargo)? | Un buen clip abre con algo que retiene en los primeros segundos |
| `h_first_sentence_s` | duración de la primera oración | Aperturas cortas enganchan más |
| `h_first_intensity` | `abs(pol)` de la primera oración | Emoción desde el inicio |
| `h_first_keybert` | similitud temática de la primera oración | Abre con el tema, no con relleno |

### Grupo F — Flujo / coherencia (9)

| Variable | Fórmula | Sustentación |
|---|---|---|
| `f_starts_dangling` | 1 si empieza con palabra colgada ("eso", "él", "and") | Un clip debe ser autocontenido |
| `f_ends_complete` | 1 si la última oración termina completa | No cortar a mitad de frase |
| `f_coherence_mean`, `f_coherence_min` | media y mínimo de similitud entre oraciones consecutivas | Hilo continuo |
| `f_topic_drift` | `1 − coseno(primera mitad, segunda mitad)` | Cambio de tema dentro del clip |
| `f_n_sentences`, `f_duration` | oraciones; duración | Tamaño del clip |
| `f_filler_rate` | muletillas / tokens | Fluidez |
| `f_word_conf` | confianza media de Whisper | Calidad de la transcripción |

### Grupo L — Lingüística de engagement (6)

| Variable | Fórmula | Sustentación |
|---|---|---|
| `l_entropy` | `−Σ p log2 p` de los tokens | Variedad léxica |
| `l_type_token` | tokens distintos / tokens | Riqueza vs. repetición |
| `l_pron_12_vs_3` | `(pron. 1.ª+2.ª + 1) / (pron. 3.ª + 1)` | Diálogo directo vs. narración sobre terceros |
| `l_fillers` | muletillas / tokens | Fluidez |
| `l_silence` | `1 − tiempo hablado / dur` | Silencio |
| `l_word_len` | largo medio de palabra | Registro (técnico vs. coloquial) |

### Grupo E — Eventos de audio (13; red PANNs CNN14, 1 muestra por segundo)

| Variable | Fórmula | Sustentación |
|---|---|---|
| `e_<ev>_max`, `e_<ev>_mean` para `laugh, applause, cheering, music, speech` | máx. y media de la probabilidad del evento en la ventana (10 variables) | Reacción del público o ambiente editado |
| `e_laugh_tail` | máx. de risa en los últimos 6 s | Remate de un chiste |
| `e_laugh_after`, `e_applause_after` | máx. de risa / aplauso en los 6 s posteriores | Reacción inmediata al final del clip |

### Grupo P — Posición (solo ablación, nunca entra al modelo)

| Variable | Fórmula | Por qué se excluye |
|---|---|---|
| `p_rel_pos`, `p_rel_pos_sq` | `centro / duración`, y su cuadrado | Daría fuga: "los finales siempre son calientes". Solo se usa para controlar (F3) y para el chequeo posicional |

### Grupo U — Habla HuBERT (32 componentes PCA, solo en pruebas)

Embedding de habla (capa 10, trozos de 10 s) reducido con un PCA congelado. Se probó en D-038/D-056/D-061: aporta direcciones
pequeñas, no confirmadas. No está en la fórmula V0.

---

## Parte 4. Qué se sabe y qué no (para sustentar con honestidad)

- **Confirmado en videos nuevos:** la combinación lineal de las 17 variables acierta un pico con ≥ 1 de 5 clips en ~47–60 %
  de los videos (azar ~30 %), en ES y EN.
- **Mecanismo plausible, no demostrado causalmente:** los picos son fronteras de segmento. Las variables con más peso (pausas,
  silencio, variación de tono) lo respaldan.
- **No confirmado:** que modelos de aprendizaje (R4, ensamble E3, deep learning, texto semántico, visual, LLM) mejoren esta fórmula.
- **Techo:** el calor de YouTube también depende de factores que no están en el video (recomendación, compartidos), por eso
  el contenido solo tiene un techo de ~50–69 %.
