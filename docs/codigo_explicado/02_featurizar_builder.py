"""
PASO 2 — FEATURIZAR: convertir cada candidato en ~115 numeros descriptivos
===================================================================================
Codigo real de: src/features/builder.py (funcion window_features y vecinos)

Copia COMENTADA, no se ejecuta. El codigo real vive en src/features/builder.py,
src/features/audio.py, src/features/extra.py, src/features/sentiment.py,
src/keywords/extractor.py.

Entrada:  un Window (candidato, ver paso 1) + el "contexto" del video completo
          (transcripcion, audio cargado, indice de keywords, etc.)
Salida:   un diccionario {nombre_columna: valor} -> se vuelve UNA FILA de la tabla.
"""


def window_features(ctx, w, hook_s, hook_words) -> dict:
    # ctx: VideoContext -> ya trae precalculado para TODO el video:
    #      - ctx.transcript  (palabras con timestamp, de faster-whisper)
    #      - ctx.audio       (AudioFrames: RMS, pitch, etc. por cada ~0.01s, de librosa)
    #      - ctx.kw_index    (KeywordIndex: pesos TF-IDF/KVSUM y embeddings KeyBERT)
    #      - ctx.sentiment   (polaridad por oracion, del modelo HF)
    # w:    el candidato puntual que estamos describiendo (p.ej. 3172.2s -> 3200.9s)

    feats = {}

    # --- Grupo K: keywords (tecnicas 5 y 6 de la tesis: KVSUM + KeyBERT) ---
    #
    #                    log(tf_ij + 1) * log(N / PF_i)
    #   w_ij = ---------------------------------------------------
    #          sqrt( SUM_k [ log(tf_kj + 1) * log(N / PF_k) ]^2 )
    #
    #   tf_ij  = cuantas veces aparece la palabra i en el pasaje j (este candidato)
    #   N      = numero total de pasajes (candidatos) del video
    #   PF_i   = en cuantos pasajes distintos aparece la palabra i ("passage frequency")
    #   el denominador (raiz de la suma de cuadrados) es solo normalizacion coseno
    #   de Salton: hace que pasajes largos y cortos queden en la misma escala.
    #
    #   EN CRIOLLO: que tan rara/especial es una palabra en ESTE pedacito del video
    #   comparado con el resto. Si la dicen mucho aqui pero poco en el resto del
    #   video, pesa mas (p.ej. en un podcast de 1h sobre negocios, si alguien dice
    #   "quiebra" solo en este tramo, esa palabra "ilumina" este candidato).
    feats.update(keyword_features(ctx, w))
    # columnas: k_kvsum_mean, k_kvsum_max, k_keybert_sim, k_novelty...

    # --- Grupo S: sentimiento / energia lexica (tecnica 7) ---
    #
    #   polaridad = P(positivo) - P(negativo)
    #
    #   P(positivo)/P(negativo) = probabilidades que escupe un modelo de IA
    #   (cardiffnlp/twitter-xlm-roberta... en el diseno, lxyuan/distilbert... en
    #   la implementacion real) ya entrenado para clasificar texto en 3 clases:
    #   positivo / neutral / negativo.
    #
    #   EN CRIOLLO: le restamos "que tan triste suena" a "que tan feliz suena".
    #   Si da +0.8, el tramo suena bastante positivo; si da -0.5, suena negativo.
    feats.update(sentiment_features(ctx, w))
    # columnas: s_polarity_mean, s_exclaim_pm (exclamaciones/min), s_second_person...

    # --- Grupo A: audio con librosa (tecnica 8) ---
    #
    #            log(volumen) - promedio(log(volumen) en TODO el video)
    #   z_rms = -----------------------------------------------------------
    #                  desviacion_estandar(log(volumen) en TODO el video)
    #
    #   EN CRIOLLO: que tan fuerte habla la persona EN ESE MOMENTO comparado con
    #   su propio promedio durante TODO el video (z-score). No importa si el
    #   micro graba flojito o fuerte en general: importa si AHI subio el tono
    #   respecto a como habla normalmente esa misma persona en ese video.
    feats.update(audio_features(ctx, w))
    # columnas: a_rms_mean/a_rms_p95 (volumen), a_pitch_std (variacion de tono),
    # a_pause_ratio (fraccion de silencio), a_speech_rate (palabras por segundo)

    # --- Grupo H: "Hook" — los primeros segundos del clip (estilo Opus Clip) ---
    #
    #   h_question = 1  si las primeras palabras tienen "?", un numero, o una
    #                   palabra de contraste ("pero", "nadie", "nunca"...)
    #              = 0  si no
    #
    #   EN CRIOLLO: es un si/no por cada señal - ¿el clip arranca con algo que
    #   atrape? (una pregunta, una cifra, un "pero resulta que..."). Cada señal
    #   es su propia columna (h_question, h_number, h_contrast...), no se suman
    #   en una sola formula; el modelo (LightGBM) decide cuanto pesa cada una.
    feats.update(hook_features(ctx, w, hook_s))
    # hook_s = cuantos segundos cuentan como "el gancho inicial" (p.ej. 3s)
    # columnas: h_question, h_number, h_second_person, h_hook_word, h_rms_onset...

    # --- Grupo F: "Flow" — fluidez/coherencia del tramo completo ---
    #
    #                    e_i . e_(i+1)
    #   coherencia_i = -------------------   = similitud coseno entre el embedding
    #                  |e_i| * |e_(i+1)|       de la oracion i y el de la oracion i+1
    #
    #   e_i = vector (embedding) de la oracion i, sacado del mismo modelo multi-
    #         lingue que se usa para KeyBERT (sentence-transformers).
    #
    #   EN CRIOLLO: comparamos el "significado" de una oracion con el de la
    #   siguiente. Si son muy parecidas (hablan de lo mismo), coherencia sale alta
    #   y el clip "fluye bien"; si saltan de tema de golpe, coherencia sale baja
    #   y se siente cortado/inconexo. f_coherence_mean = promedio de todas esas
    #   comparaciones dentro del candidato; f_coherence_min = la peor transicion.
    feats.update(flow_features(ctx, w))
    # columnas: f_starts_dangling (¿empieza con "eso", "el", algo colgado?),
    # f_coherence_mean, f_coherence_min, f_topic_drift, f_duration...

    # --- Grupo P: posicion relativa en el video (SOLO para ablacion/control) ---
    feats["p_rel_pos"] = w.start / ctx.duration
    # OJO: esta columna NO debe ayudar al modelo real (ver D-018 en DECISIONES.md:
    # en una version anterior, otra feature filtraba la posicion sin querer y eso
    # inflo los resultados de forma artificial). Por diseño, p_rel_pos solo se usa
    # como control/ablacion, nunca en el modelo desplegado.

    return feats
    # Este diccionario con ~70-90 llaves (mas E, L, U que se agregan despues,
    # ver abajo) se convierte en UNA FILA de un DataFrame de pandas.


# ------------------------------------------------------------------------------
# Despues de windows_frame() (que llama a window_features() para los 400
# candidatos y arma la tabla), otros dos modulos AGREGAN mas columnas a la
# misma tabla sin tocar lo anterior:
#
#   from src.features.extra import augment_table
#   df = augment_table(df, audio_path, cfg)
#       -> agrega columnas e_* (eventos de audio, grupo E):
#
#          p(risa) = red_neuronal_preentrenada(audio)
#
#          La "red neuronal preentrenada" es PANNs CNN14 (entrenada en AudioSet,
#          2 millones de clips de YouTube con 527 tipos de sonido). Se le pasa
#          el audio en trocitos de 1s (cada 3s, para no gastar tanto computo) y
#          devuelve una probabilidad de 0 a 1 para cada clase: risa, aplausos,
#          vitores, musica, habla.
#          EN CRIOLLO: una IA entrenada para reconocer SONIDOS (no palabras)
#          escucha el audio y dice que tan probable es que haya risas o aplausos
#          ahi. No lee la transcripcion, solo "escucha".
#
#       -> agrega columnas l_* (linguisticas, grupo L):
#
#          H = - SUM_t [ p(t) * log2(p(t)) ]      (entropia de Shannon)
#
#          p(t) = que fraccion de las palabras del candidato es la palabra t.
#          EN CRIOLLO: mide que tan VARIADO es el vocabulario. Si el candidato
#          repite las mismas 4-5 palabras una y otra vez, p(t) queda concentrada
#          en pocas palabras y H sale BAJO. Si usa puro vocabulario distinto sin
#          repetir, H sale ALTO. (Tambien se agregan aqui: type-token ratio,
#          ratio de pronombres 1a/2a persona vs 3a, tasa de muletillas y
#          silencio — papers de Reddy et al., ACL 2021).
#
#   from src.features.speech_embed import augment_table as augment_u
#   df = augment_u(df, audio_path, pca, cfg)
#       -> agrega columnas u_00...u_31 (embeddings de audio HuBERT + PCA a 32
#          dimensiones). SOLO se corre con configs/gpu.yaml, porque en CPU tarda
#          875 segundos por cada minuto de audio (inviable).
#
# Al final la tabla de un video tiene ~120 columnas: metadatos (start, end),
# ~115 de features (k_/s_/a_/h_/f_/p_/e_/l_/u_) y 4 de etiqueta (y_*, ver paso 3).
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# EJEMPLO REAL (candidato del minuto 53, el pico de la curva Most Replayed del
# video -Gzz2LZGowU), tomado directamente del parquet en:
#   D:/tesis_data/data/processed/features/sentence/-Gzz2LZGowU.parquet
#
#   k_kvsum_mean      = 0.846     (muy cargado de palabras clave)
#   s_polarity_mean   = 0.207     (ligeramente positivo)
#   a_rms_mean        = 0.173     (volumen)
#   a_pause_ratio     = 0.0       (habla fluida, sin pausas largas)
#   h_question        = 0.0       (no arranca con pregunta)
#   f_duration        = 28.74     (segundos)
#   e_laugh_mean      = 0.0008    (casi nada de risa)
# ------------------------------------------------------------------------------
