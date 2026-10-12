"""Glosario de variables: como se calcula cada una (formula) y que significa. Lo consume la pagina web (scripts/web_clips.py)."""

# var -> (grupo, formula, explicacion)
G = {
    # ---- K palabras clave
    "k_kvsum_mean": ("K Palabras clave", "media( peso_KVSUM(palabra) ) sobre las palabras de contenido de la ventana",
                     "Peso KVSUM: log(tf+1)·log(N/PF) normalizado por pasaje de 300 s. Alto = la ventana usa terminos importantes del video."),
    "k_kvsum_max": ("K Palabras clave", "max( peso_KVSUM(palabra) )", "Un solo termino muy central ya define el tramo."),
    "k_kvsum_local": ("K Palabras clave", "sum( peso_KVSUM en su pasaje de 300 s ) / n_palabras", "Importancia de las palabras relativa a su contexto cercano."),
    "k_top_terms_pm": ("K Palabras clave", "n( palabras que estan en el top de terminos del video ) / min", "Cuanto se habla del tema principal."),
    "k_content_ratio": ("K Palabras clave", "palabras de contenido / palabras totales", "Densidad informativa frente a relleno (conectores, muletillas)."),
    "k_novelty": ("K Palabras clave", "1 − fraccion de palabras ya dichas en los 300 s anteriores", "Novedad local: ¿aparecen ideas nuevas?"),
    "k_keybert_sim": ("K Palabras clave", "coseno( embedding medio de las oraciones , embedding del video )",
                      "Que tan cerca esta el tramo del tema central del video."),
    "k_keyphrase_pm": ("K Palabras clave", "n( frases clave KeyBERT contenidas en el texto ) / min", "Frases-resumen presentes en la ventana."),
    "k_noun_chunks_ps": ("K Palabras clave", "media( n_sintagmas_nominales por oracion ) (spaCy)", "Densidad explicativa: muchos sustantivos = explicacion tecnica."),
    # ---- S sentimiento
    "s_pos_mean": ("S Sentimiento", "media( P(positivo) por oracion )", "Carga positiva (DistilBERT multilingue)."),
    "s_neg_mean": ("S Sentimiento", "media( P(negativo) por oracion )", "Carga negativa."),
    "s_polarity_mean": ("S Sentimiento", "media( P(pos) − P(neg) )", "Tono emocional global."),
    "s_polarity_std": ("S Sentimiento", "desvio( P(pos) − P(neg) )", "Cambios de tono dentro del tramo."),
    "s_intensity_mean": ("S Sentimiento", "media( |P(pos) − P(neg)| )", "Intensidad emocional sin importar el signo."),
    "s_intensity_max": ("S Sentimiento", "max( |P(pos) − P(neg)| )", "La oracion mas intensa del tramo."),
    "s_questions_pm": ("S Sentimiento", "( n('?') + n('¿') ) / min", "Preguntas por minuto: abren curiosidad y cambian de tema."),
    "s_exclaim_pm": ("S Sentimiento", "( n('!') + n('¡') ) / min", "Exclamaciones por minuto: expresividad."),
    "s_second_person": ("S Sentimiento", "tokens en 2.ª persona (tu, you...) / tokens", "Interpelar al espectador."),
    "s_first_person": ("S Sentimiento", "tokens en 1.ª persona (yo, I...) / tokens", "Hablar de uno mismo: anecdota personal."),
    "s_numbers": ("S Sentimiento", "tokens que empiezan con digito / tokens", "Datos concretos (\"3 razones\")."),
    "s_intensifiers": ("S Sentimiento", "intensificadores (muy, super, really...) / tokens", "Enfasis lexico."),
    "s_hook_words": ("S Sentimiento", "palabras gancho (secreto, nunca, never...) / tokens", "Lenguaje de gancho."),
    "s_laughs": ("S Sentimiento", "n( 'jaja', 'haha'... en el texto )", "Risas escritas en la transcripcion."),
    # ---- A audio
    "a_rms_mean": ("A Audio", "media( RMS_z ) ;  RMS_z = (log(volumen) − promedio del video) / desvio", "Nivel de volumen del tramo respecto al video."),
    "a_rms_p95": ("A Audio", "percentil 95( RMS_z )", "Picos de volumen."),
    "a_rms_std": ("A Audio", "desvio( RMS_z )", "Dinamica del volumen: susurros y enfasis frente a voz plana."),
    "a_pitch_std": ("A Audio", "desvio( f0 en semitonos )", "Variacion de entonacion."),
    "a_pitch_range": ("A Audio", "percentil 95( f0 ) − percentil 5( f0 )", "Rango de tono: voz expresiva."),
    "a_voiced_ratio": ("A Audio", "frames con voz / frames totales", "Continuidad de la voz (menos = pausas, risas, musica)."),
    "a_energy_delta": ("A Audio", "media( RMS_z de la ventana ) − media( RMS_z de los 30 s previos )", "Subida de energia respecto al momento anterior."),
    "a_speech_rate": ("A Audio", "n_palabras / duracion (s)", "Ritmo del habla."),
    "a_pause_ratio": ("A Audio", "suma( huecos entre palabras > 0.3 s ) / duracion", "Proporcion en pausa. La variable de mayor peso de la formula."),
    "a_long_pauses_pm": ("A Audio", "n( huecos > 1 s ) / min", "Pausas largas: cambio de idea o remate."),
    # ---- H gancho
    "h_rms_onset": ("H Gancho", "media( RMS_z de los primeros 3 s )", "Arranque con energia."),
    "h_question": ("H Gancho", "1 si las primeras 15 palabras traen '?' ; si no 0", "Abrir con pregunta engancha."),
    "h_number": ("H Gancho", "1 si el arranque contiene un numero", "Abrir con un dato."),
    "h_second_person": ("H Gancho", "1 si el arranque interpela al espectador (tu/you)", "Interpelacion directa."),
    "h_hook_word": ("H Gancho", "1 si el arranque trae palabra gancho", "Lenguaje de gancho al inicio."),
    "h_contrast": ("H Gancho", "1 si el arranque trae contraste (pero, sin embargo...)", "Tension o giro al inicio."),
    "h_first_sentence_s": ("H Gancho", "duracion de la primera oracion (s)", "Aperturas cortas retienen mas."),
    "h_first_intensity": ("H Gancho", "|P(pos) − P(neg)| de la primera oracion", "Emocion desde el inicio."),
    "h_first_keybert": ("H Gancho", "coseno( embedding de la 1.ª oracion , embedding del video )", "Abre con el tema, no con relleno."),
    # ---- F flujo
    "f_starts_dangling": ("F Flujo", "1 si la 1.ª palabra es colgada (\"eso\", \"el\", \"and\"...)", "Un clip debe entenderse solo: penaliza empezar a mitad de idea."),
    "f_ends_complete": ("F Flujo", "1 si la ultima oracion termina completa", "No cortar a mitad de frase."),
    "f_coherence_mean": ("F Flujo", "media( coseno entre oraciones consecutivas )", "Hilo continuo."),
    "f_coherence_min": ("F Flujo", "min( coseno entre oraciones consecutivas )", "El punto mas debil del hilo."),
    "f_topic_drift": ("F Flujo", "1 − coseno( 1.ª mitad , 2.ª mitad )", "Cambio de tema dentro del clip."),
    "f_n_sentences": ("F Flujo", "numero de oraciones de la ventana", "Cantidad de contenido."),
    "f_duration": ("F Flujo", "fin − inicio (s)", "Duracion del clip."),
    "f_filler_rate": ("F Flujo", "muletillas (eh, o sea, like...) / tokens", "Fluidez del habla."),
    "f_word_conf": ("F Flujo", "media( confianza de Whisper por palabra )", "Calidad de la transcripcion."),
    # ---- L linguistica
    "l_entropy": ("L Linguistica", "−Σ p(token)·log2 p(token)", "Variedad lexica (Shannon)."),
    "l_type_token": ("L Linguistica", "tokens distintos / tokens", "Riqueza lexica frente a repeticion."),
    "l_pron_12_vs_3": ("L Linguistica", "( pron. 1.ª+2.ª + 1 ) / ( pron. 3.ª + 1 )", "Dialogo directo frente a hablar de terceros."),
    "l_fillers": ("L Linguistica", "muletillas / tokens", "Fluidez."),
    "l_silence": ("L Linguistica", "1 − tiempo_hablado / duracion", "Silencio total del tramo. Segunda variable de mayor peso."),
    "l_word_len": ("L Linguistica", "media( letras por palabra )", "Registro: tecnico frente a coloquial."),
    # ---- E eventos
    "e_laugh_max": ("E Eventos de audio", "max( P(risa) ) en la ventana (red PANNs CNN14, 1 muestra/s)", "Hubo risa."),
    "e_laugh_mean": ("E Eventos de audio", "media( P(risa) )", "Cuanta risa en promedio."),
    "e_applause_max": ("E Eventos de audio", "max( P(aplauso) )", "Hubo aplauso."),
    "e_applause_mean": ("E Eventos de audio", "media( P(aplauso) )", "Aplauso medio."),
    "e_cheering_max": ("E Eventos de audio", "max( P(vitoreo) )", "Hubo vitoreo."),
    "e_cheering_mean": ("E Eventos de audio", "media( P(vitoreo) )", "Vitoreo medio."),
    "e_music_max": ("E Eventos de audio", "max( P(musica) )", "Hay musica (intro, transicion, remate editado)."),
    "e_music_mean": ("E Eventos de audio", "media( P(musica) )", "Musica media."),
    "e_speech_max": ("E Eventos de audio", "max( P(habla) )", "Habla presente."),
    "e_speech_mean": ("E Eventos de audio", "media( P(habla) )", "Habla pura; si baja, hay otro evento sonoro."),
    "e_laugh_tail": ("E Eventos de audio", "max( P(risa) ) en los 6 s finales", "Remate de un chiste."),
    "e_laugh_after": ("E Eventos de audio", "max( P(risa) ) en los 6 s posteriores al clip", "Reaccion inmediata."),
    "e_applause_after": ("E Eventos de audio", "max( P(aplauso) ) en los 6 s posteriores", "Reaccion del publico."),
    "e_laugh_frac": ("E Eventos de audio", "fraccion de segundos con P(risa) > 0.2", "Cuanto del tramo es risa."),
}

DIM = {"hook": "Hook (gancho)", "flow": "Flow (fluidez)", "value": "Value (contenido)"}
FORMULAS = [
    ("Percentil en el video", "pct(variable) = posicion de la ventana entre todas las del mismo video, de 0 a 99",
     "Cada video tiene su linea base; lo que importa es cuanto destaca la ventana frente a sus hermanas. Es el 'puntaje' que ves de cada variable."),
    ("Score del clip (0–99)", "raw = Σ peso_i · pct_i  (17 variables) ;  score = percentil de raw frente a los 10 mejores clips de cada video de desarrollo",
     "Es el que decide el ranking y la seleccion. Los pesos salen de la correlacion parcial (Spearman) con el calor de YouTube controlando la posicion."),
    ("Aporte SHAP", "aporte(variable) = contribucion de la variable a la prediccion del modelo explicador (LightGBM, valores SHAP)",
     "Positivo suma, negativo resta. La suma de los aportes de las variables h_* da Hook, f_* da Flow, el resto da Value."),
    ("Hook / Flow / Value (0–99)", "dimension = percentil( Σ aportes SHAP de sus variables ) frente a clips de referencia",
     "Explican el clip por dimensiones. No deciden la seleccion: sirven para entender que tiene de bueno el clip."),
    ("Seleccion de los 5 clips", "ordenar por raw ; quedarse con cada candidato solo si solapa < 30 % con los ya elegidos ; solo clips <= 45 s (V1)",
     "Evita que los 5 clips sean el mismo momento y favorece clips cortos, que aciertan mas picos."),
    ("Validacion", "un clip 'toca el pico' si se solapa con algun bin del calor de YouTube > 0.5 (max 20 por video)",
     "La curva real solo se usa DESPUES, para calificar. El sistema nunca la ve al elegir."),
]


import json as _json
from pathlib import Path as _P
_L = _json.loads((_P(__file__).with_name("var_loc.json")).read_text(encoding="utf-8"))
ORDER = _L["order"]      # orden en que las variables se calculan en el codigo (builder.py -> audio.py -> extra.py)
LOC = _L["loc"]          # variable -> [archivo, linea]
