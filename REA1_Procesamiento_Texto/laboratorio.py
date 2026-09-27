"""Laboratorio REA 1: procesamiento de texto.

Cubre normalización, stemming, stopwords, TF, IDF, POS, lematización,
parsing, NER, generación con LLM, question answering, summarization,
similitud de oraciones, clasificación, traducción, generación estadística
y minería de texto. Semilla 42. Salidas: figuras/ y resultados.json.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent
FIG = BASE / "figuras"
SEED = 42
rng = np.random.default_rng(SEED)

AZUL = "#1f4e79"
VERDE = "#2e7d4f"
ROJO = "#a33b32"
ARENA = "#c4a35a"
COLORES_CLASE = {"tecnologia": AZUL, "salud": VERDE, "deporte": ROJO}

# Documentos cortos etiquetados. Los tres últimos mezclan temas a propósito.
CORPUS = [
    ("tecnologia", "El modelo de lenguaje procesa el texto y genera una respuesta."),
    ("tecnologia", "Un modelo de lenguaje representa el texto con vectores numéricos."),
    ("tecnologia", "El modelo procesa texto técnico y clasifica documentos."),
    ("tecnologia", "La red neuronal entrena un modelo de lenguaje sobre el texto."),
    ("tecnologia", "El transformer es un modelo de lenguaje que predice la palabra del texto."),
    ("salud", "El paciente presentó fiebre y el médico ordenó un examen."),
    ("salud", "El médico revisó al paciente en el hospital por la mañana."),
    ("salud", "El hospital atendió al paciente con una infección respiratoria."),
    ("salud", "El médico indicó un tratamiento cuando el paciente tenía infección."),
    ("salud", "La clínica del hospital registró la presión arterial del paciente."),
    ("deporte", "El equipo de fútbol ganó el partido con un gol."),
    ("deporte", "El equipo entrenó antes del partido de fútbol."),
    ("deporte", "El delantero del equipo marcó un gol en el partido."),
    ("deporte", "El estadio recibió al equipo de fútbol en el partido final."),
    ("deporte", "El entrenador cambió la táctica del equipo durante el partido."),
    ("tecnologia", "El médico revisó al paciente en el hospital y consultó un modelo."),
    ("salud", "El equipo de fútbol ganó el partido y luego el paciente celebró."),
    ("deporte", "El modelo de lenguaje procesa el texto del partido final."),
]

TEXTO_LARGA = (
    "El 12 de mayo de 2025, Ana Rojas, investigadora de la Universidad de Cundinamarca "
    "en Fusagasugá, presentó en Bogotá un convenio con el Ministerio de Tecnologías. "
    "El proyecto Andina-NLP entrenará modelos en español para resumir historias clínicas. "
    "El hospital San Rafael aportará el corpus anonimizado. "
    "Rojas explicó que el modelo no reemplaza al médico: solo ordena el texto para que "
    "la revisión humana sea más rápida. "
    "La gobernación de Cundinamarca financiará la primera fase hasta diciembre de 2026."
)

EJEMPLO_SUCIO = (
    "¡El Modelo de Lenguaje, en la Universidad de Cundinamarca, procesa TEXTO!!!  "
    "Visite https://ejemplo.edu/nlp"
)

FRASE_LEMMA = "Los modelos de lenguaje procesan documentos clínicos y generan resúmenes."

FRASES_SIMILITUD = [
    ("tecnologia", "El modelo de lenguaje procesa el texto y genera una respuesta."),
    ("tecnologia", "La red neuronal entrena un modelo de lenguaje sobre el texto."),
    ("salud", "El paciente presentó fiebre y el médico ordenó un examen."),
    ("salud", "El médico revisó al paciente en el hospital por la mañana."),
    ("deporte", "El equipo de fútbol ganó el partido con un gol."),
    ("deporte", "El delantero del equipo marcó un gol en el partido."),
]

PREGUNTAS = [
    {
        "pregunta": "¿Quién presentó el convenio en Bogotá?",
        "tipo": "PER",
        "oro": "Ana Rojas",
    },
    {
        "pregunta": "¿En qué ciudad se presentó el convenio?",
        "tipo": "LOC",
        "oro": "Bogotá",
    },
    {
        "pregunta": "¿Qué hospital aportará el corpus?",
        "tipo": "HOSPITAL",
        "oro": "San Rafael",
    },
    {
        "pregunta": "¿Hasta cuándo financiará la gobernación la primera fase?",
        "tipo": "FECHA",
        "oro": "diciembre de 2026",
    },
]

TRADUCIR = [
    "El modelo de lenguaje resume historias clínicas.",
    "El hospital San Rafael aportará un corpus anonimizado.",
    "La selección de fútbol entrenó en Bogotá.",
]

LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def normalizar(texto: str) -> str:
    """Minúsculas, sin URL ni puntuación. Conserva tildes y eñe."""
    texto = texto.lower().strip()
    texto = re.sub(r"https?://\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


STOPS: set[str] = set()


def tokenizar(texto: str) -> list[str]:
    return re.findall(r"[a-záéíóúüñ]{2,}", normalizar(texto))


def tokens_contenido(texto: str) -> list[str]:
    return [t for t in tokenizar(texto) if t not in STOPS]


def guardar_figura(fig: plt.Figure, nombre: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    ruta = FIG / nombre
    fig.savefig(ruta, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def cargar_stopwords() -> set[str]:
    base = set(stopwords.words("spanish"))
    base.update({"tf", "idf", "si", "solo", "cada"})
    return base


def oraciones_de(texto: str, nlp) -> list[str]:
    doc = nlp(texto)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def textrank(oraciones: list[str], k: int = 2) -> tuple[list[str], list[float]]:
    """Resumen extractivo. Mihalcea y Tarau (2004), factor de amortiguación 0.85."""
    n = len(oraciones)
    if n <= k:
        return oraciones, [1.0] * n
    vectorizador = TfidfVectorizer(tokenizer=tokens_contenido, token_pattern=None)
    matriz = vectorizador.fit_transform(oraciones)
    sim = cosine_similarity(matriz)
    np.fill_diagonal(sim, 0.0)
    suma = sim.sum(axis=1, keepdims=True)
    suma[suma == 0] = 1.0
    transicion = sim / suma
    puntaje = np.ones(n) / n
    for _ in range(40):
        puntaje = (1 - 0.85) / n + 0.85 * transicion.T @ puntaje
    elegidas = sorted(np.argsort(puntaje)[::-1][:k])
    return [oraciones[i] for i in elegidas], [round(float(p), 4) for p in puntaje]


def generar_trigrama(tokens: list[str], semilla: tuple[str, str], n: int = 12) -> str:
    """Modelo de lenguaje de trigramas con retroceso a bigrama y unigrama."""
    if len(tokens) < 3:
        return " ".join(tokens)
    uni = Counter(tokens)
    bi = Counter(zip(tokens, tokens[1:]))
    tri = Counter(zip(tokens, tokens[1:], tokens[2:]))
    w1, w2 = semilla
    salida = [w1, w2]
    local = np.random.default_rng(SEED)
    for _ in range(n):
        cands = {c: k for (a, b, c), k in tri.items() if a == w1 and b == w2}
        if not cands:
            cands = {b: k for (a, b), k in bi.items() if a == w2}
        if not cands:
            cands = dict(uni)
        palabras = list(cands)
        pesos = np.array([cands[p] for p in palabras], dtype=float)
        pesos /= pesos.sum()
        elegido = str(local.choice(palabras, p=pesos))
        salida.append(elegido)
        w1, w2 = w2, elegido
    return " ".join(salida)


def respuesta_qa(nlp, oraciones: list[str], pregunta: str, tipo: str) -> dict:
    """QA extractivo: la oración más cercana en coseno y un tramo según el tipo."""
    vectorizador = TfidfVectorizer(tokenizer=tokens_contenido, token_pattern=None)
    matriz = vectorizador.fit_transform(oraciones + [pregunta])
    sims = cosine_similarity(matriz[-1], matriz[:-1]).ravel()
    idx = int(np.argmax(sims))
    oracion = oraciones[idx]
    doc = nlp(oracion)
    if tipo == "PER":
        pers = [e.text for e in doc.ents if e.label_ == "PER"]
        tramo = pers[0] if pers else oracion
    elif tipo == "LOC":
        locs = [e for e in doc.ents if e.label_ == "LOC"]
        elegida = None
        for ent in locs:
            if ent.root.head.pos_ == "VERB":
                elegida = ent.text
        tramo = elegida or (locs[-1].text if locs else oracion)
    elif tipo == "HOSPITAL":
        m = re.search(
            r"[Hh]ospital\s+([A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)?)",
            oracion,
        )
        tramo = m.group(1).strip() if m else oracion
    else:
        m = re.search(
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|"
            r"octubre|noviembre|diciembre) de \d{4}",
            oracion,
            flags=re.IGNORECASE,
        )
        tramo = m.group(0) if m else oracion
    return {
        "oracion": oracion,
        "coseno": round(float(sims[idx]), 4),
        "respuesta": tramo,
    }


def _modelo_opus() -> Path:
    """Opus-MT es→en empaquetado por Argos (Tiedemann y Thottingal, 2020)."""
    destino = BASE / "modelos" / "translate-es_en-1_9"
    if (destino / "model" / "model.bin").exists() and (destino / "bpe.model").exists():
        return destino
    import io
    import urllib.request
    import zipfile

    destino.parent.mkdir(parents=True, exist_ok=True)
    url = "https://argos-net.com/v1/translate-es_en-1_9.argosmodel"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print("Descargando Opus-MT es→en...")
    with urllib.request.urlopen(req) as resp:
        datos = resp.read()
    with zipfile.ZipFile(io.BytesIO(datos)) as zf:
        zf.extractall(destino.parent)
    extraido = destino.parent / "translate-es_en-1_9"
    if extraido.resolve() != destino.resolve() and extraido.exists():
        extraido.rename(destino)
    return destino


def traducir_es_en(frases: list[str]) -> list[dict]:
    """Traducción neuronal con CTranslate2 y el modelo bilingüe Opus-MT."""
    import ctranslate2
    from subword_nmt.apply_bpe import BPE

    raiz = _modelo_opus()
    bpe = BPE(open(raiz / "bpe.model", encoding="utf-8"))
    traductor = ctranslate2.Translator(str(raiz / "model"), device="cpu")
    pares = []
    for frase in frases:
        tokens = bpe.process_line(frase).split()
        hipotesis = traductor.translate_batch([tokens], beam_size=5)[0].hypotheses[0]
        texto = " ".join(hipotesis).replace("@@ ", "").replace("</w>", " ")
        texto = re.sub(r"\s+", " ", texto).replace(" .", ".").strip()
        pares.append({"es": frase, "en": texto, "motor": "opus-mt-es-en"})
    return pares


def _chat(modelo, tokenizador, usuario: str, max_new: int = 80) -> str:
    mensajes = [{"role": "user", "content": usuario}]
    texto = tokenizador.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    entradas = tokenizador(texto, return_tensors="pt")
    salida = modelo.generate(
        **entradas,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tokenizador.eos_token_id,
    )
    nuevo = salida[0, entradas["input_ids"].shape[1] :]
    return tokenizador.decode(nuevo, skip_special_tokens=True).strip()


def generar_con_llm(texto_largo: str, preguntas: list[dict], resumen_extractivo: list[str]) -> dict:
    """Generación, QA generativo y resumen abstractivo con un LLM pequeño."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizador = AutoTokenizer.from_pretrained(LLM_ID)
    modelo = AutoModelForCausalLM.from_pretrained(LLM_ID)
    prompt_gen = (
        "Escribe exactamente dos oraciones en español, sin lista, sobre un modelo "
        "de lenguaje que resume historias clínicas en un hospital."
    )
    generacion = _chat(modelo, tokenizador, prompt_gen, 90)
    resumen_abs = _chat(
        modelo,
        tokenizador,
        "Resume en una sola oración en español, sin viñetas:\n" + texto_largo,
        60,
    )
    qa = []
    for item in preguntas:
        respuesta = _chat(
            modelo,
            tokenizador,
            "Responde en español solo con el dato, sin explicar.\n"
            f"Texto: {texto_largo}\nPregunta: {item['pregunta']}",
            30,
        )
        qa.append(
            {
                "pregunta": item["pregunta"],
                "oro": item["oro"],
                "respuesta_llm": respuesta,
                "acierto": item["oro"].lower() in respuesta.lower(),
            }
        )
    return {
        "modelo": LLM_ID,
        "prompt": prompt_gen,
        "generacion": generacion,
        "resumen_abstractivo": resumen_abs,
        "resumen_extractivo": resumen_extractivo,
        "qa": qa,
    }


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    global STOPS
    nlp = spacy.load("es_core_news_sm")
    stemmer = SnowballStemmer("spanish")
    STOPS = cargar_stopwords()
    stops = STOPS

    norm_ejemplo = normalizar(EJEMPLO_SUCIO)
    tokens_ejemplo = tokenizar(EJEMPLO_SUCIO)
    tokens_sin_stop = [t for t in tokens_ejemplo if t not in stops]

    doc_lemma = nlp(FRASE_LEMMA)
    tabla_stem = []
    for tok in doc_lemma:
        if not tok.is_alpha:
            continue
        tabla_stem.append(
            {
                "token": tok.text,
                "stem": stemmer.stem(tok.text.lower()),
                "lema": tok.lemma_,
                "stopword": tok.text.lower() in stops,
            }
        )

    textos = [t for _, t in CORPUS]
    etiquetas = [c for c, _ in CORPUS]
    textos_limpios = []
    for texto in textos:
        limpio = " ".join(t for t in tokenizar(texto) if t not in stops)
        textos_limpios.append(limpio)

    conteo = CountVectorizer(tokenizer=str.split, token_pattern=None)
    matriz_tf = conteo.fit_transform(textos_limpios)
    vocab = np.array(conteo.get_feature_names_out())
    tf_total = np.asarray(matriz_tf.sum(axis=0)).ravel()
    orden_tf = np.argsort(tf_total)[::-1][:12]
    tf_top = [
        {"termino": vocab[i], "tf": int(tf_total[i])} for i in orden_tf
    ]

    tfidf = TfidfVectorizer(tokenizer=str.split, token_pattern=None)
    tfidf.fit(textos_limpios)
    idf_vocab = np.array(tfidf.get_feature_names_out())
    idf_vals = tfidf.idf_
    # Términos elegidos para contrastar frecuencia documental.
    elegidos_idf = []
    for termino in ["modelo", "lenguaje", "paciente", "fútbol", "gol", "texto", "equipo", "hospital"]:
        if termino not in set(idf_vocab):
            continue
        i = int(np.where(idf_vocab == termino)[0][0])
        df = int((matriz_tf[:, conteo.vocabulary_[termino]].toarray() > 0).sum()) if termino in conteo.vocabulary_ else 0
        elegidos_idf.append(
            {"termino": termino, "df": df, "idf": round(float(idf_vals[i]), 3)}
        )
    elegidos_idf.sort(key=lambda x: x["idf"])

    frase_pos = "Ana Rojas presentó el convenio en Bogotá."
    doc_pos = nlp(frase_pos)
    pos_filas = [
        {
            "token": t.text,
            "pos": t.pos_,
            "etiqueta": t.tag_,
            "lema": t.lemma_,
            "dep": t.dep_,
            "cabeza": t.head.text,
        }
        for t in doc_pos
    ]
    doc_largo = nlp(TEXTO_LARGA)
    conteo_pos = Counter(t.pos_ for t in doc_largo if not t.is_punct)
    entidades = [
        {"texto": e.text, "etiqueta": e.label_, "inicio": e.start_char}
        for e in doc_largo.ents
    ]

    oraciones = oraciones_de(TEXTO_LARGA, nlp)
    resumen, puntajes_tr = textrank(oraciones, k=2)

    vector_sim = TfidfVectorizer(tokenizer=tokens_contenido, token_pattern=None)
    mat_sim = vector_sim.fit_transform([f for _, f in FRASES_SIMILITUD])
    cos = cosine_similarity(mat_sim)
    pares = []
    for i in range(len(FRASES_SIMILITUD)):
        for j in range(i + 1, len(FRASES_SIMILITUD)):
            pares.append(
                {
                    "i": i,
                    "j": j,
                    "clase_i": FRASES_SIMILITUD[i][0],
                    "clase_j": FRASES_SIMILITUD[j][0],
                    "coseno": round(float(cos[i, j]), 3),
                }
            )
    mismo = [p["coseno"] for p in pares if p["clase_i"] == p["clase_j"]]
    distinto = [p["coseno"] for p in pares if p["clase_i"] != p["clase_j"]]

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(tokenizer=tokens_contenido, token_pattern=None)),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )
    pred = cross_val_predict(pipe, textos, etiquetas, cv=LeaveOneOut())
    clases = ["tecnologia", "salud", "deporte"]
    matriz = confusion_matrix(etiquetas, pred, labels=clases)
    reporte = classification_report(etiquetas, pred, labels=clases, output_dict=True, zero_division=0)
    errores = []
    for texto, real, estimado in zip(textos, etiquetas, pred):
        if real != estimado:
            errores.append({"texto": texto, "real": real, "predicho": estimado})
    pipe.fit(textos, etiquetas)
    coef = pipe.named_steps["clf"].coef_
    terminos_clf = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    rasgos = {}
    for i, clase in enumerate(pipe.named_steps["clf"].classes_):
        top_idx = np.argsort(coef[i])[::-1][:5]
        rasgos[clase] = [
            {"termino": terminos_clf[j], "peso": round(float(coef[i, j]), 3)} for j in top_idx
        ]

    generacion = {}
    for clase in clases:
        tokens_clase = []
        for c, texto in CORPUS:
            if c == clase and texto not in {t for _, t in CORPUS[-3:]}:
                tokens_clase.extend(t for t in tokenizar(texto) if t not in stops)
        semilla = {"tecnologia": ("modelo", "lenguaje"), "salud": ("paciente", "médico"), "deporte": ("equipo", "fútbol")}
        # El stem del corpus no quita tildes: médico sigue con tilde en el token.
        if clase == "salud":
            semilla[clase] = ("paciente", "médico") if "médico" in tokens_clase else ("paciente", "medico")
        generacion[clase] = generar_trigrama(tokens_clase, semilla[clase], n=10)

    # Minería: TTR, bigramas y términos distintivos por media de TF-IDF.
    mineria_clases = {}
    for clase in clases:
        docs = [normalizar(t) for c, t in CORPUS if c == clase]
        toks = [tok for d in docs for tok in tokenizar(d) if tok not in stops]
        tipos = len(set(toks))
        big = CountVectorizer(
            tokenizer=tokens_contenido, token_pattern=None, ngram_range=(2, 2)
        )
        try:
            bmat = big.fit_transform(docs)
            bsum = np.asarray(bmat.sum(axis=0)).ravel()
            bvocab = np.array(big.get_feature_names_out())
            top_b = np.argsort(bsum)[::-1][:5]
            bigramas = [{"bigrama": bvocab[i], "tf": int(bsum[i])} for i in top_b if bsum[i] > 0]
        except ValueError:
            bigramas = []
        mineria_clases[clase] = {
            "tokens": len(toks),
            "tipos": tipos,
            "ttr": round(tipos / len(toks), 3) if toks else 0,
            "bigramas": bigramas,
            "rasgos_clf": rasgos[clase],
        }

    qa = []
    for item in PREGUNTAS:
        hallazgo = respuesta_qa(nlp, oraciones, item["pregunta"], item["tipo"])
        qa.append(
            {
                "pregunta": item["pregunta"],
                "oro": item["oro"],
                "respuesta": hallazgo["respuesta"],
                "oracion": hallazgo["oracion"],
                "coseno": hallazgo["coseno"],
                "acierto": item["oro"].lower() in hallazgo["respuesta"].lower(),
            }
        )

    print("Traduciendo es→en (Argos, Opus-MT)...")
    traducciones = traducir_es_en(TRADUCIR)

    # Figuras
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([t["termino"] for t in tf_top][::-1], [t["tf"] for t in tf_top][::-1], color=AZUL)
    ax.set_xlabel("Frecuencia en el corpus limpio")
    ax.set_title("Term frequency: 12 términos más frecuentes")
    guardar_figura(fig, "tf_top.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([d["termino"] for d in elegidos_idf], [d["idf"] for d in elegidos_idf], color=VERDE)
    ax.set_ylabel("IDF (suavizado de scikit-learn)")
    ax.set_title("Inverse document frequency de términos elegidos")
    guardar_figura(fig, "idf.png")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cos, cmap="Blues", vmin=0, vmax=1)
    etiquetas_cortas = [f"{i+1} {c[:3]}" for i, (c, _) in enumerate(FRASES_SIMILITUD)]
    ax.set_xticks(range(6), etiquetas_cortas, rotation=45, ha="right")
    ax.set_yticks(range(6), etiquetas_cortas)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Similitud coseno entre oraciones")
    fig.colorbar(im, ax=ax, fraction=0.046)
    guardar_figura(fig, "similitud.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(3), clases, rotation=20)
    ax.set_yticks(range(3), clases)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center")
    ax.set_title("Matriz de confusión (dejar-uno-fuera)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    guardar_figura(fig, "clasificacion.png")

    fig, ax = plt.subplots(figsize=(7.2, 4))
    items = conteo_pos.most_common()
    ax.bar([k for k, _ in items], [v for _, v in items], color=ARENA)
    ax.set_title("Part-of-speech en el texto largo")
    ax.set_ylabel("Tokens")
    guardar_figura(fig, "pos.png")

    resultados = {
        "semilla": SEED,
        "n_docs": len(CORPUS),
        "ejemplo_sucio": EJEMPLO_SUCIO,
        "ejemplo_norm": norm_ejemplo,
        "n_tokens": len(tokens_ejemplo),
        "n_tokens_sin_stop": len(tokens_sin_stop),
        "tokens_sin_stop": tokens_sin_stop,
        "stem_lema": tabla_stem,
        "tf_top": tf_top,
        "idf": elegidos_idf,
        "pos_frase": frase_pos,
        "pos": pos_filas,
        "pos_distribucion": conteo_pos.most_common(),
        "ner": entidades,
        "resumen": resumen,
        "textrank": puntajes_tr,
        "oraciones": oraciones,
        "sim_intra": round(float(np.mean(mismo)), 3) if mismo else 0,
        "sim_inter": round(float(np.mean(distinto)), 3) if distinto else 0,
        "pares": pares,
        "accuracy": round(float(accuracy_score(etiquetas, pred)), 3),
        "matriz": matriz.tolist(),
        "clases": clases,
        "f1_macro": round(float(reporte["macro avg"]["f1-score"]), 3),
        "errores": errores,
        "rasgos": rasgos,
        "generacion_trigrama": generacion,
        "mineria": mineria_clases,
        "qa": qa,
        "qa_aciertos": sum(1 for q in qa if q["acierto"]),
        "traduccion": traducciones,
        "texto_largo": TEXTO_LARGA,
    }

    llm_path = BASE / "llm.json"
    if llm_path.exists():
        resultados["llm"] = json.loads(llm_path.read_text(encoding="utf-8"))
    elif os.environ.get("SKIP_LLM") == "1":
        resultados["llm"] = {"omitido": True}
        print("LLM omitido (SKIP_LLM=1).")
    else:
        try:
            print("Cargando LLM (puede descargar el modelo la primera vez)...")
            resultados["llm"] = generar_con_llm(TEXTO_LARGA, PREGUNTAS, resumen)
            llm_path.write_text(
                json.dumps(resultados["llm"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 — se registra y el resto del lab sigue
            resultados["llm"] = {"error": f"{type(exc).__name__}: {exc}"}
            print("LLM no disponible:", resultados["llm"]["error"])

    salida = BASE / "resultados.json"
    salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Accuracy LOOCV:", resultados["accuracy"])
    print("QA aciertos:", resultados["qa_aciertos"], "/", len(qa))
    print("NER:", [(e["texto"], e["etiqueta"]) for e in entidades])
    print("Resumen:", resumen)
    print("Traducción:", traducciones)
    print("Guardado", salida)
    return resultados


try:
    STOPS = cargar_stopwords()
except LookupError:
    import nltk

    nltk.download("stopwords", quiet=True)
    STOPS = cargar_stopwords()


if __name__ == "__main__":
    main()
