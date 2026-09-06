"""Laboratorio: transforma textos en embeddings (ejecutable local y desde el notebook)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
FIG = BASE / "figuras"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

SEED = 42
np.random.seed(SEED)

# Corpus propio en español: tres temas (NLP, vivienda, deporte)
DOCUMENTOS = [
    {
        "id": "D1",
        "tema": "NLP",
        "texto": "El procesamiento de lenguaje natural convierte oraciones en vectores densos llamados embeddings.",
    },
    {
        "id": "D2",
        "tema": "NLP",
        "texto": "Los embeddings de lenguaje natural representan el significado del texto para comparar semántica y no solo palabras.",
    },
    {
        "id": "D3",
        "tema": "NLP",
        "texto": "Word2Vec y BERT aprenden embeddings y representaciones vectoriales del lenguaje natural a partir de corpus.",
    },
    {
        "id": "D4",
        "tema": "Vivienda",
        "texto": "El apartamento en Bogotá tiene tres habitaciones, garaje y un precio cercano a cuatrocientos millones.",
    },
    {
        "id": "D5",
        "tema": "Vivienda",
        "texto": "Busco un apartamento residencial con parqueadero en Bogotá, de tres habitaciones y buen precio.",
    },
    {
        "id": "D6",
        "tema": "Vivienda",
        "texto": "La casa en arriendo en Bogotá incluye tres habitaciones, dos baños y está cerca del TransMilenio.",
    },
    {
        "id": "D7",
        "tema": "Deporte",
        "texto": "El equipo de fútbol ganó el partido con dos goles en el segundo tiempo.",
    },
    {
        "id": "D8",
        "tema": "Deporte",
        "texto": "Los jugadores del equipo de fútbol entrenaron táctica antes del partido del campeonato.",
    },
]

STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "y", "o", "en", "con", "por", "para", "a", "que", "se", "su", "sus",
    "es", "son", "está", "estan", "este", "esta", "esto", "como", "más",
    "mas", "no", "si", "sí",
}


def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    tokens = [t for t in texto.split() if t and t not in STOPWORDS_ES]
    return " ".join(tokens)


def tokenizar(texto: str) -> list[str]:
    return normalizar(texto).split()


def one_hot_matriz(docs_norm: list[str]) -> tuple[np.ndarray, list[str]]:
    vocab = sorted({tok for doc in docs_norm for tok in doc.split()})
    idx = {w: i for i, w in enumerate(vocab)}
    m = np.zeros((len(docs_norm), len(vocab)), dtype=float)
    for i, doc in enumerate(docs_norm):
        for tok in set(doc.split()):
            m[i, idx[tok]] = 1.0
    return m, vocab


def vocabulario(docs_norm: list[str]) -> list[str]:
    return sorted({tok for doc in docs_norm for tok in doc.split()})


def matriz_coocurrencia(docs_norm: list[str], vocab: list[str], ventana: int = 2) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    C = np.zeros((n, n), dtype=float)
    for doc in docs_norm:
        toks = doc.split()
        for i, w in enumerate(toks):
            for j in range(max(0, i - ventana), min(len(toks), i + ventana + 1)):
                if i == j:
                    continue
                C[idx[w], idx[toks[j]]] += 1.0
    return C


def ppmi(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    total = C.sum() + eps
    p_ij = C / total
    p_i = p_ij.sum(axis=1, keepdims=True)
    p_j = p_ij.sum(axis=0, keepdims=True)
    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
    return np.maximum(pmi, 0.0)


def embeddings_palabra_svd(M: np.ndarray, k: int = 8) -> np.ndarray:
    k = min(k, min(M.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    return svd.fit_transform(M)


def embedding_documento_palabras(tokens: list[str], vocab: list[str], E: np.ndarray) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    vecs = [E[idx[t]] for t in tokens if t in idx]
    if not vecs:
        return np.zeros(E.shape[1])
    return np.mean(vecs, axis=0)


def heatmap(matriz, etiquetas, titulo, path):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(matriz, cmap="YlGnBu", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_yticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right")
    ax.set_yticklabels(etiquetas)
    ax.set_title(titulo)
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            ax.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def scatter_pca(X, temas, ids, titulo, path):
    xy = PCA(n_components=2, random_state=SEED).fit_transform(X)
    colores = {"NLP": "#3a6ea5", "Vivienda": "#c0392b", "Deporte": "#27ae60"}
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for tema in sorted(set(temas)):
        mask = np.array(temas) == tema
        ax.scatter(xy[mask, 0], xy[mask, 1], c=colores[tema], s=90, label=tema, edgecolors="black")
    for i, lab in enumerate(ids):
        ax.annotate(lab, (xy[i, 0], xy[i, 1]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return xy


def media_pares(sim, temas, mismo: bool) -> float:
    vals = []
    n = len(temas)
    for i in range(n):
        for j in range(i + 1, n):
            if (temas[i] == temas[j]) == mismo:
                vals.append(sim[i, j])
    return float(np.mean(vals)) if vals else 0.0


def main() -> dict:
    ids = [d["id"] for d in DOCUMENTOS]
    temas = [d["tema"] for d in DOCUMENTOS]
    crudos = [d["texto"] for d in DOCUMENTOS]
    docs_norm = [normalizar(t) for t in crudos]

    # 1) One-hot / presencia
    X_oh, vocab = one_hot_matriz(docs_norm)
    sim_oh = cosine_similarity(X_oh)

    # 2) BoW conteos
    bow = CountVectorizer()
    X_bow = bow.fit_transform(docs_norm).toarray()
    _ = cosine_similarity(X_bow)

    # 3) TF-IDF (embedding disperso de documento)
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(docs_norm).toarray()
    sim_tfidf = cosine_similarity(X_tfidf)

    # 4) LSA: TF-IDF + SVD → embedding denso
    svd = TruncatedSVD(n_components=3, random_state=SEED)
    X_lsa = svd.fit_transform(X_tfidf)
    sim_lsa = cosine_similarity(X_lsa)

    # 5) Embeddings de palabra: coocurrencia + PPMI + SVD, luego media por documento
    C = matriz_coocurrencia(docs_norm, vocab, ventana=2)
    M = ppmi(C)
    E_pal = embeddings_palabra_svd(M, k=3)
    X_ppmi = np.vstack(
        [embedding_documento_palabras(d.split(), vocab, E_pal) for d in docs_norm]
    )
    sim_ppmi = cosine_similarity(X_ppmi)

    # Figuras
    heatmap(sim_oh, ids, "Similitud coseno — one-hot (presencia de términos)", FIG / "fig1_sim_onehot.png")
    heatmap(sim_tfidf, ids, "Similitud coseno — TF-IDF", FIG / "fig2_sim_tfidf.png")
    heatmap(sim_lsa, ids, "Similitud coseno — embeddings LSA (TF-IDF + SVD)", FIG / "fig3_sim_lsa.png")
    scatter_pca(X_lsa, temas, ids, "Documentos en 2D (PCA sobre embeddings LSA)", FIG / "fig4_pca_lsa.png")
    scatter_pca(X_ppmi, temas, ids, "Documentos en 2D (PCA sobre media PPMI+SVD)", FIG / "fig5_pca_ppmi.png")

    # Consulta: paráfrasis de vivienda vs ruido deportivo
    consulta = "Necesito un apartamento de tres habitaciones con parqueadero en Bogotá"
    q_norm = normalizar(consulta)
    q_tfidf = tfidf.transform([q_norm]).toarray()
    q_lsa = svd.transform(q_tfidf)
    ranking = np.argsort(-cosine_similarity(q_lsa, X_lsa)[0])

    metricas = {
        "n_docs": len(DOCUMENTOS),
        "vocab_size": len(vocab),
        "dim_onehot": int(X_oh.shape[1]),
        "dim_tfidf": int(X_tfidf.shape[1]),
        "dim_lsa": int(X_lsa.shape[1]),
        "dim_ppmi": int(X_ppmi.shape[1]),
        "onehot_intra": round(media_pares(sim_oh, temas, True), 4),
        "onehot_inter": round(media_pares(sim_oh, temas, False), 4),
        "tfidf_intra": round(media_pares(sim_tfidf, temas, True), 4),
        "tfidf_inter": round(media_pares(sim_tfidf, temas, False), 4),
        "lsa_intra": round(media_pares(sim_lsa, temas, True), 4),
        "lsa_inter": round(media_pares(sim_lsa, temas, False), 4),
        "ppmi_intra": round(media_pares(sim_ppmi, temas, True), 4),
        "ppmi_inter": round(media_pares(sim_ppmi, temas, False), 4),
        "lsa_varianza": [round(float(x), 4) for x in svd.explained_variance_ratio_],
        "consulta": consulta,
        "ranking_lsa": [
            {"id": ids[i], "tema": temas[i], "sim": round(float(cosine_similarity(q_lsa, X_lsa)[0, i]), 4)}
            for i in ranking.tolist()
        ],
        "sim_d4_d5_tfidf": round(float(sim_tfidf[3, 4]), 4),
        "sim_d4_d7_tfidf": round(float(sim_tfidf[3, 6]), 4),
        "sim_d1_d2_lsa": round(float(sim_lsa[0, 1]), 4),
        "sim_d1_d7_lsa": round(float(sim_lsa[0, 6]), 4),
    }

    (BASE / "resultados.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(f"Figuras en {FIG}")
    return metricas


if __name__ == "__main__":
    main()
