# REA 1 — Procesamiento de texto

**Procesamiento de Lenguaje Natural** · CAD2202023206 · Wilson Alfonso Díaz Capador

## Entrega en Moodle

1. Subir el PDF `Informe_Procesamiento_Texto_Wilson_Diaz.pdf`
2. Pegar la URL del repositorio: https://github.com/SonicWD/PROCESAMIENTO-DE-LENGUAJE-NATURAL

Notebook: `Laboratorio_Procesamiento_Texto.ipynb`

## Qué cubre

Normalización, stemming, stopwords, term frequency, inverse document frequency, part-of-speech, lematización, parsing, named-entity, generación con LLM, question answering, summarization, similitud de oraciones, clasificación, traducción, generación estadística (trigramas) y minería de texto.

## Cómo reproducir

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python laboratorio.py
```

La primera ejecución descarga Opus-MT es→en (unos 280 MB) en `modelos/`. La generación con Qwen2.5-0.5B-Instruct está en `generar_llm.py` y exige PyTorch; escribe `llm.json`, que `laboratorio.py` incorpora si el archivo ya existe.

Semilla 42.
