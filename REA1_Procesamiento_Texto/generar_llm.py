"""Generación, resumen abstractivo y QA con Qwen2.5-0.5B-Instruct.

Se ejecuta con el entorno que tiene PyTorch (ruta corta en Windows).
Escribe llm.json para que laboratorio.py lo incorpore al informe.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).parent
MODELO = "Qwen/Qwen2.5-0.5B-Instruct"
TEXTO = (
    "El 12 de mayo de 2025, Ana Rojas, investigadora de la Universidad de Cundinamarca "
    "en Fusagasugá, presentó en Bogotá un convenio con el Ministerio de Tecnologías. "
    "El proyecto Andina-NLP entrenará modelos en español para resumir historias clínicas. "
    "El hospital San Rafael aportará el corpus anonimizado. "
    "Rojas explicó que el modelo no reemplaza al médico: solo ordena el texto para que "
    "la revisión humana sea más rápida. "
    "La gobernación de Cundinamarca financiará la primera fase hasta diciembre de 2026."
)
PREGUNTAS = [
    ("¿Quién presentó el convenio en Bogotá?", "Ana Rojas"),
    ("¿En qué ciudad se presentó el convenio?", "Bogotá"),
    ("¿Qué hospital aportará el corpus?", "San Rafael"),
    ("¿Hasta cuándo financiará la gobernación la primera fase?", "diciembre de 2026"),
]


def chat(modelo, tokenizador, usuario: str, max_new: int) -> str:
    mensajes = [{"role": "user", "content": usuario}]
    texto = tokenizador.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    entradas = tokenizador(texto, return_tensors="pt")
    with torch.no_grad():
        salida = modelo.generate(
            **entradas,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizador.eos_token_id,
        )
    nuevo = salida[0, entradas["input_ids"].shape[1] :]
    return tokenizador.decode(nuevo, skip_special_tokens=True).strip()


def main() -> None:
    print("Cargando", MODELO)
    tokenizador = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO)
    modelo.eval()
    prompt = (
        "Escribe exactamente dos oraciones en español, sin lista ni título, "
        "sobre un modelo de lenguaje que resume historias clínicas en un hospital."
    )
    generacion = chat(modelo, tokenizador, prompt, 160)
    print("GEN:", generacion)
    resumen = chat(
        modelo,
        tokenizador,
        "Resume en una sola oración completa en español, sin viñetas y sin cortar la frase:\n"
        + TEXTO,
        120,
    )
    print("RES:", resumen)
    qa = []
    for pregunta, oro in PREGUNTAS:
        respuesta = chat(
            modelo,
            tokenizador,
            "Responde en español solo con el dato pedido, sin explicar.\n"
            f"Texto: {TEXTO}\nPregunta: {pregunta}",
            40,
        )
        print("QA:", pregunta, "->", respuesta)
        qa.append(
            {
                "pregunta": pregunta,
                "oro": oro,
                "respuesta_llm": respuesta,
                "acierto": oro.lower() in respuesta.lower(),
            }
        )
    payload = {
        "modelo": MODELO,
        "prompt": prompt,
        "generacion": generacion,
        "resumen_abstractivo": resumen,
        "qa": qa,
        "qa_aciertos": sum(1 for item in qa if item["acierto"]),
    }
    destino = BASE / "llm.json"
    destino.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Guardado", destino)


if __name__ == "__main__":
    main()
