"""Genera el informe PDF (vía Word) del laboratorio de embeddings — formato APA del curso."""
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Cm, Inches, Pt

BASE = Path(__file__).parent
FIG = BASE / "figuras_colab"
RES = json.loads((BASE / "resultados.json").read_text(encoding="utf-8"))
REPO = "https://github.com/SonicWD/PROCESAMIENTO-DE-LENGUAJE-NATURAL"
COLAB = (
    "https://colab.research.google.com/github/SonicWD/"
    "PROCESAMIENTO-DE-LENGUAJE-NATURAL/blob/main/REA1_Embeddings/"
    "Laboratorio_Transforma_Texto_Embeddings.ipynb"
)

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)


def add_p(text, indent=True, align="justify"):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 2.0
    if indent and align == "justify":
        p.paragraph_format.first_line_indent = Cm(1.27)
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }[align]
    p.add_run(text)
    return p


def add_heading(text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 2.0
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(14 if level == 0 else 12)
    if level == 0:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


def add_tabla(headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
                r.font.name = "Times New Roman"
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(10)
                    r.font.name = "Times New Roman"
    cap_space = doc.add_paragraph()
    cap_space.paragraph_format.space_after = Pt(10)


def add_figura(path, caption, width_in=5.6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width_in))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.line_spacing = 1.15
    cap.paragraph_format.space_after = Pt(12)
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(10)


def add_space(n=1):
    for _ in range(n):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)


def add_portada_line(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    run.bold = True
    return p


add_space(2)
add_portada_line('Informe: Transforma texto en embeddings')
add_space(8)
add_portada_line("PROCESAMIENTO DE LENGUAJE NATURAL")
add_space(10)
add_portada_line("WILSON ALFONSO DÍAZ CAPADOR")
add_space(10)
add_portada_line("UNIVERSIDAD DE CUNDINAMARCA")
add_portada_line("PROGRAMA ESPECIALIZACIÓN EN INTELIGENCIA ARTIFICIAL")
add_portada_line("FACULTAD DE INGENIERÍA")
ultima = add_portada_line("2026")
ultima.add_run().add_break(WD_BREAK.PAGE)

add_heading(
    "Del token al vector: laboratorio de transformación de texto en embeddings",
    0,
)
add_p("")

add_heading("Enlaces de evidencia")
add_p(
    f"Repositorio GitHub (público): {REPO}. "
    f"Notebook del laboratorio, ya ejecutado en Google Colab con salidas y figuras: {COLAB}. "
    "Las imágenes de este informe se extraen de esas celdas (mapas de calor y PCA). "
    "La semilla es 42; el corpus son las ocho oraciones del cuaderno."
)

add_heading("1. Introducción y objetivo")
add_p(
    "Los modelos de aprendizaje automático no leen palabras: leen números. "
    "Transformar texto en embeddings consiste en mapear oraciones a vectores "
    "de números reales de modo que la geometría del espacio —distancias y "
    "ángulos— refleje semántica (Mikolov et al., 2013; Jurafsky y Martin, 2023). "
    "Este informe documenta el laboratorio del REA 1: se parte de un corpus "
    "propio en español con tres temas (NLP, vivienda y deporte), se construyen "
    "representaciones dispersas (one-hot y TF-IDF) y densas (LSA; PPMI + SVD) "
    "y se evalúa si el espacio sirve para comparar documentos y recuperar "
    "avisos con una consulta nueva."
)

add_heading("2. Marco técnico: el proceso de embedding")
add_p(
    "El flujo tiene cinco etapas. Primera, normalización: minúsculas, retiro "
    "de puntuación y stopwords del español, para no gastar dimensiones en "
    "artículos. Segunda, tokenización por espacios sobre el texto ya limpio. "
    "Tercera, vectorización. En one-hot y TF-IDF cada eje es un término del "
    "vocabulario; el laboratorio obtuvo |V| = "
    f"{RES['vocab_size']} dimensiones. TF-IDF baja el peso de palabras muy "
    "comunes en el corpus y sube el de las distintivas (Salton y Buckley, 1988). "
    "Cuarta, densificación. LSA aplica SVD a la matriz TF-IDF y deja cada "
    "documento en "
    f"{RES['dim_lsa']} factores latentes (Deerwester et al., 1990). En paralelo, "
    "una matriz de coocurrencia con ventana 2 se convierte a PPMI y se reduce "
    "con SVD: es la familia count-based emparentada con GloVe (Pennington et al., "
    "2014). El vector de un documento es la media de los embeddings de sus "
    "palabras. Quinta, comparación con similitud coseno, invariante a la norma "
    "del vector y estándar en recuperación de información."
)
add_p(
    "La diferencia conceptual es clave para la rúbrica. Un one-hot de "
    "“apartamento en Bogotá” y “inmueble en la capital” puede ser ortogonal si "
    "no comparten tokens. Un embedding denso busca un eje de “vivienda” donde "
    "ambos se proyectan cerca. Word2Vec y BERT siguen esa idea con redes; aquí "
    "se usa álgebra lineal reproducible, sin claves de API, para que el "
    "profesor clone el repo y obtenga las mismas figuras."
)

add_heading("3. Diseño experimental")
add_p(
    f"Se usaron {RES['n_docs']} documentos etiquetados por tema. El bloque NLP "
    "repite embeddings, lenguaje y natural; el de vivienda, apartamento, "
    "Bogotá, habitaciones y precio; el de deporte, equipo, fútbol y partido. "
    "Esa repetición controlada permite medir coseno intra-tema (pares del mismo "
    "tema) frente a inter-tema (pares de temas distintos). La consulta de "
    "prueba fue: «"
    f"{RES['consulta']}». Se proyectó con el mismo vectorizador TF-IDF y el "
    "mismo SVD de LSA, y se ordenó el corpus por coseno. Semilla 42 en PCA y SVD."
)

add_heading("4. Resultados (salida de Colab)")
add_p(
    "La Tabla 1 copia la celda 7 del notebook. Al pasar de 57 dimensiones dispersas "
    "a 3 densas, el coseno intra-tema pasa de 0.24–0.35 a ≈ 0.99, y el inter-tema "
    "sigue cerca de 0. En TF-IDF, D4 y D5 (vivienda) se parecen 0.40 y D4 vs D7 "
    "(deporte) queda en 0.00: sin palabras compartidas no hay señal. LSA no inventa "
    "temas; comprime las correlaciones de TF-IDF. La varianza de los tres "
    "componentes LSA es 0.1317, 0.1303 y 0.1630 (suma 0.425): basta para separar "
    "tres tópicos, no para reconstruir el vocabulario completo."
)
add_tabla(
    ["Representación", "Dim.", "Coseno intra-tema", "Coseno inter-tema"],
    [
        ["one-hot", "57", "0.3460", "0.0053"],
        ["TF-IDF", "57", "0.2363", "0.0045"],
        ["LSA", "3", "0.9879", "0.0155"],
        ["PPMI+SVD", "3", "0.9872", "0.0351"],
    ],
)
add_p(
    "Tabla 1. Separación por tema medida en el notebook ejecutado en Colab. "
    "Elaboración a partir de la celda 7.",
    indent=False,
    align="center",
)
add_figura(
    FIG / "colab_fig1.png",
    "Figura 1. Similitud coseno TF-IDF. Salida de Google Colab (celda 8).",
)
add_p(
    "En la Figura 1 se ven tres bloques: D1–D3 (NLP, cosenos 0.17–0.19), "
    "D4–D6 (vivienda; el par D4–D5 llega a 0.40) y D7–D8 (deporte, 0.30). "
    "Entre NLP y el resto el mapa es amarillo (0.00). Es exactamente el límite "
    "de una representación léxica: si no hay tokens en común, el ángulo es 90°."
)
add_figura(
    FIG / "colab_fig2.png",
    "Figura 2. Similitud coseno LSA (3 dimensiones). Salida de Google Colab (celda 8).",
)
add_p(
    "La Figura 2 es el mismo corpus en el espacio denso. Los bloques intra-tema "
    "pasan a azul oscuro (≈ 1.00) y el cruce entre temas permanece claro. D1, D2 y "
    "D3 quedan casi indistinguibles entre sí; D4, D5 y D6 forman el otro cubo; "
    "D7 y D8 el tercero. Ahí se ve el oficio del embedding: documentos que en "
    "TF-IDF apenas se rozaban ahora comparten un factor latente de tema."
)
add_figura(
    FIG / "colab_fig3.png",
    "Figura 3. PCA de documentos sobre embeddings LSA. Salida de Google Colab (celda 9).",
)
add_p(
    "La Figura 3 proyecta esos vectores a 2D. Azul (NLP: D1–D3) se agrupa abajo "
    "a la izquierda; rojo (vivienda: D4 y D5 casi superpuestos, D6 un poco "
    "desplazado) a la derecha; verde (deporte: D7–D8) arriba. El eje X separa "
    "vivienda del resto; el eje Y separa deporte. No es un t-SNE industrial: "
    "ocho puntos, tres temas, separación legible."
)
add_figura(
    FIG / "colab_fig4.png",
    "Figura 4. PCA sobre la media PPMI+SVD. Salida de Google Colab (celda 9).",
)
add_p(
    "La Figura 4, construida con coocurrencia local y no con TF-IDF, reproduce "
    "las mismas tres nubes. El embedding de documento como promedio de palabras "
    "hereda la estructura de contexto: “apartamento” y “habitaciones” empujan "
    "hacia vivienda; “fútbol” y “partido” hacia deporte."
)
add_p(
    "La Tabla 2 es el ranking de la consulta «Necesito un apartamento de tres "
    "habitaciones con parqueadero en Bogotá», proyectada con el mismo TF-IDF+SVD "
    "(celda 10). Los tres avisos de vivienda salen primero (0.9998, 0.9997 y "
    "0.9656). El resto del corpus queda en torno a cero. D6 no menciona "
    "parqueadero y aun así entra al bloque: eso no lo haría un filtro booleano."
)
add_tabla(
    ["Puesto", "Id", "Tema", "Coseno LSA"],
    [
        ["1", "D5", "Vivienda", "0.9998"],
        ["2", "D4", "Vivienda", "0.9997"],
        ["3", "D6", "Vivienda", "0.9656"],
        ["4", "D7", "Deporte", "0.0219"],
        ["5", "D3", "NLP", "0.0000"],
        ["6", "D2", "NLP", "0.0000"],
        ["7", "D1", "NLP", "0.0000"],
        ["8", "D8", "Deporte", "−0.0553"],
    ],
)
add_p(
    "Tabla 2. Recuperación semántica de la consulta de vivienda. Salida de Colab, celda 10.",
    indent=False,
    align="center",
)

add_heading("5. Análisis y límites")
add_p(
    "Tres lecturas. Una, el cuello de botella del NLP clásico no es “pasar "
    "texto a números”, sino elegir una geometría que no colapse la semántica. "
    "Dos oraciones sin solapamiento léxico son invisibles para TF-IDF y visibles "
    "para LSA si pertenecen al mismo factor. Dos, la dimensionalidad no es un "
    "lujo: 57 ejes one-hot vs 3 densos. En un corpus real el vocabulario llega "
    "a decenas de miles; los embeddings (Word2Vec 300-D, BERT 768-D) son la "
    "misma apuesta a mayor escala (Mikolov et al., 2013; Devlin et al., 2019). "
    "Tres, el laboratorio es deliberadamente pequeño. Un modelo contextual "
    "distinguiría “banco” (finanzas) de “banco” (asiento); LSA sobre ocho "
    "oraciones no. El valor académico está en dejar el pipeline transparente "
    "y medido, no en competir con un API comercial."
)

add_heading("6. Conclusiones")
add_p(
    "Se transformó texto en embeddings siguiendo un proceso explícito: "
    "limpieza, vectores dispersos, reducción densa, coseno y consulta. Los "
    "números del laboratorio —sobre todo LSA intra 0.99 frente a inter 0.02, "
    "y un ranking de consulta que recupera los tres avisos de vivienda— "
    "muestran que el espacio vectorial organizó el significado. El código "
    "modular está en GitHub y se puede reejecutar en Colab. Esa es la evidencia "
    "que pide la rúbrica: comprensión técnica, notebook funcional y análisis "
    "de resultados, no una captura aislada."
)

doc.add_page_break()
add_heading("Referencias", 0)
add_p("")
refs = [
    "Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., y Harshman, "
    "R. (1990). Indexing by latent semantic analysis. Journal of the American "
    "Society for Information Science, 41(6), 391–407.",
    "Devlin, J., Chang, M.-W., Lee, K., y Toutanova, K. (2019). BERT: "
    "Pre-training of deep bidirectional transformers for language understanding. "
    "En Proceedings of NAACL-HLT 2019 (pp. 4171–4186).",
    "Jurafsky, D., y Martin, J. H. (2023). Speech and language processing "
    "(3.ª ed. en borrador). https://web.stanford.edu/~jurafsky/slp3/",
    "Mikolov, T., Chen, K., Corrado, G., y Dean, J. (2013). Efficient estimation "
    "of word representations in vector space. https://arxiv.org/abs/1301.3781",
    "Pennington, J., Socher, R., y Manning, C. D. (2014). GloVe: Global vectors "
    "for word representation. En Proceedings of EMNLP 2014 (pp. 1532–1543).",
    "Salton, G., y Buckley, C. (1988). Term-weighting approaches in automatic "
    "text retrieval. Information Processing & Management, 24(5), 513–523.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 2.0
    p.paragraph_format.left_indent = Cm(1.27)
    p.paragraph_format.first_line_indent = Cm(-1.27)
    p.add_run(ref)

out = BASE / "Informe_Transforma_Texto_Embeddings_Wilson_Diaz.docx"
doc.save(out)
palabras = len(" ".join(p.text for p in doc.paragraphs).split())
print(out)
print("palabras", palabras)
