# 🔍 Buscador Multimodal de Productos

Una aplicación web interactiva que permite buscar productos de supermercado utilizando texto e imágenes, combinando embeddings vectoriales y búsqueda por similitud con FAISS.

---

## 📋 Descripción

Este proyecto implementa un buscador multimodal de productos construido con **Streamlit**. Utiliza modelos de embeddings (inspirado en CLIP) para representar productos a través de su nombre, descripción e imagen en un espacio vectorial común, lo que permite realizar búsquedas semánticas eficientes mediante **FAISS** (Facebook AI Similarity Search).

---

## ✨ Características

- **Búsqueda por texto**: Encuentra productos escribiendo el nombre o descripción.
- **Búsqueda por imagen**: Sube una imagen para encontrar productos visualmente similares.
- **Modo separado**: Busca usando únicamente nombre, descripción o imagen por separado.
- **Modo combinado**: Combina embeddings de texto e imagen para una búsqueda más rica.
- **Resultados rápidos**: Índices FAISS con similitud del coseno (producto interno sobre vectores normalizados) para una recuperación eficiente.

---

## 🛠️ Tecnologías utilizadas

| Herramienta | Uso |
|---|---|
| [Streamlit](https://streamlit.io/) | Interfaz web interactiva |
| [FAISS](https://github.com/facebookresearch/faiss) | Búsqueda de similitud vectorial |
| [PyTorch](https://pytorch.org/) | Backend de deep learning |
| [Transformers (HuggingFace)](https://huggingface.co/docs/transformers) | Modelos de embeddings multimodales |
| [Pandas](https://pandas.pydata.org/) | Manipulación de datos |
| [NumPy](https://numpy.org/) | Operaciones con vectores |
| [Pillow](https://python-pillow.org/) | Procesamiento de imágenes |

---

## 📂 Estructura del proyecto

```
Multimodal-Product-Search/
├── app.py                      # Aplicación principal de Streamlit
├── EDA.ipynb                   # Análisis exploratorio de datos
├── supermercados.csv           # Dataset de productos de supermercado
├── entorno_sinproblemas.yml    # Entorno Conda con dependencias
└── README.md                   # Este archivo
```

---

## ⚙️ Instalación

### Requisitos previos

- [Anaconda](https://www.anaconda.com/) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Pasos

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/paolosalazarp/Multimodal-Product-Search.git
   cd Multimodal-Product-Search
   ```

2. **Crea el entorno Conda con todas las dependencias:**
   ```bash
   conda env create -f entorno_sinproblemas.yml
   ```

3. **Activa el entorno:**
   ```bash
   conda activate faiss-env
   ```

4. **Ejecuta la aplicación:**
   ```bash
   streamlit run app.py
   ```

5. Abre tu navegador en `http://localhost:8501`

---

## 🚀 Uso

1. **Selecciona el modo de búsqueda:**
   - **Separado**: Elige buscar por nombre, descripción o imagen de forma individual.
   - **Combinado**: Usa texto e imagen simultáneamente para obtener resultados más precisos.

2. **Introduce tu consulta:**
   - Escribe texto en el campo de búsqueda, o
   - Sube una imagen en formato PNG, JPG o JPEG.

3. **Explora los resultados**: La aplicación mostrará los productos más similares según tu consulta.

---

## 📊 Dataset

El archivo `supermercados.csv` contiene información de productos de supermercado que se utilizan en el análisis exploratorio del notebook `EDA.ipynb`.

---

## 📝 Notas

- La versión actual de la demo utiliza **embeddings simulados** (vectores aleatorios normalizados) en lugar de un modelo CLIP real, con el fin de facilitar las pruebas sin necesidad de GPU ni descarga de modelos pesados.
- Para producción, los embeddings simulados deben reemplazarse por un modelo real como `openai/clip-vit-base-patch32` de HuggingFace.

---

## 📄 Licencia

Este proyecto es de uso educativo y de investigación.
