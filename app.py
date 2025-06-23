import streamlit as st
import pandas as pd
import numpy as np
import faiss
from PIL import Image
import torch

# --- Simulación de modelo CLIP ---
# Aquí simplemente devolvemos vectores aleatorios para testear
def obtener_embedding_simulado(tipo="texto"):
    np.random.seed(42 if tipo == "texto" else 99)
    emb = np.random.rand(512).astype("float32")
    return emb / np.linalg.norm(emb)

# --- Simular productos con embeddings aleatorios ---
@st.cache_data
def cargar_datos_simulados(n=10):
    productos = []
    for i in range(n):
        productos.append({
            "id": i,
            "nombre": f"Producto {i}",
            "descripcion": f"Este es el producto número {i}",
            "embedding_nombre": np.random.rand(512).astype("float32"),
            "embedding_descripcion": np.random.rand(512).astype("float32"),
            "embedding_imagen": np.random.rand(512).astype("float32"),
        })

    df = pd.DataFrame(productos)

    # Normalizar
    for col in ["embedding_nombre", "embedding_descripcion", "embedding_imagen"]:
        df[col] = df[col].apply(lambda v: v / np.linalg.norm(v))

    emb_nombre = np.vstack(df["embedding_nombre"])
    emb_desc = np.vstack(df["embedding_descripcion"])
    emb_img = np.vstack(df["embedding_imagen"])
    emb_combi = (emb_nombre + emb_desc + emb_img) / 3

    # Crear índices FAISS
    idx_nombre = faiss.IndexFlatIP(512)
    idx_nombre.add(emb_nombre)

    idx_desc = faiss.IndexFlatIP(512)
    idx_desc.add(emb_desc)

    idx_img = faiss.IndexFlatIP(512)
    idx_img.add(emb_img)

    idx_combi = faiss.IndexFlatIP(512)
    idx_combi.add(emb_combi)

    return df, idx_nombre, idx_desc, idx_img, idx_combi

df, idx_nombre, idx_desc, idx_img, idx_combi = cargar_datos_simulados()

# --- Buscar ---
def buscar_faiss(embedding, index, topk=5):
    D, I = index.search(embedding.reshape(1, -1).astype("float32"), topk)
    return df.iloc[I[0]]

# --- Interfaz ---
st.title("🔍 Buscador Multimodal de Productos (Demo)")

modo = st.radio("Modo de búsqueda", ["Separado", "Combinado"])

if modo == "Separado":
    tipo = st.selectbox("Buscar por:", ["nombre", "descripcion", "imagen"])

    if tipo == "imagen":
        img = st.file_uploader("Sube una imagen (simulado)", type=["png", "jpg", "jpeg"])
        if img:
            st.image(Image.open(img), caption="Imagen subida", use_column_width=True)
            emb = obtener_embedding_simulado("imagen")
            index = idx_img
            resultados = buscar_faiss(emb, index)
            st.subheader("Resultados simulados:")
            for _, row in resultados.iterrows():
                st.markdown(f"**{row['nombre']}** — {row['descripcion']}")
    else:
        texto = st.text_input("Escribe el texto (simulado)")
        if texto:
            emb = obtener_embedding_simulado("texto")
            index = idx_nombre if tipo == "nombre" else idx_desc
            resultados = buscar_faiss(emb, index)
            st.subheader("Resultados simulados:")
            for _, row in resultados.iterrows():
                st.markdown(f"**{row['nombre']}** — {row['descripcion']}")

elif modo == "Combinado":
    st.info("Usamos el promedio de embeddings (simulado).")
    texto = st.text_input("Texto opcional")
    img = st.file_uploader("Imagen opcional", type=["png", "jpg", "jpeg"])

    if texto or img:
        embeddings = []
        if texto:
            embeddings.append(obtener_embedding_simulado("texto"))
        if img:
            st.image(Image.open(img), use_column_width=True)
            embeddings.append(obtener_embedding_simulado("imagen"))

        if embeddings:
            emb_combi = np.mean(np.vstack(embeddings), axis=0)
            resultados = buscar_faiss(emb_combi, idx_combi)
            st.subheader("Resultados simulados:")
            for _, row in resultados.iterrows():
                st.markdown(f"**{row['nombre']}** — {row['descripcion']}")
