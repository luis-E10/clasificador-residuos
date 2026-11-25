"""Clasificación de residuos con cámara o archivo."""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo():
    classes = json.loads((ROOT / "clases.json").read_text(encoding="utf-8"))
    model = keras.models.load_model(ROOT / "modelo.keras", compile=False)
    if model.output_shape[-1] != len(classes):
        raise ValueError("El modelo y el archivo de clases no coinciden.")
    return model, classes


def clasificar(image, model):
    height, width = model.input_shape[1:3]
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    batch = np.asarray(resized, dtype=np.float32)[None, ...]
    return model(batch, training=False).numpy()[0]


def main():

    st.markdown(
        """
        <h1 style="text-align: center;">♻️ Clasificador de residuos</h1>
        <h2 style="text-align: center;">Luis Eduardo Sagastume</h2>
        """,
        unsafe_allow_html=True,
    )
    st.set_page_config(page_title="Clasificador de residuos", page_icon="♻️")
    st.title("♻️ Modelo predictivo Reciclaje clase de IA-UTH -2026-Luis Sagastume")
    st.write("Fotografía un solo residuo de vidrio, papel, cartón o plástico.")
    source = st.radio("Origen de la imagen", ("Cámara", "Subir foto"), horizontal=True)
    if source == "Cámara":
        picture = st.camera_input("Tomar una fotografía")
    else:
        picture = st.file_uploader("Selecciona una fotografía", type=["jpg", "jpeg", "png", "webp"])
    if picture is None:
        return

    try:
        with Image.open(picture) as original:
            img = ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("No se pudo abrir la fotografía. Prueba otra imagen.")
        return

    st.image(img, caption="Fotografía analizada", width=440)
    model, classes = cargar_modelo()
    scores = clasificar(img, model)
    index = int(np.argmax(scores))
    st.subheader("Predicción principal")
    st.success(f"{classes[index]} · {scores[index]:.2%} de confianza")
    with st.expander("Confianza por clase"):
        for label, score in zip(classes, scores):
            st.progress(float(score), text=f"{label} · {score:.2%}")


if __name__ == "__main__":
    main()
