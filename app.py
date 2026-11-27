import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image

st.set_page_config(
    page_title="Detector de Género con IA",
    page_icon="👤",
    layout="centered",
)

st.title("👤 Detector de Género Facial con IA")
st.write("Toma una foto con tu cámara para estimar el género con DeepFace.")


def mejorar_iluminacion(img_bgr: np.ndarray) -> np.ndarray:
    # Corrige contraluz usando ecualización adaptativa CLAHE en el canal L
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


@st.cache_resource
def precargar_modelo() -> bool:
    _ = DeepFace.analyze(
        img_path=np.zeros((224, 224, 3), dtype=np.uint8),
        actions=["gender"],
        detector_backend="opencv",
        enforce_detection=False,
        silent=True,
    )
    return True


with st.spinner("Cargando modelo de reconocimiento facial..."):
    precargar_modelo()

img_file = st.camera_input("📸 Toma una foto")

if img_file is not None:
    image = Image.open(img_file)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_mejorada = mejorar_iluminacion(img_bgr)

    with st.spinner("🤖 Analizando rostro..."):
        try:
            analisis = DeepFace.analyze(
                img_path=img_mejorada,
                actions=["gender"],
                detector_backend="opencv",
                enforce_detection=True,
                silent=True,
            )

            if analisis:
                datos_rostro = analisis[0]
                region = datos_rostro["region"]
                genero_pred = datos_rostro["gender"]

                rx = max(0, int(region["x"]))
                ry = max(0, int(region["y"]))
                rw = max(0, int(region["w"]))
                rh = max(0, int(region["h"]))

                mujer_prob = float(genero_pred.get("Woman", 0.0))
                hombre_prob = float(genero_pred.get("Man", 0.0))

                if mujer_prob >= 50.0:
                    etiqueta = f"Mujer ({mujer_prob:.1f}%)"
                    color = (255, 105, 180)
                else:
                    etiqueta = f"Hombre ({hombre_prob:.1f}%)"
                    color = (30, 144, 255)

                img_dibujada = img_np.copy()
                cv2.rectangle(img_dibujada, (rx, ry), (rx + rw, ry + rh), color, 4)

                st.image(img_dibujada, caption="Rostro Detectado", use_container_width=True)

                confianza_rostro = float(datos_rostro.get("face_confidence") or 0.0) * 100.0
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Género Predicho", etiqueta)
                with col2:
                    st.metric("Confianza del Rostro", f"{confianza_rostro:.1f}%")

        except ValueError:
            st.warning(
                "⚠️ No se detectó ningún rostro de forma clara. "
                "Asegúrate de mirar de frente a la cámara y tener buena iluminación."
            )
        except Exception as e:
            st.error(f"❌ Ocurrió un error al analizar la imagen: {e}")