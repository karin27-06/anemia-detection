
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="Detección de Anemia", page_icon="🔬", layout="wide")
st.title("🔬 Detección de Anemia mediante Conjuntiva Palpebral")
st.markdown("**Modelo:** DenseNet-201 con explicabilidad Grad-CAM")
st.markdown("---")

@st.cache_resource
def cargar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = models.densenet201(weights=None)
    num_features = modelo.classifier.in_features
    modelo.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    modelo.load_state_dict(torch.load("densenet201_anemia_v3.pth", map_location=device))
    modelo.to(device)
    modelo.eval()
    return modelo, device

modelo, device = cargar_modelo()
clases = ["Anémico", "No Anémico"]

def preprocesar(img_pil):
    img_np = np.array(img_pil.convert("RGB"))
    gris = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gris, 15, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        x, y, w, h = cv2.boundingRect(max(contornos, key=cv2.contourArea))
        if w > 50 and h > 50:
            img_np = img_np[y:y+h, x:x+w]
    img_recortada = Image.fromarray(img_np)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img_recortada, transform(img_recortada).unsqueeze(0)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Cargar imagen")
    archivo = st.file_uploader("Sube una imagen de conjuntiva palpebral", type=["jpg","jpeg","png"])
    if archivo:
        img_pil = Image.open(archivo).convert("RGB")
        st.image(img_pil, caption="Imagen original cargada", use_column_width=True)

with col2:
    if archivo:
        st.subheader("📊 Resultado del análisis")
        img_recortada, input_t = preprocesar(img_pil)
        input_t = input_t.to(device)
        img_224 = img_recortada.resize((224, 224))
        img_np  = np.array(img_224) / 255.0

        with torch.no_grad():
            salida = modelo(input_t)
            probs  = torch.softmax(salida, dim=1)[0]
            pred   = salida.argmax(dim=1).item()

        prob_anemic     = probs[0].item()
        prob_non_anemic = probs[1].item()

        if pred == 0:
            st.error(f"⚠️ Resultado: **ANÉMICO**")
        else:
            st.success(f"✅ Resultado: **NO ANÉMICO**")

        st.markdown("**Probabilidades:**")
        st.progress(prob_anemic,     text=f"Anémico: {prob_anemic*100:.1f}%")
        st.progress(prob_non_anemic, text=f"No Anémico: {prob_non_anemic*100:.1f}%")

        st.subheader("🗺️ Mapa de activación Grad-CAM")
        target_layer  = [modelo.features.denseblock4]
        cam_obj       = GradCAM(model=modelo, target_layers=target_layer)
        targets       = [ClassifierOutputTarget(pred)]
        grayscale_cam = cam_obj(input_tensor=input_t, targets=targets)[0]
        heatmap       = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(img_224, caption="Imagen preprocesada")
        with col_b:
            fig, ax = plt.subplots()
            ax.imshow(grayscale_cam, cmap="jet")
            ax.axis("off")
            ax.set_title("Mapa de activación")
            st.pyplot(fig)
            plt.close()
        with col_c:
            st.image(heatmap, caption="Grad-CAM superpuesto")

        st.markdown("---")
        st.subheader("📋 Reporte")
        st.markdown(f"""
- **Resultado:** {clases[pred]}
- **Probabilidad de anemia:** {prob_anemic*100:.2f}%
- **Probabilidad sin anemia:** {prob_non_anemic*100:.2f}%
- **Modelo:** DenseNet-201 con Grad-CAM (v3 balanceado)
- **Nota:** Este sistema es una herramienta de apoyo diagnóstico. Consulte siempre a un profesional de salud.
        """)
    else:
        st.info("👈 Sube una imagen de conjuntiva palpebral para comenzar.")

with st.expander("ℹ️ Información del modelo"):
    st.markdown("""
- **Arquitectura:** DenseNet-201 con Transfer Learning
- **Explicabilidad:** Grad-CAM
- **Dataset:** CP-AnemiC + Eyes-Defy-Anemia (4,733 imágenes)
- **Accuracy:** 77.8% | **AUC-ROC:** 0.80
- **Umbral de anemia:** Hemoglobina < 12 g/dL
    """)
