
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

st.set_page_config(
    page_title="AnemiaDetect AI",
    page_icon="🔬",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    /* Oculta elementos de Streamlit */
    footer {
        visibility: hidden;
    }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        max-width: 720px !important;
    }

    /* HERO */
    .hero {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-radius: 0 0 16px 16px;
        padding: 2rem 1.5rem 1.5rem;
        text-align: center;
        margin-bottom: 1.2rem;
        border-bottom: 3px solid #e94560;
    }
    .hero h1 { color: #fff; font-size: 1.6rem; font-weight: 700; margin: 0 0 0.3rem; }
    .hero p  { color: #a0aec0; font-size: 0.85rem; margin: 0; }
    .hero .badge {
        display: inline-block; background: #e94560; color: #fff;
        padding: 2px 12px; border-radius: 99px; font-size: 0.7rem;
        font-weight: 600; margin-top: 0.6rem;
    }

    /* CARDS */
    .card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.2);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    color: var(--text-color);
}
    .card-title {
    color: var(--text-color);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.8rem;
}

    /* RESULTADO */
    .result-box {
        border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.8rem;
    }
    .result-anemic {
    background: var(--secondary-background-color);
    border-left: 4px solid #e94560;
}
    .result-normal {
    background: var(--secondary-background-color);
    border-left: 4px solid #38a169;
}
    .result-title  { font-size: 1.05rem; font-weight: 700; margin: 0 0 0.2rem; }
    .result-sub    { font-size: 0.8rem; opacity: 0.85; margin: 0; }

    /* MÉTRICAS */
    .metric-row { display: flex; gap: 0.5rem; margin-bottom: 0.8rem; }
    .metric-box {
    flex: 1;
    background: var(--secondary-background-color);
    border-radius: 8px;
    padding: 0.6rem 0.4rem;
    text-align: center;
    color: var(--text-color);
}
    .metric-val   { font-size: 1.1rem; font-weight: 700; }
    .metric-label {
    color: var(--text-color);
    opacity: .7;
    font-size: 0.65rem;
    text-transform: uppercase;
}

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
    background: var(--secondary-background-color);
    border-radius: 8px;
    padding: 3px;
    gap: 3px;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-color);
}
    .stTabs [aria-selected="true"] { background: #e94560 !important; color: #fff !important; }

    /* FOOTER */
    .footer {
    text-align: center;
    color: var(--text-color);
    opacity: .7;
    font-size: 0.75rem;
    border-top: 1px solid rgba(128,128,128,.2);
    padding-top: 1rem;
    margin-top: 1rem;
}

    /* RESPONSIVE */
    @media (max-width: 600px) {
        .hero h1 { font-size: 1.25rem; }
        .metric-val { font-size: 0.95rem; }
        .metric-label { font-size: 0.6rem; }
    }
    .result-title{
    color: var(--text-color);
    font-weight:700;
}

.result-sub{
    color: var(--text-color);
    opacity:.8;
}
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔬 AnemiaDetect AI</h1>
    <p>Detección no invasiva de anemia mediante conjuntiva palpebral</p>
    <span class="badge">DenseNet-201 · Grad-CAM · Transfer Learning</span>
</div>
""", unsafe_allow_html=True)

# ── Modelo ─────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = models.densenet201(weights=None)
    num_features = modelo.classifier.in_features
    modelo.classifier = nn.Sequential(
        nn.Linear(num_features, 256), nn.ReLU(),
        nn.Dropout(0.4), nn.Linear(256, 2)
    )
    modelo.load_state_dict(torch.load(
        "densenet201_anemia_v3.pth", map_location=device
    ))
    modelo.to(device).eval()
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
    img_r = Image.fromarray(img_np)
    t = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return img_r, t(img_r).unsqueeze(0)

# ── INPUT ──────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📤 Ingresa una imagen</div>',
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Subir archivo", "📷 Tomar foto"])
imagen_final = None

with tab1:
    archivo = st.file_uploader("img", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
    if archivo:
        imagen_final = Image.open(archivo).convert("RGB")
        col_img, _ = st.columns([1, 1])
        with col_img:
            st.image(imagen_final, caption="Imagen cargada", width=200)

with tab2:
    foto = st.camera_input("foto", label_visibility="collapsed")
    if foto:
        imagen_final = Image.open(foto).convert("RGB")
        col_img, _ = st.columns([1, 1])
        with col_img:
            st.image(imagen_final, caption="Foto capturada", width=200)

st.markdown('</div>', unsafe_allow_html=True)

# ── ANÁLISIS ───────────────────────────────────────────────────
if not imagen_final:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem;color:var(--text-color);opacity:.7;">
        <div style="font-size:2rem;">👆</div>
        <p style="font-size:0.85rem;margin-top:0.3rem;">
            Sube una imagen o toma una foto para comenzar
        </p>
    </div>""", unsafe_allow_html=True)
else:
    img_recortada, input_t = preprocesar(imagen_final)
    input_t = input_t.to(device)
    img_224 = img_recortada.resize((224, 224))
    img_np  = np.array(img_224) / 255.0

    with torch.no_grad():
        salida = modelo(input_t)
        probs  = torch.softmax(salida, dim=1)[0]
        pred   = salida.argmax(dim=1).item()

    prob_anemic     = probs[0].item()
    prob_non_anemic = probs[1].item()

    # Resultado
    if pred == 0:
        st.markdown(f"""
        <div class="result-box result-anemic">
            <p class="result-title" style="color:#fc8181;">⚠️ ANEMIA DETECTADA</p>
            <p class="result-sub" style="color:#feb2b2;">
                Se detectaron indicios de anemia. Consulte a un profesional de salud.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-normal">
            <p class="result-title">✅ SIN ANEMIA DETECTADA</p>
            <p class="result-sub">
                No se detectaron indicios de anemia. Esto no reemplaza un diagnóstico clínico.
            </p>
        </div>""", unsafe_allow_html=True)
    
    # Probabilidades
    st.markdown("### 📊 Probabilidades")

    st.progress(
        prob_anemic,
        text=f"Anémico: {prob_anemic*100:.1f}%"
    )

    st.progress(
        prob_non_anemic,
        text=f"No Anémico: {prob_non_anemic*100:.1f}%"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Métricas
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-val" style="color:#fc8181;">{prob_anemic*100:.1f}%</div>
            <div class="metric-label">P. Anemia</div>
        </div>
        <div class="metric-box">
            <div class="metric-val" style="color:#68d391;">{prob_non_anemic*100:.1f}%</div>
            <div class="metric-label">P. Normal</div>
        </div>
        <div class="metric-box">
            <div class="metric-val" style="color:#63b3ed;">0.80</div>
            <div class="metric-label">AUC-ROC</div>
        </div>
        <div class="metric-box">
            <div class="metric-val" style="color:#b794f4;">77.8%</div>
            <div class="metric-label">Accuracy</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Grad-CAM
    st.markdown('<div class="card"><div class="card-title">🗺️ Mapa de Activación Grad-CAM</div>',
                unsafe_allow_html=True)

    cam_obj = GradCAM(model=modelo, target_layers=[modelo.features.denseblock4])
    grayscale_cam = cam_obj(input_tensor=input_t,
                            targets=[ClassifierOutputTarget(pred)])[0]
    heatmap = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.image(img_224, caption="Preprocesada", width=160)
    with col_b:
        st.markdown("**🗺️ Mapa de activación**")
        
        fig, ax = plt.subplots(figsize=(2.2, 2.2))
        # fig.patch.set_facecolor(...)
        ax.imshow(grayscale_cam, cmap="jet")
        ax.axis("off")
        plt.tight_layout(pad=0.1)
        st.pyplot(fig, use_container_width=False)
        plt.close()
    with col_c:
        st.image(heatmap, caption="Grad-CAM", width=160)

    st.markdown('</div>', unsafe_allow_html=True)

    # Reporte
    st.markdown('<div class="card"><div class="card-title">📋 Reporte</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
| Campo | Valor |
|---|---|
| **Resultado** | {clases[pred]} |
| **P. Anemia** | {prob_anemic*100:.2f}% |
| **P. Normal** | {prob_non_anemic*100:.2f}% |
| **Modelo** | DenseNet-201 + Grad-CAM v3 |
| **Umbral** | Hb < 12 g/dL |
    """)
    st.caption("⚠️ Herramienta de apoyo diagnóstico. No reemplaza la evaluación clínica.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Info ───────────────────────────────────────────────────────
with st.expander("ℹ️ Información técnica"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Modelo**
- DenseNet-201 + Transfer Learning
- Fine-tuning en denseblock4
- Grad-CAM para explicabilidad
        """)
    with c2:
        st.markdown("""
**Dataset y métricas**
- CP-AnemiC + Eyes-Defy-Anemia
- 4,733 imágenes con augmentation
- Accuracy 77.8% · AUC-ROC 0.80
        """)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    AnemiaDetect AI · Universidad Señor de Sipán · 2026 · KCS<br>
    Santamaria Chafloc Jonathan Levi · DenseNet-201 + Grad-CAM
</div>
""", unsafe_allow_html=True)