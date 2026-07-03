# Detección de Anemia - Conjuntiva Palpebral

## Archivos
- app.py                      → aplicación web Streamlit
- densenet201_anemia_v3.pth   → modelo entrenado
- requirements.txt            → dependencias

## Cómo ejecutar localmente
1. pip install -r requirements.txt
2. streamlit run app.py
3. Abre: http://localhost:8501

## Métricas
- Accuracy: 77.8% | AUC-ROC: 0.80
- Arquitectura: DenseNet-201 + Transfer Learning + Grad-CAM
- Dataset: CP-AnemiC + Eyes-Defy-Anemia (4,733 imágenes)
