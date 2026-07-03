# Sistema de deteccion de anemia mediante conjuntiva palpebral #
**Sistema de deteccion de anemia para diferentes tipo de personas** es una aplicación full-stack desarrollada por  
[JonathanSantamaria](https://github.com/JonathanSantamaria)
utilizando **Python 3.14**, **Opencv-python-headless**, **Grad-cam** y la librería de componentes **Matplotlib** y **Numpy** para la interfaz de numeración.


# Detección de Anemia - Conjuntiva Palpebral - Desarrollado por Jonathan Santamaria

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
