import streamlit as st
from utils.brain_utils import classify_slice, detect_tumor
from PIL import Image
import requests
from io import BytesIO

st.title("🧠 Срезы мозга и детекция опухоли")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    source = st.radio("Источник изображений:", ["Загрузка файлов", "Ссылка (URL)"])

    if source == "Загрузка файлов":
        uploaded_files = st.file_uploader("Загрузите MRI срезы", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        for file in uploaded_files:
            image = Image.open(file)
            slice_type = classify_slice(image)
            detection_image = detect_tumor(image, slice_type)
            st.image(detection_image, caption=f"{slice_type.upper()} срез — результат YOLO", use_column_width=True)
    else:
        url = st.text_input("Введите URL изображения:")
        if st.button("Обработать по ссылке"):
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                slice_type = classify_slice(image)
                detection_image = detect_tumor(image, slice_type)
                st.image(detection_image, caption=f"{slice_type.upper()} срез — результат YOLO", use_column_width=True)
            except:
                st.error("Ошибка при загрузке изображения.")

with tab2:
    st.header("Классификация срезов (EfficientNet-B0)")
    st.markdown("""
    - **Модель классификации**: EfficientNet-B0
    - **Эпох**: 5
    - **Точность классификации**: 95%
    """)
    st.image("assets/conf_class.png", caption="Confusion Matrix (Slice Type Classification)")

    st.markdown("---")
    st.subheader("YOLOv12m модели для детекции опухоли")
    st.markdown("- 3 отдельных YOLOv12m модели (по типу среза)")
    st.markdown("- **Эпох**: 150 (у каждой)")
    st.markdown("- **mAP по срезам**: Axial 0.81, Sagittal 0.78, Coronal 0.80")

    with st.expander("📊 Axial — Метрики и описание датасета"):
        st.image("assets/ax_data.png", caption="Data — Axial")
        st.image("assets/confusion_ax.png", caption="Confusion Matrix — Axial")
        st.image("assets/pr_ax.png", caption="PR Curve — Axial")

    with st.expander("📊 Sagittal — Метрики и описание датасета"):
        st.image("assets/data_sag.png", caption="Data — Sagittal")
        st.image("assets/conf_sag.png", caption="Confusion Matrix — Sagittal")
        st.image("assets/pr_sag.png", caption="PR Curve — Sagittal")

    with st.expander("📊 Coronal — Метрики и описание датасета"):
        st.image("assets/data_cor.png", caption="Data — Coronal")
        st.image("assets/confusion_cor.png", caption="Confusion Matrix — Coronal")
        st.image("assets/pr_cor.png", caption="PR Curve — Coronal")

