import streamlit as st
from utils.unet_utils import run_unet_segmentation
from PIL import Image
import requests
from io import BytesIO

st.title("🛰️ Сегментация спутниковых снимков (Unet)")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    source = st.radio("Источник изображений:", ["Загрузка файлов", "Ссылка (URL)"])

    if source == "Загрузка файлов":
        uploaded_files = st.file_uploader("Загрузите изображения", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        for file in uploaded_files:
            image = Image.open(file)
            mask = run_unet_segmentation(image)
            st.image(mask, caption=f"Сегментация: {file.name}", use_column_width=True)
    else:
        url = st.text_input("Введите URL изображения:")
        if st.button("Обработать по ссылке"):
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                mask = run_unet_segmentation(image)
                st.image(mask, caption="Сегментация (по ссылке)", use_column_width=True)
            except:
                st.error("Ошибка при загрузке изображения.")

with tab2:
    st.header("Unet для семантической сегментации")
    st.markdown("""
    - **Модель**: UNet (на основе ResNet34 encoder)
    - **Обучение**: 25 эпох
    - **IoU**: 0.77
    - **mAP**: 0.74  
    """)
    st.image("assets/map_stats_unet.png", caption="PR / IoU / F1-графики")