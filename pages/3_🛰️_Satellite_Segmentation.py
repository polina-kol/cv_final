import streamlit as st
from utils.unet_utils import run_unet_segmentation
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Сегментация спутниковых снимков", page_icon="🛰️")
st.title("🛰️ Сегментация спутниковых снимков (U-Net)")

tab1, tab2 = st.tabs(["Сервис", "Информация о модели"])

with tab1:
    st.subheader("Выберите способ загрузки изображения")
    source = st.radio("Источник изображения:", ["Загрузить файл", "Указать URL"])

    if source == "Загрузить файл":
        uploaded_files = st.file_uploader("Загрузите изображение(я)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                image = Image.open(file).convert("RGB")
                mask = run_unet_segmentation(image)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Оригинал", use_container_width=True)
                with col2:
                    st.image(mask, caption="Маска сегментации", use_container_width=True)

                buf = BytesIO()
                mask.save(buf, format="PNG")
                st.download_button(
                    label="📥 Скачать маску",
                    data=buf.getvalue(),
                    file_name=f"mask_{file.name}.png",
                    mime="image/png"
                )

    else:
        url = st.text_input("Введите URL изображения:")
        if st.button("Обработать по ссылке") and url:
            try:
                response = requests.get(url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                mask = run_unet_segmentation(image)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Оригинал", use_container_width=True)
                with col2:
                    st.image(mask, caption="Маска сегментации", use_container_width=True)

                buf = BytesIO()
                mask.save(buf, format="PNG")
                st.download_button(
                    label="📥 Скачать маску",
                    data=buf.getvalue(),
                    file_name="mask_from_url.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Ошибка при загрузке изображения: {e}")

with tab2:
    st.header("U-Net для семантической сегментации")
    st.markdown("""
    - **Модель**: UNet (на основе ResNet34 encoder)
    - **Тип задачи**: Бинарная сегментация (например, лес / не лес)
    - **mAP**: 0.74  
    - **IoU (Jaccard)**: 0.77
    - **Формат входного изображения**: RGB (256x256)
    """)
    st.image("assets/map_stats_unet.png", caption="PR / IoU / F1-графики")
