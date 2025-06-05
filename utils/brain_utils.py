import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# === Загрузка моделей ===
# Классификационная модель
classification_model = torch.load("models/classifier.pt", map_location="cpu")
classification_model.eval()

# YOLO модели по типу среза
yolo_models = {
    "axial": YOLO("models/yolo_axial.pt"),
    "sagittal": YOLO("models/yolo_sag.pt"),
    "coronal": YOLO("models/yolo_coronal.pt"),
}

# Классы срезов (в порядке обучения)
slice_classes = ["axial", "coronal", "sagittal"]

# === Преобразования ===
# Только для классификации
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # оптимальный размер для EfficientNet-B0
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def classify_slice(image: Image.Image) -> str:
    """
    Классифицирует тип среза мозга: axial, coronal, sagittal.
    Использует уменьшенную копию изображения (224x224).
    """
    input_tensor = classification_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = classification_model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()
    return slice_classes[predicted]


def detect_tumor(original_image: Image.Image, slice_type: str) -> Image.Image:
    """
    Детектирует опухоль на изображении с помощью соответствующей YOLO модели.
    Работает с 640x640 изображением.
    """
    model = yolo_models.get(slice_type.lower())
    if model is None:
        raise ValueError(f"No YOLO model found for slice type: {slice_type}")

    resized_image = original_image.resize((512, 512))
    results = model.predict(resized_image, conf=0.25, save=False, imgsz=640)
    annotated_image = results[0].plot()
    return Image.fromarray(annotated_image)
