import torch
import torchvision.transforms as transforms
from PIL import Image
from models.my_unet import UNet
import streamlit as st

MODEL_PATH = "models/unet_forest.pth"

@st.cache_resource
def load_model():
    model = UNet(in_channels=3, out_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def postprocess(output, original_size):
    output = output.squeeze().cpu().numpy()
    output = (output > 0.5).astype("uint8") * 255
    mask = Image.fromarray(output).resize(original_size)
    return mask.convert("L")

def run_unet_segmentation(image: Image.Image) -> Image.Image:
    model, device = load_model()
    input_tensor = preprocess(image).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    return postprocess(output, image.size)
