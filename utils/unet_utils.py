from PIL import Image, ImageOps

def run_unet_segmentation(image: Image.Image) -> Image.Image:
    # Инверсия — как будто предсказана маска
    return ImageOps.invert(image.convert("RGB"))