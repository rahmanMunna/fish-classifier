"""
Image preprocessing pipeline.

Identical to the transform used in training and in the original Gradio
app.py: Resize -> ToTensor -> Normalize, using the mean/std/img_size from
class_names.json (via app.config).
"""

from PIL import Image
from torchvision import transforms

from app.config import IMG_SIZE, MEAN, STD

preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def preprocess_image(pil_image: Image.Image):
    """
    Convert a PIL image into a normalized tensor ready for model input.
    Does NOT add the batch dimension — callers (inference service) do that,
    since batching may differ between single-image and future batch endpoints.
    """
    rgb_image = pil_image.convert("RGB")
    return preprocess_transform(rgb_image)