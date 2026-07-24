"""
Central configuration for the backend.

Model weights (Models/*.pt) are downloaded from Hugging Face at startup
rather than committed to git — Git LFS support was unreliable across
hosting platforms (Railway's builder checked out LFS pointer files
instead of the actual binaries). Hosting on Hugging Face and downloading
once at container startup sidesteps that entirely.
"""

import json
import os
import urllib.request

import torch

# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
STATIC_TEMP_DIR = os.path.join(BASE_DIR, "app", "static", "temp")

os.makedirs(STATIC_TEMP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# Download model weights from Hugging Face if not already present locally
# ─────────────────────────────────────────────────────────────────────────
HF_BASE_URL = "https://huggingface.co/munna1919/fish-classifier-weights/resolve/main"

for _filename in ["efficientnet.pt", "efficientnetb0.pt", "mobilenet.pt", "hybrid.pt"]:
    _local_path = os.path.join(MODELS_DIR, _filename)
    if not os.path.exists(_local_path):
        print(f"Downloading {_filename} from Hugging Face...")
        urllib.request.urlretrieve(f"{HF_BASE_URL}/{_filename}", _local_path)
        print(f"Downloaded {_filename} ({os.path.getsize(_local_path)} bytes)")
    else:
        print(f"{_filename} already present locally, skipping download.")

# ─────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────
# Class metadata (loaded once, at import time)
# ─────────────────────────────────────────────────────────────────────────
with open(CLASS_NAMES_PATH) as f:
    _META = json.load(f)

CLASS_NAMES: list[str] = _META["class_names"]
NUM_CLASSES: int = _META["num_classes"]
IMG_SIZE: int = _META.get("img_size", 224)
MEAN: list[float] = _META["mean"]
STD: list[float] = _META["std"]

# ─────────────────────────────────────────────────────────────────────────
# Model registry filenames
# ─────────────────────────────────────────────────────────────────────────
MODEL_FILES = {
    "EfficientNet-B3": "efficientnet.pt",
    "EfficientNet-B0": "efficientnetb0.pt",
    "MobileNetV3": "mobilenet.pt",
    "Hybrid (DFCViT-4C)": "hybrid.pt",
}