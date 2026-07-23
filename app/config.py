"""
Central configuration for the backend.

Mirrors exactly what the original Gradio app.py read from class_names.json
and how it resolved paths — nothing about these values has changed.
"""

import json
import os

import torch

# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────
# app/config.py -> app/ -> project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
STATIC_TEMP_DIR = os.path.join(BASE_DIR, "app", "static", "temp")

os.makedirs(STATIC_TEMP_DIR, exist_ok=True)

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
# Model registry filenames (used by model_loader.py in Phase 4)
# ─────────────────────────────────────────────────────────────────────────
MODEL_FILES = {
    "EfficientNet-B3": "efficientnet.pt",
    "EfficientNet-B0": "efficientnetb0.pt",
    "MobileNetV3": "mobilenet.pt",
    "Hybrid (DFCViT-4C)": "hybrid.pt",
}