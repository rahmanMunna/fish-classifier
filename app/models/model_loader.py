"""
Model registry + lazy-loading cache.

Reuses the exact same builder logic as the original Gradio app.py:
  - EfficientNet-B3 / EfficientNet-B0 -> timm, num_classes swapped
  - MobileNetV3                        -> torchvision mobilenet_v3_large,
                                           classifier[-1] swapped
  - Hybrid (DFCViT-4C)                  -> custom architecture from
                                           app/models/hybrid_model.py

Models are only loaded into memory the first time they're requested
(get_model), then cached in _MODEL_CACHE for the lifetime of the process.
This is important on Render's free/low tiers — don't reload a model on
every request.
"""

import os

import timm
import torch
import torch.nn as nn
from torchvision import models

from app.config import DEVICE, MODEL_FILES, MODELS_DIR, NUM_CLASSES
from app.models.hybrid_model import DFCViT4C

# ─────────────────────────────────────────────────────────────────────────
# Builders — one per architecture family
# ─────────────────────────────────────────────────────────────────────────
def build_efficientnet(variant: str, weight_file: str) -> nn.Module:
    m = timm.create_model(variant, pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(os.path.join(MODELS_DIR, weight_file), map_location=DEVICE)
    m.load_state_dict(state)
    return m


def build_mobilenetv3(weight_file: str) -> nn.Module:
    m = models.mobilenet_v3_large(weights=None)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    state = torch.load(os.path.join(MODELS_DIR, weight_file), map_location=DEVICE)
    m.load_state_dict(state)
    return m


def build_hybrid(weight_file: str) -> nn.Module:
    m = DFCViT4C(NUM_CLASSES)
    state = torch.load(os.path.join(MODELS_DIR, weight_file), map_location=DEVICE)
    m.load_state_dict(state)
    return m


# ─────────────────────────────────────────────────────────────────────────
# Registry: display name -> (builder, Grad-CAM target layer getter)
# The display names here are the source of truth for GET /models and for
# the `model_name` field accepted by POST /predict.
# ─────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "EfficientNet-B3": {
        "builder": lambda: build_efficientnet("efficientnet_b3", MODEL_FILES["EfficientNet-B3"]),
        "target_layer": lambda m: [m.conv_head],
    },
    "EfficientNet-B0": {
        "builder": lambda: build_efficientnet("efficientnet_b0", MODEL_FILES["EfficientNet-B0"]),
        "target_layer": lambda m: [m.conv_head],
    },
    "MobileNetV3": {
        "builder": lambda: build_mobilenetv3(MODEL_FILES["MobileNetV3"]),
        "target_layer": lambda m: [m.features[-1][0]],
    },
    "Hybrid (DFCViT-4C)": {
        "builder": lambda: build_hybrid(MODEL_FILES["Hybrid (DFCViT-4C)"]),
        "target_layer": lambda m: [m.proj_conv],
    },
}

_MODEL_CACHE: dict[str, nn.Module] = {}


def get_model(name: str) -> nn.Module:
    """Return a cached, eval-mode model instance for the given display name.
    Builds + loads weights only on first call for that name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    if name not in _MODEL_CACHE:
        model = MODEL_REGISTRY[name]["builder"]()
        model.to(DEVICE).eval()
        _MODEL_CACHE[name] = model

    return _MODEL_CACHE[name]


def get_target_layers(name: str, model: nn.Module):
    """Return the Grad-CAM target layer(s) for a given model name."""
    return MODEL_REGISTRY[name]["target_layer"](model)


def list_available_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())