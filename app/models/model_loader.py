"""
Model registry + lazy-loading cache.

Same builder logic as before (EfficientNet-B3/B0 via timm, MobileNetV3 via
torchvision, Hybrid DFCViT-4C custom architecture).

IMPORTANT CHANGE: Render's free tier only has 512MB RAM. Caching all 4
models simultaneously (as the original version did) can exceed that once
a user tries more than one or two models in a session, causing the
instance to be OOM-killed (502 Bad Gateway).

This version keeps only the MOST RECENTLY USED model in memory
(MAX_CACHED_MODELS = 1). Switching models will re-load from disk each
time — slightly slower, but keeps the app alive on the free tier. Raise
MAX_CACHED_MODELS if you upgrade to a paid Render plan with more RAM.
"""

import gc
import os
from collections import OrderedDict

import timm
import torch
import torch.nn as nn
from torchvision import models

from app.config import DEVICE, MODEL_FILES, MODELS_DIR, NUM_CLASSES
from app.models.hybrid_model import DFCViT4C

# Keep PyTorch's own thread pool small too — on a small Render instance,
# torch defaulting to using many threads for CPU inference can itself add
# memory/CPU overhead. 1-2 is plenty for single-image inference.
torch.set_num_threads(1)


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

# ─────────────────────────────────────────────────────────────────────────
# LRU cache — keeps at most MAX_CACHED_MODELS models in memory at once.
# On a 512MB Render free instance, 1 is the safe default.
# ─────────────────────────────────────────────────────────────────────────
MAX_CACHED_MODELS = int(os.environ.get("MAX_CACHED_MODELS", "1"))

_MODEL_CACHE: "OrderedDict[str, nn.Module]" = OrderedDict()


def get_model(name: str) -> nn.Module:
    """Return a cached, eval-mode model instance for the given display name.
    Evicts the least-recently-used model if the cache is full."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    if name in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(name)  # mark as most-recently-used
        return _MODEL_CACHE[name]

    model = MODEL_REGISTRY[name]["builder"]()
    model.to(DEVICE).eval()
    _MODEL_CACHE[name] = model

    while len(_MODEL_CACHE) > MAX_CACHED_MODELS:
        evicted_name, evicted_model = _MODEL_CACHE.popitem(last=False)
        del evicted_model
        gc.collect()

    return _MODEL_CACHE[name]


def get_target_layers(name: str, model: nn.Module):
    """Return the Grad-CAM target layer(s) for a given model name."""
    return MODEL_REGISTRY[name]["target_layer"](model)


def list_available_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())