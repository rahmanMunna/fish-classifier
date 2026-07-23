"""
Grad-CAM (Explainable AI) service.

Reuses the exact Grad-CAM logic from the original Gradio app.py:
  - pytorch_grad_cam.GradCAM on the model's designated target layer(s)
  - show_cam_on_image() to overlay the heatmap on the resized input image

The only change from the Gradio version is the output format: instead of
returning a PIL.Image for gr.Image, this returns a Base64-encoded PNG
string, since that's what a JSON API response needs to carry an image.
"""

import base64
import io

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.config import IMG_SIZE
from app.models.model_loader import get_target_layers


def generate_gradcam_base64(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    pred_index: int,
    original_pil_image: Image.Image,
) -> str:
    """
    Run Grad-CAM for the given model/prediction and return the heatmap
    overlay as a Base64-encoded PNG string (no "data:image/png;base64,"
    prefix — the frontend can add that if needed for an <img src>).
    """
    target_layers = get_target_layers(model_name, model)

    # Same resize the model/preprocessing used, normalized to [0, 1] float
    # for show_cam_on_image's expected input format.
    resized = original_pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb_np = np.array(resized).astype(np.float32) / 255.0

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_index)])[0]

    cam_image_np = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
    cam_pil = Image.fromarray(cam_image_np)

    buffer = io.BytesIO()
    cam_pil.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")