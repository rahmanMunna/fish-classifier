"""
Grad-CAM (Explainable AI) service — pure PyTorch + NumPy/Pillow implementation.

Previously used the `pytorch_grad_cam` package, but that library imports
`cv2` at module load time (even if we never call its OpenCV-dependent
functions), and `cv2` requires system-level X11 libraries (libxcb, etc.)
that aren't present by default on minimal container platforms like
Railway/Render's Nixpacks builders. Rather than chase system-package
fixes across every hosting platform, Grad-CAM is reimplemented directly
here using standard forward/backward hooks — same algorithm, zero
extra system dependencies.

Algorithm (standard Grad-CAM, Selvaraju et al. 2017):
  1. Forward hook captures the target layer's activations.
  2. Backward hook captures gradients of the predicted class w.r.t.
     those activations.
  3. Global-average-pool the gradients -> per-channel importance weights.
  4. Weighted sum of activation channels -> ReLU -> normalize to [0, 1].
  5. Resize to the input image size, apply a jet colormap, blend over
     the original image.
"""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.config import IMG_SIZE
from app.models.model_loader import get_target_layers


class _GradCAMHook:
    """Registers forward/backward hooks on a single target layer to
    capture activations and gradients for one forward+backward pass."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def compute_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Runs one forward+backward pass and returns a normalized (H, W)
        grayscale CAM in [0, 1], matching the target layer's spatial size."""
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=False)

        activations = self.activations[0]  # (C, H, W)
        gradients = self.gradients[0]       # (C, H, W)

        weights = gradients.mean(dim=(1, 2))  # (C,) global-average-pool
        cam = torch.einsum("c,chw->hw", weights, activations)
        cam = F.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.cpu().numpy()


def _apply_jet_colormap(grayscale: np.ndarray) -> np.ndarray:
    """Standard analytic 'jet' colormap approximation (blue -> cyan -> green
    -> yellow -> red), avoiding any dependency on OpenCV/matplotlib for this.
    Input: float array in [0, 1]. Output: uint8 RGB array, same shape + (3,).
    """
    r = np.clip(1.5 - np.abs(4 * grayscale - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * grayscale - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * grayscale - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def generate_gradcam_base64(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    pred_index: int,
    original_pil_image: Image.Image,
    alpha: float = 0.45,
) -> str:
    """
    Run Grad-CAM for the given model/prediction and return the heatmap
    overlay as a Base64-encoded PNG string (no "data:image/png;base64,"
    prefix — the frontend adds that if needed for an <img src>).
    """
    target_layers = get_target_layers(model_name, model)
    target_layer = target_layers[0]

    hook = _GradCAMHook(model, target_layer)
    try:
        grayscale_cam = hook.compute_cam(input_tensor, pred_index)
    finally:
        hook.remove()

    # Resize the low-res CAM (e.g. 7x7) up to the display image size using PIL
    cam_img = Image.fromarray((grayscale_cam * 255).astype(np.uint8))
    cam_resized = cam_img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    grayscale_resized = np.array(cam_resized).astype(np.float32) / 255.0

    heatmap_rgb = _apply_jet_colormap(grayscale_resized)

    resized_original = original_pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    original_rgb = np.array(resized_original).astype(np.float32)

    overlay = (heatmap_rgb.astype(np.float32) * alpha) + (original_rgb * (1 - alpha))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    cam_pil = Image.fromarray(overlay)

    buffer = io.BytesIO()
    cam_pil.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")