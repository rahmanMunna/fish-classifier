"""
Inference service.

Reuses the exact prediction logic from the original Gradio app.py's
run_inference(): preprocess -> forward pass -> softmax -> argmax.
Adds inference timing, since the API response includes it.

Grad-CAM is deliberately NOT done here — see gradcam_service.py (Phase 6).
Keeping them separate means /predict can run without Grad-CAM overhead
if we ever want a lightweight endpoint, and keeps this file single-purpose.
"""

import time
from dataclasses import dataclass

import torch
from PIL import Image

from app.config import CLASS_NAMES, DEVICE, NUM_CLASSES
from app.models.model_loader import get_model
from app.utils.preprocess import preprocess_image


@dataclass
class InferenceResult:
    pred_class: str
    pred_index: int
    confidence: float
    probabilities: dict[str, float]
    inference_time_ms: float
    input_tensor: torch.Tensor  # kept for reuse by gradcam_service (avoids re-preprocessing)


def run_prediction(pil_image: Image.Image, model_name: str) -> InferenceResult:
    """Run a forward pass for the given model and return prediction + probabilities."""
    model = get_model(model_name)

    start = time.perf_counter()

    input_tensor = preprocess_image(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    pred_index = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index])
    probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    return InferenceResult(
        pred_class=pred_class,
        pred_index=pred_index,
        confidence=confidence,
        probabilities=probabilities,
        inference_time_ms=elapsed_ms,
        input_tensor=input_tensor,
    )