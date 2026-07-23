"""
Pydantic response models for the REST API.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class ModelsResponse(BaseModel):
    models: list[str]


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Predicted species name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of the top prediction")
    probabilities: dict[str, float] = Field(..., description="Per-class probability breakdown")
    gradcam: str = Field(..., description="Base64-encoded PNG of the Grad-CAM heatmap overlay")
    inference_time: float = Field(..., description="Inference time in milliseconds")
    model_used: str = Field(..., description="Display name of the model that produced this result")


class ErrorResponse(BaseModel):
    detail: str