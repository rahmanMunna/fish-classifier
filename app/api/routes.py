"""
API routes.

GET  /health         -> defined in main.py (Phase 1), stays there
GET  /models          -> list of available model display names
POST /predict          -> multipart image + model_name -> prediction + gradcam
POST /generate-pdf      -> multipart image + model_name -> downloadable PDF

/generate-pdf deliberately re-runs the full predict + gradcam pipeline
rather than requiring the frontend to resend the previous prediction's
data. This keeps the API stateless and each endpoint self-contained —
the frontend just needs the same (image, model_name) pair it already has,
not a copy of a large base64 blob it received from /predict.
"""

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from app.models.model_loader import get_model, list_available_models
from app.schemas.response import ModelsResponse, PredictResponse
from app.services.gradcam_service import generate_gradcam_base64
from app.services.inference import run_prediction
from app.services.pdf_service import generate_pdf_report
from app.utils.image_utils import read_upload_as_pil

router = APIRouter()


def _validate_model_name(model_name: str) -> None:
    available = list_available_models()
    if model_name not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_name '{model_name}'. Available models: {available}",
        )


@router.get("/models", response_model=ModelsResponse, tags=["Models"])
def get_models() -> ModelsResponse:
    return ModelsResponse(models=list_available_models())


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
    image: UploadFile = File(...),
    model_name: str = Form(...),
) -> PredictResponse:
    _validate_model_name(model_name)
    pil_image = await read_upload_as_pil(image)

    result = run_prediction(pil_image, model_name)
    model = get_model(model_name)

    gradcam_b64 = generate_gradcam_base64(
        model=model,
        model_name=model_name,
        input_tensor=result.input_tensor,
        pred_index=result.pred_index,
        original_pil_image=pil_image,
    )

    return PredictResponse(
        prediction=result.pred_class,
        confidence=result.confidence,
        probabilities=result.probabilities,
        gradcam=gradcam_b64,
        inference_time=result.inference_time_ms,
        model_used=model_name,
    )


@router.post("/generate-pdf", tags=["Report"])
async def generate_pdf(
    image: UploadFile = File(...),
    model_name: str = Form(...),
) -> FileResponse:
    _validate_model_name(model_name)
    pil_image = await read_upload_as_pil(image)

    result = run_prediction(pil_image, model_name)
    model = get_model(model_name)

    gradcam_b64 = generate_gradcam_base64(
        model=model,
        model_name=model_name,
        input_tensor=result.input_tensor,
        pred_index=result.pred_index,
        original_pil_image=pil_image,
    )

    pdf_path = generate_pdf_report(
        model_name=model_name,
        pred_class=result.pred_class,
        confidence=result.confidence,
        probabilities=result.probabilities,
        original_pil_image=pil_image,
        gradcam_base64=gradcam_b64,
    )

    filename = pdf_path.split("/")[-1].split("\\")[-1]
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)