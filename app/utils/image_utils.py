"""
Small helper for turning an uploaded file into a PIL.Image, with clean
HTTP-friendly errors for the common failure modes: wrong content-type,
oversized upload, corrupt/unreadable image data.
"""

import io

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


async def read_upload_as_pil(upload_file: UploadFile) -> Image.Image:
    """Read a FastAPI UploadFile and return it as a PIL.Image (RGB).

    Raises HTTPException(400) for: wrong content-type, oversized file,
    or unreadable/corrupt image data.
    """
    if upload_file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{upload_file.content_type}'. "
                f"Allowed types: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    raw_bytes = await upload_file.read()

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max allowed size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()  # force-read now, so corrupt files fail here, not later
        return image.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}")