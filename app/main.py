"""
FastAPI entrypoint.

Wires together:
  - CORS (for the future Next.js frontend)
  - GET /health
  - api/routes.py -> GET /models, POST /predict, POST /generate-pdf
  - Global exception handlers, so every error the frontend receives is
    consistent JSON ({"detail": "..."}) instead of a raw traceback/HTML
    page on unexpected 500s.
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router

logger = logging.getLogger("fish_classifier")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Fish Classifier Backend",
    description="REST API backend for the Carp Fish Species Classification thesis project.",
    version="0.1.0",
)

# CORS — wide open for now during local development.
# Restrict allow_origins to the deployed Vercel URL in Phase 10.
# Replace allow_origins=["*"] with your actual frontend domain(s) once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",                     # local Next.js dev
        "*",      # replace with your real Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Simple liveness check used by Render and local testing."""
    return {"status": "running"}


# ─────────────────────────────────────────────────────────────────────────
# Global exception handlers
# ─────────────────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Missing/invalid form fields (e.g. no image, no model_name) -> clean 422."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Pass through our own intentional HTTPExceptions (400s from routes/
    image_utils) unchanged, just logged."""
    logger.warning(f"HTTPException on {request.url.path}: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Anything truly unexpected (e.g. a bug, an OOM, a torch runtime error)
    -> log the full traceback server-side, return a clean generic 500 to
    the client instead of leaking internals."""
    logger.exception(f"Unhandled exception on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again or contact support."},
    )