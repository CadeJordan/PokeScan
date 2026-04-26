"""POST /grade endpoint: front + back image -> {grade, grade_int, confidence}."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.app.ml.inference import GradePrediction, get_inference_service

log = logging.getLogger(__name__)

router = APIRouter(tags=["grade"])

MAX_BYTES = 12 * 1024 * 1024  # 12 MB per image
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


class CenteringSubGrade(BaseModel):
    grade: float | None = None
    left_right: str | None = None
    top_bottom: str | None = None


class SubGrades(BaseModel):
    """Reserved slots for per-factor breakdown.

    Phase B fills `centering`. Phase C fills `corners`, `edges`, `surface`.
    """

    centering: CenteringSubGrade | None = None
    corners: float | None = None
    edges: float | None = None
    surface: float | None = None


class GradeResponse(BaseModel):
    grade: float = Field(..., description="Soft expected grade in [1, 10].")
    grade_int: int = Field(..., ge=1, le=10, description="Hard predicted grade rank.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    rank_probs: list[float] = Field(
        ..., description="Cumulative P(grade > k) for k = 1..9."
    )
    sub_grades: SubGrades = Field(default_factory=SubGrades)
    notes: list[str] = Field(default_factory=list)


async def _read_image(upload: UploadFile, label: str) -> bytes:
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{label}: unsupported content type {upload.content_type!r}",
        )
    data = await upload.read()
    if len(data) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"{label}: empty upload")
    if len(data) > MAX_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"{label}: too large")
    return data


@router.post("/grade", response_model=GradeResponse)
async def grade_card(
    front: Annotated[UploadFile, File(description="Card front image (JPEG/PNG/WebP).")],
    back: Annotated[UploadFile, File(description="Card back image (JPEG/PNG/WebP).")],
) -> GradeResponse:
    service = get_inference_service()
    if not service.ready:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="grade model not loaded - run `python -m ml.training.export` first",
        )
    front_bytes = await _read_image(front, "front")
    back_bytes = await _read_image(back, "back")

    try:
        pred: GradePrediction = service.predict(front_bytes, back_bytes)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    notes: list[str] = []
    if not pred.found_quad_front:
        notes.append("front: card outline not auto-detected; used center-padded fallback")
    if not pred.found_quad_back:
        notes.append("back: card outline not auto-detected; used center-padded fallback")

    sub: SubGrades = SubGrades()
    try:
        from backend.app.ml.centering import compute_centering_subgrade

        sub.centering = compute_centering_subgrade(front_bytes)
    except Exception as exc:  # noqa: BLE001
        log.debug("centering subgrade unavailable: %s", exc)

    return GradeResponse(
        grade=round(pred.grade, 2),
        grade_int=pred.grade_int,
        confidence=round(pred.confidence, 4),
        rank_probs=[round(p, 4) for p in pred.rank_probs],
        sub_grades=sub,
        notes=notes,
    )


@router.get("/health")
async def health() -> dict:
    service = get_inference_service()
    return {
        "status": "ok",
        "model_loaded": service.ready,
        "metadata": service.metadata,
    }
