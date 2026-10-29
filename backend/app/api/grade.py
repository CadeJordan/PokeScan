"""POST /grade endpoint: front + back image -> {grade, grade_int, confidence}."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.app.ml.inference import GradePrediction, get_inference_service
from backend.app.ml.preprocess import ImageDecodeError

log = logging.getLogger(__name__)

router = APIRouter(tags=["grade"])

MAX_BYTES = 12 * 1024 * 1024  # 12 MB per image
# Keep in sync with `preprocess.UPLOAD_FORMATS`, the formats decoded from bytes.
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


class CenteringSubGrade(BaseModel):
    """Front ratios in `left_right` / `top_bottom`, back in `back_*`; None when that
    face's borders couldn't be measured reliably. `grade` is the worse face."""

    grade: float | None = None
    left_right: str | None = None
    top_bottom: str | None = None
    back_left_right: str | None = None
    back_top_bottom: str | None = None


class FactorSubGrade(BaseModel):
    grade: float | None = None
    hint: str | None = None
    flags: list[str] = Field(default_factory=list)


class SubGrades(BaseModel):
    """Per-factor breakdown.

    `centering` is classical-CV geometry. With the probe grade model,
    `corners`, `edges`, `surface` are the soft grades its corner, edge-strip
    and centre-region probes give (both faces); they are learned from overall
    PSA grades, not PSA sub-grades. Without it they fall back to the
    classical-CV condition analysis (whitening, wear, cut, scratches/creases).
    """

    centering: CenteringSubGrade | None = None
    corners: FactorSubGrade | None = None
    edges: FactorSubGrade | None = None
    surface: FactorSubGrade | None = None


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
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail=f"{label}: too large")
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
            detail="grade model not loaded - see /api/health for the reason "
            "(run `python -m ml.training.train_probe` if no model is trained)",
        )
    front_bytes = await _read_image(front, "front")
    back_bytes = await _read_image(back, "back")

    # Model inference and the CV sub-grades are seconds of CPU work; run them
    # off the event loop so other requests (e.g. /health) aren't blocked.
    try:
        pred: GradePrediction = await run_in_threadpool(service.predict, front_bytes, back_bytes)
    except ImageDecodeError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"{exc.face or 'image'}: not a readable image (corrupt or truncated file)",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    rejection = pred.rejection()
    if rejection is not None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=rejection[1])

    # Face guard notes first (swap / rotation): the notes below describe the
    # faces as graded, i.e. after the guard.
    notes: list[str] = list(pred.notes)
    if not pred.found_quad_front:
        notes.append("front: card outline not auto-detected; used center-padded fallback")
    if not pred.found_quad_back:
        notes.append("back: card outline not auto-detected; used center-padded fallback")
    # Reuse the model's crops so each face is detected once per request.
    faces = pred.crops if pred.crops is not None else (front_bytes, back_bytes)
    if pred.crops is not None:
        for label, crop in zip(("front", "back"), pred.crops, strict=True):
            if crop.rotation == 90:
                notes.append(f"{label}: photo was sideways; rotated it")

    sub: SubGrades = SubGrades()
    try:
        from backend.app.ml.centering import compute_centering_subgrade

        sub.centering = await run_in_threadpool(compute_centering_subgrade, *faces)
        if sub.centering is None:
            notes.append("centering: border edges not found reliably on either side; not graded")
        elif sub.centering.left_right is None:
            notes.append("centering: front borders unclear; graded from the back only")
        elif sub.centering.back_left_right is None:
            notes.append("centering: back borders unclear; graded from the front only")
    except Exception as exc:  # noqa: BLE001
        log.debug("centering subgrade unavailable: %s", exc)

    if pred.region_grades is not None:
        for name in ("corners", "edges", "surface"):
            grade = round(pred.region_grades[name], 1)  # hint matches the shown grade
            hint, flags = _region_hint(grade)
            setattr(sub, name, FactorSubGrade(grade=grade, hint=hint, flags=flags))
        return _response(pred, sub, notes)

    try:
        from backend.app.ml.condition import compute_condition_subgrades

        cond = await run_in_threadpool(compute_condition_subgrades, *faces)
        for name in ("corners", "edges", "surface"):
            factor = getattr(cond, name)
            setattr(
                sub,
                name,
                FactorSubGrade(
                    grade=round(factor.grade, 1),
                    hint=factor.hint,
                    flags=factor.flags,
                ),
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("condition subgrades unavailable: %s", exc)
        notes.append("corner/edge/surface analysis unavailable for these images")

    return _response(pred, sub, notes)


# Region soft grade -> UI hint. A region grade is the overall grade that region
# alone suggests, so typical gem-mint cards sit around 9.3-9.7.
_REGION_HINTS = ((9.3, "clean", []), (8.7, "slight wear", []), (7.5, "wear", ["wear"]))


def _region_hint(grade: float) -> tuple[str, list[str]]:
    for floor, hint, flags in _REGION_HINTS:
        if grade >= floor:
            return hint, list(flags)
    return "heavy wear", ["wear"]


def _response(pred: GradePrediction, sub: SubGrades, notes: list[str]) -> GradeResponse:
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
