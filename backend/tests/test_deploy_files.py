"""Container build inputs (intent/cloud-run-deploy): the Dockerfile, the serving
requirements and the two allow-list ignore files stay in step.

`.dockerignore` (local `docker build`) and `.gcloudignore` (`gcloud run deploy
--source .`) must let through the same files: backend/ without its tests, the
serving requirements and the six model files; never data/images, crops, the DB,
backups, .env or the frontend.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES = {
    "card_features.onnx",
    "card_features.json",
    "grade_head.npz",
    "grade_head.json",
    "face_guard.npz",
    "face_guard.json",
}


def _lines(name: str) -> list[str]:
    text = (ROOT / name).read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def _reincluded(lines: list[str]) -> set[str]:
    return {ln[1:].strip("/") for ln in lines if ln.startswith("!")}


def test_dockerignore_is_an_allow_list():
    lines = _lines(".dockerignore")
    assert lines[0] == "*"
    assert _reincluded(lines) == {
        "Dockerfile",
        "deploy/requirements-serve.txt",
        "backend",
        *(f"data/models/{f}" for f in MODEL_FILES),
    }
    assert "backend/tests/" in lines


def test_gcloudignore_mirrors_the_dockerignore():
    lines = _lines(".gcloudignore")
    assert lines[0] == "/*"
    # Directories are re-opened level by level (gitignore can't re-include a file
    # under an excluded directory); the files let through are the same.
    opened_dirs = {"data", "data/models", "deploy"}
    assert _reincluded(lines) - opened_dirs == _reincluded(_lines(".dockerignore"))
    for d in opened_dirs:  # each re-opened directory is closed again just below
        assert f"/{d}/*" in lines
    assert "/backend/tests/" in lines


def test_dockerfile_copies_the_model_files_and_runs_as_non_root():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    copied = set(re.findall(r"data/models/([\w.]+)", text))
    assert copied == MODEL_FILES
    users = re.findall(r"^USER\s+(\S+)", text, flags=re.MULTILINE)
    assert users and users[-1] not in ("root", "0")
    assert "--workers 1" in text
    assert "${PORT" in text
    assert ".env" not in text


def test_serving_requirements_are_pinned_and_torch_free():
    reqs = _lines("deploy/requirements-serve.txt")
    names = set()
    for req in reqs:
        assert "==" in req, req
        names.add(re.split(r"[\[=]", req, maxsplit=1)[0].lower().replace("_", "-"))
    assert {"fastapi", "uvicorn", "onnxruntime", "opencv-python-headless", "numpy"} <= names
    assert not names & {"torch", "torchvision", "timm", "opencv-python", "pokescan"}
