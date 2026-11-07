# PokeScan API image for Cloud Run (docs/deploy.md). Build context: the repo
# root, filtered by .dockerignore (local `docker build`) and .gcloudignore
# (`gcloud run deploy --source .`), which allow only backend/ (no tests), this
# file, deploy/requirements-serve.txt and the six serving model files.
# headless OpenCV and ONNX Runtime import and grade on the slim base as is: no
# extra system libraries are needed (checked 2026-11-07).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /app

# Serving packages only (headless OpenCV, ONNX Runtime CPU; no torch).
COPY deploy/requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install -r /tmp/requirements-serve.txt && rm /tmp/requirements-serve.txt

# Non-root user. It owns /app/data (not the files in it) because Settings
# creates data/images and data/crops at startup; both stay empty.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/data/models \
    && chown app /app/data

COPY backend/ /app/backend/
# The probe model (head + DINOv2 feature graph + its sidecar) and the face guard.
COPY data/models/card_features.onnx data/models/card_features.json \
     data/models/grade_head.npz data/models/grade_head.json \
     data/models/face_guard.npz data/models/face_guard.json \
     /app/data/models/

USER app
EXPOSE 8080

# One worker: GRADE_CONCURRENCY bounds grades within it. Cloud Run sets $PORT.
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port \"${PORT:-8080}\" --workers 1"]
