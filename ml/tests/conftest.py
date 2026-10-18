from __future__ import annotations

import pytest

from backend.app.core.config import get_settings


@pytest.fixture(scope="session")
def models_dir():
    d = get_settings().models_dir
    if not (d / "grade_model.onnx").exists():
        pytest.skip(f"no exported model in {d}")
    return d
