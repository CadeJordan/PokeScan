from __future__ import annotations

import pytest

from backend.app.core.config import get_settings


@pytest.fixture(scope="session")
def models_dir():
    s = get_settings()
    if not (s.grade_head_path.exists() and s.feature_model_path.exists()):
        pytest.skip(f"probe model not trained in {s.models_dir}")
    return s.models_dir
