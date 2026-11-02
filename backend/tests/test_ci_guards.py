"""CI must fail when a test step runs no tests (intent/serving-hardening).

The frontend's `npm test` checks this itself (`frontend/scripts/test.mjs`);
the backend step relies on pytest's own exit code, pinned here.
"""

from __future__ import annotations

import subprocess
import sys


def test_pytest_exits_5_when_no_tests_are_collected(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_empty.py").write_text("# no tests\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", "pytest.ini", "."],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 5, result.stdout + result.stderr
