"""Model provenance: `train_probe._git_commit` (also used by `train_face_guard`)
returns HEAD, or None with a WARNING saying why (intent/serving-hardening)."""

from __future__ import annotations

import logging
import subprocess

import pytest

from ml.training import train_probe


def _fake_run(**result):
    def run(cmd, **kwargs):
        if "raise" in result:
            raise result["raise"]
        return subprocess.CompletedProcess(cmd, result["returncode"], result["stdout"], result["stderr"])

    return run


def test_returns_head(monkeypatch, caplog):
    monkeypatch.setattr(
        train_probe.subprocess, "run", _fake_run(returncode=0, stdout="abc123\n", stderr="")
    )
    with caplog.at_level(logging.WARNING):
        assert train_probe._git_commit() == "abc123"
    assert caplog.records == []


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        ({"raise": FileNotFoundError(2, "The system cannot find the file specified")}, "FileNotFoundError"),
        ({"raise": subprocess.TimeoutExpired(["git"], 30)}, "TimeoutExpired"),
        (
            {"returncode": 128, "stdout": "", "stderr": "fatal: detected dubious ownership in repository\n"},
            "dubious ownership",
        ),
        ({"returncode": 0, "stdout": "", "stderr": ""}, "exit 0"),
    ],
    ids=["git_not_on_path", "timeout", "refused", "empty_output"],
)
def test_failure_is_logged_not_silent(monkeypatch, caplog, result, reason):
    monkeypatch.setattr(train_probe.subprocess, "run", _fake_run(**result))
    with caplog.at_level(logging.WARNING):
        assert train_probe._git_commit() is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "git_commit unavailable" in warnings[0].getMessage()
    assert reason in warnings[0].getMessage()


def test_face_guard_uses_the_same_helper():
    from ml.training import train_face_guard

    assert train_face_guard._git_commit is train_probe._git_commit


def test_real_repo_answers():
    """In this checkout git resolves HEAD (skips when run outside a git repo)."""
    commit = train_probe._git_commit()
    if commit is None:
        pytest.skip("not a git checkout / git unavailable")
    assert len(commit) == 40
