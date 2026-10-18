# Small fixes

Changes too small for their own spec. Each one should leave behaviour unchanged and keep the fast suite at 248 passed / 3 xfailed:
`python -m pytest -m "not slow and not visual and not model"`

## 1. Lint baseline (ruff: 25 errors → 0)

Needed before [[ci-fast-suite]] can gate on `ruff check .`.

Config (`pyproject.toml`): these patterns are idiomatic here, so allow them rather than rewriting code.
- [ ] `[tool.ruff.lint.flake8-bugbear] extend-immutable-calls = ["typer.Option", "typer.Argument"]` clears all 14 B008 errors (typer CLI defaults).
- [ ] Allow N812 for the conventional aliases `import albumentations as A` (`ml/data/dataset.py:16`) and `import torch.nn.functional as F` (`ml/training/calibrate.py:25`), via `ignore` or `# noqa: N812`.
- [ ] `backend/app/ml/preprocess.py:491` `M` (OpenCV transform-matrix convention): `# noqa: N806`, or rename to `warp_m`.

Code (auto-fixable with `ruff check --fix`; review the diff):
- [ ] UP035 in `backend/app/main.py:11`, `ml/data/db.py:8`, `ml/data/ebay_client.py:20`: import from `collections.abc`.
- [ ] UP037 in `ml/data/ebay_client.py:99`: unquote the annotation.
- [ ] I001 + F401 in `backend/app/ml/multitask.py:18,25`: sort imports, drop the unused `NUM_THRESHOLDS`.

Code (manual):
- [ ] E741 in `backend/app/ml/centering.py:40`: rename `l` to `left` (or similar).
- [ ] SIM105 in `backend/app/core/config.py:65`: use `contextlib.suppress(OSError)`.

Done when `ruff check .` exits 0.

## 2. Deprecated 413 status constant

- [ ] `backend/app/api/grade.py:67`: `HTTP_413_REQUEST_ENTITY_TOO_LARGE` → `HTTP_413_CONTENT_TOO_LARGE`. It's the only warning from our own code in the test run. Check that the installed Starlette has the new name (fastapi 0.136.1 does).

## 3. Misleading comment in the model

- [ ] `backend/app/ml/model.py`, in `GradeModel.__init__`: the comment says outputs are "guaranteed monotonic via softplus", but there's no softplus. Monotonicity comes from `cummin` in `decode_corn_logits`. Fix the comment only; the decoding question itself is [[corn-decode-verification]].

## 4. Commit the agent-skills setup

- [ ] Commit the already-written `docs/agents/` (issue tracker, triage labels, domain docs), the `CLAUDE.md` updates and this `intent/` folder on `CadeJordan/ScoreImprovements`. The draft `.github/workflows/ci.yml` is covered by [[ci-fast-suite]].

## 5. GitHub CLI auth (user action)

- [ ] Run `! gh auth login`. `gh` is currently unauthenticated, so the issue-tracker workflow in `docs/agents/issue-tracker.md` can't work, and repo visibility (relevant to [[git-history-image-purge]]) can't be checked.
- [ ] Then create the five triage labels if they're missing: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.

## 6. README venv instructions

- [ ] `README.md` setup tells you to create and activate `.venv`. Until [[pinned-dev-environment]] lands, add a one-line note that the current working env is the global pyenv 3.11.6, or fold this into that spec.

## 7. Stop tracking graphify cache

- [ ] `graphify-out/cache/` (dozens of AST/semantic JSON blobs) is committed. Add `graphify-out/cache/` to `.gitignore` and `git rm -r --cached graphify-out/cache`. Keep `graph.json` / `GRAPH_REPORT.md` tracked if they're wanted. Ask before removing them.

## 8. Frontend lint script

- [ ] `frontend/package.json` still has `"lint": "next lint"`, but the draft CI notes `next lint` was removed in Next 16 and calls `npx eslint src` instead. If the installed Next is 16+, change the script to `eslint src` so `npm run lint` (documented in `CLAUDE.md`) works locally.

## 9. Global env sanity check

- [ ] Run `python -m pip check` on the global pyenv env and record the result in [[pinned-dev-environment]] before freezing constraints from it.
