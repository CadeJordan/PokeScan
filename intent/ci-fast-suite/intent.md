# CI: lint + fast test suite

## Intent

Every push to `main` and every PR to `CadeJordan/PokeScan` runs lint plus the fast pytest suite, and the frontend lint/typecheck/build, on GitHub Actions, and **fails** on any regression, so the current baseline can't silently slip.

## Current state (2026-10-17)

A draft `.github/workflows/ci.yml` exists in the working tree (untracked, written 23:51, not yet run on GitHub). It already:
- runs on `push` to `main`, on `pull_request` and on `workflow_dispatch`, with concurrency cancel and `contents: read`;
- in the Python job: Ubuntu, Python 3.11, `libgl1`, CPU torch from the PyTorch CPU index, `pip install -e ".[dev]"`, ruff (**`continue-on-error: true`**), and `pytest -m "not slow and not model and not visual"` with `WANDB_MODE=disabled`;
- in the frontend job: Node 22, `npm ci`, `npx eslint src`, `npx tsc --noEmit`, `npm run build`.

So this spec is about finishing and verifying it, not writing it.

## Why

- Nothing has been checked on GitHub yet: the workflow is uncommitted and has never run.
- Ruff doesn't block the build, so lint can regress freely.
- Dependencies are unpinned (`>=` floors plus whatever CPU torch is latest), so CI can test a different stack than local. See [[pinned-dev-environment]].

## Scope

- Commit the workflow and get both jobs green on a PR from `CadeJordan/ScoreImprovements`.
- After the lint baseline (`smallfix.md` §1) reaches 0 errors, remove `continue-on-error` from the Ruff step.
- Once [[pinned-dev-environment]] lands, install with `-c constraints.txt`, with CPU torch pinned to the same version as local (2.5.1) from the CPU index.
- Point the pip cache's `cache-dependency-path` at the constraints file once it exists.

## Out of scope

- `slow`, `model` and `visual` tests. They need real data or weights, which must never be uploaded to CI.
- GPU runners, deploys (`npm run deploy`), wandb.

## Acceptance criteria

- [x] `ci.yml` committed; both jobs green on a PR with no secrets configured. (PR #3, 2026-10-18)
- [x] The Python job collects the same tests as locally: 248 passed / 3 xfailed / 31 deselected, give or take xpass changes. (CI: 248 passed / 3 xfailed / 31 deselected.)
- [ ] No `continue-on-error` remains on Ruff; a PR introducing a ruff error fails.
- [x] A failing test fails the run. (First PR #3 run failed on a real torch 2.14 ONNX-export break, since fixed by pinning CPU torch 2.5.1.)
- [ ] The workflow never reads `.env`, `data/` or secrets; `model` tests skip because no weights exist.
- [x] The Python job takes under 30 min (the configured timeout). (~1.5 min.)

## Facts vs assumptions

- **Fact:** the draft workflow contents above, and the local fast-suite result (170s, Windows, GPU present but unused).
- **Assumption:** the suite passes on Linux. Only verified on Windows; OpenCV codec or path differences may show up.
- **Assumption:** tests build timm models with `pretrained=False`, so there are no weight downloads in CI. Verify.
- **Assumption:** the frontend job passes as written. `next lint` was removed in Next 16, which the workflow already works around; local `npm run lint` hasn't been checked (see `smallfix.md`).

## Open questions

- Is the repo public? That affects free Actions minutes and is relevant to [[git-history-image-purge]].
