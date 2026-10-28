# Pinned dev environment

## Intent

Anyone (human or agent) can rebuild the exact Python environment the test suite currently passes in, inside `PokeScan/.venv`, from files committed to the repo.

## Why

- `.venv` exists but only contains `pip`, `setuptools` and `python-dotenv`. `CLAUDE.md` and `README.md` say to activate it, but nothing would run.
- Everything actually runs from the global pyenv interpreter (`~/.pyenv/pyenv-win/versions/3.11.6`), where `pokescan` is installed editable.
- `pyproject.toml` only has `>=` floors and there is no lockfile, so a fresh install can resolve different torch / timm / albumentations / onnxruntime versions than the known-good set.

Known-good set (global env, 2026-10-17, 248 passed / 3 xfailed on the fast suite):
`torch==2.5.1+cu121`, `timm==1.0.26`, `pytorch-lightning==2.6.1`, `onnxruntime==1.25.0`, `albumentations==2.0.8`, `opencv-python==4.11.0.86`, `numpy==1.26.4`, `fastapi==0.136.1`.

## Scope

- Generate a constraints file (e.g. `constraints.txt`) from the global env, limited to packages reachable from `pokescan[dev]`, so unrelated globals don't leak in.
- Rebuild `.venv` with `pip install -e ".[dev]" -c constraints.txt`. torch needs the cu121 index (`--extra-index-url https://download.pytorch.org/whl/cu121`); record that in the install command.
- Document the install and the regenerate-constraints command in `README.md` / `CLAUDE.md`.

## Out of scope

- Changing or uninstalling anything in the global pyenv environment.
- Upgrading any dependency. Pin what works first; upgrades are separate changes.
- Switching tooling (uv, poetry). Can be revisited later; plain pip + constraints is the smallest step.
- Frontend `node_modules` (already has `package-lock.json`).

## Acceptance criteria

- [ ] `constraints.txt` committed; every package in `pokescan[dev]`'s dependency closure is pinned with `==`.
- [ ] From a deleted-and-recreated `.venv`, the documented command installs without errors and `python -m pip check` is clean.
- [ ] Inside `.venv`: `python -m pytest -m "not slow and not visual and not model"` → 248 passed, 3 xfailed (or better).
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` is `True` in `.venv` on this machine (GPU training still works).
- [ ] `CLAUDE.md` names `.venv` as the interpreter and gives the install command.

## Facts vs assumptions

- **Fact:** venv contents, global versions and the passing test run, all checked 2026-10-17.
- **Assumption:** the cu121 wheel for torch 2.5.1 is still downloadable from the PyTorch index.
- **Fact (2026-10-18):** `python -m pip check` on the global env reports one conflict:
  `opencv-contrib-python 4.13.0.92 has requirement numpy>=2; python_version >= "3.9", but you have numpy 1.26.4.`
  `opencv-contrib-python` is pulled in by `mediapipe` (a global package, not in `pokescan[dev]`); `pokescan` pins `numpy<2.0` and `import cv2` currently resolves to 4.11.0 (`opencv-python`). So the conflict is outside our dependency closure. Exclude `opencv-contrib-python` / `mediapipe` when generating constraints, and expect `pip check` to be clean in the rebuilt `.venv`.

## Open questions

- Should CI use the same constraints with CPU torch, or a separate CPU constraints file? (See [[ci-fast-suite]].)
