# Git history image purge (needs decision)

## Intent

Decide whether the card images committed in the past should be removed from git history. If yes, do it safely, without touching the working-tree `data/` folder.

**Status: needs a decision from the repo owner. Do not execute without explicit approval.** It rewrites history and needs a force-push.

## Why

- Commit `608c429 Model` added 1,807 JPEGs under `data/crops/images/` (~271 MiB of blobs). `9c23530 removed images` deleted them from the tree, but they're still in history on `origin/main`, `CadeJordan/setup` and `CadeJordan/ScoreImprovements`.
- The images are PSA cert scans and eBay listing photos. If the repo is public, they're redistributed to anyone who clones it. Every clone also pays the ~271 MiB.

## Options

1. **Do nothing.** Accept the size and exposure. Fine if the repo is private and size doesn't matter.
2. **Rewrite history** with `git filter-repo --path data/crops/images --invert-paths` on a fresh mirror clone, then force-push all branches. Collaborators must re-clone, and open PRs break.
3. **Start a new repo** from the current tree, archiving the old one. It's the simplest rewrite but loses history.

## Scope (if option 2 is chosen)

- Work on a fresh `git clone --mirror` in a scratch directory, never in `PokeScan/`. The working-tree `data/` must be untouched.
- Verify: the blob count of `*.jpg` in history is 0; every branch's tip tree equals its pre-rewrite tree (`git diff <old> <new>` is empty for each branch).
- Force-push only after the owner confirms. Afterwards, reset the local clone to the rewritten branches. Uncommitted `CLAUDE.md`, `docs/` and `intent/` work must be committed or stashed first.

## Acceptance criteria

- [ ] The owner has chosen an option, recorded in `docs/adr/`.
- [ ] If rewriting: the jpg blobs in `git rev-list --objects --all` drop to 0, branch tip trees are unchanged, and the local `data/` file count is unchanged (5,275 images, 5,260 crops on 2026-10-17).

## Facts vs assumptions

- **Fact:** blob count and size, and that the images are reachable from all three remote branches (checked 2026-10-17).
- **Unknown:** repo visibility. `gh` isn't authenticated, so it couldn't be checked.
- **Unknown:** whether GitHub has cached or forked copies. A rewrite doesn't purge those; that would need GitHub support.
