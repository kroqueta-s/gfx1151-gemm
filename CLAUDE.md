# CLAUDE.md — working on gfx1151-gemm

**GEMM measurements and, eventually, kernels for AMD Strix Halo (gfx1151) on
Windows ROCm.** This repository holds what is useful *without* knowing any
particular model runner: benchmark harnesses, environment notes, and measured
facts about this GPU. Anything that only makes sense inside one runner
(per-stage profiles, upstream pins) lives in that runner's repository.

Read [`README.md`](README.md) next; it carries the measurements.

**If `docs/local/` exists, read `docs/local/00_operator_notes.md` too.** It is
deliberately not tracked.

## Language: **what ships is English, what explains is not**

This repository is public. Everything tracked by git — code, comments, every
runtime string, README, commit messages — is **English**. `docs/local/` is the
author's language and is never tracked. Text shown to a person is not an
exception; assume whoever runs this reads English.

## Rules

1. **Report what you measured, never an estimate.** Every number in the README
   says when and under what environment it was taken.
2. **A comparison is only valid within one environment.** Numbers taken under
   different ROCm/torch versions are never put in the same table without
   saying so.
3. **Record the GPU clock with every measurement.** It idles near 700 MHz and
   ramps above 2.3 GHz at the driver's discretion; `gpuclock.py` reads it
   straight from the driver, and a reference GEMM alongside is the cheap
   cross-check.
4. **No model names in code.** Shapes are data; a harness takes them as input.
5. **Third-party code stays in `third_party/`,** with its LICENSE unmodified
   and the pinned upstream commit stated in the README.

## Style

- Python: `ruff` and `black`, line length 100, type hints everywhere.
- Comments say why, not what.
- Tests, if any, are hand-written scripts under `tests/`. No `pytest`.
