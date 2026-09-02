# SPDX-License-Identifier: MIT
"""Run a generation pipeline N times and report stage-time medians with clock evidence.

Model-agnostic: it runs whatever command you give it (each repetition a fresh
process), pulls the `metrics` object out of the JSON the command prints, and
wraps every repetition in evidence — a reference GEMM before and after (run
with `--python`, normally the pipeline's own venv), and a `gpuclock.py` trace
during. Single wall-clock figures are not publishable on a machine whose
clock the driver controls; five medians with the clock attached are.

    python bench_pipeline.py --runs 5 --json out.json ^
        --python .venv\\Scripts\\python.exe ^
        -- .venv\\Scripts\\python.exe tools\\run_single.py --image assets\\sample.png --out C:\\out

The command's stdout must contain a JSON object with a `"metrics"` key (the
*-strix-halo runners' `run_single.py` prints one).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpuclock import ClockWatch  # noqa: E402

BENCH_GEMM = Path(__file__).resolve().parent / "bench_gemm.py"


def reference_gemm(python: str, shape: str) -> dict[str, Any] | None:
    """One quick reference GEMM in a fresh process; returns the merged JSON entry."""
    proc = subprocess.run(
        [
            python,
            str(BENCH_GEMM),
            "--shape",
            shape,
            "--warmup",
            "5",
            "--iters",
            "50",
            "--blocks",
            "3",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return None
    last = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not last:
        return None
    run = json.loads(last[-1])
    entry = run["results"][0]
    return {"median_tflops": entry["median_tflops"], "blocks": entry["blocks_tflops"]}


def extract_metrics(stdout: str) -> dict[str, Any] | None:
    """Find the last JSON object in the output that carries a `metrics` key."""
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    for match in re.finditer(r"\{", stdout):
        try:
            obj, _ = decoder.raw_decode(stdout, match.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and "metrics" in obj:
            found = obj["metrics"]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--python", required=True, help="python used for the reference GEMM")
    parser.add_argument("--ref-shape", default="4096,4096,4096")
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the pipeline command after --")

    repetitions: list[dict[str, Any]] = []
    for i in range(args.runs):
        ref_before = reference_gemm(args.python, args.ref_shape)
        started = time.perf_counter()
        with ClockWatch() as watch:
            proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        wall = time.perf_counter() - started
        ref_after = reference_gemm(args.python, args.ref_shape)
        if proc.returncode != 0:
            print(f"run {i + 1}/{args.runs} failed:\n{proc.stderr[-2000:]}")
            return 1
        metrics = extract_metrics(proc.stdout)
        if metrics is None:
            print(f"run {i + 1}/{args.runs}: no metrics object in the output")
            return 1
        repetitions.append(
            {
                "metrics": metrics,
                "wall_sec": round(wall, 1),
                "ref_gemm_before": ref_before,
                "ref_gemm_after": ref_after,
                "clock": watch.summary(),
            }
        )
        print(
            f"run {i + 1}/{args.runs}: gen {metrics.get('gen_sec')}s, "
            f"clock {repetitions[-1]['clock'].get('gfxclk_mhz')}, "
            f"ref {ref_before and ref_before['median_tflops']}"
            f"->{ref_after and ref_after['median_tflops']} TFLOPS",
            flush=True,
        )

    numeric: dict[str, list[float]] = {}
    for rep in repetitions:
        for key, value in rep["metrics"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric.setdefault(key, []).append(float(value))
    stages = {
        key: {
            "median": round(statistics.median(values), 2),
            "min": min(values),
            "max": max(values),
        }
        for key, values in numeric.items()
        if len(values) == len(repetitions)
    }

    out = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": args.runs,
        "command": command,
        "stages": stages,
        "repetitions": repetitions,
    }
    print(json.dumps({"stages": stages}, indent=2))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
