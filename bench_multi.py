# SPDX-License-Identifier: MIT
"""Run a benchmark in N fresh processes and merge the results.

A single process gives one library initialisation, one allocator state, one
algorithm-selection history — publishable numbers want the spread across
processes too. This wrapper runs the given command N times, expects the last
stdout line of each run to be the benchmark's JSON object (which is what
`bench_gemm.py` and `bench_membw.py` print), and merges the per-block figures
of matching entries across processes into one median and range.

    python bench_multi.py --processes 5 --json merged.json -- ^
        .venv\\Scripts\\python.exe bench_gemm.py --shape 4096,4096,4096
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from typing import Any


def _entry_key(entry: dict[str, Any]) -> str:
    """Identity of one result row, shared across processes."""
    if "kernel" in entry:
        return f"{entry['kernel']}/{entry.get('buffer_mb')}"
    if "sq" in entry:
        return (
            f"sdpa B{entry.get('b')}H{entry.get('h')}"
            f"SQ{entry.get('sq')}SKV{entry.get('skv')}D{entry.get('d')}"
        )
    return (
        f"M{entry.get('m')}xN{entry.get('n')}xK{entry.get('k')}"
        f"B{entry.get('batch')} {entry.get('dtype')} {entry.get('layout', '')}"
    )


def _blocks(entry: dict[str, Any]) -> list[float]:
    return entry.get("blocks_tflops") or entry.get("blocks_gbps") or []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the benchmark command after --")

    runs: list[dict[str, Any]] = []
    for i in range(args.processes):
        proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            print(f"process {i + 1}/{args.processes} failed:\n{proc.stderr[-2000:]}")
            return 1
        last = [line for line in proc.stdout.splitlines() if line.startswith("{")][-1]
        runs.append(json.loads(last))
        print(f"process {i + 1}/{args.processes} done", flush=True)

    merged: dict[str, dict[str, Any]] = {}
    for run in runs:
        for entry in run.get("results", []):
            key = _entry_key(entry)
            slot = merged.setdefault(
                key, {"entry": entry, "per_process_medians": [], "all_blocks": []}
            )
            median = entry.get("median_tflops", entry.get("median_gbps"))
            slot["per_process_medians"].append(median)
            slot["all_blocks"].extend(_blocks(entry))

    summary = []
    for key, slot in merged.items():
        blocks = slot["all_blocks"]
        row = {
            "key": key,
            "median": statistics.median(blocks),
            "min": min(blocks),
            "max": max(blocks),
            "per_process_medians": slot["per_process_medians"],
        }
        summary.append(row)
        print(
            f"{key}: median {row['median']} (range {row['min']}-{row['max']}, "
            f"process medians {row['per_process_medians']})",
            flush=True,
        )

    out = {
        "processes": args.processes,
        "command": command,
        "environment": {
            k: runs[0].get(k) for k in ("date", "gpu", "torch", "hip", "preferred_blas", "env")
        },
        "summary": summary,
        "runs": runs,
    }
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
