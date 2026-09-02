# SPDX-License-Identifier: MIT
"""Device memory bandwidth microbenchmark (copy and triad), timed with device events.

Companion to `bench_gemm.py`, same discipline: warmup, then timed blocks,
median and range reported, and the GPU clock recorded alongside
(`gpuclock.py`). Exists to anchor a roofline: a GEMM ceiling only means
something next to the bandwidth that feeds it.

    python bench_membw.py
    python bench_membw.py --mb 2048 --iters 20 --blocks 5

Bytes accounted: copy moves 2N bytes per call (read + write), triad
(c = a + 2*b) moves 3N.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpuclock import ClockWatch  # noqa: E402


def bench_kernel(name: str, mb: int, warmup: int, iters: int, blocks: int) -> dict[str, Any]:
    """Time one bandwidth kernel and return GB/s per block with median and range."""
    import torch

    n = mb * 1024 * 1024 // 4  # fp32 elements
    bufs = {
        "a": torch.randn(n, device="cuda"),
        "b": torch.randn(n, device="cuda"),
        "c": torch.empty(n, device="cuda"),
    }
    if name == "copy":
        bytes_per_call = 2 * n * 4

        def op() -> None:
            bufs["c"].copy_(bufs["a"])
    else:  # triad
        bytes_per_call = 3 * n * 4

        def op() -> None:
            torch.add(bufs["a"], bufs["b"], alpha=2.0, out=bufs["c"])

    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    gbps: list[float] = []
    for _ in range(blocks):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            op()
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1000.0
        gbps.append(round(bytes_per_call * iters / sec / 1e9, 1))
    bufs.clear()
    torch.cuda.empty_cache()
    ordered = sorted(gbps)
    return {
        "kernel": name,
        "buffer_mb": mb,
        "blocks_gbps": gbps,
        "median_gbps": ordered[len(ordered) // 2],
        "min_gbps": ordered[0],
        "max_gbps": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mb", type=int, default=1024, help="buffer size per operand, MiB")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20, help="calls per timed block")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    import torch

    results = []
    with ClockWatch() as watch:
        for kernel in ("copy", "triad"):
            r = bench_kernel(kernel, args.mb, args.warmup, args.iters, args.blocks)
            results.append(r)
            print(
                f"{kernel} ({args.mb} MiB): median {r['median_gbps']} GB/s "
                f"(range {r['min_gbps']}-{r['max_gbps']}, blocks {r['blocks_gbps']})",
                flush=True,
            )

    run = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "env": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(("TORCH_", "ROCBLAS_", "HIPBLASLT_", "MIOPEN_", "PYTORCH_", "HSA_"))
        },
        "clock": watch.summary(),
        "warmup": args.warmup,
        "iters": args.iters,
        "blocks": args.blocks,
        "results": results,
    }
    print(json.dumps(run))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
