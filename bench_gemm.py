# SPDX-License-Identifier: MIT
"""GEMM microbenchmark for one set of shapes, timed with device events.

Made for AMD Strix Halo (gfx1151) on Windows ROCm, but nothing here is
specific to it. Shapes are data: pass the ones your workload actually runs
(a profiler tells you), not the ones that look impressive.

    python bench_gemm.py --shape 4096,4096,4096 --shape 25600,1024,1024,fp16
    python bench_gemm.py --shape 4096,4096,4096 --env TORCH_BLAS_PREFER_HIPBLASLT=1

Method, chosen to survive this hardware's quirks:

- **Warmup, then blocks.** Each shape runs `--warmup` calls first (the first
  call may compile or pick algorithms), then `--blocks` blocks of `--iters`
  calls, each block timed with device events. The result is the **median block
  and the min-max range** — a single peak says nothing on a machine whose
  clock moves.
- **The GPU clock is part of the result.** It idles near 700 MHz and ramps
  above 2.3 GHz under load, at the driver's discretion. Record it alongside
  every run (`gpuclock.py` reads it straight from the driver) and treat a run
  whose blocks disagree wildly as a clock artifact.
- `--env KEY=VALUE` is applied **before torch is imported**, because BLAS
  backend selection reads the environment at import time.

Output is one JSON object per shape on stdout (after a human-readable line),
plus `--json PATH` to write the whole run to a file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def parse_shape(text: str) -> tuple[int, int, int, int, str]:
    """Parse M,N,K[,B][,dtype] into (m, n, k, batch, dtype)."""
    parts = [p.strip() for p in text.split(",")]
    dtype = "fp16"
    if parts and parts[-1] in DTYPES:
        dtype = parts.pop()
    if len(parts) == 3:
        m, n, k = (int(p) for p in parts)
        return m, n, k, 1, dtype
    if len(parts) == 4:
        m, n, k, b = (int(p) for p in parts)
        return m, n, k, b, dtype
    raise ValueError(f"--shape wants M,N,K[,B][,dtype], got: {text}")


def bench_one(
    m: int,
    n: int,
    k: int,
    batch: int,
    dtype_name: str,
    warmup: int,
    iters: int,
    blocks: int,
    layout: str = "nn",
) -> dict[str, Any]:
    """Run one shape and return per-block TFLOPS with the median and range.

    `layout="nt"` hands B over as a transposed view, the way `F.linear` passes
    a weight matrix. Backends pick different kernels for it, so benchmark the
    layout your workload actually uses.
    """
    import torch

    dtype = getattr(torch, DTYPES[dtype_name])
    if batch > 1:
        a = torch.randn(batch, m, k, device="cuda", dtype=dtype)
        if layout == "nt":
            b = torch.randn(batch, n, k, device="cuda", dtype=dtype).transpose(1, 2)
        else:
            b = torch.randn(batch, k, n, device="cuda", dtype=dtype)
        op = torch.bmm
    else:
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        if layout == "nt":
            b = torch.randn(n, k, device="cuda", dtype=dtype).t()
        else:
            b = torch.randn(k, n, device="cuda", dtype=dtype)
        op = torch.mm
    for _ in range(warmup):
        op(a, b)
    torch.cuda.synchronize()

    flop_per_call = 2.0 * batch * m * n * k
    tflops: list[float] = []
    for _ in range(blocks):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            op(a, b)
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1000.0
        tflops.append(round(flop_per_call * iters / sec / 1e12, 2))
    del a, b
    torch.cuda.empty_cache()
    ordered = sorted(tflops)
    return {
        "m": m,
        "n": n,
        "k": k,
        "batch": batch,
        "dtype": dtype_name,
        "layout": layout,
        "blocks_tflops": tflops,
        "median_tflops": ordered[len(ordered) // 2],
        "min_tflops": ordered[0],
        "max_tflops": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        required=True,
        metavar="M,N,K[,B][,dtype]",
        help="repeatable; dtype is one of fp16/bf16/fp32, default fp16",
    )
    parser.add_argument(
        "--layout",
        choices=("nn", "nt", "both"),
        default="nn",
        help="how B is laid out: contiguous (nn), transposed view as F.linear passes "
        "weights (nt), or both",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100, help="calls per timed block")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set before torch is imported (BLAS backends read these at import)",
    )
    args = parser.parse_args()

    shapes = [parse_shape(s) for s in args.shape]
    for pair in args.env:
        key, _, value = pair.partition("=")
        os.environ[key] = value

    import torch

    preferred = ""
    try:
        preferred = str(torch.backends.cuda.preferred_blas_library())
    except (AttributeError, RuntimeError):
        pass

    layouts = ("nn", "nt") if args.layout == "both" else (args.layout,)
    results = []
    for m, n, k, batch, dtype_name in shapes:
        for layout in layouts:
            r = bench_one(m, n, k, batch, dtype_name, args.warmup, args.iters, args.blocks, layout)
            results.append(r)
            print(
                f"M={m} N={n} K={k} B={batch} {dtype_name} {layout}: "
                f"median {r['median_tflops']} TFLOPS "
                f"(range {r['min_tflops']}-{r['max_tflops']}, blocks {r['blocks_tflops']})",
                flush=True,
            )

    run = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "preferred_blas": preferred,
        "env": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(("TORCH_", "ROCBLAS_", "HIPBLASLT_", "MIOPEN_", "PYTORCH_", "HSA_"))
        },
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
