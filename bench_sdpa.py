# SPDX-License-Identifier: MIT
"""Scaled-dot-product-attention microbenchmark, same discipline as bench_gemm.

The 3D pipelines profiled in docs/profiles.md spend more device time in
attention than in GEMM, so the attention ceiling matters as much as the GEMM
one. This measures `F.scaled_dot_product_attention` at given shapes, timed
with device events, clock recorded alongside.

    python bench_sdpa.py --shape 1,16,4096,4096,64
    python bench_sdpa.py --shape 1,16,4096,1374,64 --env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

Shape is B,H,SQ,SKV,D (batch, heads, query length, key/value length, head
dim). FLOPs are accounted as 4*B*H*SQ*SKV*D (the two matmuls of one forward
pass; softmax not counted).
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


def parse_shape(text: str) -> tuple[int, int, int, int, int]:
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 5:
        raise ValueError(f"--shape wants B,H,SQ,SKV,D, got: {text}")
    return tuple(parts)  # type: ignore[return-value]


def bench_one(
    b: int, h: int, sq: int, skv: int, d: int, warmup: int, iters: int, blocks: int
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    q = torch.randn(b, h, sq, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, skv, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, skv, d, device="cuda", dtype=torch.float16)
    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    flop_per_call = 4.0 * b * h * sq * skv * d
    tflops: list[float] = []
    ms: list[float] = []
    for _ in range(blocks):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1000.0
        tflops.append(round(flop_per_call * iters / sec / 1e12, 2))
        ms.append(round(1000.0 * sec / iters, 3))
    del q, k, v
    torch.cuda.empty_cache()
    ordered = sorted(tflops)
    return {
        "b": b,
        "h": h,
        "sq": sq,
        "skv": skv,
        "d": d,
        "blocks_tflops": tflops,
        "median_tflops": ordered[len(ordered) // 2],
        "min_tflops": ordered[0],
        "max_tflops": ordered[-1],
        "median_ms_per_call": sorted(ms)[len(ms) // 2],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", required=True, metavar="B,H,SQ,SKV,D")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    shapes = [parse_shape(s) for s in args.shape]
    for pair in args.env:
        key, _, value = pair.partition("=")
        os.environ[key] = value

    import torch

    backends = {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }

    results = []
    with ClockWatch() as watch:
        for b, h, sq, skv, d in shapes:
            r = bench_one(b, h, sq, skv, d, args.warmup, args.iters, args.blocks)
            results.append(r)
            print(
                f"B={b} H={h} SQ={sq} SKV={skv} D={d}: "
                f"median {r['median_tflops']} TFLOPS, {r['median_ms_per_call']} ms/call "
                f"(range {r['min_tflops']}-{r['max_tflops']})",
                flush=True,
            )

    run = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "sdp_backends_enabled": backends,
        "clock": watch.summary(),
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
