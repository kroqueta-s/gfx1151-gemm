# Where the ~30 TFLOPS ceiling comes from

Three GEMM backends (rocBLAS 7.2.1, hipBLASLt 7.2.1, the ROCm 10.0 default
path) all converge on 25–31 TFLOPS fp16 for well-proportioned shapes and none
exceeds ~31. This page pins down what that ceiling is. Method as everywhere
in this repository: 5 processes × 5 blocks × 100 calls, medians, clock
recorded by [`gpuclock.py`](../gpuclock.py). Environment: torch
2.13.0+rocm10.0.0, Windows 11, measured 2026-09-03.

## It is not bandwidth

- Device bandwidth ([`bench_membw.py`](../bench_membw.py), 1 GiB fp32
  buffers): **copy 210.6 GB/s, triad 226.2 GB/s** (5-process medians, spread
  under 1 %).
- A 4096³ fp16 GEMM has an arithmetic intensity of ~1 365 FLOP/byte at
  perfect reuse; at 226 GB/s the bandwidth roofline sits above 300 TFLOPS —
  an order of magnitude away from the observed 30.
- The K sweep confirms it empirically (M=N=4096 fixed, fp16, nn):

  | K | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
  |---|--:|--:|--:|--:|--:|--:|
  | TFLOPS | 23.3 | 25.4 | 25.2 | 30.2 | 30.4 | 28.1 |

  Throughput saturates once K reaches 4096 instead of climbing with
  arithmetic intensity — the signature of a compute-side limit, not a
  bandwidth one. (Even K=512 sits far below its bandwidth roofline.)

## The compute peak moves with the clock, and the clock drops under GEMM

The marketing figure for this silicon — 59.4 TFLOPS fp16 — assumes 2.9 GHz
(40 CU × 512 fp16 FLOP/CU/cycle = 20.48 TFLOPS per GHz). The GPU does reach
2.9 GHz: `gpuclock.py` recorded 2 900 MHz peaks during TRELLIS generation
(mixed attention/GEMM/elementwise work). **But under sustained dense GEMM the
driver holds it much lower: median 1 914 MHz, max 2 126 MHz, at 18–25 W GPU
power, across the whole K sweep.**

At the measured sustained clock the attainable peak is:

```
20.48 TFLOPS/GHz × 1.91 GHz ≈ 39 TFLOPS
```

so the libraries' 30.2 TFLOPS is **~77 % of the clock-adjusted peak** — an
unremarkable, healthy number for a vendor BLAS, not a mystery.

Cross-checks:

- **bf16 30.7, fp16 30.0** at 4096³ — identical, as WMMA predicts. (fp32:
  3.5, on a path that clearly does not use the matrix units; not comparable.)
- The externally reported 41.3 TFLOPS from a hand-tuned WMMA kernel on Linux
  (ROCm 7.14, same silicon) is consistent with this model: it needs either
  ~2.4 GHz sustained at ~85 % efficiency or ~2.0 GHz at 100 % — i.e. **the
  Linux result is largely a clock/power-management result, not purely a
  kernel-quality result.**

## What this means for a hand-written kernel on Windows

- The verdict of the three-way split: **compute-bound, with the effective
  ceiling set by the sustained clock (~39 TFLOPS at today's 1.9 GHz), and
  the libraries already at ~77 % of it.**
- A hand-written WMMA kernel that reaches 85–90 % of the same clock-adjusted
  peak would land at **33–35 TFLOPS** — a real but modest margin over 30,
  and right at the >33 bar that separates "worth shipping" from "not".
- Anything near 40 on Windows would have to come from the *clock*, not the
  kernel: sustaining 2.3–2.4 GHz under WMMA load is worth +20 % by itself.
  Whether the driver's DVFS can be coaxed there (power limits, workload
  shaping) is a separate question from GEMM code quality — and nothing in
  the GEMM instruction stream controls it.
