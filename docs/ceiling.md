# Where the ~30 TFLOPS ceiling comes from

Three GEMM backends (rocBLAS 7.2.1, hipBLASLt 7.2.1, the ROCm 10.0 default
path) all converge on 25–31 TFLOPS fp16 for well-proportioned shapes and none
exceeds ~31. This page pins down what that ceiling is. Method as everywhere
in this repository: 5 processes × 5 blocks × 100 calls, medians, clock
recorded by [`gpuclock.py`](../gpuclock.py). Environment: ASUS ProArt PX13
(HN7306EA-AI9641W) — Ryzen AI MAX+ 395, Radeon 8060S (gfx1151, 40 CU) —
**a 13-inch laptop at its factory power limits**, torch 2.13.0+rocm10.0.0,
Windows 11, measured 2026-09-03.

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

What that 18–25 W is *not* is the chassis limit. The primary sources: ASUS
configures this laptop's SoC at **up to 85 W** for the Ryzen AI MAX+ 395
("Up to 85 W TDP" on the [product page][asus-px13], CPU+GPU combined;
[tech specs][asus-spec] list a 200 W adapter), against AMD's
[default 55 W / cTDP 45–120 W][amd-395] for the chip — all stock here,
nothing unlocked. During the sustained-GEMM measurements the CPU was close
to idle, so even a generous allowance for it leaves the SoC well inside its
85 W envelope while the GPU's PM-sensor power reads 18–25 W. **The clock is
therefore being set by a per-domain DVFS policy (SMU/driver) below the
chassis budget, not by the chassis budget itself.** Two consequences:

- A bigger-chassis Strix Halo machine would not automatically lift this
  ceiling: it was not the wall being hit. Whether the policy behaves
  differently at other cTDP configurations is untested here.
- The clock visibly depends on the *workload's character* — dense WMMA GEMM
  1.9 GHz, flash attention 2.1 GHz, mixed pipeline work bursting to 2.9 —
  which is DVFS responding to electrical density, the way power-virus loads
  always clock lowest. A more efficient kernel raises FLOPs per cycle and
  can push the operating clock further *down*, partially spending its own
  gains.

The *method* (measure the sustained clock, multiply by 20.48 TFLOPS/GHz)
transfers to any Strix Halo machine; the 39 is this platform's number.

[asus-px13]: https://www.asus.com/us/laptops/for-creators/proart/proart-px13-hn7306/
[asus-spec]: https://www.asus.com/us/laptops/for-creators/proart/proart-px13-hn7306/techspec/
[amd-395]: https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html

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
  Linux result is largely a clock/power result, not purely a kernel-quality
  result** (and the platform behind it, with its own power budget, is not
  this 13-inch laptop; the comparison spans OS *and* chassis).

## What this means for a hand-written kernel on Windows

- The verdict of the three-way split: **compute-bound, with the effective
  ceiling set by the sustained clock (~39 TFLOPS at today's 1.9 GHz), and
  the libraries already at ~77 % of it.**
- A hand-written WMMA kernel that reaches 85–90 % of the same clock-adjusted
  peak would land at **33–35 TFLOPS** — a real but modest margin over 30,
  and right at the >33 bar that separates "worth shipping" from "not". And
  that projection is optimistic: DVFS clocks electrically denser work lower
  (see above), so raising arithmetic efficiency tends to lower the operating
  clock and hand back part of the gain.
- Anything near 40 on Windows would have to come from the *clock*, not the
  kernel — and since the clock is set by SMU/driver policy well inside the
  chassis power budget, neither a bigger machine nor a better GEMM
  instruction stream reaches it. What might is the platform firmware/driver
  itself (the plausible source of the Linux 41.3 gap), which is outside
  anything a kernel controls.
