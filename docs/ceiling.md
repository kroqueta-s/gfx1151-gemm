# Where the ~30 TFLOPS ceiling comes from

Three GEMM backends (rocBLAS 7.2.1, hipBLASLt 7.2.1, the ROCm 10.0 default
path) all converge on 25–31 TFLOPS fp16 for well-proportioned shapes and none
exceeds ~31. This page pins down what that ceiling is. Method as everywhere
in this repository: 5 processes × 5 blocks × 100 calls, medians, clock
recorded by [`gpuclock.py`](../gpuclock.py). Environment: ASUS ProArt PX13
(HN7306EA-AI9641W) — Ryzen AI MAX+ 395, Radeon 8060S (gfx1151, 40 CU) —
**a 13-inch laptop at its factory power limits**, torch 2.13.0+rocm10.0.0,
Windows 11, measured 2026-09-02.

One scope note: everything on this page is the **display-on** world. With
the console display off, the driver pins the GPU near 600 MHz regardless of
workload — a separate phenomenon with its own page,
[displayoff.md](displayoff.md).

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

**What holds the clock there is the chassis's sustained power limit.** An
earlier revision of this page claimed otherwise from a single sensor; a
two-source verification (2026-09-03) settled it. The ADL PM-log value first
quoted (18–25 W) is `ADL_PMLOG_GFX_POWER` — **the GFX rail alone**. Reading
the package with two independent series at once (ADL `ASIC_POWER` and
HWiNFO64's CPU Package Power / APU STAPM / per-limit usage counters, 1 Hz,
phases of idle / 5-min GEMM / flash attention / CPU-only):

| Phase | GFX clock | GFX rail | Package (ADL / HWiNFO) | PPT-slow usage |
|---|--:|--:|--:|--:|
| idle | ~950 MHz | 3 W | 16 / 15.7 W | 25 % |
| **GEMM 4096³ (5 min)** | 1 940 MHz | 19 W | **69 / 70.0 W** | **100 %** |
| **flash attention** | 2 087 MHz | 34 W | **70 / 70.0 W** | **100 %** |
| CPU-only load | — | 3 W | 47 / 47 W | 69 % |

Both GEMM and attention run with **PPT-slow pinned at exactly 100 %** and
the package flat at 70 W; TDC (62 %), EDC (45 %) and thermal (62 %) all
have margin, so it is the sustained power limit — not current, not
temperature, and not a free-floating driver policy. The fast limit reads
82.3 % used at 70 W, putting PPT-fast at ~85 W — matching ASUS's published
"up to 85 W" ([product page][asus-px13]) as the *boost* budget, with ~70 W
as this chassis's sustained one (AMD's chip-level range is
[45–120 W cTDP][amd-395]). Consequences:

- **A bigger-budget Strix Halo machine genuinely could sit higher.** This
  is the wall being hit, and it is configured per chassis. The externally
  reported Linux 41.3 TFLOPS becomes easier to read as platform power
  budget × OS, rather than kernel quality.
- At a power wall, the clock differences between workloads (GEMM 1.94 GHz,
  attention 2.09 GHz, mixed bursts to 2.9) are power physics: denser work
  costs more per cycle, so it clocks lower inside the same 70 W.
- For kernels the currency changes: **at a fixed 70 W, throughput gains are
  energy-efficiency gains** (FLOPs per joule). Raising utilization without
  raising efficiency lowers the clock and hands the gain back.

The *method* (log package power and the limit-usage counters alongside the
clock) transfers to any Strix Halo machine; the 70 W, the 1.9 GHz and the
39 TFLOPS derived from them are this chassis's numbers.

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
  Linux result is largely a power-budget result, not purely a
  kernel-quality result** (the platform behind it, with its own sustained
  limit, is not this 13-inch laptop; the comparison spans chassis *and* OS).

## What this means for a hand-written kernel on Windows

- The verdict of the three-way split: **compute-bound, with the effective
  ceiling set by the sustained clock the 70 W power budget allows
  (~39 TFLOPS at 1.9 GHz), and the libraries already at ~77 % of it.**
- A hand-written WMMA kernel that reaches 85–90 % of the same clock-adjusted
  peak would land at **33–35 TFLOPS** — a real but modest margin over 30,
  and right at the >33 bar that separates "worth shipping" from "not". And
  that projection is optimistic: DVFS clocks electrically denser work lower
  (see above), so raising arithmetic efficiency tends to lower the operating
  clock and hand back part of the gain.
- Anything near 40 on Windows would have to come from the *power budget*,
  not the kernel: the sustained clock is set by the 70 W PPT-slow limit,
  so the honest routes up are a chassis with a bigger budget, a vendor
  performance mode that raises the limit (if one exists here), or a kernel
  that genuinely does more FLOPs per joule — not one that merely keeps the
  units busier.
- **Decision (2026-09-02): the hand-written kernel was not attempted.** The
  projected best case grazes the bar that would justify shipping it, the
  DVFS behaviour above pushes the expectation below that bar, and the two
  things that would actually move the ceiling — driver DVFS policy and
  attention efficiency ([attention.md](attention.md)) — are not GEMM
  kernels. A negative result, published as one.
- Cross-check on the policy: switching the **Windows power plan** (custom
  Performance / Balanced / even Power saver) does not move GEMM throughput
  at all (31.2 / 31.5 / 32.3 TFLOPS, same clock band) — the operating point
  is chosen by the AMD driver/SMU, not by the OS power plan.
