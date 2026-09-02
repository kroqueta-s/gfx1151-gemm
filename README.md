# gfx1151-gemm

**GEMM measurements, benchmark harness and environment notes for AMD Strix Halo
(gfx1151, Radeon 8060S) on Windows ROCm.**

This repository holds what is useful without knowing any particular workload:
how fast this GPU actually multiplies matrices, which switches change that, and
the traps this platform sets for anyone doing compute on it. It grew out of
three 3D-generation runners ([hi3dgen-strix-halo],
[trellis-strix-halo], [hunyuan3d-strix-halo]) but nothing here depends on them —
the same numbers and switches apply to any torch workload on this hardware,
local LLMs included.

[hi3dgen-strix-halo]: https://github.com/kroqueta-s/hi3dgen-strix-halo
[trellis-strix-halo]: https://github.com/kroqueta-s/trellis-strix-halo
[hunyuan3d-strix-halo]: https://github.com/kroqueta-s/hunyuan3d-strix-halo

## Contents

- [`bench_gemm.py`](bench_gemm.py) — GEMM microbenchmark (device-event timing,
  warmup + timed blocks, median and range, clock trace attached). Shapes are
  arguments; nothing is hard-coded.
- [`bench_sdpa.py`](bench_sdpa.py) / [`bench_membw.py`](bench_membw.py) —
  the same discipline for flash attention and for memory bandwidth.
- [`bench_multi.py`](bench_multi.py) — runs any of the above in N fresh
  processes and merges the blocks into one median and range; published
  numbers here are 5-process medians.
- [`bench_pipeline.py`](bench_pipeline.py) — runs a whole generation
  pipeline N times, pulls its stage metrics, and brackets every run with a
  reference GEMM and a clock trace.
- [`gpuclock.py`](gpuclock.py) — reads the live GPU clock, activity and power
  straight from the AMD driver's own `atiadlxx.dll` (no third-party tool, no
  unsigned binary). Every measurement here carries its output as evidence.
- [`gfxlight.py`](gfxlight.py) — a hidden render loop, kept as an experiment
  switch for testing whether a machine's power management treats compute-only
  work differently. On this machine it currently changes nothing (see below).
- [`docs/profiles.md`](docs/profiles.md) — measured GEMM profiles of the three
  3D-generation pipelines named above, and how much their dominant shapes
  overlap.
- [`docs/hipblaslt.md`](docs/hipblaslt.md) — rocBLAS against hipBLASLt on the
  shapes those pipelines actually run: the Lt path works on gfx1151 under
  Windows, wins by 8× on skinny GEMMs, and loses on fat ones.
- [`docs/rocm10.md`](docs/rocm10.md) — the ROCm 10.0 wheels: four install
  traps (Smart App Control, kernel-pack device packages, torchvision,
  MIOpen batch norm), and the measurements after the upgrade (1.13–1.79×
  end to end).
- [`docs/ceiling.md`](docs/ceiling.md) — why every backend stops at
  ~30 TFLOPS: not bandwidth (K sweep + measured 226 GB/s), but the sustained
  clock under GEMM load (~1.9 GHz → ~39 TFLOPS attainable, libraries at
  ~77 % of it).
- [`docs/attention.md`](docs/attention.md) — the flash-attention side of the
  same question: 19–24 TFLOPS, flat in sequence length, ~55 % of the
  clock-adjusted peak.

## What is known about gfx1151 on Windows

Everything below was measured on one machine, in September 2026:
**ASUS ProArt PX13 (HN7306EA-AI9641W)** — AMD Ryzen AI MAX+ 395 (Strix
Halo), Radeon 8060S iGPU (gfx1151, 40 CU), 32 GB of dedicated VRAM,
Windows 11, **a 13-inch laptop running at its factory power limits**
(nothing unlocked, nothing overclocked). That last part matters: Strix Halo
ships in machines with very different power budgets, and the sustained
clocks — and therefore every ceiling derived from them — are properties of
this chassis, not of the silicon. Re-measure before trusting any of it in a
different environment.

### The clock moves, so record it with every number

Measured 2026-09-03 with [`gpuclock.py`](gpuclock.py) (which reads the
driver's own PM sensors): the GPU idles at 709–745 MHz and reaches
**2.39 GHz** under compute-only GEMM load — no rendering required. A render
loop alive alongside (`gfxlight.py`) changes neither the clock nor GEMM
throughput (A/B/A: 31.7 / 32.3 / 31.8 TFLOPS at 4096³). Because power
management is driver- and state-dependent, **every measurement in this
repository carries clock evidence**: a reference GEMM before and after, and
where it matters, a `gpuclock.py` trace.

### GEMM: the floor moved, the ceiling did not

Three backends were measured on the shapes real pipelines run (always with
the shape and layout attached — a TFLOPS figure without them means nothing
here, because the same silicon spans 1.8 to 30.8 depending on both):

- **The floor.** rocBLAS 7.2.1 ran one real pipeline shape
  (M=25600, N=128, K=2048, nn) at **1.8 TFLOPS**; hipBLASLt 7.2.1 lifted it
  to 14–15 ([`docs/hipblaslt.md`](docs/hipblaslt.md)) and the ROCm 10.0
  default path delivers 15.8 with no switches
  ([`docs/rocm10.md`](docs/rocm10.md)). The pathology was per-shape, never
  general.
- **The ceiling.** All three backends converge on 25–31 TFLOPS for
  well-proportioned shapes and none exceeds ~31. That is not a bandwidth
  limit and not far from what the hardware allows: under sustained GEMM the
  driver holds the clock near 1.9 GHz, which caps the attainable peak at
  ~39 TFLOPS, and the libraries sit at ~77 % of that
  ([`docs/ceiling.md`](docs/ceiling.md)).

The paper number for this silicon (59.4 TFLOPS fp16) assumes 2.9 GHz — a
clock this GPU only touches in bursts. Hand-tuned WMMA HIP kernels reach
41–46 TFLOPS at 4096³ on Linux (ROCm 7.14, FP32 accumulate, external
measurement), which the clock model in `docs/ceiling.md` attributes largely
to power management rather than kernel quality.

### hipBLASLt ships gfx1151 kernels on Windows 7.2.1

The `rocm_sdk_libraries` wheel that backs torch 2.9.1+rocm7.2.1 carries a
hipBLASLt kernel library for gfx1151 (95 files, alongside gfx110x/gfx120x).
The Lt path actually engages, and what it is worth is measured in
[`docs/hipblaslt.md`](docs/hipblaslt.md). Torch selects it with
`TORCH_BLAS_PREFER_HIPBLASLT=1` set **before torch is imported**; verify it
did not fall back by logging with `HIPBLASLT_LOG_MASK=32` and
`HIPBLASLT_LOG_FILE`.

### Attention needs AOTriton, and the switch precedes torch

`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, set before torch is imported,
makes flash/mem-efficient SDPA available on gfx1151 — measured 10–20× faster
than the fallback (4096-token attention: 135 ms → 12 ms). Set after import it
does nothing. Without it the math backend materialises `q @ kᵀ` in fp16 and can
overflow.

### First runs lie

MIOpen tunes convolution kernels once per machine; a first run through a
conv-heavy model can be an order of magnitude slower than every later one
(measured: 793 s against 1.7 s for the same stage). Benchmark from the second
run onward, always.

### `torch.cuda.mem_get_info` counts shared memory

It reports 43.87 GB on a 32 GB-dedicated machine. Exceeding dedicated VRAM
raises nothing and silently spills into shared memory, several times slower.
Cap torch yourself (`torch.cuda.set_per_process_memory_fraction`) so overflow
fails fast as `torch.OutOfMemoryError` instead.

## License

MIT (see [LICENSE](LICENSE)).
