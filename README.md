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
  warmup + timed blocks, median and range). Shapes are arguments; nothing is
  hard-coded.
- [`gfxlight.py`](gfxlight.py) — clock keepalive: a hidden render loop that
  stops the Windows driver from parking the GPU at 600 MHz during compute
  (see below). Run it alongside any benchmark or workload.
- [`docs/profiles.md`](docs/profiles.md) — measured GEMM profiles of the three
  3D-generation pipelines named above, and how much their dominant shapes
  overlap.
- [`docs/hipblaslt.md`](docs/hipblaslt.md) — rocBLAS against hipBLASLt on the
  shapes those pipelines actually run: the Lt path works on gfx1151 under
  Windows, wins by 8× on skinny GEMMs, and loses on fat ones.
- [`docs/rocm10.md`](docs/rocm10.md) — the ROCm 10.0 wheels: four install
  traps (Smart App Control, kernel-pack device packages, torchvision,
  MIOpen batch norm), and the measurements after the upgrade (1.13–1.68×
  end to end).

## What is known about gfx1151 on Windows

Everything below was measured on one machine — Strix Halo, Radeon 8060S, 32 GB
of dedicated VRAM, Windows 11, driver with ROCm 7.2.1 support, torch
2.9.1+rocm7.2.1 — in September 2026. Re-measure before trusting any of it in a
different environment.

### The GPU idles at 600 MHz unless something renders

The AMD Windows driver does not raise the GPU power state for compute-only
work: at 99 % compute utilisation the clock sits at **600 MHz** (measured
2026-09-01: 4.8 TFLOPS on a GEMM that reaches 20.9 TFLOPS at 2.35 GHz, a 4.3×
difference). Any live 3D rendering — including a hidden window — raises it.
`gfxlight.py` exists for exactly this; keep it alive while measuring anything,
and record a reference GEMM next to every measurement so a clock drop cannot
masquerade as a regression. **Every number in this repository was taken with
the keepalive on.**

### What rocBLAS delivers (the baseline to beat)

fp16 square GEMM through `torch.mm`, keepalive on, 2026-09-02:

| Shape | rocBLAS TFLOPS |
|---|--:|
| 2048³ | ~24 |
| 4096³ | ~31 |

On the ROCm 10.0 wheels the default path reaches ~30 TFLOPS across the same
shapes — including the ones rocBLAS 7.2.1 ran at 1.8 — and hipBLASLt
preference stops mattering; see [`docs/rocm10.md`](docs/rocm10.md).

The paper number for this silicon (59.4 TFLOPS fp16) is a clock calculation,
not a measurement. Hand-tuned WMMA HIP kernels reach 41–46 TFLOPS at 4096³ on
Linux (ROCm 7.14, external measurement); treat everything above that as
unreachable.

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
