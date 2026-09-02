# hipBLASLt on gfx1151 / Windows: measured, and it is real

**Environment:** Strix Halo (Radeon 8060S, gfx1151), Windows 11, ROCm 7.2.1,
torch 2.9.1+rocm7.2.1, measured 2026-09-02 with
[`bench_gemm.py`](../bench_gemm.py) (fp16, warmup 10, 5 blocks × 100 calls,
device-event timing, medians reported).

## The Lt path engages on this architecture

Contrary to the folk wisdom that Windows hipBLASLt has no gfx1151 kernels: the
`rocm_sdk_libraries` wheel behind torch 2.9.1+rocm7.2.1 ships a gfx1151
hipBLASLt kernel library (95 files), and with

```
TORCH_BLAS_PREFER_HIPBLASLT=1
ROCBLAS_USE_HIPBLASLT=1
```

set **before torch is imported**, `HIPBLASLT_LOG_MASK=32` logging shows
hipBLASLt solutions being selected (17 340 log lines over one benchmark run,
`--solution_index` present, fp32 compute) and **no
"Attempting to use hipBLASLt on an unsupported architecture" fallback**.

## Microbenchmark: neither backend wins outright

Median TFLOPS, fp16. `nn` = both operands contiguous; `nt` = B passed as a
transposed view, the way `F.linear` passes weights. Bold marks the winner
where the gap is beyond noise.

| M, N, K | rocBLAS nn | rocBLAS nt | hipBLASLt nn | hipBLASLt nt |
|---|--:|--:|--:|--:|
| 4096, 1024, 4096 | 19.4 | 23.1 | **25.5** | 23.4 |
| 4096, 4096, 1024 | **28.5** | 18.0 | 22.9 | 23.9 |
| 4096, 3072, 1024 | **31.3** | 27.2 | 24.3 | 25.2 |
| 4096, 1024, 1024 | 25.8 | **29.5** | 21.9 | 22.6 |
| 1374, 2048, 1024 | **28.1** | 25.9 | 18.7 | 18.4 |
| 25600, 128, 2048 | 1.8 | 8.2 | 14.3 | **15.3** |
| 25600, 128, 128 | **19.2** | 18.8 | 5.5 | 8.3 |
| 65536, 96, 192 | **14.2** | 13.0 | 8.7 | 10.7 |
| 5086, 1024, 4096 | 19.1 | 22.7 | **24.1** | 21.2 |
| 8194, 2048, 2048 | 20.1 | 22.5 | 23.4 | **24.7** |
| 8194, 2048, 8192 | **28.2** | 18.5 | 23.8 | 21.5 |
| 8194, 8192, 2048 | **28.3** | 18.8 | 23.3 | 24.4 |
| 8194, 2048, 4096 | 26.8 | 22.6 | 24.8 | 26.7 |
| 2740, 2048, 1024 | 25.4 | **27.3** | 19.8 | 17.9 |
| 13824, 1024, 4096 | 20.1 | 22.5 | **24.8** | 22.1 |
| 2048³ | 23.0 | **28.4** | 21.3 | 23.6 |
| 4096³ | **30.8** | 21.4 | 24.2 | 24.0 |

Patterns:

- **hipBLASLt rescues the skinny GEMM.** M large, N = 128, K = 2048 — the
  sparse-convolution projection in TRELLIS-family pipelines — goes from
  1.8 to 14–15 TFLOPS (**8×** in the layout the models actually use).
- **hipBLASLt is steadier across layouts** (its nn/nt columns nearly agree);
  rocBLAS swings up to 1.6× between layouts on the same shape.
- **rocBLAS keeps the crown on fat shapes in its preferred layout**
  (4096³: 30.8 vs 24.2) and on very small-N shapes other than the one above.

## End to end, torch decides per call — and it comes out ahead

Each pipeline generation measured twice (its runner's `tools/profile_gemm.py`),
identical settings, only the two environment variables changed:

| Pipeline | Stage | rocBLAS | hipBLASLt | Speedup |
|---|---|--:|--:|--:|
| TRELLIS | slat (dominant) | 52.4 s | 39.1 s | **1.34×** |
| TRELLIS | whole generation | 79.4 s | 66.1 s | **1.20×** |
| Hi3DGen | slat | 17.1 s | 10.7 s | **1.60×** |
| Hi3DGen | structure (dominant, attention-bound) | 44.1 s | 44.0 s | 1.00× |
| Hunyuan3D | shape | 82.6 s | 80.4 s | ~1.03× (within variance) |

In-model observations: the skinny sparse-conv `mm` fell from 14.5 s to 1.0 s
in the TRELLIS slat stage; Hunyuan3D's unbiased `mm` projections improved
(21.7 → 26.9 TFLOPS) while its biased `addmm` rows did not move.

All three runners in this family therefore default to
`*_PREFER_HIPBLASLT=on` and record the backend actually in effect in
`metrics.blas_backend`.
