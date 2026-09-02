# ROCm 10.0 wheels on gfx1151 / Windows: what changes, what breaks

**Environment:** the same machine as [profiles.md](profiles.md), upgraded from
torch 2.9.1+rocm7.2.1 to **torch 2.13.0+rocm10.0.0** (`whl-next` channel),
Adrenalin driver unchanged (2026-08, ROCm 7.2.1 era — the wheels bundle their
own runtime). Measured 2026-09-02.

## Install: four traps, in the order they bite

1. **Smart App Control blocks the 7.14.1 wheels.** The multi-arch channel's
   `rocm-sdk-libraries` 7.14.1 ships unsigned `rocrand.dll`/`hiprand.dll`
   that SAC rejects (WinError 4551), taking MIOpen and rocFFT down with them
   as dependency failures. The same DLLs from the 7.14.0 and 10.0.0 builds
   load fine — SAC is reputation-based, so this is per-build-hash luck, and
   reinstalling the same version does not help. torch 2.12 requires 7.14.1
   on that channel, which is why this machine went to 10.0 instead.
2. **The exact-arch device package is mandatory.** torch 2.13 ships its GPU
   kernels as an external kernel pack (`torch/.kpack/torch_gfx1151.kpack`),
   delivered by `amd-torch-device-gfx1151` (which also pulls
   `rocm-sdk-device-gfx1151`, the kernels for rocBLAS/hipBLASLt/MIOpen). The
   family package `amd-torch-device-gfx115x` carries only the flash-attention
   images. Install **both**; with only the family package every kernel launch
   fails with `hipErrorInvalidImage` (`kpack_load_code_object failed`).
3. **torchvision has its own exact-arch package** —
   `amd-torchvision-device-gfx1151` (no family variant exists). Without it,
   torchvision's compiled ops (`deform_conv2d` and friends) fail with
   `hipErrorInvalidDeviceFunction`.
4. **MIOpen cannot run batch norm at all.** It compiles its batch-norm
   kernels with hiprtc at first use, and the compile dies with
   `'type_traits' file not found`: the wheels ship no C++ standard library
   for hiprtc to include (`rocm-sdk-devel` does not fix it). Surfaces as
   `miopenStatusUnknownError` from `F.batch_norm`. Convolutions use
   precompiled solvers and are unaffected. The model runners in this family
   probe for this at load time and reroute `F.batch_norm` to torch's native
   kernel when MIOpen fails (see their `shims.py`).

The working install line (Python 3.12, Windows):

```powershell
pip install "torch==2.13.0+rocm10.0.0" "torchvision==0.28.0+rocm10.0.0" `
    "amd-torch-device-gfx115x==2.13.0+rocm10.0.0" `
    "amd-torch-device-gfx1151==2.13.0+rocm10.0.0" `
    "amd-torchvision-device-gfx1151==0.28.0+rocm10.0.0" `
    --index-url https://stable.repo.amd.com/rocm/whl-next/ `
    --extra-index-url https://pypi.org/simple
```

## GEMM: the default path caught up with hipBLASLt

Same [`bench_gemm.py`](../bench_gemm.py) method as
[hipblaslt.md](hipblaslt.md), keepalive on. On ROCm 10.0 the numbers with and
without `TORCH_BLAS_PREFER_HIPBLASLT=1` agree within noise — the pathological
rocBLAS cases are gone from the default path. Selected medians (fp16, layout
that the pipelines use):

| M, N, K | 7.2.1 rocBLAS | 7.2.1 hipBLASLt | 10.0 default |
|---|--:|--:|--:|
| 25600, 128, 2048 (sparse-conv shim) | 1.8 | 14.3 | **15.8** |
| 4096, 1024, 4096 (MLP down) | 19.4 | 25.5 | **25.7** |
| 4096, 3072, 1024 (QKV, nt) | 27.2 | 25.2 | **29.7** |
| 8194, 8192, 2048 (Hunyuan MLP up) | 28.3 | 23.3 | **30.1** |
| 8194, 2048, 8192 (nt) | 18.5 | 21.5 | **24.6** |
| 65536, 96, 192 (decode) | 14.2 | 8.7 | 11.8 |
| 25600, 128, 128 | 19.2 | 5.5 | 12.8 |
| 2048³ | 23.0 | 21.3 | 26.8–27.7 |
| 4096³ | **30.8** | 24.2 | 30.2 |

The library ceiling on this silicon is **~30 TFLOPS** either way; what ROCm 10
fixed is the floor. The two small-N decode shapes regressed slightly; they are
worth fractions of a second per generation.

Reference GEMM alongside every run: 30.2–30.4 at 2048³, 30.6–30.9 at 4096³.

## End to end: 1.13–1.68× per pipeline

Second-run stage walls (first runs pay MIOpen tuning), keepalive on,
hipBLASLt preference on (harmless here, still needed on 7.2.1):

| Pipeline | Stage | 7.2.1 rocBLAS | 7.2.1 +hipBLASLt | 10.0 |
|---|---|--:|--:|--:|
| Hi3DGen | structure (attention-bound) | 44.1 s | 44.0 s | **26.3 s** |
| Hi3DGen | slat | 17.1 s | 10.7 s | **9.4 s** |
| Hi3DGen | whole generation | 69.7 s | 62.9 s | **41.5 s (1.68×)** |
| TRELLIS | structure | 22.9 s | 22.9 s | **14.3 s** |
| TRELLIS | slat | 52.4 s | 39.1 s | **32.8 s** |
| TRELLIS | whole generation | 79.4 s | 66.1 s | **~56 s (1.42×)** |
| Hunyuan3D | shape | 82.6 s | 80.4 s | **73.2 s (1.13×)** |

The attention-bound structure stages nearly halved (newer AOTriton flash
kernels), and the sparse-conv skinny GEMM fix carried over. One open item:
TRELLIS's conditioning stage grew from ~1 s to ~5 s on the new stack —
unexplained, small, noted for later.

## A caution about profiling on torch 2.13

The per-op profiler on this build double-counts calls (counts exactly 2× per
op, per-shape TFLOPS come out above the hardware ceiling). Stage walls and
`bench_gemm.py` device-event timings are unaffected and are what the tables
above use. Do not trust `torch.profiler` per-op device time on this stack
without revalidating it.
