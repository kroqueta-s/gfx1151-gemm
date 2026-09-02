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
[hipblaslt.md](hipblaslt.md). On ROCm 10.0 the numbers with and
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
fixed is the floor (why the ceiling sits where it does:
[ceiling.md](ceiling.md)). The two small-N decode shapes regressed slightly;
they are worth fractions of a second per generation.

**Layouts:** rocBLAS 7.2.1 swung up to 1.6× between nn and nt on the same
shape; on 10.0 the asymmetry mostly closed (4096³: 30.2 nn / 29.4 nt). Where
it survives, it is ≤9 % and confined to shapes with small stage shares —
`F.linear` passes weights as nt, and pre-transposing them to force nn would
buy ~+9 % on Hunyuan3D's MLP-up GEMM (nn 30.1 / nt 27.7) and ~+6 % on the
TRELLIS-family MLP-down, which works out to **well under 1 % of any stage**.
Not worth the weight duplication; measured and declined.

Reference GEMM alongside every run: 30.2–30.4 at 2048³, 30.6–30.9 at 4096³.

## End to end: 1.13–1.79× per pipeline

The 10.0 column is the **median of 5 runs** (each a fresh process, tuned
MIOpen caches verified present beforehand, a 4096³ reference GEMM before and
after every run — all stayed within 29.7–31.5 TFLOPS — and a `gpuclock.py`
trace during). The 7.2.1 columns are single measurements from before the
upgrade and are labelled as such. hipBLASLt preference on everywhere
(a no-op on 10.0, still needed on 7.2.1).

| Pipeline | Stage | 7.2.1 rocBLAS (single) | 7.2.1 +hipBLASLt (single) | 10.0 (median of 5) |
|---|---|--:|--:|--:|
| Hi3DGen | structure (attention-bound) | 44.1 s | 44.0 s | **26.2 s** (26.1–27.0) |
| Hi3DGen | slat | 17.1 s | 10.7 s | **9.4 s** (8.3–9.5) |
| Hi3DGen | whole generation | 69.7 s | 62.9 s | **39.0 s (1.79×)** (38.2–39.3) |
| TRELLIS | conditioning | 1.1 s | — | 0.8 s (0.8–0.9) |
| TRELLIS | structure | 22.9 s | 22.9 s | **13.8 s** (13.8–14.0) |
| TRELLIS | slat | 52.4 s | 39.1 s | **32.0 s** (31.9–32.1) |
| TRELLIS | whole generation | 79.4 s | 66.1 s | **49.6 s (1.60×)** (49.4–49.8) |
| Hunyuan3D | shape | 82.6 s | 80.4 s | **73.1 s (1.13×)** (73.0–73.3) |

The attention-bound structure stages nearly halved — that is the newer
AOTriton flash kernels ([attention.md](attention.md)), not GEMM — and the
sparse-conv skinny GEMM fix carried over. An apparent regression resolved
itself along the way: TRELLIS's conditioning stage read ~5 s on the first
runs after the upgrade, which turned out to be MIOpen 3.6.0's one-time
tuning of the DINOv2 convolutions; with the tuning cache in place it is back
to 0.8 s.

## A caution about profiling on torch 2.13

On this build, running `torch.profiler` **with a `TorchDispatchMode` active**
double-counts every op (exactly 2× per call; per-shape TFLOPS then come out
above the hardware ceiling). The profiler alone counts correctly — minimal
repro: 100 `torch.mm` calls report `count=100` bare, `count=200` under a
pass-through dispatch mode; torch 2.9.1 counted correctly in both cases.
Stage walls and `bench_gemm.py` device-event timings are unaffected and are
what the tables above use.
