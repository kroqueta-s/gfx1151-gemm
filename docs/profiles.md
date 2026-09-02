# GEMM profiles of three 3D-generation pipelines on gfx1151 / Windows

**Environment (identical for all three):** ASUS ProArt PX13
(HN7306EA-AI9641W) — Ryzen AI MAX+ 395 / Strix Halo, Radeon 8060S (gfx1151,
32 GB dedicated VRAM), a 13-inch laptop at factory power limits — Windows
11, ROCm 7.2.1, torch 2.9.1+rocm7.2.1,
Python 3.12.10, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, rocBLAS backend
(no hipBLASLt), measured 2026-09-02. Reference fp16 GEMM
alongside every run: 21–25 TFLOPS at 2048³, 29–31 TFLOPS at 4096³.

**How:** each runner's `tools/profile_gemm.py` — one unprofiled generation for
honest stage walls, then the same stages under `torch.profiler` with a
`TorchDispatchMode` riding along for dtypes. One run per pipeline on the same
reference image (an SDXL robot). This torch build has no Kineto, so per-op
device times carry a few microseconds of overhead per call and parent/child
double counting is not fully eliminated; **class shares are decision-grade,
not paper-grade**, and per-shape TFLOPS (kernel-time based) are solid.

## Where the device time goes

Share of profiled device time (bookkeeping noise excluded), per stage. *Wall*
is from the unprofiled run.

| Pipeline | Stage | Wall | GEMM | Attention | Other (elementwise, gather/scatter, norms) |
|---|---|--:|--:|--:|--:|
| Hi3DGen | structure (dominant) | 44.1 s | 11.6 s (25 %) | 24.8 s (54 %) | 9.6 s (21 %) |
| Hi3DGen | slat | 17.1 s | 4.8 s (28 %) | 3.4 s (20 %) | 9.1 s (53 %) |
| Hi3DGen | decode | 3.5 s | 0.5 s (10 %) | 1.1 s | 3.6 s |
| TRELLIS | structure | 22.9 s | 5.7 s (25 %) | 12.7 s (56 %) | 4.9 s |
| TRELLIS | slat (dominant) | 52.4 s | 21.4 s (33 %) | 10.3 s (16 %) | 34.0 s |
| TRELLIS | decode | 3.0 s | 0.4 s (9 %) | 1.0 s | 3.1 s |
| Hunyuan3D | shape (single stage) | 82.6 s | 42.7 s (52 %) | 18.8 s (23 %) | 20.5 s |

Two different worlds:

- **The TRELLIS family (Hi3DGen is built on TRELLIS) is attention- and
  elementwise-bound.** GEMM is 25–33 % of any stage; the structure stages spend
  more than half their time in flash attention (4096 tokens × 100–200 forward
  passes), and the sparse stages spend as much again in the boolean masks,
  gathers and copies of the submanifold-convolution shim.
- **Hunyuan3D is GEMM-bound (52 %)**, with large, well-proportioned shapes
  that rocBLAS already runs at 21–27 TFLOPS.

## Dominant GEMM shapes

All fp16 with fp16 accumulation dtype at the torch level (fp32 rows are
negligible scheduler/embedding work). `M_vox` is the active-voxel count of the
sample (23 215 for TRELLIS, 28 605 for Hi3DGen — sample-dependent).

### Hi3DGen and TRELLIS: **the same shapes** (same upstream architecture)

| Role | M | N | K | rocBLAS TFLOPS | Time share of its stage |
|---|--:|--:|--:|--:|---|
| MLP down | 4096 (tokens) | 1024 | 4096 | 18.5–18.7 | biggest transformer GEMM |
| MLP up | 4096 | 4096 | 1024 | 25–26 | |
| QKV | 4096 | 3072 | 1024 | 31 | |
| Out proj | 4096 | 1024 | 1024 | 29–30 | |
| Cross-attn KV | 1374 | 2048 | 1024 | 19–20 | |
| **Sparse-conv shim** | **M_vox (23k–29k)** | **128** | **2048** | **1.0–1.5** | **27 % of TRELLIS slat (14.5 s), 17 % of Hi3DGen slat** |
| Sparse-conv shim | M_vox | 128 | 128/256 | 11–13 | |
| Decode conv-ish | 65536 | 96–192 | 96–768 | 11–20 | |

The slat-stage transformer runs the same four shapes with M = M_vox subsampled
(≈ 4 383–5 086). **The single worst GEMM on either pipeline is the skinny
sparse-conv projection (N = 128): rocBLAS delivers 1 TFLOPS**, a 30× gap to
what the same silicon does at 4096³.

### Hunyuan3D: different architecture, different shapes

| Role | M | N | K | rocBLAS TFLOPS |
|---|--:|--:|--:|--:|
| DiT attn proj (mm) | 8194 (tokens) | 2048 | 2048 | 21.7 |
| DiT MLP down | 8194 | 2048 | 8192 | 22.0 |
| DiT MLP up | 8194 | 8192 | 2048 | 26.1 |
| DiT (fused pair) | 8194 | 2048 | 2048×2 | 27.7 |
| Cross-attn | 2740 | 2048 | 1024 | 23.5 |
| VAE / FlashVDM | 13824 | 1024 | 4096 | 20.3 |

## What the overlap means for a shared kernel

- **Hi3DGen and TRELLIS overlap ~100 %**: one shape set serves both.
- **Hunyuan3D shares nothing with them.** A kernel that serves all three needs
  shape dispatch from day one.
- **The two workloads want different things.** Hunyuan3D wants a classic
  large-tile fp16 GEMM (its shapes are already at 21–27 TFLOPS — and the
  realistic headroom above that on this platform is set by the sustained
  clock, not by kernel quality; see [ceiling.md](ceiling.md)). The TRELLIS
  family's largest single win is not a big-tile GEMM at all but the **N=128
  skinny GEMM at 1 TFLOPS**, plus attention and elementwise work that no
  GEMM kernel touches.

Per-stage detail lives in each runner's repository
(`docs/gemm_profile.md`); raw JSON stays with the machine that measured it.
