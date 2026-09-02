# Attention on gfx1151 / Windows: the other ceiling

The pipelines profiled in [profiles.md](profiles.md) spend more device time
in attention than in GEMM (Hi3DGen's dominant stage is 54 % attention), so
the attention ceiling matters at least as much as the GEMM one. Measured
2026-09-03 with [`bench_sdpa.py`](../bench_sdpa.py): fp16
`F.scaled_dot_product_attention`, AOTriton flash backend
(`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`), torch 2.13.0+rocm10.0.0,
5 processes × 5 blocks × 50 calls, medians; FLOPs counted as the two matmuls
(4·B·H·SQ·SKV·D).

| B, H, SQ, SKV, D | TFLOPS | ms/call | Corresponds to |
|---|--:|--:|---|
| 1, 16, 1024, 1024, 64 | 19.2 | 0.22 | |
| 1, 16, 2048, 2048, 64 | 21.1 | 0.78 | |
| 1, 16, 4096, 4096, 64 | 22.8 | 2.97 | TRELLIS/Hi3DGen structure self-attn |
| 1, 16, 4096, 1374, 64 | 23.0 | 0.98 | their cross-attention |
| 1, 16, 8192, 8192, 64 | 24.0 | 11.2 | |
| 1, 16, 8194, 8194, 128 | 24.3 | 21.4 | Hunyuan3D DiT-like |

Observations:

- **Throughput is nearly flat in sequence length** (19 → 24 TFLOPS over
  8× seq): the flash kernels are already in their steady state at the sizes
  these pipelines use; there is no small-sequence cliff to fix.
- **Clock during the sweep: median 2 129 MHz** (`gpuclock.py`) — higher than
  under dense GEMM (1 914). Clock-adjusted peak at 2.13 GHz is
  ~43.6 TFLOPS, so flash attention runs at **~55 % of attainable peak**,
  against the GEMM libraries' ~77 % ([ceiling.md](ceiling.md)). In relative
  terms, attention has more headroom left than GEMM does.
- On the ROCm 7.2.1 stack the same self-attention shape cost ~5–6 ms/call
  (from in-pipeline profiling); the 10.0 wheels' newer AOTriton images cut
  that to 2.97 ms, which is most of why the attention-bound structure stages
  nearly halved in [rocm10.md](rocm10.md).

What was *not* measured here: backward passes (inference only), head-count
scaling, and non-flash fallbacks. The `sdp_backends_enabled` flags and the
clock trace ride along in the JSON for every run.
