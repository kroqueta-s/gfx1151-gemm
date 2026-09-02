# The 600 MHz story: it was the display, not the workload

On this machine (ASUS ProArt PX13, Ryzen AI MAX+ 395 / gfx1151, Windows 11),
**the AMD driver pins the GPU near 600 MHz whenever the console display is
off** — lid closed, or the display-off timeout expiring, locked or not. GEMM
drops from ~31 to ~8 TFLOPS, a 4× loss, and stays there until the display
comes back. Anyone running unattended compute on this platform — LLM
inference included — is likely running at a quarter speed without knowing.

This page tells the story in order, because the attribution changed twice
and the intermediate versions were published here.

## Timeline

- **2026-09-01** (Adrenalin 26.8.1 was installed 22:34 that night; earlier
  measurements ran on the prior driver). During remote benchmarking, GEMM
  measured **4.8 TFLOPS with the clock at 600 MHz**; with a render loop
  alive it reached 20.9. Attribution at the time: "the driver does not
  raise the power state for compute-only work." A hidden-window render
  keepalive (`gfxlight.py`) was built on that reading, and an A/B on a real
  pipeline (179 s → 86 s) seemed to confirm it. **What was not recorded:
  the lock and display state during those runs.**
- **2026-09-02 evening.** Re-verification under an unlocked session with
  the display on: the keepalive changed nothing (A/B/A at 4096³:
  31.7 / 32.3 / 31.8 TFLOPS) and compute alone reached 2.39 GHz. The
  original claim was retired from these documents. The observations of
  09-01 were never wrong — the attribution was.
- **2026-09-02 night.** With `gpuclock.py` (driver PM sensors) and a
  session-state watcher running, the slow state reproduced on demand and
  the cause pinned:

## What was measured (all on Adrenalin 26.8.1, ROCm 10.0 wheels)

| Console state | GPU clock under GEMM | 4096³ fp16 |
|---|--:|--:|
| Unlocked, display on | ~1.9 GHz | 31.4 TFLOPS |
| **Locked**, display on | ~1.9 GHz | 31.2 |
| Locked, **display off** (timeout) | **600–632 MHz** | **7.6–8.5** |
| **Lid closed** | **600 MHz** | **8.3–8.5** |
| Lid open, no input past the 60 s timeout | **609–610 MHz** | **7.6** |
| RDP / remote-desktop sessions (connected, disconnected) | not measured | not measured |

The lock is a bystander; **the display power state is the trigger**, by any
route. One run shows the whole mechanism end to end: with
`SetThreadExecutionState(ES_DISPLAY_REQUIRED)` held, 240 s pass at full
speed (locked, no input); the moment it is released, the 60 s timeout runs
out and the clock lands at 609 MHz within the next probe.

## What does and does not help

| Measure | Effect |
|---|---|
| `gfxlight.py` render keepalive | **None.** No effect with the display on (nothing to fix), and no effect with it off — 600 MHz with the loop alive (8.3 TFLOPS). The state it was built for is exactly the state it cannot touch. |
| Windows power plan (Performance / Balanced / Power saver) | **None** (31.2 / 31.5 / 32.3 TFLOPS). |
| **`SetThreadExecutionState(ES_CONTINUOUS \| ES_DISPLAY_REQUIRED)`, held before the display sleeps** | **Full prevention.** The standard media-player API; keeps the display (and the clock) up through lock and timeout. It cannot *wake* a display that is already off (611 MHz after setting it in the slow state). |
| Injecting one 1-px mouse move (`SendInput`) | **Instant rescue**: display wakes to the lock screen and the very next probe reads 30.8 TFLOPS, still locked. |
| Unlocking | Works (it wakes the display). |

## It is not Modern Standby

This is an S0 Low Power Idle (Modern Standby) machine, which made the OS's
screen-off power phase the obvious suspect. Tested 2026-09-03: with Modern
Standby disabled (`PlatformAoAcOverride=0`, value verified in the same log,
after a reboot, probing from a boot-time task), the pin still occurred —
607–633 MHz / 7.5–8.4 TFLOPS across five probes in two locked/display-off
windows, full speed whenever the console was unlocked in between. The
trigger lives in the driver's own display-state handling, not in the OS
standby machinery.

## Notes

- The numbers 4.8 (09-01) and 7.6–8.5 (09-02) come from different stacks
  (rocBLAS 7.2.1 vs ROCm 10.0 defaults) — consistent with the floor being a
  fixed ~600 MHz clock under two different libraries.
- Reproduced on the current driver, so this is not fixed by updating
  Adrenalin 26.8.1; whether it is intended power management or a bug is a
  question for AMD. In the pinned state the package draws ~22 W and the GFX
  rail single digits — a hard clock floor far below the 70 W sustained
  limit that caps display-on performance ([ceiling.md](ceiling.md)).
- Every other measurement in this repository was taken with the display on;
  the ceilings in [ceiling.md](ceiling.md) describe the display-on world.
