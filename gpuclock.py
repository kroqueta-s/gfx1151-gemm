# SPDX-License-Identifier: MIT
"""Read the GPU clock (and friends) from the AMD driver's own ADL library.

On gfx1151 under Windows there is no rocm-smi; the wheels ship no telemetry
tool, and installing third-party monitors is unattractive on a machine with
Smart App Control enforcing (unsigned binaries are refused by build hash).
The Adrenalin driver, however, already exposes everything needed through
`atiadlxx.dll` (signed, present on every machine with the driver), and its
`ADL2_New_QueryPMLogData_Get` returns the live PM sensors.

Sensor indices follow the public ADL SDK `ADL_PMLOG_SENSORS` enum; the ones
surfaced here were also verified empirically on this machine (gfx clock:
hundreds of MHz idle, 2.3+ GHz under GEMM load).

Standalone (any Python, torch not needed)::

    python gpuclock.py --seconds 10 --interval 0.5

Or from a harness::

    from gpuclock import ClockWatch
    with ClockWatch() as w:
        ...  # GPU work
    print(w.summary())  # median/min/max gfx clock over the window

Reading the sensors is a driver query, not GPU work; sampling at 2 Hz has no
measurable effect on a running benchmark.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import threading
import time
from ctypes import Structure, byref, c_int, c_void_p, windll
from typing import Any

# ADL_PMLOG_SENSORS indices (ADL SDK, adl_defines.h), cross-checked empirically
# on this machine: gfx clock 709-745 MHz idle / 2392 under GEMM load, activity
# 1-2 % idle / 73 % load, power 1-2 W idle / 27 W load. Indices 28/29 look like
# temperatures (44-45 idle, 66-71 load) but are left unnamed until confirmed.
SENSOR_NAMES = {
    1: "gfxclk_mhz",
    2: "memclk_mhz",
    3: "socclk_mhz",
    19: "gfx_activity_pct",
    30: "power_w",
}


class _ADLSingleSensorData(Structure):
    _fields_ = [("supported", c_int), ("value", c_int)]


class _ADLPMLogDataOutput(Structure):
    _fields_ = [("size", c_int), ("sensors", _ADLSingleSensorData * 256)]


_ADL_MAIN_MALLOC_CALLBACK = ctypes.WINFUNCTYPE(c_void_p, c_int)

_msvcrt = ctypes.cdll.msvcrt
_msvcrt.malloc.restype = c_void_p
_msvcrt.malloc.argtypes = [ctypes.c_size_t]


@_ADL_MAIN_MALLOC_CALLBACK
def _alloc(size: int) -> int:
    return _msvcrt.malloc(size)


class AdlSession:
    """One ADL context. Never raises after construction; `ok` says whether it works."""

    def __init__(self) -> None:
        self.ok = False
        self._adl: Any = None
        self._context = c_void_p()
        self._adapter = -1
        try:
            self._adl = windll.LoadLibrary("atiadlxx.dll")
            if self._adl.ADL2_Main_Control_Create(_alloc, 1, byref(self._context)) != 0:
                return
            n = c_int()
            self._adl.ADL2_Adapter_NumberOfAdapters_Get(self._context, byref(n))
            for idx in range(n.value):
                out = _ADLPMLogDataOutput()
                if self._adl.ADL2_New_QueryPMLogData_Get(self._context, idx, byref(out)) == 0:
                    if any(s.supported for s in out.sensors):
                        self._adapter = idx
                        self.ok = True
                        return
        except OSError:
            return

    def read(self) -> dict[str, int]:
        """Return the named sensors, plus every supported raw index as `raw_<i>`."""
        if not self.ok:
            return {}
        out = _ADLPMLogDataOutput()
        if self._adl.ADL2_New_QueryPMLogData_Get(self._context, self._adapter, byref(out)) != 0:
            return {}
        result: dict[str, int] = {}
        for i, s in enumerate(out.sensors):
            if not s.supported:
                continue
            name = SENSOR_NAMES.get(i)
            if name:
                result[name] = s.value
            result[f"raw_{i}"] = s.value
        return result


class ClockWatch:
    """Sample the gfx clock on a thread and summarise it afterwards."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.samples: list[dict[str, int]] = []
        self._session = AdlSession()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            data = self._session.read()
            if data:
                self.samples.append(data)
            self._stop.wait(self.interval)

    def __enter__(self) -> ClockWatch:
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        """Median / min / max of the gfx clock and power over the window."""
        if not self.samples:
            return {"supported": self._session.ok, "samples": 0}
        out: dict[str, Any] = {"supported": True, "samples": len(self.samples)}
        for key in ("gfxclk_mhz", "power_w"):
            values = [s[key] for s in self.samples if key in s]
            if values:
                out[key] = {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--raw", action="store_true", help="print every supported sensor index")
    args = parser.parse_args()

    session = AdlSession()
    if not session.ok:
        print("ADL not available (no AMD driver, or the query is unsupported)")
        return 1
    end = time.time() + args.seconds
    while time.time() < end:
        data = session.read()
        if args.raw:
            print(json.dumps(data))
        else:
            named = {k: v for k, v in data.items() if not k.startswith("raw_")}
            print(json.dumps(named))
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
