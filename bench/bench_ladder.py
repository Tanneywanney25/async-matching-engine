"""Phase 3 benchmark: feature kernels, the optimization ladder, baselines, bandwidth ceiling.

    python -m bench.bench_ladder --archive <dir> --out <dir> --source synthetic|real [--quick]

Writes three sections (shapes in ``bench/CONTRACT.md``):

``bandwidth``  device-to-device copy kernel and ``cudaMemcpy`` over 1 GiB, read + write bytes;
``ladder``     v0 AoS per-metric kernels, v1 SoA per-metric, v2 SoA fused, v3 fused with
               128-bit loads (four snapshots per thread), v4 = v3 with a grid-stride loop and a
               full launch-configuration sweep; per-kernel time from CUDA events around the
               launches only, end-to-end time from the wall clock including the host-to-device
               copy of the inputs (pageable host memory) and the copy back of the outputs;
``baselines``  single-threaded C++ (g++ -O3 -march=native), the same with OpenMP on every core,
               vectorized NumPy and a CuPy expression chain, all on this machine in this job.

Every implementation is verified against the float64 NumPy reference before anything is timed.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from bench import ref_features as ref
from bench.common import (REPO_ROOT, T4_THEORETICAL_BW_GBS, Bench, cupy, effective_bandwidth, event_time_ms,
                          load_archive, raw_module, to_aos, wall_time_ms)

BLOCK = 256
SWEEP_BLOCKS = (64, 128, 256, 512, 1024)
SWEEP_GRID_MULTIPLES = (1, 2, 4, 8, 16, 32)
QUICK_SWEEP_BLOCKS = (128, 256, 1024)
QUICK_SWEEP_GRID_MULTIPLES = (4, 16)
QUICK_N = 50_000
BW_BYTES = 1 << 30
QUICK_BW_BYTES = 1 << 28
COPY_GRID, COPY_BLOCK = 1280, 256

RUNGS = [
    ("v0_aos", "v0 AoS, one kernel per metric", 4),
    ("v1_soa", "v1 SoA, one kernel per metric", 4),
    ("v2_soa_fused", "v2 SoA, all metrics fused", 1),
    ("v3_soa_fused_vec4", "v3 fused, 128-bit loads, 4 snapshots per thread", 1),
    ("v4_soa_fused_vec4_sweep", "v4 v3 + grid-stride loop, best of the launch sweep", 1),
]


def log(*args) -> None:
    print("[ladder]", *args, flush=True)


# --------------------------------------------------------------------------- #
# Bandwidth ceiling
# --------------------------------------------------------------------------- #


def device_theoretical_gbs() -> float:
    cp = cupy()
    p = cp.cuda.runtime.getDeviceProperties(0)
    return 2.0 * p["memoryClockRate"] * 1e3 * p["memoryBusWidth"] / 8 / 1e9


def measure_bandwidth(bench: Bench, quick: bool) -> dict:
    cp = cupy()
    nbytes = QUICK_BW_BYTES if quick else BW_BYTES
    src = cp.ones(nbytes // 4, dtype=cp.int32)
    dst = cp.empty_like(src)
    module = raw_module(["copy.cu"])
    copy = module.get_function("copy_int4")
    read = module.get_function("read_int4")
    sink = cp.empty(COPY_GRID * COPY_BLOCK, dtype=cp.uint32)
    stream = cp.cuda.get_current_stream()
    n_int4 = np.int64(nbytes // 16)
    rt = cp.cuda.runtime

    def kernel() -> None:
        copy((COPY_GRID,), (COPY_BLOCK,), (src, dst, n_int4))

    def memcpy() -> None:
        rt.memcpyAsync(dst.data.ptr, src.data.ptr, nbytes, rt.memcpyDeviceToDevice, stream.ptr)

    def read_only() -> None:
        read((COPY_GRID,), (COPY_BLOCK,), (src, n_int4, sink))

    kernel()
    stream.synchronize()
    if not bool((dst[:1024] == 1).all()):
        raise RuntimeError("copy kernel produced wrong data")
    res = bench.block("bandwidth", {"copy_kernel": lambda: event_time_ms(stream, kernel),
                                    "cudamemcpy_d2d": lambda: event_time_ms(stream, memcpy),
                                    "read_kernel": lambda: event_time_ms(stream, read_only)})
    del src, dst, sink
    cp.get_default_memory_pool().free_all_blocks()
    k_gbs = 2 * nbytes / (res["copy_kernel"]["median_ms"] * 1e-3) / 1e9
    m_gbs = 2 * nbytes / (res["cudamemcpy_d2d"]["median_ms"] * 1e-3) / 1e9
    r_gbs = nbytes / (res["read_kernel"]["median_ms"] * 1e-3) / 1e9
    ceilings = {"copy_kernel": k_gbs, "cudamemcpy": m_gbs, "read_kernel": r_gbs}
    section = {
        "bytes": nbytes, "copy_kernel": res["copy_kernel"], "cudamemcpy_d2d": res["cudamemcpy_d2d"],
        "read_kernel": res["read_kernel"], "read_kernel_gbs": r_gbs,
        "copy_ceiling_gbs": max(k_gbs, m_gbs), "ceiling_kind": max(ceilings, key=ceilings.get),
        "copy_kernel_gbs": k_gbs, "cudamemcpy_gbs": m_gbs, "ceiling_gbs": max(ceilings.values()),
        "theoretical_gbs": device_theoretical_gbs(), "theoretical_spec_gbs": T4_THEORETICAL_BW_GBS,
        "counting": "copies: read + write bytes (2 x buffer); read kernel: bytes read (1 x buffer)",
        "ceiling_rule": "the highest of the three measured figures; the feature kernels read ~95% of their traffic, "
                        "so a copy (half writes) understates what they can reach",
        "copy_launch": {"grid": COPY_GRID, "block": COPY_BLOCK, "load_width_bits": 128},
    }
    bench.section("bandwidth", section)
    log(f"bandwidth: copy kernel {k_gbs:.1f} GB/s, cudaMemcpy {m_gbs:.1f} GB/s, read-only {r_gbs:.1f} GB/s, "
        f"theoretical {section['theoretical_gbs']:.1f}")
    return section


def pinned_copy(arr: np.ndarray) -> np.ndarray:
    """A page-locked host copy of ``arr`` (CuPy pinned memory viewed as a NumPy array)."""
    cp = cupy()
    arr = np.ascontiguousarray(arr)
    mem = cp.cuda.alloc_pinned_memory(max(1, arr.nbytes))
    out = np.frombuffer(mem, dtype=arr.dtype, count=arr.size).reshape(arr.shape)
    out[...] = arr
    return out


# --------------------------------------------------------------------------- #
# C++ baseline
# --------------------------------------------------------------------------- #


def build_cpu_library() -> ctypes.CDLL:
    out = REPO_ROOT / "build" / "libfeatures_cpu.so"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", "-o", str(out),
           str(REPO_ROOT / "cpu" / "features_cpu.cpp")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"g++ failed: {proc.stderr[-2000:]}")
    lib = ctypes.CDLL(str(out))
    lib.features_cpu.restype = ctypes.c_int
    lib.features_cpu.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int64, ctypes.c_int32, ctypes.c_int32]
                                 + [ctypes.c_void_p] * 6)
    lib.features_cpu_max_threads.restype = ctypes.c_int
    return lib


def gxx_version() -> str:
    try:
        return subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return repr(exc)


# --------------------------------------------------------------------------- #
# CuPy chain baseline
# --------------------------------------------------------------------------- #

CUPY_OPS = 22  # array expressions below, each one or more CuPy kernels


def cupy_features(bpx, bsz, apx, asz, prod, tick) -> Dict[str, object]:
    cp = cupy()
    bb, ba = bpx[0], apx[0]
    both = (bb > 0) & (ba > 0)
    spread = cp.where(both, ba - bb, 0).astype(cp.int32)
    b10 = bsz[: ref.TOP].sum(axis=0)
    a10 = asz[: ref.TOP].sum(axis=0)
    tot = b10 + a10
    imb = cp.where(tot > 0, (b10 - a10) / cp.where(tot > 0, tot, 1), 0).astype(cp.float32)
    bs0, as0 = bsz[0], asz[0]
    tot0 = bs0 + as0
    micro = cp.where(both & (tot0 > 0), 0.5 * spread.astype(cp.float32) * ((bs0 - as0) / cp.where(tot0 > 0, tot0, 1)),
                     0).astype(cp.float32)
    return {"spread_ticks": spread, "spread_usd": spread.astype(cp.float32) * tick[prod],
            "depth_bid": bsz.sum(axis=0), "depth_ask": asz.sum(axis=0), "imbalance10": imb,
            "microprice_offset_ticks": micro}


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #


class Ladder:
    def __init__(self, arch: dict, n: int) -> None:
        cp = self.cp = cupy()
        meta = arch["meta"]
        self.levels = int(arch["bid_px"].shape[0])
        self.n = int(n)
        self.n_stride = int(math.ceil(self.n / 4) * 4)
        self.tick = ref.tick_usd_array(meta["products"], meta["tick_sizes"])
        L, n, ns = self.levels, self.n, self.n_stride
        # Host: the archive's own level-major arrays (unpadded, for NumPy and C++) ...
        self.h = {k: np.ascontiguousarray(arch[k][:, :n]) for k in ("bid_px", "bid_sz", "ask_px", "ask_sz")}
        self.h_prod = np.ascontiguousarray(arch["snap_product"][:n])
        # ... and the padded SoA copies the GPU reads (row stride n_stride, 16-byte aligned rows).
        self.hp = {}
        for k, arr in self.h.items():
            pad = np.zeros((L, ns), dtype=arr.dtype)
            pad[:, :n] = arr
            self.hp[k] = pad
        self.hp_prod = np.zeros(ns, dtype=np.uint8)
        self.hp_prod[:n] = self.h_prod
        self.h_aos = to_aos(self.h).view(np.uint8)
        self.d = {k: cp.asarray(v) for k, v in self.hp.items()}
        self.d_prod = cp.asarray(self.hp_prod)
        self.d_tick = cp.asarray(self.tick)
        self.d_aos = cp.asarray(self.h_aos)
        self.out = {name: cp.empty(ns, dtype=ref.OUTPUT_DTYPES[name]) for name in ref.OUTPUTS}
        self.h_out = {name: np.empty(ns, dtype=ref.OUTPUT_DTYPES[name]) for name in ref.OUTPUTS}
        # Pinned (page-locked) copies of exactly what the kernels read and write, for the pinned
        # end-to-end variant; the staging copy into them is not timed (the NumPy baseline's
        # inputs are also already in memory).
        self.pin = {"bid_sz": pinned_copy(self.hp["bid_sz"]), "ask_sz": pinned_copy(self.hp["ask_sz"]),
                    "bid_px0": pinned_copy(self.hp["bid_px"][0]), "ask_px0": pinned_copy(self.hp["ask_px"][0]),
                    "product": pinned_copy(self.hp_prod), "aos": pinned_copy(self.h_aos)}
        self.pin_out = {name: pinned_copy(np.empty(ns, dtype=ref.OUTPUT_DTYPES[name])) for name in ref.OUTPUTS}
        self.mod = raw_module(["features.cu"])
        self.k = {name: self.mod.get_function(name) for name in (
            "v0_spread", "v0_depth", "v0_imbalance", "v0_microprice", "v1_spread", "v1_depth", "v1_imbalance",
            "v1_microprice", "v2_fused", "v3_fused_vec4", "v4_fused_vec4_gs")}
        self.stream = cp.cuda.get_current_stream()
        self.n64, self.ns64 = np.int64(n), np.int64(ns)
        self.grid1 = max(1, math.ceil(n / BLOCK))
        self.groups = ns // 4
        self.grid3 = max(1, math.ceil(self.groups / BLOCK))
        self.v4_config = (BLOCK, 40 * 8)

    def outs(self) -> tuple:
        o = self.out
        return (o["spread_ticks"], o["spread_usd"], o["depth_bid"], o["depth_ask"], o["imbalance10"],
                o["microprice_offset_ticks"])

    def soa_in(self) -> tuple:
        d = self.d
        return (d["bid_px"], d["bid_sz"], d["ask_px"], d["ask_sz"], self.d_prod, self.d_tick)

    # -- launches ---------------------------------------------------------- #

    def v0(self) -> None:
        k, o, g = self.k, self.out, (self.grid1,)
        k["v0_spread"](g, (BLOCK,), (self.d_aos, self.d_prod, self.d_tick, self.n64, o["spread_ticks"], o["spread_usd"]))
        k["v0_depth"](g, (BLOCK,), (self.d_aos, self.n64, o["depth_bid"], o["depth_ask"]))
        k["v0_imbalance"](g, (BLOCK,), (self.d_aos, self.n64, o["imbalance10"]))
        k["v0_microprice"](g, (BLOCK,), (self.d_aos, self.n64, o["microprice_offset_ticks"]))

    def v1(self) -> None:
        k, o, d, g = self.k, self.out, self.d, (self.grid1,)
        k["v1_spread"](g, (BLOCK,), (d["bid_px"], d["ask_px"], self.d_prod, self.d_tick, self.n64,
                                     o["spread_ticks"], o["spread_usd"]))
        k["v1_depth"](g, (BLOCK,), (d["bid_sz"], d["ask_sz"], self.n64, self.ns64, o["depth_bid"], o["depth_ask"]))
        k["v1_imbalance"](g, (BLOCK,), (d["bid_sz"], d["ask_sz"], self.n64, self.ns64, o["imbalance10"]))
        k["v1_microprice"](g, (BLOCK,), (d["bid_px"], d["bid_sz"], d["ask_px"], d["ask_sz"], self.n64,
                                         o["microprice_offset_ticks"]))

    def v2(self) -> None:
        self.k["v2_fused"]((self.grid1,), (BLOCK,), (*self.soa_in(), self.n64, self.ns64, *self.outs()))

    def v3(self) -> None:
        self.k["v3_fused_vec4"]((self.grid3,), (BLOCK,), (*self.soa_in(), self.ns64, *self.outs()))

    def v4(self, block: Optional[int] = None, grid: Optional[int] = None) -> None:
        b, g = (block, grid) if block else self.v4_config
        self.k["v4_fused_vec4_gs"]((g,), (b,), (*self.soa_in(), self.ns64, *self.outs()))

    def launcher(self, rung: str) -> Callable[[], None]:
        return {"v0_aos": self.v0, "v1_soa": self.v1, "v2_soa_fused": self.v2, "v3_soa_fused_vec4": self.v3,
                "v4_soa_fused_vec4_sweep": self.v4}[rung]

    # -- end to end ---------------------------------------------------------- #

    def upload_soa(self, pinned: bool = False) -> None:
        """Host-to-device copy of exactly what the SoA kernels read."""
        if pinned:
            src = (self.pin["bid_sz"], self.pin["ask_sz"], self.pin["bid_px0"], self.pin["ask_px0"], self.pin["product"])
        else:
            src = (self.hp["bid_sz"], self.hp["ask_sz"], self.hp["bid_px"][0], self.hp["ask_px"][0], self.hp_prod)
        self.d["bid_sz"].set(src[0])
        self.d["ask_sz"].set(src[1])
        self.d["bid_px"][0].set(src[2])
        self.d["ask_px"][0].set(src[3])
        self.d_prod.set(src[4])

    def download(self, pinned: bool = False) -> None:
        dst = self.pin_out if pinned else self.h_out
        for name in ref.OUTPUTS:
            self.out[name].get(out=dst[name])

    def e2e(self, rung: str, pinned: bool = False) -> Callable[[], None]:
        launch = self.launcher(rung)
        if rung == "v0_aos":
            def run() -> None:
                self.d_aos.set(self.pin["aos"] if pinned else self.h_aos)
                self.d_prod.set(self.pin["product"] if pinned else self.hp_prod)
                launch()
                self.download(pinned)
        else:
            def run() -> None:
                self.upload_soa(pinned)
                launch()
                self.download(pinned)
        return run

    # -- verification -------------------------------------------------------- #

    def poison(self) -> None:
        for name, arr in self.out.items():
            arr.fill(-7)

    def outputs_host(self) -> Dict[str, np.ndarray]:
        self.stream.synchronize()
        return {name: self.cp.asnumpy(arr) for name, arr in self.out.items()}


def triton_rung(bench: Bench, lad: "Ladder", reference: dict, n: int, quick: bool):
    """Set up, sweep and verify the Triton rung; ``(None, info)`` when Triton is unavailable or fails."""
    from bench import triton_features as tf

    info: dict = {"attempted": True}
    module, version = tf.load(install=True)
    if module is None:
        info.update({"ok": False, "reason": version})
        log("Triton rung skipped:", version)
        return None, info
    info["version"] = version
    try:
        tr = tf.TritonRung(lad)
        lad.poison()
        lad.v2()
        v2_out = lad.outputs_host()
        configs = tf.QUICK_CONFIGS if quick else tf.CONFIGS
        for cfg in configs:  # compile every configuration before anything is timed
            tr.launch(cfg)
        lad.stream.synchronize()
        sweep = bench.block("ladder_triton_sweep", {f"b{b}_w{w}": (lambda b=b, w=w: event_time_ms(lad.stream, tr.launcher((b, w))))
                                                    for b, w in configs})
        table = [{"block_size": b, "num_warps": w, "median_ms": sweep[f"b{b}_w{w}"]["median_ms"],
                  "iqr_ms": sweep[f"b{b}_w{w}"]["iqr_ms"]} for b, w in configs]
        best = min(table, key=lambda t: t["median_ms"])
        tr.config = (best["block_size"], best["num_warps"])
        lad.poison()
        tr.launch()
        got = lad.outputs_host()
        ver = ref.compare(got, reference, n)
        same = {name: bool(np.array_equal(got[name][:n].view(np.uint32) if got[name].dtype.itemsize == 4 else got[name][:n],
                                          v2_out[name][:n].view(np.uint32) if v2_out[name].dtype.itemsize == 4 else v2_out[name][:n]))
                for name in ref.OUTPUTS}
        info.update({"ok": bool(ver["verified"]), "verification": ver, "identical_to_v2": same,
                     "identical_to_v2_all": all(same.values()), "interop": tr.interop, "sweep": table,
                     "stream": {"triton": tr.stream_ptr, "ladder": tr.ladder_stream_ptr},
                     "best": best, "kernel_mirrors": "v2_fused (cuda/features.cu)"})
        log(f"Triton {version}: verified {ver['verified']}, identical to v2 {all(same.values())}, best {best}")
        return tr, info
    except Exception as exc:  # noqa: BLE001 - an extra rung must never cost the required ones
        info.update({"ok": False, "error": repr(exc)[:1500]})
        log("Triton rung failed:", repr(exc)[:300])
        return None, info


def run(bench: Bench, arch: dict, quick: bool = False) -> None:
    cp = cupy()
    t0 = time.perf_counter()
    bw = measure_bandwidth(bench, quick)
    ceiling, theoretical = bw["ceiling_gbs"], bw["theoretical_gbs"]

    n_total = int(arch["bid_px"].shape[1])
    n = min(n_total, QUICK_N) if quick else n_total
    lad = Ladder(arch, n)
    L = lad.levels
    log(f"n={n:,} levels={L} n_stride={lad.n_stride:,}")
    reference = ref.features_reference(lad.h["bid_px"], lad.h["bid_sz"], lad.h["ask_px"], lad.h["ask_sz"],
                                       lad.h_prod, lad.tick)
    min_io = ref.bytes_min_io(n, L)
    flops = ref.FLOPS_PER_SNAPSHOT * n
    intensity = flops / min_io

    # 1. verify every rung before timing
    verification: Dict[str, dict] = {}
    for name, _label, _ in RUNGS:
        lad.poison()
        lad.launcher(name)()
        verification[name] = ref.compare(lad.outputs_host(), reference, n)
        log(f"verify {name}: {verification[name]['verified']} max_abs {verification[name]['max_abs_err']}")

    # 2. launch-configuration sweep for v4 (grid-stride), every configuration interleaved
    blocks = QUICK_SWEEP_BLOCKS if quick else SWEEP_BLOCKS
    mults = QUICK_SWEEP_GRID_MULTIPLES if quick else SWEEP_GRID_MULTIPLES
    configs = [(b, 40 * m) for b in blocks for m in mults]
    sweep_variants = {f"b{b}_g{g}": (lambda b=b, g=g: event_time_ms(lad.stream, lambda: lad.v4(b, g)))
                      for b, g in configs}
    sweep_res = bench.block("ladder_v4_sweep", sweep_variants)
    table = []
    for b, g in configs:
        m = sweep_res[f"b{b}_g{g}"]
        table.append({"block_size": b, "grid": g, "grid_stride": True, "median_ms": m["median_ms"],
                      "iqr_ms": m["iqr_ms"], "effective_bw_gbs": effective_bandwidth(min_io, m["median_ms"]),
                      "throttle_flag": m.get("throttle_flag")})
    best = ref.pick_best(table)
    lad.v4_config = (best["block_size"], best["grid"])
    lad.poison()
    lad.v4()
    verification["v4_soa_fused_vec4_sweep"] = ref.compare(lad.outputs_host(), reference, n)
    log(f"sweep best: block {best['block_size']} grid {best['grid']} {best['median_ms']:.4f} ms")

    # 2b. extra rung: the fused kernel in Triton, if Triton runs here (never the best CUDA rung)
    tr, triton_info = triton_rung(bench, lad, reference, n, quick)
    if tr is not None:
        verification["triton_fused"] = triton_info["verification"]

    # 3. baselines that need preparation: C++ library and the CuPy chain (verified first)
    baselines: dict = {}
    lib = None
    try:
        lib = build_cpu_library()
    except Exception as exc:  # noqa: BLE001 - recorded, ladder still runs
        baselines["cpp_error"] = repr(exc)
        log("C++ baseline unavailable:", exc)
    cpu_out = {name: np.empty(n, dtype=ref.OUTPUT_DTYPES[name]) for name in ref.OUTPUTS}
    threads_all = os.cpu_count() or 1

    def cpp(threads: int) -> None:
        h = lad.h
        rc = lib.features_cpu(h["bid_px"].ctypes.data, h["bid_sz"].ctypes.data, h["ask_px"].ctypes.data,
                              h["ask_sz"].ctypes.data, lad.h_prod.ctypes.data, lad.tick.ctypes.data,
                              ctypes.c_int64(n), ctypes.c_int32(L), ctypes.c_int32(threads),
                              *(cpu_out[name].ctypes.data for name in ref.OUTPUTS))
        if rc < 0:
            raise RuntimeError(f"features_cpu returned {rc}")

    def timed_cpu(fn: Callable[[], None]) -> Callable[[], float]:
        def go() -> float:
            t = time.perf_counter()
            fn()
            return (time.perf_counter() - t) * 1e3
        return go

    if lib is not None:
        cpp(1)
        baselines["cpp_single_verify"] = ref.compare(cpu_out, reference, n)
        cpp(threads_all)
        baselines["cpp_omp_verify"] = ref.compare(cpu_out, reference, n)
    np_out = ref.features_numpy(lad.h["bid_px"], lad.h["bid_sz"], lad.h["ask_px"], lad.h["ask_sz"], lad.h_prod, lad.tick)
    baselines["numpy_verify"] = ref.compare(np_out, reference, n)
    del np_out
    dpad = lad.d

    def cupy_kernel() -> None:
        cupy_features(dpad["bid_px"], dpad["bid_sz"], dpad["ask_px"], dpad["ask_sz"], lad.d_prod, lad.d_tick)

    def cupy_e2e() -> None:
        bsz = cp.asarray(lad.hp["bid_sz"])
        asz = cp.asarray(lad.hp["ask_sz"])
        bpx0 = cp.asarray(lad.hp["bid_px"][:1])
        apx0 = cp.asarray(lad.hp["ask_px"][:1])
        prod = cp.asarray(lad.hp_prod)
        res = cupy_features(bpx0, bsz, apx0, asz, prod, lad.d_tick)
        for name in ref.OUTPUTS:
            res[name].get(out=lad.h_out[name])

    cupy_res = {k: cp.asnumpy(v) for k, v in cupy_features(dpad["bid_px"], dpad["bid_sz"], dpad["ask_px"],
                                                           dpad["ask_sz"], lad.d_prod, lad.d_tick).items()}
    baselines["cupy_verify"] = ref.compare(cupy_res, reference, n)
    del cupy_res

    # 4. per-kernel block: every rung plus the CuPy chain, CUDA events around the launches only
    kernel_variants = {name: (lambda name=name: event_time_ms(lad.stream, lad.launcher(name))) for name, _, _ in RUNGS}
    kernel_variants["cupy_chain"] = lambda: event_time_ms(lad.stream, cupy_kernel)
    if tr is not None:
        kernel_variants["triton_fused"] = lambda: event_time_ms(lad.stream, tr.launch)
    kres = bench.block("ladder_kernel", kernel_variants)

    # 5. end-to-end block: every rung with transfers, CuPy with transfers, NumPy and C++ (wall clock)
    e2e_variants = {f"{name}_e2e": (lambda name=name: wall_time_ms(lad.e2e(name, pinned=True))) for name, _, _ in RUNGS}
    e2e_variants.update({f"{name}_e2e_pageable": (lambda name=name: wall_time_ms(lad.e2e(name, pinned=False)))
                         for name, _, _ in RUNGS})
    e2e_variants["cupy_e2e"] = lambda: wall_time_ms(cupy_e2e)
    if tr is not None:
        def triton_e2e(pinned: bool) -> Callable[[], None]:
            def go() -> None:
                lad.upload_soa(pinned)
                tr.launch()
                lad.download(pinned)
            return go
        e2e_variants["triton_fused_e2e"] = lambda: wall_time_ms(triton_e2e(True))
        e2e_variants["triton_fused_e2e_pageable"] = lambda: wall_time_ms(triton_e2e(False))
    e2e_variants["numpy"] = timed_cpu(lambda: ref.features_numpy(lad.h["bid_px"], lad.h["bid_sz"], lad.h["ask_px"],
                                                                 lad.h["ask_sz"], lad.h_prod, lad.tick))
    if lib is not None:
        e2e_variants["cpp_single"] = timed_cpu(lambda: cpp(1))
        e2e_variants["cpp_omp"] = timed_cpu(lambda: cpp(threads_all))
    eres = bench.block("ladder_e2e_and_cpu", e2e_variants)

    # 6. assemble
    rungs = []
    for name, label, launches in RUNGS:
        km = kres[name]
        eff = effective_bandwidth(min_io, km["median_ms"])
        block_size, grid = ((lad.v4_config[0], lad.v4_config[1]) if name.startswith("v4")
                            else (BLOCK, lad.grid3 if name.startswith("v3") else lad.grid1))
        rungs.append({
            "name": name, "label": label, "kernel_ms": km, "e2e_ms": eres[f"{name}_e2e"],
            "e2e_pageable_ms": eres[f"{name}_e2e_pageable"],
            "effective_bw_gbs": eff, "pct_of_ceiling": eff / ceiling * 100.0,
            "pct_of_theoretical": eff / theoretical * 100.0, "block_size": int(block_size), "grid": int(grid),
            "launches_per_pass": launches, "verified": verification[name]["verified"],
            "max_abs_err": verification[name]["max_abs_err"], "violations": verification[name]["violations"],
            "arithmetic_intensity_flop_per_byte": intensity,
            "gflops": flops / (km["median_ms"] * 1e-3) / 1e9,
            "e2e_bytes_h2d": int(lad.h_aos.nbytes + lad.hp_prod.nbytes) if name == "v0_aos" else
            int(lad.hp["bid_sz"].nbytes + lad.hp["ask_sz"].nbytes + 2 * 4 * lad.n_stride + lad.hp_prod.nbytes),
        })
    if tr is not None:
        km = kres["triton_fused"]
        eff = effective_bandwidth(min_io, km["median_ms"])
        rungs.append({
            "name": "triton_fused", "label": f"Triton {triton_info.get('version')}, fused (extra rung)", "extra": True,
            "kernel_ms": km, "e2e_ms": eres["triton_fused_e2e"], "e2e_pageable_ms": eres["triton_fused_e2e_pageable"],
            "effective_bw_gbs": eff, "pct_of_ceiling": eff / ceiling * 100.0, "pct_of_theoretical": eff / theoretical * 100.0,
            "block_size": int(tr.config[0]), "num_warps": int(tr.config[1]), "grid": int(math.ceil(n / tr.config[0])),
            "launches_per_pass": 1, "verified": verification["triton_fused"]["verified"],
            "max_abs_err": verification["triton_fused"]["max_abs_err"], "violations": verification["triton_fused"]["violations"],
            "arithmetic_intensity_flop_per_byte": intensity, "gflops": flops / (km["median_ms"] * 1e-3) / 1e9,
            "e2e_bytes_h2d": int(lad.hp["bid_sz"].nbytes + lad.hp["ask_sz"].nbytes + 2 * 4 * lad.n_stride + lad.hp_prod.nbytes),
            "block_size_meaning": "snapshots per Triton program",
        })
    # the headline compares CUDA rungs; the Triton rung is reported beside them, win or lose
    best_rung = min((r for r in rungs if not r.get("extra")), key=lambda r: r["kernel_ms"]["median_ms"])
    bk, be = best_rung["kernel_ms"]["median_ms"], best_rung["e2e_ms"]["median_ms"]
    bep = best_rung["e2e_pageable_ms"]["median_ms"]
    speedups = {"best_rung": best_rung["name"],
                "e2e_vs_numpy": eres["numpy"]["median_ms"] / be,
                "e2e_vs_numpy_pageable": eres["numpy"]["median_ms"] / bep,
                "kernel_vs_cupy": kres["cupy_chain"]["median_ms"] / bk,
                "e2e_vs_cupy": eres["cupy_e2e"]["median_ms"] / bep,
                "v0_to_best_kernel": rungs[0]["kernel_ms"]["median_ms"] / bk}
    if lib is not None:
        speedups["kernel_vs_cpp_single"] = eres["cpp_single"]["median_ms"] / bk
        speedups["kernel_vs_cpp_omp"] = eres["cpp_omp"]["median_ms"] / bk
        speedups["e2e_vs_cpp_single"] = eres["cpp_single"]["median_ms"] / be

    bench.section("ladder", {
        "n_snapshots": n, "n_snapshots_archive": n_total, "levels": L, "n_stride": lad.n_stride, "quick": bool(quick),
        "bytes_min_io": min_io, "bytes_min_io_per_snapshot": min_io // n, "flops_per_snapshot": ref.FLOPS_PER_SNAPSHOT,
        "rungs": rungs,
        "sweep": {"rung": "v4_soa_fused_vec4_sweep", "table": table,
                  "best": {"block_size": best["block_size"], "grid": best["grid"], "median_ms": best["median_ms"]}},
        "speedups": speedups, "outputs": list(ref.OUTPUTS), "triton": triton_info,
        "ceiling_gbs": ceiling, "theoretical_gbs": theoretical,
        "timing": {"kernel": "CUDA events around the launch(es) of one full pass, no transfers",
                   "e2e": "wall clock: host-to-device copy of exactly the inputs the kernels read from pinned host "
                          "buffers, the launch(es), and the device-to-host copy of the six outputs into pinned buffers",
                   "e2e_pageable": "the same from pageable NumPy memory (cudaMemcpy stages through a bounce buffer)",
                   "e2e_vs_cupy": "both from pageable memory"},
        "wall_seconds": time.perf_counter() - t0,
    })
    bench.section("baselines", {
        "cpp_single": {"ms": eres.get("cpp_single"), "compiler": "g++ -O3 -march=native", "gxx": gxx_version(),
                       "threads": 1, "verified": baselines.get("cpp_single_verify", {}).get("verified", False)},
        "cpp_omp": {"ms": eres.get("cpp_omp"), "compiler": "g++ -O3 -march=native -fopenmp", "threads": threads_all,
                    "verified": baselines.get("cpp_omp_verify", {}).get("verified", False)},
        "numpy": {"ms": eres["numpy"], "numpy_version": np.__version__, "threads": 1,
                  "verified": baselines["numpy_verify"]["verified"]},
        "cupy": {"kernel_ms": kres["cupy_chain"], "e2e_ms": eres["cupy_e2e"], "cupy_version": cp.__version__,
                 "n_elementwise_ops": CUPY_OPS, "verified": baselines["cupy_verify"]["verified"]},
        "verification": {k: v for k, v in baselines.items() if k.endswith("_verify")},
        "cpp_error": baselines.get("cpp_error"),
        "same_machine": True, "same_archive_hash": (arch["meta"] or {}).get("content_hash"),
    })
    log(f"best rung {best_rung['name']}: kernel {bk:.4f} ms ({best_rung['effective_bw_gbs']:.1f} GB/s, "
        f"{best_rung['pct_of_ceiling']:.1f}% of ceiling), e2e {be:.2f} ms; speedups {speedups}")
    del lad
    cp.get_default_memory_pool().free_all_blocks()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 3: feature kernel ladder")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", required=True, choices=("synthetic", "real"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    arch = load_archive(Path(args.archive))
    bench = Bench(Path(args.out), args.source, arch["meta"])
    run(bench, arch, quick=args.quick)


if __name__ == "__main__":
    main()
