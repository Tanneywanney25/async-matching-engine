"""Kernel resource facts without hardware counters: ptxas statistics and theoretical occupancy.

Pure functions, importable on a machine with no GPU (the tests run them on CPU):

* :func:`parse_ptxas` reads ``-Xptxas -v`` output (nvcc's or NVRTC's) into per-kernel registers,
  spill stores and loads, stack frame, static shared memory and constant memory;
* :func:`parse_cuobjdump` reads ``cuobjdump --dump-resource-usage`` (used for CUB, whose kernels
  are compiled into ``build/libscan_cub.so`` rather than from a ``cuda/*.cu`` entry point);
* :func:`parse_cub_log` reads the launch lines a ``-DCUB_DEBUG_LOG`` build of CUB prints;
* :func:`occupancy` reproduces the CUDA occupancy calculator for one launch configuration and
  names the resource that binds, so the table can say *why* a kernel sits where it does. The
  GPU job compares its block count with ``cudaOccupancyMaxActiveBlocksPerMultiprocessor``.
* :func:`nvrtc_compile` compiles a source through NVRTC with ptxas verbose output (locally from
  the ``nvidia-cuda-nvrtc-cu12`` wheel, used to compile-check kernels before a GPU job).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Compute capability 7.5 (Tesla T4), from the CUDA occupancy calculator's hardware table. The GPU
# job reads the device-property values (registers, threads, shared memory, blocks per SM) from
# cudaGetDeviceProperties and passes them in; the allocation units are not device properties.
SM75 = {
    "warp_size": 32,
    "max_threads_per_sm": 1024,
    "max_blocks_per_sm": 16,
    "regs_per_sm": 65536,
    "shared_mem_per_sm": 65536,
    "reserved_shared_mem_per_block": 0,
    "reg_alloc_unit": 256,          # registers are allocated per warp in units of 256
    "warp_alloc_granularity": 4,    # warps per SM limited by registers round down to a multiple of 4
    "shared_alloc_unit": 256,       # shared memory per block rounds up to 256 bytes
}

_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '(sm_\d+)'")
_PROPS = re.compile(r"Function properties for (\S+)")
_FRAME = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_USED = re.compile(r"Used (\d+) registers")
_BARRIERS = re.compile(r"used (\d+) barriers")
_BYTES = re.compile(r"(\d+) bytes (smem|cmem\[\d+\]|lmem)")


def parse_ptxas(log: str) -> Dict[str, dict]:
    """``{entry function: {registers, spill_stores, spill_loads, stack_bytes, static_smem_bytes, ...}}``.

    Handles both nvcc's layout and NVRTC's (which prefixes the frame line with ``ptxas .``).
    Frame lines for non-inlined device functions are kept under ``callees`` of the entry being
    compiled, since their spills happen inside that kernel too.
    """
    kernels: Dict[str, dict] = {}
    entry: Optional[str] = None
    props_for: Optional[str] = None
    for line in log.splitlines():
        m = _ENTRY.search(line)
        if m:
            entry = m.group(1)
            kernels[entry] = {"arch": m.group(2), "registers": None, "spill_stores": 0, "spill_loads": 0,
                              "stack_bytes": 0, "static_smem_bytes": 0, "barriers": 0, "cmem": {}, "callees": {}}
            props_for = None
            continue
        m = _PROPS.search(line)
        if m:
            props_for = m.group(1)
            continue
        m = _FRAME.search(line)
        if m and entry is not None:
            frame = {"stack_bytes": int(m.group(1)), "spill_stores": int(m.group(2)), "spill_loads": int(m.group(3))}
            if props_for in (None, entry):
                kernels[entry].update(frame)
            else:
                kernels[entry]["callees"][props_for] = frame
            continue
        m = _USED.search(line)
        if m and entry is not None:
            k = kernels[entry]
            k["registers"] = int(m.group(1))
            b = _BARRIERS.search(line)
            if b:
                k["barriers"] = int(b.group(1))
            for size, kind in _BYTES.findall(line):
                if kind == "smem":
                    k["static_smem_bytes"] = int(size)
                elif kind == "lmem":
                    k["lmem_bytes"] = int(size)
                else:
                    k["cmem"][kind] = int(size)
    for k in kernels.values():
        k["total_spill_bytes"] = k["spill_stores"] + k["spill_loads"] + sum(
            c["spill_stores"] + c["spill_loads"] for c in k["callees"].values())
    return kernels


_CUOBJ_FN = re.compile(r"^\s*Function (\S+):\s*$")
_CUOBJ_FIELD = re.compile(r"([A-Z]+(?:\[\d+\])?):(\d+)")


def parse_cuobjdump(text: str) -> Dict[str, dict]:
    """``cuobjdump --dump-resource-usage``: ``{mangled name: {REG, STACK, SHARED, LOCAL, ...}}``."""
    out: Dict[str, dict] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = _CUOBJ_FN.match(line)
        if m:
            current = m.group(1)
            out[current] = {}
            continue
        if current is not None and "REG:" in line:
            out[current] = {k: int(v) for k, v in _CUOBJ_FIELD.findall(line)}
            current = None
    return out


_CUB_INIT = re.compile(r"Invoking init_kernel<<<(\d+), (\d+), 0, -?\d+>>>")
_CUB_SCAN = re.compile(r"Invoking (\d+) scan_kernel<<<(\d+), (\d+), 0, -?\d+>>>\(\), (\d+) items per thread, (\d+) SM occupancy")


def parse_cub_log(text: str) -> List[dict]:
    """Launch lines from a ``-DCUB_DEBUG_LOG`` build of ``cub::DeviceScan``, in order."""
    launches = []
    for line in text.splitlines():
        m = _CUB_INIT.search(line)
        if m:
            launches.append({"kernel": "init_kernel", "grid": int(m.group(1)), "block": int(m.group(2))})
            continue
        m = _CUB_SCAN.search(line)
        if m:
            launches.append({"kernel": "scan_kernel", "start_tile": int(m.group(1)), "grid": int(m.group(2)),
                             "block": int(m.group(3)), "items_per_thread": int(m.group(4)),
                             "cub_blocks_per_sm": int(m.group(5))})
    return launches


def _ceil_to(x: int, unit: int) -> int:
    return int(math.ceil(x / unit) * unit) if unit else x


def occupancy(registers: int, static_smem: int, block: int, dynamic_smem: int = 0, hw: Optional[dict] = None) -> dict:
    """Theoretical occupancy of one launch configuration on compute capability 7.5.

    Returns the active blocks per SM allowed by each resource, the binding one(s), the resulting
    warps per SM and the occupancy as a fraction of the SM's warp slots.
    """
    h = dict(SM75, **(hw or {}))
    warps_per_block = int(math.ceil(block / h["warp_size"]))
    max_warps = h["max_threads_per_sm"] // h["warp_size"]
    limits: Dict[str, int] = {"blocks": h["max_blocks_per_sm"], "warps": max_warps // warps_per_block if warps_per_block else 0}
    if registers:
        regs_per_warp = _ceil_to(registers * h["warp_size"], h["reg_alloc_unit"])
        warps_by_regs = (h["regs_per_sm"] // regs_per_warp) // h["warp_alloc_granularity"] * h["warp_alloc_granularity"]
        limits["registers"] = warps_by_regs // warps_per_block
    else:
        limits["registers"] = h["max_blocks_per_sm"]
    smem = static_smem + dynamic_smem
    if smem:
        per_block = _ceil_to(smem + h["reserved_shared_mem_per_block"], h["shared_alloc_unit"])
        limits["shared_memory"] = h["shared_mem_per_sm"] // per_block
    else:
        limits["shared_memory"] = h["max_blocks_per_sm"]
    blocks = min(limits.values())
    binding = [k for k in ("registers", "shared_memory", "warps", "blocks") if limits[k] == blocks]
    warps = blocks * warps_per_block
    return {"block": block, "warps_per_block": warps_per_block, "limits": limits, "blocks_per_sm": blocks,
            "active_warps_per_sm": warps, "max_warps_per_sm": max_warps,
            "occupancy": warps / max_warps if max_warps else 0.0, "limited_by": binding}


def primary_config(configs: Sequence[dict], prefer_block: Optional[int] = None) -> Optional[dict]:
    """The configuration a kernel's reported number was measured with.

    Most kernels have one. A swept kernel's reported figure uses the sweep's best block size,
    passed as ``prefer_block``; otherwise the most-launched configuration wins.
    """
    if not configs:
        return None
    if prefer_block is not None:
        hits = [c for c in configs if c["block"][0] == prefer_block]
        if hits:
            return max(hits, key=lambda c: c["launches"])
    return max(configs, key=lambda c: (c["launches"], -c["block"][0]))


def block_sizes(configs: Iterable[dict]) -> List[int]:
    """Distinct total threads per block among ``configs``, ascending."""
    return sorted({int(c["block"][0]) * int(c["block"][1]) * int(c["block"][2]) for c in configs})


# --------------------------------------------------------------------------- #
# NVRTC (compile checks; CPU-only machines included)
# --------------------------------------------------------------------------- #


def find_nvrtc() -> Optional[str]:
    """Path of ``libnvrtc``: ``$NVRTC_LIB``, the ``nvidia-cuda-nvrtc-cu12`` wheel, or the CUDA toolkit."""
    candidates: List[str] = []
    if os.environ.get("NVRTC_LIB"):
        candidates.append(os.environ["NVRTC_LIB"])
    try:
        import nvidia.cuda_nvrtc as wheel  # type: ignore[import-not-found]

        for root in list(getattr(wheel, "__path__", [])):
            candidates += sorted(glob.glob(os.path.join(root, "lib", "libnvrtc.so.*")))
            candidates += sorted(glob.glob(os.path.join(root, "bin", "nvrtc64_*.dll")))
    except ImportError:
        pass
    candidates += sorted(glob.glob("/opt/cuda-pip/nvidia/cuda_nvrtc/lib/libnvrtc.so.*"))
    candidates += sorted(glob.glob("/usr/local/cuda/lib64/libnvrtc.so*"))
    found = ctypes.util.find_library("nvrtc")
    if found:
        candidates.append(found)
    for path in candidates:
        if "builtins" in os.path.basename(path) or ".alt." in os.path.basename(path):
            continue
        try:
            ctypes.CDLL(path)
            return path
        except OSError:
            continue
    return None


def nvrtc_version(lib_path: str) -> Tuple[int, int]:
    lib = ctypes.CDLL(lib_path)
    major, minor = ctypes.c_int(), ctypes.c_int()
    lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    return major.value, minor.value


def nvrtc_compile(source: str, options: Sequence[str], lib_path: Optional[str] = None,
                  name: str = "kernel.cu") -> Tuple[bool, str, int]:
    """Compile ``source`` with NVRTC; returns ``(ok, program log, cubin bytes)``.

    Pass ``-arch=sm_75`` to get a CUBIN (and ptxas statistics with ``--ptxas-options=-v``).
    """
    lib = ctypes.CDLL(lib_path or find_nvrtc() or "libnvrtc.so")
    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), 0, None, None)
    if rc != 0:
        return False, f"nvrtcCreateProgram failed ({rc})", 0
    try:
        opts = [o.encode() for o in options]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
        size = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        lib.nvrtcGetProgramLog(prog, buf)
        cubin = ctypes.c_size_t(0)
        if rc == 0 and lib.nvrtcGetCUBINSize(prog, ctypes.byref(cubin)) != 0:
            cubin = ctypes.c_size_t(0)
        return rc == 0, buf.value.decode(errors="replace"), int(cubin.value)
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
