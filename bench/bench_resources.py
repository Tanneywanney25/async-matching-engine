"""Section ``resources``: registers, spills, shared memory and theoretical occupancy per kernel.

No hardware counters are needed (Nsight Compute is blocked on Kaggle). Runs after every other
section in the same job, because ``bench.common.raw_module`` records each kernel's launch
configurations as the benchmarks run (in this process and in the Nsight Systems children, which
write ``raw/launches/launches_<pid>.json``). Then, for every ``cuda/*.cu`` file:

1. ``nvcc -cubin -arch=sm_75 -Xptxas -v`` gives each entry function's registers per thread, spill
   stores and loads, stack frame and static shared memory;
2. a generated host program (``#include "<file>.cu"``) prints ``cudaFuncGetAttributes`` and
   ``cudaOccupancyMaxActiveBlocksPerMultiprocessor`` for every block size this job launched;
3. the NVRTC binary that was actually benchmarked (the CuPy kernel's ``num_regs`` and
   ``shared_size_bytes``) is compared with nvcc's counts, and :func:`bench.kernel_resources.occupancy`
   names the limiting resource and must agree with the API's block count.

CUB's scan kernels are compiled into ``build/libscan_cub.so``: their resources come from
``cuobjdump --dump-resource-usage`` on that library, their launch configuration and the occupancy
CUB itself computes (with ``cudaOccupancyMaxActiveBlocksPerMultiprocessor``) from a
``-DCUB_DEBUG_LOG`` build of the same source run at both scan sizes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from bench import kernel_resources as kr
from bench.common import LAUNCHES, REPO_ROOT, Bench, merge_launch_logs

CUDA_DIR = REPO_ROOT / "cuda"
BUILD = REPO_ROOT / "build" / "resources"
NVCC_FLAGS = ["-O3", "-arch=sm_75", "-I", str(CUDA_DIR)]
HOST_PROPS = ("regsPerMultiprocessor", "maxThreadsPerMultiProcessor", "sharedMemPerMultiprocessor",
              "maxBlocksPerMultiProcessor", "reservedSharedMemPerBlock", "warpSize", "multiProcessorCount")
CUB_SIZES = (863_153, 16_777_216)


def _sh(cmd: List[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)


def kernel_files() -> Dict[str, List[str]]:
    """``{file: [extern "C" kernel names]}`` for every ``cuda/*.cu`` with device entry points."""
    out = {}
    for path in sorted(CUDA_DIR.glob("*.cu")):
        names = re.findall(r'extern "C" __global__ void(?:\s+__launch_bounds__\([^)]*\))?\s+(\w+)\s*\(', path.read_text(encoding="utf-8"))
        if names:
            out[path.name] = names
    return out


def host_program(file: str, kernels: Dict[str, List[int]]) -> str:
    """C++ that prints device limits, then attributes and occupancy for each kernel, as JSON lines."""
    lines = [f'#include "{file}"', "#include <cstdio>", "#include <cuda_runtime.h>", "",
             "static void report(const char* name, const void* fn, const int* blocks, int nb) {",
             "    cudaFuncAttributes a{}; cudaError_t e = cudaFuncGetAttributes(&a, fn);",
             '    std::printf("{\\"kernel\\":\\"%s\\",\\"err\\":%d,\\"numRegs\\":%d,\\"sharedSizeBytes\\":%zu,\\"localSizeBytes\\":%zu,'
             '\\"constSizeBytes\\":%zu,\\"maxThreadsPerBlock\\":%d,\\"binaryVersion\\":%d,\\"ptxVersion\\":%d,\\"occupancy\\":[",',
             "                name, (int)e, a.numRegs, a.sharedSizeBytes, a.localSizeBytes, a.constSizeBytes, a.maxThreadsPerBlock,",
             "                a.binaryVersion, a.ptxVersion);",
             "    for (int i = 0; i < nb; ++i) {",
             "        int n = 0; cudaError_t oe = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, blocks[i], 0);",
             '        std::printf("%s{\\"block\\":%d,\\"blocks_per_sm\\":%d,\\"err\\":%d}", i ? "," : "", blocks[i], n, (int)oe);',
             "    }",
             '    std::printf("]}\\n");',
             "}", "", "int main() {",
             "    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);",
             '    std::printf("{\\"device\\":\\"%s\\",' + ",".join(f'\\"{k}\\":%d' for k in HOST_PROPS) + '}\\n", p.name, '
             + ", ".join(f"(int)p.{k}" for k in HOST_PROPS) + ");"]
    for name, blocks in kernels.items():
        arr = ", ".join(str(b) for b in blocks) or "0"
        lines.append(f'    {{ const int b[] = {{{arr}}}; report("{name}", (const void*){name}, b, {len(blocks) if blocks else 0}); }}')
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)


def nvcc_resources(file: str, launched: Dict[str, List[int]], names: List[str]) -> dict:
    """ptxas statistics plus the runtime-API attributes and occupancy for one ``cuda/*.cu``."""
    BUILD.mkdir(parents=True, exist_ok=True)
    stem = Path(file).stem
    res: dict = {"file": file}
    proc = _sh(["nvcc", *NVCC_FLAGS, "-cubin", "-Xptxas", "-v", "-o", str(BUILD / f"{stem}.cubin"), str(CUDA_DIR / file)])
    res["ptxas_log"] = (proc.stdout + proc.stderr)[-20000:]
    res["ptxas"] = kr.parse_ptxas(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        res["error"] = "nvcc -cubin failed: " + (proc.stderr or proc.stdout)[-1500:]
        return res
    src = BUILD / f"occ_{stem}.cu"
    src.write_text(host_program(file, {n: launched.get(n, []) for n in names}), encoding="utf-8")
    exe = BUILD / f"occ_{stem}"
    proc = _sh(["nvcc", *NVCC_FLAGS, "-o", str(exe), str(src)])
    if proc.returncode != 0:
        res["error"] = "nvcc host program failed: " + (proc.stderr or proc.stdout)[-1500:]
        return res
    proc = _sh([str(exe)], timeout=300)
    rows = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.startswith("{")]
    res["device"] = next((r for r in rows if "device" in r), {})
    res["api"] = {r["kernel"]: r for r in rows if "kernel" in r}
    if proc.returncode != 0:
        res["error"] = f"host program exit {proc.returncode}: {proc.stderr[-800:]}"
    return res


def nvrtc_attributes(kernel) -> dict:
    """Attributes of the NVRTC binary that was benchmarked (CuPy ``RawKernel`` properties)."""
    raw = getattr(kernel, "raw", kernel)
    out = {}
    for attr in ("num_regs", "shared_size_bytes", "local_size_bytes", "const_size_bytes", "max_threads_per_block",
                 "binary_version", "ptx_version"):
        try:
            out[attr] = int(getattr(raw, attr))
        except Exception as exc:  # noqa: BLE001 - attribute names differ across CuPy versions
            out[attr + "_error"] = repr(exc)[:200]
    return out


def benchmarked_attributes(launches: Dict[str, dict]) -> Dict[str, dict]:
    """Recompile each launched kernel's module exactly as ``raw_module`` did and read its attributes."""
    from bench.common import raw_module

    out: Dict[str, dict] = {}
    modules: Dict[tuple, object] = {}
    for name, entry in launches.items():
        key = tuple(entry["sources"])
        try:
            if key not in modules:
                modules[key] = raw_module(list(key))
            out[name] = nvrtc_attributes(modules[key].get_function(name))
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": repr(exc)[:300]}
    return out


def cub_resources(raw_dir: Path) -> dict:
    """CUB scan kernels: cuobjdump resources of the benchmarked library and CUB's own launch log."""
    lib = REPO_ROOT / "build" / "libscan_cub.so"
    out: dict = {"library": str(lib.relative_to(REPO_ROOT))}
    if not lib.exists():
        from bench.bench_scans import build_cub

        build_cub()
    dump = _sh(["cuobjdump", "--dump-resource-usage", str(lib)])
    names = kr.parse_cuobjdump(dump.stdout)
    demangled = {}
    if names and shutil.which("c++filt"):
        proc = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, timeout=60)
        demangled = dict(zip(names, proc.stdout.splitlines()))
    kernels = {}
    for mangled, fields in names.items():
        pretty = demangled.get(mangled, mangled)
        if "double" not in pretty:  # the float instantiations exist but were not benchmarked
            continue
        kind = "scan_kernel" if "DeviceScanKernel" in pretty else "init_kernel" if "DeviceScanInitKernel" in pretty else pretty[:60]
        kernels[kind] = {"mangled": mangled, "demangled": pretty[:400], "registers": fields.get("REG"),
                         "static_smem_bytes": fields.get("SHARED"), "stack_bytes": fields.get("STACK"),
                         "local_bytes": fields.get("LOCAL")}
    out["kernels"] = kernels
    (raw_dir / "resources").mkdir(parents=True, exist_ok=True)
    (raw_dir / "resources" / "cub_cuobjdump.txt").write_text(dump.stdout[-200000:], encoding="utf-8")
    main = BUILD / "cub_log_main.cu"
    BUILD.mkdir(parents=True, exist_ok=True)
    main.write_text("\n".join([
        "#include <cstdio>", "#include <cstdlib>", "#include <cuda_runtime.h>",
        'extern "C" int cub_inclusive_sum_f64(const double* in, double* out, long long n, void* stream);',
        "int main(int argc, char** argv) {",
        "    for (int i = 1; i < argc; ++i) {",
        "        long long n = std::atoll(argv[i]); double *a, *b;",
        "        cudaMalloc(&a, n * sizeof(double)); cudaMalloc(&b, n * sizeof(double)); cudaMemset(a, 0, n * sizeof(double));",
        '        std::printf("SIZE %lld\\n", n); std::fflush(stdout);',
        "        int rc = cub_inclusive_sum_f64(a, b, n, nullptr); cudaDeviceSynchronize();",
        '        std::printf("RC %d\\n", rc); std::fflush(stdout); cudaFree(a); cudaFree(b);',
        "    }", "    return 0;", "}", ""]), encoding="utf-8")
    exe = BUILD / "cub_log"
    proc = _sh(["nvcc", "-O3", "-arch=sm_75", "-DCUB_DEBUG_LOG", "-o", str(exe), str(REPO_ROOT / "cuda" / "scan_cub.cu"), str(main)])
    if proc.returncode != 0:
        out["log_error"] = (proc.stderr or proc.stdout)[-1500:]
        return out
    proc = _sh([str(exe), *map(str, CUB_SIZES)], timeout=300)
    (raw_dir / "resources" / "cub_debug_log.txt").write_text(proc.stdout[-200000:], encoding="utf-8")
    per_size, current = {}, None
    for line in proc.stdout.splitlines():
        if line.startswith("SIZE "):
            current = int(line.split()[1])
            per_size[current] = []
        elif current is not None:
            per_size[current] += kr.parse_cub_log(line)
    for launches in per_size.values():
        for launch in launches:
            k = kernels.get(launch["kernel"])
            if k and k.get("registers") is not None:
                calc = kr.occupancy(k["registers"], k.get("static_smem_bytes") or 0, launch["block"])
                launch["occupancy"] = calc
                if "cub_blocks_per_sm" in launch:
                    launch["calculator_agrees"] = calc["blocks_per_sm"] == launch["cub_blocks_per_sm"]
    out["launches"] = {str(n): v for n, v in per_size.items()}
    return out


def kernel_rows(launches: Dict[str, dict], per_file: Dict[str, dict], nvrtc: Dict[str, dict],
                prefer: Dict[str, int]) -> List[dict]:
    """One row per kernel that this job launched, with every launched block size's occupancy."""
    rows = []
    for file, res in sorted(per_file.items()):
        device = res.get("device") or {}
        hw = {"regs_per_sm": device.get("regsPerMultiprocessor"), "max_threads_per_sm": device.get("maxThreadsPerMultiProcessor"),
              "shared_mem_per_sm": device.get("sharedMemPerMultiprocessor"), "max_blocks_per_sm": device.get("maxBlocksPerMultiProcessor"),
              "reserved_shared_mem_per_block": device.get("reservedSharedMemPerBlock"), "warp_size": device.get("warpSize")}
        hw = {k: v for k, v in hw.items() if v is not None}
        for name, stats in res.get("ptxas", {}).items():
            if name not in launches:
                continue
            api = (res.get("api") or {}).get(name, {})
            configs = launches[name]["configs"]
            primary = kr.primary_config(configs, prefer.get(name))
            occ = []
            api_occ = {o["block"]: o["blocks_per_sm"] for o in api.get("occupancy", []) if o.get("err") == 0}
            for block in kr.block_sizes(configs):
                calc = kr.occupancy(stats["registers"] or 0, stats["static_smem_bytes"], block, 0, hw)
                calc["api_blocks_per_sm"] = api_occ.get(block)
                calc["calculator_agrees"] = api_occ.get(block) == calc["blocks_per_sm"]
                calc["launches"] = sum(c["launches"] for c in configs if c["block"][0] * c["block"][1] * c["block"][2] == block)
                occ.append(calc)
            nv = nvrtc.get(name, {})
            rows.append({
                "kernel": name, "file": file, "registers": stats["registers"], "spill_stores": stats["spill_stores"],
                "spill_loads": stats["spill_loads"], "stack_bytes": stats["stack_bytes"], "total_spill_bytes": stats["total_spill_bytes"],
                "static_smem_bytes": stats["static_smem_bytes"], "barriers": stats["barriers"],
                "api_num_regs": api.get("numRegs"), "api_shared_bytes": api.get("sharedSizeBytes"), "api_local_bytes": api.get("localSizeBytes"),
                "nvrtc_num_regs": nv.get("num_regs"), "nvrtc_shared_bytes": nv.get("shared_size_bytes"), "nvrtc_local_bytes": nv.get("local_size_bytes"),
                "nvrtc_matches_nvcc": nv.get("num_regs") == stats["registers"] and nv.get("shared_size_bytes") == stats["static_smem_bytes"],
                "primary_block": primary["block"][0] if primary else None, "primary_grid": primary["grid"][0] if primary else None,
                "configs": configs, "occupancy": occ,
            })
    return rows


def run(bench: Bench, arch: dict, quick: bool) -> None:
    raw = bench.raw_dir
    logs = [LAUNCHES.snapshot()]
    for path in sorted((raw / "launches").glob("launches_*.json")) if (raw / "launches").exists() else []:
        if path.name != f"launches_{os.getpid()}.json":
            logs.append(json.loads(path.read_text(encoding="utf-8")))
    launches = merge_launch_logs(logs)
    bench.save_raw("resources_launches", launches)
    files = kernel_files()
    launched_blocks = {name: kr.block_sizes(e["configs"]) for name, e in launches.items()}
    per_file = {f: nvcc_resources(f, launched_blocks, names) for f, names in files.items()}
    (raw / "resources").mkdir(parents=True, exist_ok=True)
    for f, res in per_file.items():
        (raw / "resources" / f"{Path(f).stem}_ptxas.txt").write_text(res.pop("ptxas_log", ""), encoding="utf-8")
    nvrtc = benchmarked_attributes(launches)
    best = (bench.metrics.get("sections", {}).get("ladder", {}).get("sweep", {}).get("best") or {}).get("block_size")
    rows = kernel_rows(launches, per_file, nvrtc, {"v4_fused_vec4_gs": best} if best else {})
    try:
        cub = cub_resources(raw)
    except Exception as exc:  # noqa: BLE001 - CUB is one row; never lose the rest of the table
        cub = {"error": repr(exc)[:500]}
    nvcc_version = _sh(["nvcc", "--version"]).stdout.strip().splitlines()[-2:]
    device = next((r["device"] for r in per_file.values() if r.get("device")), {})
    bench.section("resources", {
        "method": "nvcc -cubin -arch=sm_75 -Xptxas -v; cudaFuncGetAttributes and cudaOccupancyMaxActiveBlocksPerMultiprocessor "
                  "in a host program per file; NVRTC attributes of the benchmarked binaries for comparison",
        "nvcc": nvcc_version, "device": device, "quick": quick,
        "kernels": rows, "not_launched": sorted(n for names in files.values() for n in names if n not in launches),
        "errors": {f: r["error"] for f, r in per_file.items() if r.get("error")},
        "all_nvrtc_match": all(r["nvrtc_matches_nvcc"] for r in rows),
        "all_calculator_agree": all(o["calculator_agrees"] for r in rows for o in r["occupancy"]),
        "cub": cub, "launch_log": "raw/resources_launches.json",
    })
