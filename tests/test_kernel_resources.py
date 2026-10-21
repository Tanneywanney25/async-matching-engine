"""Kernel resource parsing, the occupancy calculator and the generated occupancy host program (CPU only)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from bench import bench_resources as br
from bench import kernel_resources as kr

NVRTC_LOG = """ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'slippage_warp' for 'sm_75'
ptxas info    : Function properties for slippage_warp
ptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 48 registers, used 0 barriers, 10240 bytes smem, 456 bytes cmem[0], 8 bytes cmem[2]
ptxas info    : Compile time = 49.957 ms
ptxas info    : Compiling entry function 'slippage_naive' for 'sm_75'
ptxas info    : Function properties for slippage_naive
ptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 38 registers, used 0 barriers, 456 bytes cmem[0]
"""

NVCC_LOG = """ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'heavy' for 'sm_75'
ptxas info    : Function properties for _Z6helperPd
    16 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads
ptxas info    : Function properties for heavy
    40 bytes stack frame, 24 bytes spill stores, 32 bytes spill loads
ptxas info    : Used 255 registers, 2048 bytes smem, 380 bytes cmem[0]
"""


def test_parse_ptxas_reads_nvrtc_and_nvcc_layouts() -> None:
    k = kr.parse_ptxas(NVRTC_LOG)
    assert set(k) == {"slippage_warp", "slippage_naive"}
    assert k["slippage_warp"]["registers"] == 48 and k["slippage_warp"]["static_smem_bytes"] == 10240
    assert k["slippage_warp"]["cmem"] == {"cmem[0]": 456, "cmem[2]": 8}
    assert k["slippage_naive"]["registers"] == 38 and k["slippage_naive"]["static_smem_bytes"] == 0
    h = kr.parse_ptxas(NVCC_LOG)["heavy"]
    assert (h["registers"], h["spill_stores"], h["spill_loads"], h["stack_bytes"]) == (255, 24, 32, 40)
    assert h["callees"]["_Z6helperPd"]["spill_stores"] == 8
    assert h["total_spill_bytes"] == 24 + 32 + 8 + 8


@pytest.mark.parametrize("regs,smem,block,blocks,limited,occ", [
    (48, 10240, 256, 4, "warps", 1.0),          # slippage_warp: warp slots bind before registers or shared memory
    (38, 0, 256, 4, "warps", 1.0),
    (128, 0, 256, 2, "registers", 0.5),        # 4096 registers per warp: 4 warps per sub-partition
    (32, 49152, 128, 1, "shared_memory", 0.125),
    (16, 0, 32, 16, "blocks", 0.5),            # 16-block cap with one-warp blocks
    (64, 0, 1024, 1, "registers", 1.0),        # 2048 per warp, 32 warps fit exactly; ties with warps
])
def test_occupancy_matches_the_calculator(regs, smem, block, blocks, limited, occ) -> None:
    r = kr.occupancy(regs, smem, block)
    assert r["blocks_per_sm"] == blocks
    assert limited in r["limited_by"]
    assert r["occupancy"] == pytest.approx(occ)


def test_occupancy_zero_when_a_block_cannot_fit() -> None:
    assert kr.occupancy(255, 0, 1024)["blocks_per_sm"] == 0


def test_parse_cuobjdump_and_cub_log() -> None:
    dump = """
Resource usage:
 Common:
  GLOBAL:0
 Function _ZN3cub6detail4scan16DeviceScanKernelIdEEvPd:
  REG:40 STACK:0 SHARED:6152 LOCAL:0 CONSTANT[0]:528 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN3cub20DeviceScanInitKernelIdEEvi:
  REG:12 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:364 TEXTURE:0 SURFACE:0 SAMPLER:0
"""
    parsed = kr.parse_cuobjdump(dump)
    assert parsed["_ZN3cub6detail4scan16DeviceScanKernelIdEEvPd"]["REG"] == 40
    assert parsed["_ZN3cub20DeviceScanInitKernelIdEEvi"]["SHARED"] == 0
    log = ("Invoking init_kernel<<<7, 128, 0, 0>>>()\n"
           "Invoking 0 scan_kernel<<<844, 128, 0, 0>>>(), 8 items per thread, 6 SM occupancy\n")
    launches = kr.parse_cub_log(log)
    assert launches[0] == {"kernel": "init_kernel", "grid": 7, "block": 128}
    assert launches[1]["items_per_thread"] == 8 and launches[1]["cub_blocks_per_sm"] == 6


def test_primary_config_prefers_the_sweep_best() -> None:
    cfgs = [{"grid": [40, 1, 1], "block": [64, 1, 1], "launches": 33},
            {"grid": [80, 1, 1], "block": [128, 1, 1], "launches": 33},
            {"grid": [80, 1, 1], "block": [128, 1, 1], "launches": 66}]
    assert kr.primary_config(cfgs, prefer_block=128)["launches"] == 66
    assert kr.primary_config(cfgs[:2])["block"][0] == 64  # tie on launches: the smaller block
    assert kr.block_sizes(cfgs) == [64, 128]


def test_kernel_files_finds_every_entry_point() -> None:
    files = br.kernel_files()
    assert "scan_cub.cu" not in files  # host code, compiled into a shared library
    assert {"slippage_naive", "slippage_warp"} <= set(files["slippage.cu"])
    assert "dd_block_scan" in files["ddscan.cu"]  # name on the line after __launch_bounds__


STUB_RUNTIME = """#pragma once
#include <cstddef>
#define __global__
#define __launch_bounds__(...)
typedef int cudaError_t;
struct cudaFuncAttributes { int numRegs; size_t sharedSizeBytes, localSizeBytes, constSizeBytes; int maxThreadsPerBlock, binaryVersion, ptxVersion; };
struct cudaDeviceProp { char name[256]; int regsPerMultiprocessor, maxThreadsPerMultiProcessor, sharedMemPerMultiprocessor,
                        maxBlocksPerMultiProcessor, reservedSharedMemPerBlock, warpSize, multiProcessorCount; };
inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, const void*) { a->numRegs = 48; a->sharedSizeBytes = 10240; return 0; }
inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, const void*, int block, size_t) { *n = 1024 / block; return 0; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { p->name[0] = 'T'; p->name[1] = '4'; p->name[2] = 0;
  p->regsPerMultiprocessor = 65536; p->maxThreadsPerMultiProcessor = 1024; p->sharedMemPerMultiprocessor = 65536;
  p->maxBlocksPerMultiProcessor = 16; p->reservedSharedMemPerBlock = 0; p->warpSize = 32; p->multiProcessorCount = 40; return 0; }
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
def test_generated_host_program_compiles_and_prints_json(tmp_path: Path) -> None:
    (tmp_path / "cuda_runtime.h").write_text(STUB_RUNTIME, encoding="utf-8")
    (tmp_path / "k.cu").write_text('extern "C" __global__ void k1(int* x) { (void)x; }\n'
                                   'extern "C" __global__ void __launch_bounds__(256) k2(float* y) { (void)y; }\n', encoding="utf-8")
    src = tmp_path / "occ.cpp"
    src.write_text(br.host_program("k.cu", {"k1": [128, 256], "k2": []}), encoding="utf-8")
    exe = tmp_path / "occ"
    # nvcc includes cuda_runtime.h before any source; -include does the same for g++
    proc = subprocess.run(["g++", "-std=c++17", "-include", "cuda_runtime.h", "-I", str(tmp_path), "-o", str(exe), str(src)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(ln) for ln in subprocess.run([str(exe)], capture_output=True, text=True).stdout.splitlines()]
    assert rows[0]["device"] == "T4" and rows[0]["regsPerMultiprocessor"] == 65536
    k1 = next(r for r in rows if r.get("kernel") == "k1")
    assert k1["numRegs"] == 48 and [o["blocks_per_sm"] for o in k1["occupancy"]] == [8, 4]
    assert next(r for r in rows if r.get("kernel") == "k2")["occupancy"] == []


def test_kernel_rows_join_ptxas_api_and_launches() -> None:
    launches = {"slippage_warp": {"sources": ["slippage.cu"], "options": [],
                                  "configs": [{"grid": [3375, 1, 1], "block": [256, 1, 1], "shared_mem": 0, "launches": 40}]}}
    per_file = {"slippage.cu": {"ptxas": kr.parse_ptxas(NVRTC_LOG), "device": {"regsPerMultiprocessor": 65536},
                                "api": {"slippage_warp": {"numRegs": 48, "sharedSizeBytes": 10240, "localSizeBytes": 0,
                                                          "occupancy": [{"block": 256, "blocks_per_sm": 4, "err": 0}]}}}}
    rows = br.kernel_rows(launches, per_file, {"slippage_warp": {"num_regs": 48, "shared_size_bytes": 10240}}, {})
    assert len(rows) == 1  # slippage_naive was not launched, so it has no row
    row = rows[0]
    assert row["registers"] == 48 and row["nvrtc_matches_nvcc"] and row["primary_block"] == 256
    assert row["occupancy"][0]["calculator_agrees"] and row["occupancy"][0]["limited_by"] == ["warps"]
