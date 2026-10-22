"""Phase 7: end-to-end pipeline with pinned memory, chunking and three CUDA streams.

    python -m bench.bench_pipeline --archive data/synth --out results/<run> --source synthetic [--quick]
                                   [--chunks 16] [--subset N] [--int16] [--no-nsys]
    python -m bench.bench_pipeline --nsys-child --archive <dir> [--chunks 16] [--subset N]

The second form is what ``run()`` launches under ``nsys profile``: one warmed-up
overlapped pass wrapped in NVTX ranges.

What is measured (section ``pipeline`` of metrics.json, shape in bench/CONTRACT.md):

* The four level-major archive arrays are copied once into a chunk-major pinned
  host buffer (``cupy.cuda.alloc_pinned_memory`` wrapped as NumPy). Chunk i covers
  snapshots ``[i*n/K, (i+1)*n/K)`` and is one contiguous 256-byte-aligned block
  holding ``bid_px, bid_sz, ask_px, ask_sz`` for those snapshots, each segment
  still level-major inside the chunk, so a chunk is one ``cudaMemcpyAsync`` in
  and one out.
* ``overlapped``: three non-blocking streams (copy in, compute, copy out), triple
  buffered device buffers and ``cupy.cuda.Event`` dependencies: the host-to-device
  copy of chunk i+1 runs while the fused feature kernel (``cuda/pipeline.cu``)
  processes chunk i and the outputs of chunk i-1 are copied back.
* ``serial``: the same chunks on one stream, copy in / compute / copy out
  strictly in order, a single device buffer.

Both are wall-clock times of one pass over the archive, interleaved by
``Bench.block`` (warmup discarded, at least 30 iterations). An instrumented pass
records CUDA events around every chunk's H2D, kernel and D2H and sums them per
stage. The Nsight Systems proof runs one overlapped pass in a child process
under ``nsys profile``, exports the report to SQLite and renders
``raw/pipeline_timeline.svg`` with ``bench/timeline_svg.py``. If ``nsys`` is
missing the section is written without the SVG and says so in ``nsys_error``.

Stretch (``--int16`` or ``PIPELINE_INT16=1``): prices travel as int16 tick
offsets from the best price of their side (host-side NumPy encode per chunk,
checked for overflow with an int32 fallback per chunk) and are decoded inside
the kernel; measured end to end, interleaved with the int32 pass, and recorded
under ``int16_encoding``. Any failure there is recorded and leaves the required
results untouched.

CPU-only machines can import this module (CuPy is imported lazily); the chunk
math, the encode/decode round trip and the NumPy reference are unit-tested in
``tests/test_timeline_svg.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bench import timeline_svg
from bench.common import REPO_ROOT, Bench, load_archive, raw_module, wall_time_ms
from bench.common import cupy as _cupy

DEFAULT_CHUNKS = 16
ALIGN = 256                      # every segment of a chunk block starts 256-byte aligned
BLOCK = 256                      # threads per block (cuda/pipeline.cu PIPELINE_BLOCK)
STREAMS = 3
BUFFERS = 3                      # triple buffering
INT16_MISSING = -32768           # sentinel offset for a missing level (px = 0)
INT16_MAX_ABS = 32767
QUICK_SUBSET = 131_072
QUICK_ITERS = 5
NVTX_OVERLAPPED = "overlapped_pass"
NVTX_SERIAL = "serial_pass"
LEVEL_ARRAYS = ("bid_px", "bid_sz", "ask_px", "ask_sz")
INPUT_FIELDS_I32: Tuple[Tuple[str, type], ...] = (
    ("bid_px", np.int32), ("bid_sz", np.float32), ("ask_px", np.int32), ("ask_sz", np.float32))
OUTPUT_FIELDS: Tuple[Tuple[str, type], ...] = (
    ("spread_ticks", np.int32), ("depth_bid", np.float32), ("depth_ask", np.float32),
    ("imbalance10", np.float32), ("microprice_offset", np.float32))
LAYOUT_NOTE = ("chunk-major pinned copy built once: chunk i = snapshots [i*n/K, (i+1)*n/K); one contiguous "
               "256-byte-aligned block per chunk holding bid_px, bid_sz, ask_px, ask_sz (each level-major inside "
               "the chunk), so every chunk is one cudaMemcpyAsync in and one out")


def log(*args) -> None:
    print("[bench_pipeline]", *args, flush=True)


# --------------------------------------------------------------------------- #
# Chunk math and buffer layouts (pure NumPy / integers; unit-tested on CPU)
# --------------------------------------------------------------------------- #


def chunk_bounds(n: int, k: int) -> List[Tuple[int, int]]:
    """Chunk i covers ``[i*n//k, (i+1)*n//k)``; empty chunks (k > n) are dropped."""
    if n <= 0 or k <= 0:
        return []
    k = min(int(k), int(n))
    bounds = [(i * n // k, (i + 1) * n // k) for i in range(k)]
    return [(s, e) for s, e in bounds if e > s]


def align_up(nbytes: int, align: int = ALIGN) -> int:
    return (int(nbytes) + align - 1) // align * align


def layout_int32(levels: int, length: int) -> dict:
    """Byte offsets of the four level-major segments of an int32 chunk block."""
    seg = align_up(4 * levels * length)
    return {"kind": "int32", "length": int(length), "segment_bytes": seg, "block_bytes": 4 * seg,
            "payload_bytes": 16 * levels * length,
            "offsets": {"bid_px": 0, "bid_sz": seg, "ask_px": 2 * seg, "ask_sz": 3 * seg}}


def layout_int16(levels: int, length: int) -> dict:
    """Offsets of an int16-encoded chunk: best prices, int16 offsets, float32 sizes."""
    best = align_up(4 * length)
    off = align_up(2 * levels * length)
    sz = align_up(4 * levels * length)
    offsets = {"best_bid": 0, "best_ask": best, "bid_off": 2 * best, "ask_off": 2 * best + off,
               "bid_sz": 2 * best + 2 * off, "ask_sz": 2 * best + 2 * off + sz}
    return {"kind": "int16", "length": int(length), "block_bytes": 2 * best + 2 * off + 2 * sz,
            "payload_bytes": 8 * length + 4 * levels * length + 8 * levels * length, "offsets": offsets}


def layout_output(length: int) -> dict:
    """Five contiguous output arrays (20 bytes per snapshot), one D2H copy per chunk."""
    return {"length": int(length), "block_bytes": 4 * length * len(OUTPUT_FIELDS),
            "offsets": {name: i * 4 * length for i, (name, _dt) in enumerate(OUTPUT_FIELDS)}}


def encode_int16_offsets(px: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Encode level-major int32 prices as (best price, int16 offsets, overflow flag).

    ``px`` is ``(levels, L)``; level 0 is the best price. A present level (px > 0)
    becomes ``px - best``; a missing level becomes :data:`INT16_MISSING`. ``overflow``
    is True when any present offset has magnitude above 32767, in which case the
    caller must send the chunk as int32 (the returned offsets are clipped and must
    not be used).
    """
    px = np.asarray(px)
    best = px[0].astype(np.int32, copy=True)
    present = px > 0
    diff = px.astype(np.int64) - best.astype(np.int64)[None, :]
    overflow = bool(np.any(np.abs(diff[present]) > INT16_MAX_ABS)) if present.any() else False
    off = np.where(present, np.clip(diff, -INT16_MAX_ABS, INT16_MAX_ABS), INT16_MISSING).astype(np.int16)
    return best, off, overflow


def decode_int16_offsets(best: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_int16_offsets` (what the kernel does per level)."""
    off = np.asarray(off, dtype=np.int16)
    px = np.asarray(best, dtype=np.int64)[None, :] + off.astype(np.int64)
    return np.where(off == INT16_MISSING, 0, px).astype(np.int32)


# --------------------------------------------------------------------------- #
# NumPy reference of the fused kernel (float64; the kernel is float32)
# --------------------------------------------------------------------------- #


def reference_features(bid_px, bid_sz, ask_px, ask_sz, imb_levels: int = 10) -> Dict[str, np.ndarray]:
    """Spread (ticks), depth per side, top-10 imbalance and microprice offset per snapshot.

    Same conventions as ``cuda/pipeline.cu`` and ``metrics.book_imbalance``:
    imbalance ``(bid - ask) / (bid + ask)`` over the top ``imb_levels`` levels, 0 when
    the total is 0; microprice offset ``microprice - mid`` in ticks, which equals
    ``0.5 * spread * (bid_sz0 - ask_sz0) / (bid_sz0 + ask_sz0)``.
    """
    bb = np.asarray(bid_px[0]).astype(np.int64)
    ba = np.asarray(ask_px[0]).astype(np.int64)
    both = (bb > 0) & (ba > 0)
    spread = np.where(both, ba - bb, 0).astype(np.int32)
    depth_bid = np.sum(bid_sz, axis=0, dtype=np.float64)
    depth_ask = np.sum(ask_sz, axis=0, dtype=np.float64)
    b10 = np.sum(bid_sz[:imb_levels], axis=0, dtype=np.float64)
    a10 = np.sum(ask_sz[:imb_levels], axis=0, dtype=np.float64)
    tot10 = b10 + a10
    imbalance = np.zeros_like(tot10)
    np.divide(b10 - a10, tot10, out=imbalance, where=tot10 > 0)
    bs0 = np.asarray(bid_sz[0]).astype(np.float64)
    as0 = np.asarray(ask_sz[0]).astype(np.float64)
    tot0 = bs0 + as0
    ratio = np.zeros_like(tot0)
    np.divide(bs0 - as0, tot0, out=ratio, where=tot0 > 0)
    microprice_offset = np.where(both & (tot0 > 0), 0.5 * spread.astype(np.float64) * ratio, 0.0)
    return {"spread_ticks": spread, "depth_bid": depth_bid, "depth_ask": depth_ask,
            "imbalance10": imbalance, "microprice_offset": microprice_offset}


def compare_features(got: Dict[str, np.ndarray], ref: Dict[str, np.ndarray]) -> dict:
    """Exact for ticks, float32 tolerance (1e-3 absolute + 1e-4 relative) for the rest."""
    errors: Dict[str, float] = {}
    ok = True
    for name, _dt in OUTPUT_FIELDS:
        g = np.asarray(got[name])
        r = np.asarray(ref[name])
        if g.shape != r.shape:
            ok = False
            errors[name] = float("inf")
            continue
        if name == "spread_ticks":
            err = int(np.max(np.abs(g.astype(np.int64) - r.astype(np.int64)))) if g.size else 0
            errors[name] = err
            ok = ok and err == 0
        else:
            diff = np.abs(g.astype(np.float64) - r)
            errors[name] = float(diff.max()) if g.size else 0.0
            ok = ok and bool(np.all(diff <= 1e-3 + 1e-4 * np.abs(r)))
    return {"verified": bool(ok), "max_abs_err": errors}


def consequence_text(h2d_ms: float, kernel_ms: float, d2h_ms: float, h2d_gbs: float,
                     transfer_share_pct: float, overlap_saved_pct: float) -> str:
    """Plain-English design consequence derived from the measured breakdown."""
    if transfer_share_pct >= 50.0:
        return (
            f"Transfers dominate: the host-to-device and device-to-host copies are {transfer_share_pct:.0f}% of the "
            f"pipeline's GPU time (H2D {h2d_ms:.1f} ms at {h2d_gbs:.1f} GB/s, kernel {kernel_ms:.1f} ms, "
            f"D2H {d2h_ms:.1f} ms). Overlapping three streams can only hide the kernel and the copy-back behind "
            f"the input copy, which saved {overlap_saved_pct:.1f}% of the serial wall time. The design consequence "
            "is that the archive should be copied to the card once and stay resident, with every analysis run in "
            "that one residency instead of streaming the data per analysis."
        )
    return (
        f"Compute dominates: the kernel is {100.0 - transfer_share_pct:.0f}% of the pipeline's GPU time "
        f"(H2D {h2d_ms:.1f} ms at {h2d_gbs:.1f} GB/s, kernel {kernel_ms:.1f} ms, D2H {d2h_ms:.1f} ms), so "
        f"streaming the archive through three streams hides most of the transfer and saved {overlap_saved_pct:.1f}% "
        f"of the serial wall time. Keeping the archive resident still removes the remaining {transfer_share_pct:.0f}% "
        "but the kernel, not the PCIe link, is the bottleneck for this feature set."
    )


def subset_archive(arch: dict, subset: Optional[int]) -> dict:
    """The four level-major arrays (first ``subset`` snapshots) plus meta/path."""
    n = int(arch["bid_px"].shape[1])
    out = {k: arch[k] for k in ("meta", "path") if k in arch}
    if subset is not None and 0 < int(subset) < n:
        for name in LEVEL_ARRAYS:
            out[name] = arch[name][:, : int(subset)]
    else:
        for name in LEVEL_ARRAYS:
            out[name] = arch[name]
    return out


# --------------------------------------------------------------------------- #
# GPU side (CuPy imported lazily)
# --------------------------------------------------------------------------- #


class _NoNvtx:
    @staticmethod
    def RangePush(*_args, **_kwargs) -> None:  # noqa: N802 - mirrors cupy.cuda.nvtx
        return None

    @staticmethod
    def RangePop() -> None:  # noqa: N802
        return None


def nvtx_module():
    """``cupy.cuda.nvtx`` when available, otherwise a no-op stand-in."""
    try:
        from cupy.cuda import nvtx  # type: ignore

        if getattr(nvtx, "available", True):
            return nvtx
    except Exception:  # noqa: BLE001
        pass
    return _NoNvtx()


class HostStaging:
    """Chunk-major pinned copies of the archive plus the pinned output buffer."""

    def __init__(self, arch: dict, chunks: int = DEFAULT_CHUNKS, int16: bool = False, pinned: bool = True) -> None:
        self.levels, self.n = (int(x) for x in arch["bid_px"].shape)
        self.bounds = chunk_bounds(self.n, chunks)
        if not self.bounds:
            raise ValueError("archive has no snapshots")
        self.K = len(self.bounds)
        self.pinned = bool(pinned)
        self._pinned_blocks: list = []
        self.layouts32 = [layout_int32(self.levels, e - s) for s, e in self.bounds]
        self.in_offsets32 = _prefix([lay["block_bytes"] for lay in self.layouts32])
        self.bytes_h2d = sum(lay["block_bytes"] for lay in self.layouts32)
        self.payload_h2d = sum(lay["payload_bytes"] for lay in self.layouts32)
        t0 = time.perf_counter()
        self.host_in32 = self._alloc(self.bytes_h2d)
        for c, (s, e) in enumerate(self.bounds):
            lay = self.layouts32[c]
            base = self.in_offsets32[c]
            for name, dtype in INPUT_FIELDS_I32:
                view = self.view(self.host_in32, base + lay["offsets"][name], dtype, (self.levels, e - s))
                view[...] = arch[name][:, s:e]
        self.layouts_out = [layout_output(e - s) for s, e in self.bounds]
        self.out_offsets = _prefix([lay["block_bytes"] for lay in self.layouts_out])
        self.bytes_d2h = sum(lay["block_bytes"] for lay in self.layouts_out)
        self.host_out = self._alloc(self.bytes_d2h)
        self.staging_ms = (time.perf_counter() - t0) * 1e3
        self.int16: Optional[dict] = None
        if int16:
            self.build_int16(arch)

    # -- memory -------------------------------------------------------------- #

    def _alloc(self, nbytes: int) -> np.ndarray:
        nbytes = max(int(nbytes), ALIGN)
        if self.pinned:
            cp = _cupy()
            mem = cp.cuda.alloc_pinned_memory(nbytes)
            self._pinned_blocks.append(mem)
            arr = np.frombuffer(mem, dtype=np.uint8, count=nbytes)
            arr[...] = 0
            return arr
        return np.zeros(nbytes, dtype=np.uint8)

    @staticmethod
    def view(buf: np.ndarray, offset: int, dtype, shape) -> np.ndarray:
        shape = tuple(int(s) for s in shape)
        count = int(np.prod(shape)) if shape else 1
        itemsize = np.dtype(dtype).itemsize
        return buf[offset: offset + count * itemsize].view(dtype).reshape(shape)

    # -- int16 encoding (stretch) ------------------------------------------- #

    def build_int16(self, arch: dict) -> dict:
        """Encode every chunk's prices as int16 offsets; overflowing chunks keep int32."""
        t0 = time.perf_counter()
        layouts: List[Optional[dict]] = []
        encoded: List[Optional[tuple]] = []
        fallback: List[bool] = []
        for c, (s, e) in enumerate(self.bounds):
            best_bid, bid_off, over_b = encode_int16_offsets(arch["bid_px"][:, s:e])
            best_ask, ask_off, over_a = encode_int16_offsets(arch["ask_px"][:, s:e])
            if over_b or over_a:
                layouts.append(None)
                encoded.append(None)
                fallback.append(True)
            else:
                layouts.append(layout_int16(self.levels, e - s))
                encoded.append((best_bid, best_ask, bid_off, ask_off))
                fallback.append(False)
        total = sum(lay["block_bytes"] for lay in layouts if lay is not None)
        host = self._alloc(total)
        offsets: List[Optional[int]] = []
        cursor = 0
        for c, lay in enumerate(layouts):
            if lay is None:
                offsets.append(None)
                continue
            offsets.append(cursor)
            s, e = self.bounds[c]
            length = e - s
            best_bid, best_ask, bid_off, ask_off = encoded[c]  # type: ignore[misc]
            offs = lay["offsets"]
            self.view(host, cursor + offs["best_bid"], np.int32, (length,))[...] = best_bid
            self.view(host, cursor + offs["best_ask"], np.int32, (length,))[...] = best_ask
            self.view(host, cursor + offs["bid_off"], np.int16, (self.levels, length))[...] = bid_off
            self.view(host, cursor + offs["ask_off"], np.int16, (self.levels, length))[...] = ask_off
            self.view(host, cursor + offs["bid_sz"], np.float32, (self.levels, length))[...] = arch["bid_sz"][:, s:e]
            self.view(host, cursor + offs["ask_sz"], np.float32, (self.levels, length))[...] = arch["ask_sz"][:, s:e]
            cursor += lay["block_bytes"]
        bytes_h2d = sum((lay["block_bytes"] if lay is not None else self.layouts32[c]["block_bytes"])
                        for c, lay in enumerate(layouts))
        self.int16 = {
            "layouts": layouts, "offsets": offsets, "host": host, "fallback": fallback,
            "overflow_fallbacks": int(sum(fallback)), "bytes_h2d": int(bytes_h2d),
            "max_block_bytes": max([lay["block_bytes"] for lay in layouts if lay is not None] or [0]),
            "encode_ms": (time.perf_counter() - t0) * 1e3,
        }
        return self.int16

    # -- outputs -------------------------------------------------------------- #

    def outputs(self) -> Dict[str, np.ndarray]:
        """Assemble the per-chunk output blocks into whole arrays (copies)."""
        out: Dict[str, np.ndarray] = {}
        for name, dtype in OUTPUT_FIELDS:
            parts = [self.view(self.host_out, self.out_offsets[c] + lay["offsets"][name], dtype, (lay["length"],))
                     for c, lay in enumerate(self.layouts_out)]
            out[name] = np.concatenate(parts)
        return out


def _prefix(sizes: Sequence[int]) -> List[int]:
    offsets: List[int] = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += int(size)
    return offsets


class DevicePipeline:
    """Three streams, triple-buffered device blocks, event dependencies, one fused kernel."""

    def __init__(self, staging: HostStaging, nvtx: bool = False) -> None:
        cp = self.cp = _cupy()
        self.st = staging
        self.rt = cp.cuda.runtime
        self.H2D = self.rt.memcpyHostToDevice
        self.D2H = self.rt.memcpyDeviceToHost
        self.module = raw_module(["pipeline.cu"])
        self.k32 = self.module.get_function("pipeline_features_i32")
        self.k16 = self.module.get_function("pipeline_features_i16")
        self.max_in = 0
        self.dev_in: list = []
        self.ensure_input_capacity(max(lay["block_bytes"] for lay in staging.layouts32))
        if staging.int16 is not None:
            self.ensure_input_capacity(staging.int16["max_block_bytes"])
        self.max_out = max(lay["block_bytes"] for lay in staging.layouts_out)
        self.dev_out = [cp.cuda.alloc(self.max_out) for _ in range(BUFFERS)]
        self.bad_books = cp.zeros(1, dtype=cp.uint32)
        self.s_in = cp.cuda.Stream(non_blocking=True)
        self.s_comp = cp.cuda.Stream(non_blocking=True)
        self.s_out = cp.cuda.Stream(non_blocking=True)
        self.s_serial = cp.cuda.Stream(non_blocking=True)
        self.ev = {name: [cp.cuda.Event(block=False, disable_timing=False) for _ in range(staging.K)]
                   for name in ("h_start", "h_end", "k_start", "k_end", "d_start", "d_end")}
        self.nvtx = nvtx_module() if nvtx else None
        self.host_in_ptr = int(staging.host_in32.ctypes.data)
        self.host_out_ptr = int(staging.host_out.ctypes.data)

    @property
    def device_bytes(self) -> int:
        return BUFFERS * (self.max_in + getattr(self, "max_out", 0)) + 4

    def ensure_input_capacity(self, nbytes: int) -> None:
        if nbytes <= self.max_in:
            return
        self.cp.cuda.Device().synchronize()
        self.dev_in = [self.cp.cuda.alloc(int(nbytes)) for _ in range(BUFFERS)]
        self.max_in = int(nbytes)

    # -- per chunk ------------------------------------------------------------ #

    def _plan(self, c: int, encoding: str):
        """(host source pointer, layout, kernel) of chunk ``c`` for an encoding."""
        st = self.st
        if encoding == "int16" and st.int16 is not None and not st.int16["fallback"][c]:
            lay = st.int16["layouts"][c]
            src = int(st.int16["host"].ctypes.data) + int(st.int16["offsets"][c])
            return src, lay, self.k16
        lay = st.layouts32[c]
        return self.host_in_ptr + st.in_offsets32[c], lay, self.k32

    def _launch(self, c: int, lay: dict, kernel, buf_in, buf_out, stream) -> None:
        cp = self.cp
        length = lay["length"]
        levels = self.st.levels
        offs = lay["offsets"]

        def arr(base, offset, dtype, count):
            return cp.ndarray((int(count),), dtype=dtype, memptr=base + int(offset))

        if lay["kind"] == "int32":
            inputs = (arr(buf_in, offs["bid_px"], np.int32, levels * length),
                      arr(buf_in, offs["bid_sz"], np.float32, levels * length),
                      arr(buf_in, offs["ask_px"], np.int32, levels * length),
                      arr(buf_in, offs["ask_sz"], np.float32, levels * length))
        else:
            inputs = (arr(buf_in, offs["best_bid"], np.int32, length),
                      arr(buf_in, offs["best_ask"], np.int32, length),
                      arr(buf_in, offs["bid_off"], np.int16, levels * length),
                      arr(buf_in, offs["ask_off"], np.int16, levels * length),
                      arr(buf_in, offs["bid_sz"], np.float32, levels * length),
                      arr(buf_in, offs["ask_sz"], np.float32, levels * length))
        olay = self.st.layouts_out[c]
        outputs = tuple(arr(buf_out, olay["offsets"][name], dtype, length) for name, dtype in OUTPUT_FIELDS)
        grid = (length + BLOCK - 1) // BLOCK
        with stream:  # the launch goes to `stream` (works on every CuPy version)
            kernel((grid,), (BLOCK,), (*inputs, np.int32(length), np.int32(levels), *outputs, self.bad_books))

    # -- passes --------------------------------------------------------------- #

    def issue_overlapped(self, c: int, encoding: str = "int32", instrument: bool = False) -> None:
        """Enqueue chunk ``c`` of an overlapped pass: H2D on s_in, kernel on s_comp, D2H on s_out.

        Nothing here blocks the host, so several pipelines (one per card) can be issued chunk by chunk.
        """
        st, ev = self.st, self.ev
        s_in, s_comp, s_out = self.s_in, self.s_comp, self.s_out
        b = c % BUFFERS
        src, lay, kernel = self._plan(c, encoding)
        if self.nvtx is not None:
            self.nvtx.RangePush(f"chunk {c:02d}")
        # copy in: the buffer was last read by the kernel of chunk c-3
        if c >= BUFFERS:
            s_in.wait_event(ev["k_end"][c - BUFFERS])
        if instrument:
            ev["h_start"][c].record(s_in)
        self.rt.memcpyAsync(self.dev_in[b].ptr, src, lay["block_bytes"], self.H2D, s_in.ptr)
        ev["h_end"][c].record(s_in)
        # compute: needs its input, and the output buffer drained by D2H of chunk c-3
        s_comp.wait_event(ev["h_end"][c])
        if c >= BUFFERS:
            s_comp.wait_event(ev["d_end"][c - BUFFERS])
        if instrument:
            ev["k_start"][c].record(s_comp)
        self._launch(c, lay, kernel, self.dev_in[b], self.dev_out[b], s_comp)
        ev["k_end"][c].record(s_comp)
        # copy out
        s_out.wait_event(ev["k_end"][c])
        if instrument:
            ev["d_start"][c].record(s_out)
        olay = st.layouts_out[c]
        self.rt.memcpyAsync(self.host_out_ptr + st.out_offsets[c], self.dev_out[b].ptr,
                            olay["block_bytes"], self.D2H, s_out.ptr)
        ev["d_end"][c].record(s_out)
        if self.nvtx is not None:
            self.nvtx.RangePop()

    def issue_h2d(self, c: int) -> None:
        """Enqueue only chunk ``c``'s host-to-device copy (bandwidth measurement)."""
        src, lay, _kernel = self._plan(c, "int32")
        self.rt.memcpyAsync(self.dev_in[c % BUFFERS].ptr, src, lay["block_bytes"], self.H2D, self.s_in.ptr)

    def sync(self) -> None:
        self.s_in.synchronize()
        self.s_comp.synchronize()
        self.s_out.synchronize()

    def pass_overlapped(self, encoding: str = "int32", instrument: bool = False) -> None:
        """H2D(i+1) on s_in, kernel(i) on s_comp, D2H(i-1) on s_out; triple buffers."""
        for c in range(self.st.K):
            self.issue_overlapped(c, encoding, instrument)
        self.sync()

    def pass_h2d(self) -> None:
        for c in range(self.st.K):
            self.issue_h2d(c)
        self.s_in.synchronize()

    def pass_serial(self, encoding: str = "int32", instrument: bool = False) -> None:
        """Same chunks, one stream, one buffer: copy in, compute, copy out, in order."""
        st, ev = self.st, self.ev
        s = self.s_serial
        for c in range(st.K):
            src, lay, kernel = self._plan(c, encoding)
            if instrument:
                ev["h_start"][c].record(s)
            self.rt.memcpyAsync(self.dev_in[0].ptr, src, lay["block_bytes"], self.H2D, s.ptr)
            if instrument:
                ev["h_end"][c].record(s)
                ev["k_start"][c].record(s)
            self._launch(c, lay, kernel, self.dev_in[0], self.dev_out[0], s)
            if instrument:
                ev["k_end"][c].record(s)
                ev["d_start"][c].record(s)
            olay = st.layouts_out[c]
            self.rt.memcpyAsync(self.host_out_ptr + st.out_offsets[c], self.dev_out[0].ptr,
                                olay["block_bytes"], self.D2H, s.ptr)
            if instrument:
                ev["d_end"][c].record(s)
        s.synchronize()

    def run_pass(self, kind: str, encoding: str = "int32", instrument: bool = False) -> None:
        if kind == "overlapped":
            self.pass_overlapped(encoding, instrument)
        elif kind == "serial":
            self.pass_serial(encoding, instrument)
        else:
            raise ValueError(kind)

    def stage_sums_ms(self) -> Dict[str, float]:
        """Per-stage sums over all chunks of the last instrumented pass."""
        cp, ev = self.cp, self.ev
        sums = {}
        for stage, (a, b) in (("h2d_ms", ("h_start", "h_end")), ("kernel_ms", ("k_start", "k_end")),
                              ("d2h_ms", ("d_start", "d_end"))):
            sums[stage] = float(sum(cp.cuda.get_elapsed_time(ev[a][c], ev[b][c]) for c in range(self.st.K)))
        return sums

    def read_bad_books(self) -> int:
        self.cp.cuda.Device().synchronize()
        return int(self.bad_books.get()[0])

    def reset_bad_books(self) -> None:
        self.bad_books.fill(0)
        self.cp.cuda.Device().synchronize()


# --------------------------------------------------------------------------- #
# Two cards
# --------------------------------------------------------------------------- #


def parse_topology(text: str) -> dict:
    """The link between each pair of cards and each card's CPU and NUMA affinity, from ``nvidia-smi topo -m``.

    The header row has no cell for the row labels, so column ``i`` of the header is cell ``i`` of
    a row once its label is removed.
    """
    import re

    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # nvidia-smi underlines the header row with ANSI escapes
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    header = lines[0].split()
    gpus = [h for h in header if re.fullmatch(r"GPU\d+", h)]
    out: dict = {"gpus": gpus, "links": {}, "cpu_affinity": {}, "numa_affinity": {}}
    cpu_col = header.index("CPU") if "CPU" in header else None
    for ln in lines[1:]:
        parts = ln.split()
        if not parts or parts[0] not in gpus:
            continue
        name, cells = parts[0], parts[1:]
        for i, g in enumerate(gpus):
            if i < len(cells) and g != name:
                out["links"][f"{name}-{g}"] = cells[i]
        if cpu_col is not None and cpu_col < len(cells):
            out["cpu_affinity"][name] = cells[cpu_col]
            if cpu_col + 1 < len(cells):
                out["numa_affinity"][name] = cells[cpu_col + 1]
    return out


TOPOLOGY_MEANING = {
    "X": "the same card", "SYS": "through the CPU interconnect between NUMA nodes",
    "NODE": "through PCIe host bridges within one NUMA node", "PHB": "through one PCIe host bridge (the CPU)",
    "PXB": "through more than one PCIe switch", "PIX": "through a single PCIe switch",
}


def wall_all_ms(fn, devices: Sequence[int]) -> float:
    """Wall clock of ``fn()`` with every listed card idle before and synchronised after."""
    cp = _cupy()
    for d in devices:
        with cp.cuda.Device(d):
            cp.cuda.Device(d).synchronize()
    t0 = time.perf_counter()
    fn()
    for d in devices:
        with cp.cuda.Device(d):
            cp.cuda.Device(d).synchronize()
    return (time.perf_counter() - t0) * 1e3


class MultiGpuPipeline:
    """The archive split in contiguous halves, one :class:`DevicePipeline` per card.

    Each card gets its own pinned host staging (allocated with that card current), its own
    module, streams, events and triple-buffered device blocks; the two cards' chunks are issued
    alternately so both run at once, and the host only waits at the end of the pass.
    """

    def __init__(self, arch: dict, chunks_per_device: int, devices: Sequence[int]) -> None:
        cp = self.cp = _cupy()
        self.devices = list(devices)
        n = int(arch["bid_px"].shape[1])
        self.parts = []
        for d, (s, e) in zip(self.devices, chunk_bounds(n, len(self.devices))):
            part = {name: arch[name][:, s:e] for name in LEVEL_ARRAYS}
            with cp.cuda.Device(d):
                staging = HostStaging(part, chunks_per_device, int16=False, pinned=True)
                pipe = DevicePipeline(staging)
            self.parts.append({"device": d, "start": s, "end": e, "staging": staging, "pipe": pipe})
        self.bytes_h2d = sum(p["staging"].bytes_h2d for p in self.parts)

    def _issue(self, method: str) -> None:
        k_max = max(p["staging"].K for p in self.parts)
        for c in range(k_max):
            for part in self.parts:
                if c < part["staging"].K:
                    with self.cp.cuda.Device(part["device"]):
                        getattr(part["pipe"], method)(c)

    def pass_overlapped(self) -> None:
        self._issue("issue_overlapped")
        for part in self.parts:
            with self.cp.cuda.Device(part["device"]):
                part["pipe"].sync()

    def pass_h2d(self) -> None:
        self._issue("issue_h2d")
        for part in self.parts:
            with self.cp.cuda.Device(part["device"]):
                part["pipe"].s_in.synchronize()

    def outputs(self) -> Dict[str, np.ndarray]:
        outs = [p["staging"].outputs() for p in self.parts]
        return {name: np.concatenate([o[name] for o in outs]) for name, _dtype in OUTPUT_FIELDS}


def identical_outputs(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> dict:
    """Bit-for-bit comparison of two output sets (NaN equal to NaN)."""
    per = {}
    for name, _dtype in OUTPUT_FIELDS:
        x, y = np.asarray(a[name]), np.asarray(b[name])
        same = x.shape == y.shape and bool(np.array_equal(x.view(np.uint32) if x.dtype.itemsize == 4 else x,
                                                          y.view(np.uint32) if y.dtype.itemsize == 4 else y))
        per[name] = same
    return {"identical": all(per.values()), "per_output": per}


def host_memcpy_gbs(nbytes: int, repeats: int = 5) -> float:
    """Single-threaded host memcpy bandwidth (read + write counted once, as a copy's payload)."""
    src = np.ones(max(int(nbytes), 1 << 20), dtype=np.uint8)
    dst = np.empty_like(src)
    np.copyto(dst, src)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        best = min(best, time.perf_counter() - t0)
    return src.nbytes / best / 1e9


def run_two_gpu(bench: Bench, pipe: DevicePipeline, staging: HostStaging, sub: dict, ref: Dict[str, np.ndarray],
                n_iters: Optional[int]) -> dict:
    """One card against two on the same chunks: end to end, copy-only bandwidth, equal outputs."""
    cp = _cupy()
    n_dev = cp.cuda.runtime.getDeviceCount()
    topo_text = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30).stdout
    links = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,pcie.link.gen.max,pcie.link.width.max",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    out: dict = {"attempted": True, "device_count": n_dev, "topology_text": topo_text, "topology": parse_topology(topo_text),
                 "pcie": [ln.strip() for ln in links.splitlines()]}
    if n_dev < 2:
        out.update({"ok": False, "reason": f"{n_dev} CUDA device(s) visible"})
        return out
    devices = (0, 1)
    if staging.K % 2:
        out.update({"ok": False, "reason": f"{staging.K} chunks do not split evenly across two cards"})
        return out
    multi = MultiGpuPipeline(sub, staging.K // 2, devices)
    out.update({"devices": list(devices), "chunks_per_device": staging.K // 2,
                "bytes_h2d_per_device": [p["staging"].bytes_h2d for p in multi.parts], "bytes_h2d_total": multi.bytes_h2d,
                "split": [[p["start"], p["end"]] for p in multi.parts]})
    # correctness: both cards' outputs against the reference and bit for bit against one card's
    pipe.pass_overlapped()
    cp.cuda.Device(0).synchronize()
    one = staging.outputs()
    multi.pass_overlapped()
    two = multi.outputs()
    out["verification"] = compare_features(two, ref)
    out["identical_to_one_gpu"] = identical_outputs(one, two)
    block = bench.block("pipeline_two_gpu", {
        "one_gpu": lambda: wall_all_ms(pipe.pass_overlapped, devices),
        "two_gpu": lambda: wall_all_ms(multi.pass_overlapped, devices),
    }, n_iters=n_iters, devices=devices)
    h2d = bench.block("pipeline_h2d_only", {
        "one_gpu": lambda: wall_all_ms(pipe.pass_h2d, devices),
        "two_gpu": lambda: wall_all_ms(multi.pass_h2d, devices),
    }, n_iters=n_iters, devices=devices)
    # the negotiated link right after the copies (an idle card may drop to a lower generation);
    # the "pcie" lines above are each card's maximum, what it is rated for
    current = subprocess.run(["nvidia-smi", "--query-gpu=index,pcie.link.gen.current,pcie.link.width.current",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    out["pcie_current_after_copies"] = [ln.strip() for ln in current.splitlines()]
    out["pcie_fields"] = "index, pci.bus_id, pcie.link.gen.max, pcie.link.width.max"
    one_ms, two_ms = block["one_gpu"]["median_ms"], block["two_gpu"]["median_ms"]
    out.update({
        "one_gpu_ms": block["one_gpu"], "two_gpu_ms": block["two_gpu"],
        "speedup": one_ms / two_ms if two_ms > 0 else 0.0,
        "scaling_efficiency_pct": (one_ms / two_ms) / 2 * 100.0 if two_ms > 0 else 0.0,
        "h2d_only_one_gpu_ms": h2d["one_gpu"], "h2d_only_two_gpu_ms": h2d["two_gpu"],
        "h2d_gbs_one_gpu": gbs(staging.bytes_h2d, h2d["one_gpu"]["median_ms"]),
        "h2d_gbs_two_gpu_aggregate": gbs(multi.bytes_h2d, h2d["two_gpu"]["median_ms"]),
        "host_memcpy_gbs": host_memcpy_gbs(staging.bytes_h2d // 4),
    })
    out["h2d_aggregate_vs_one_gpu"] = (out["h2d_gbs_two_gpu_aggregate"] / out["h2d_gbs_one_gpu"]
                                       if out["h2d_gbs_one_gpu"] else 0.0)
    out["ok"] = bool(out["verification"]["verified"] and out["identical_to_one_gpu"]["identical"])
    log(f"two cards: {one_ms:.2f} ms on one, {two_ms:.2f} ms on two ({out['speedup']:.2f}x); copy-only "
        f"{out['h2d_gbs_one_gpu']:.1f} GB/s on one, {out['h2d_gbs_two_gpu_aggregate']:.1f} GB/s on two; "
        f"identical outputs {out['identical_to_one_gpu']['identical']}")
    del multi
    return out


def verify_pass(pipe: DevicePipeline, ref: Dict[str, np.ndarray], kind: str, encoding: str = "int32") -> dict:
    """Run one pass and compare the pinned outputs with the NumPy reference."""
    pipe.reset_bad_books()
    pipe.run_pass(kind, encoding)
    pipe.cp.cuda.Device().synchronize()
    result = compare_features(pipe.st.outputs(), ref)
    result["bad_books_device"] = pipe.read_bad_books()
    result["pass"] = kind
    result["encoding"] = encoding
    return result


def measure_breakdown(pipe: DevicePipeline, kind: str, passes: int, encoding: str = "int32") -> dict:
    """Median per-stage event sums over ``passes`` instrumented passes."""
    samples: List[Dict[str, float]] = []
    pipe.run_pass(kind, encoding, instrument=True)  # warm
    for _ in range(max(1, passes)):
        pipe.run_pass(kind, encoding, instrument=True)
        samples.append(pipe.stage_sums_ms())
    out = {stage: float(np.median([s[stage] for s in samples])) for stage in ("h2d_ms", "kernel_ms", "d2h_ms")}
    out["sum_ms"] = out["h2d_ms"] + out["kernel_ms"] + out["d2h_ms"]
    out["passes"] = len(samples)
    out["samples"] = samples
    return out


def gbs(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9 if ms > 0 else 0.0


# --------------------------------------------------------------------------- #
# Nsight Systems proof
# --------------------------------------------------------------------------- #


def nsys_child(archive_dir: Path, chunks: int, subset: Optional[int], warmup: int = 2) -> dict:
    """Body of ``--nsys-child``: warm up, then one serial and ONE overlapped pass in NVTX ranges."""
    cp = _cupy()
    arch = load_archive(Path(archive_dir), names=list(LEVEL_ARRAYS))
    sub = subset_archive(arch, subset)
    staging = HostStaging(sub, chunks)
    pipe = DevicePipeline(staging, nvtx=True)
    nvtx = nvtx_module()
    nvtx.RangePush("warmup")
    for _ in range(max(0, warmup)):
        pipe.pass_serial()
        pipe.pass_overlapped()
    cp.cuda.Device().synchronize()
    nvtx.RangePop()
    nvtx.RangePush(NVTX_SERIAL)
    serial_ms = wall_time_ms(pipe.pass_serial)
    nvtx.RangePop()
    nvtx.RangePush(NVTX_OVERLAPPED)
    overlapped_ms = wall_time_ms(pipe.pass_overlapped)
    nvtx.RangePop()
    cp.cuda.Device().synchronize()
    summary = {"n_snapshots": staging.n, "chunks": staging.K, "serial_ms": serial_ms, "overlapped_ms": overlapped_ms,
               "bytes_h2d": staging.bytes_h2d, "bytes_d2h": staging.bytes_d2h, "nvtx": not isinstance(nvtx, _NoNvtx)}
    print("NSYS_CHILD " + json.dumps(summary), flush=True)
    return summary


def run_nsys(out_dir: Path, archive_dir: Optional[Path], chunks: int, subset: Optional[int],
             title: Optional[str] = None) -> dict:
    """Profile the child under nsys, export SQLite, render the SVG. Never raises."""
    out_dir = Path(out_dir)
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    result: dict = {"nsys_report": None, "nsys_sqlite": None, "timeline_svg": None}
    nsys = os.environ.get("NSYS") or shutil.which("nsys")
    if not nsys:
        result["nsys_error"] = "nsys not found on PATH (Nsight Systems unavailable; set NSYS=/path/to/nsys)"
        log(result["nsys_error"])
        return result
    if archive_dir is None:
        result["nsys_error"] = "archive directory unknown, cannot launch the nsys child"
        return result
    rep_base = raw / "pipeline"
    rep = raw / "pipeline.nsys-rep"
    sqlite = raw / "pipeline.sqlite"
    svg = raw / "pipeline_timeline.svg"
    child = [sys.executable, "-m", "bench.bench_pipeline", "--nsys-child", "--archive", str(archive_dir),
             "--chunks", str(chunks)]
    if subset:
        child += ["--subset", str(subset)]
    cmd = [nsys, "profile", "--trace=cuda,nvtx", "--stats=false", "--force-overwrite", "true",
           "-o", str(rep_base), *child]
    result["nsys_command"] = " ".join(cmd)
    log_path = raw / "pipeline_nsys.log"
    try:
        version = subprocess.run([nsys, "--version"], capture_output=True, text=True, timeout=120)
        result["nsys_version"] = (version.stdout or version.stderr).strip()
        log("nsys profile:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800)
        log_path.write_text(f"$ {' '.join(cmd)}\n[exit {proc.returncode}]\n{proc.stdout}\n{proc.stderr}\n", encoding="utf-8")
        for line in (proc.stdout or "").splitlines():
            if line.startswith("NSYS_CHILD "):
                try:
                    result["nsys_child"] = json.loads(line[len("NSYS_CHILD "):])
                except ValueError:
                    pass
        if proc.returncode != 0 or not rep.exists():
            tail = ((proc.stderr or "") + (proc.stdout or ""))[-600:]
            result["nsys_error"] = f"nsys profile exited {proc.returncode}, report {'present' if rep.exists() else 'missing'}: {tail}"
            log(result["nsys_error"])
            return result
        result["nsys_report"] = "raw/pipeline.nsys-rep"
        export = subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true",
                                 "--output", str(sqlite), str(rep)], capture_output=True, text=True, timeout=1800)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"$ nsys export [exit {export.returncode}]\n{export.stdout}\n{export.stderr}\n")
        if export.returncode != 0 or not sqlite.exists():
            result["nsys_error"] = f"nsys export exited {export.returncode}: {((export.stderr or '') + (export.stdout or ''))[-600:]}"
            log(result["nsys_error"])
            return result
        result["nsys_sqlite"] = "raw/pipeline.sqlite"
        stats = timeline_svg.render(sqlite, svg, nvtx_range=NVTX_OVERLAPPED, title=title)
        result["timeline_svg"] = "raw/pipeline_timeline.svg"
        result["nsys_timeline"] = stats
        try:
            serial_stats = timeline_svg.render(
                sqlite, raw / "pipeline_timeline_serial.svg", nvtx_range=NVTX_SERIAL,
                title="Serial reference: one stream, copy in / compute / copy out in order")
            result["timeline_svg_serial"] = "raw/pipeline_timeline_serial.svg"
            result["nsys_timeline_serial"] = serial_stats
        except Exception as exc:  # noqa: BLE001 - the serial picture is a bonus
            result["timeline_svg_serial_error"] = repr(exc)
        log(f"nsys timeline: {stats['bars']} bars, overlap {stats['overlap_pct']:.1f}% of {stats['window_ms']:.2f} ms")
    except subprocess.TimeoutExpired as exc:
        result["nsys_error"] = f"nsys timed out: {exc}"
        log(result["nsys_error"])
    except Exception as exc:  # noqa: BLE001 - never fail the job over the profiler
        result["nsys_error"] = f"nsys step failed: {exc!r}"
        log(result["nsys_error"])
    return result


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def single_device() -> bool:
    """True when fewer than two CUDA devices are visible (the two-card step is then recorded as skipped)."""
    try:
        return _cupy().cuda.runtime.getDeviceCount() < 2
    except Exception:  # noqa: BLE001
        return True


def run(bench: Bench, arch: dict, quick: bool = False, int16: Optional[bool] = None,
        chunks: Optional[int] = None, nsys: bool = True, subset: Optional[int] = None,
        archive_dir: Optional[Path] = None) -> None:
    """Measure the pipeline and write the ``pipeline`` section.

    ``int16=None`` reads ``PIPELINE_INT16`` (default off); ``chunks=None`` reads
    ``PIPELINE_CHUNKS`` (default 16). ``quick`` runs a few iterations on a subset.
    """
    cp = _cupy()
    K = int(chunks or os.environ.get("PIPELINE_CHUNKS") or DEFAULT_CHUNKS)
    if int16 is None:
        int16 = _env_flag("PIPELINE_INT16", default=True)
    two_gpu = _env_flag("PIPELINE_TWO_GPU", default=True)
    n_total = int(arch["bid_px"].shape[1])
    if subset is None and quick:
        subset = min(n_total, QUICK_SUBSET)
    sub = subset_archive(arch, subset)
    levels, n = (int(x) for x in sub["bid_px"].shape)
    n_iters = QUICK_ITERS if quick else None
    n_break = QUICK_ITERS if quick else bench.n_iters
    archive_dir = Path(archive_dir) if archive_dir is not None else (Path(arch["path"]) if arch.get("path") else None)
    log(f"n={n} (archive {n_total}), levels={levels}, chunks={K}, quick={quick}, int16={int16}")

    section: dict = {"chunks": K, "streams": STREAMS, "pinned": True, "n_snapshots": n, "n_snapshots_archive": n_total,
                     "subset": subset, "levels": levels, "quick": bool(quick), "layout": LAYOUT_NOTE,
                     "kernel": "pipeline_features_i32", "block_size": BLOCK, "triple_buffered": True}

    # 1. staging: pinned chunk-major copy (pageable fallback is recorded, never silent)
    try:
        staging = HostStaging(sub, K, int16=False, pinned=True)
    except Exception as exc:  # noqa: BLE001 - cudaHostAlloc can fail on a loaded machine
        log("pinned allocation failed, falling back to pageable host memory:", repr(exc))
        staging = HostStaging(sub, K, int16=False, pinned=False)
        section["pinned"] = False
        section["pinned_error"] = repr(exc)
    pipe = DevicePipeline(staging)
    section.update({
        "chunks": staging.K, "chunk_snapshots": max(e - s for s, e in staging.bounds),
        "chunk_bytes": max(lay["block_bytes"] for lay in staging.layouts32),
        "bytes_h2d_per_pass": staging.bytes_h2d, "bytes_d2h_per_pass": staging.bytes_d2h,
        "payload_bytes_per_pass": staging.payload_h2d, "host_pinned_bytes": staging.bytes_h2d + staging.bytes_d2h,
        "device_bytes": pipe.device_bytes, "staging_ms": staging.staging_ms,
    })
    if pipe.device_bytes > 3 * 1024 ** 3:
        log(f"WARNING device footprint {pipe.device_bytes / 1e9:.2f} GB exceeds the 3 GB budget")
    log(f"staged {staging.bytes_h2d / 1e6:.1f} MB pinned in {staging.staging_ms:.0f} ms; "
        f"device buffers {pipe.device_bytes / 1e6:.1f} MB")

    # 2. correctness before timing
    ref = reference_features(sub["bid_px"], sub["bid_sz"], sub["ask_px"], sub["ask_sz"])
    ver_serial = verify_pass(pipe, ref, "serial")
    ver_over = verify_pass(pipe, ref, "overlapped")
    section["verification"] = {"serial": ver_serial, "overlapped": ver_over,
                               "verified": bool(ver_serial["verified"] and ver_over["verified"])}
    section["verified"] = section["verification"]["verified"]
    section["bad_books_device"] = int(ver_over["bad_books_device"])
    log(f"verified serial={ver_serial['verified']} overlapped={ver_over['verified']} "
        f"max_abs_err={ver_over['max_abs_err']} bad_books={ver_over['bad_books_device']}")

    # 3. interleaved wall-clock block: overlapped vs serial
    block = bench.block("pipeline", {
        "overlapped": lambda: wall_time_ms(pipe.pass_overlapped),
        "serial": lambda: wall_time_ms(pipe.pass_serial),
    }, n_iters=n_iters)
    overlapped, serial = block["overlapped"], block["serial"]
    saved_pct = ((serial["median_ms"] - overlapped["median_ms"]) / serial["median_ms"] * 100.0
                 if serial["median_ms"] > 0 else 0.0)
    section.update({"overlapped_ms": overlapped, "serial_ms": serial, "overlap_saved_pct": saved_pct})
    log(f"overlapped {overlapped['median_ms']:.2f} ms, serial {serial['median_ms']:.2f} ms, saved {saved_pct:.1f}%")

    # 4. per-stage breakdown from CUDA events (overlapped pass is the contract's; serial kept beside it)
    br_over = measure_breakdown(pipe, "overlapped", n_break)
    br_serial = measure_breakdown(pipe, "serial", n_break)
    bench.save_raw("pipeline__breakdown", {"overlapped": br_over["samples"], "serial": br_serial["samples"],
                                           "bytes_h2d": staging.bytes_h2d, "bytes_d2h": staging.bytes_d2h})
    breakdown = {k: br_over[k] for k in ("h2d_ms", "kernel_ms", "d2h_ms", "sum_ms")}
    breakdown["overlapped_wall_ms"] = overlapped["median_ms"]
    breakdown["passes"] = br_over["passes"]
    breakdown["measured_in"] = ("overlapped pass: CUDA events recorded on each stream around every chunk's H2D, "
                                "kernel and D2H, summed per stage, median over the passes")
    breakdown_serial = {k: br_serial[k] for k in ("h2d_ms", "kernel_ms", "d2h_ms", "sum_ms")}
    breakdown_serial["serial_wall_ms"] = serial["median_ms"]
    breakdown_serial["passes"] = br_serial["passes"]
    h2d_gbs = gbs(staging.bytes_h2d, breakdown["h2d_ms"])
    d2h_gbs = gbs(staging.bytes_d2h, breakdown["d2h_ms"])
    transfer_share = ((breakdown["h2d_ms"] + breakdown["d2h_ms"]) / breakdown["sum_ms"] * 100.0
                      if breakdown["sum_ms"] > 0 else 0.0)
    section.update({
        "breakdown": breakdown, "breakdown_serial": breakdown_serial,
        "h2d_gbs": h2d_gbs, "d2h_gbs": d2h_gbs,
        "h2d_gbs_serial": gbs(staging.bytes_h2d, breakdown_serial["h2d_ms"]),
        "d2h_gbs_serial": gbs(staging.bytes_d2h, breakdown_serial["d2h_ms"]),
        "kernel_effective_gbs": gbs(staging.payload_h2d + staging.bytes_d2h, breakdown["kernel_ms"]),
        "transfer_share_pct": transfer_share,
        "consequence": consequence_text(breakdown["h2d_ms"], breakdown["kernel_ms"], breakdown["d2h_ms"],
                                        h2d_gbs, transfer_share, saved_pct),
        "timeline_svg": None, "nsys_report": None, "nsys_sqlite": None,
        "int16_encoding": {"attempted": False, "ok": False,
                           "reason": "not requested (pass --int16 or set PIPELINE_INT16=1)"},
    })
    log(f"breakdown h2d {breakdown['h2d_ms']:.2f} ms ({h2d_gbs:.1f} GB/s), kernel {breakdown['kernel_ms']:.2f} ms, "
        f"d2h {breakdown['d2h_ms']:.2f} ms ({d2h_gbs:.1f} GB/s), transfer share {transfer_share:.1f}%")
    bench.section("pipeline", section)

    # 5. Nsight Systems proof (child process); failure is recorded, not raised
    if nsys:
        title = (f"Three-stream pipeline, {staging.K} chunks of {section['chunk_bytes'] / 1e6:.0f} MB: "
                 "{overlap}")  # placeholder replaced below once the overlap is known
        ns = run_nsys(bench.out_dir, archive_dir, staging.K, subset, title=None)
        if ns.get("nsys_timeline"):
            # re-render with a title carrying the chunking and the measured overlap fraction
            try:
                pct = ns["nsys_timeline"]["overlap_pct"]
                final_title = title.replace("{overlap}", f"{pct:.1f}% of the trace has two or more lanes active")
                ns["nsys_timeline"] = timeline_svg.render(bench.out_dir / "raw" / "pipeline.sqlite",
                                                          bench.out_dir / "raw" / "pipeline_timeline.svg",
                                                          nvtx_range=NVTX_OVERLAPPED, title=final_title)
            except Exception as exc:  # noqa: BLE001
                ns["nsys_timeline"]["retitle_error"] = repr(exc)
        section.update(ns)
    else:
        section["nsys_error"] = "nsys step disabled (--no-nsys)"
    bench.section("pipeline", section)

    # 6. the same chunks split across both cards, each with its own streams and pinned buffers
    if two_gpu and not single_device():
        try:
            section["two_gpu"] = run_two_gpu(bench, pipe, staging, sub, ref, n_iters)
        except Exception as exc:  # noqa: BLE001 - recorded; the one-card results stand on their own
            log("two-card run failed:", repr(exc))
            section["two_gpu"] = {"attempted": True, "ok": False, "error": repr(exc)}
        bench.section("pipeline", section)

    # 7. int16 transfer encoding, isolated
    if int16:
        try:
            section["int16_encoding"] = run_int16(bench, pipe, staging, sub, ref, n_iters)
        except Exception as exc:  # noqa: BLE001 - stretch must never break the required results
            log("int16 encoding failed:", repr(exc))
            section["int16_encoding"] = {"attempted": True, "ok": False, "error": repr(exc)}
        bench.section("pipeline", section)

    del pipe, staging
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001
        pass
    log("done")


def run_int16(bench: Bench, pipe: DevicePipeline, staging: HostStaging, sub: dict,
              ref: Dict[str, np.ndarray], n_iters: Optional[int]) -> dict:
    """The int16 offset encoding, measured end to end against the int32 pass in one block."""
    info = staging.build_int16(sub)
    pipe.ensure_input_capacity(info["max_block_bytes"])
    log(f"int16 encode: {info['encode_ms']:.0f} ms on the host, {info['overflow_fallbacks']} of {staging.K} chunks "
        f"fell back to int32, bytes {staging.bytes_h2d} -> {info['bytes_h2d']}")
    pipe.pass_overlapped("int32")
    pipe.cp.cuda.Device().synchronize()
    out32 = staging.outputs()
    ver = verify_pass(pipe, ref, "overlapped", encoding="int16")
    same = identical_outputs(out32, staging.outputs())
    block = bench.block("pipeline_int16", {
        "int32": lambda: wall_time_ms(lambda: pipe.pass_overlapped("int32")),
        "int16": lambda: wall_time_ms(lambda: pipe.pass_overlapped("int16")),
    }, n_iters=n_iters)
    br16 = measure_breakdown(pipe, "overlapped", QUICK_ITERS if n_iters else max(10, bench.n_iters // 3), "int16")
    bytes_saved_pct = (staging.bytes_h2d - info["bytes_h2d"]) / staging.bytes_h2d * 100.0 if staging.bytes_h2d else 0.0
    ok = bool(ver["verified"] and ver["bad_books_device"] == 0 and same["identical"])
    return {
        "attempted": True, "ok": ok, "e2e_ms": block["int16"], "int32_e2e_ms": block["int32"],
        "identical_to_int32": same, "snapshots_in_fallback_chunks": int(sum(
            e - s for (s, e), fb in zip(staging.bounds, info["fallback"]) if fb)),
        "speedup_vs_int32": (block["int32"]["median_ms"] / block["int16"]["median_ms"]
                             if block["int16"]["median_ms"] > 0 else 0.0),
        "bytes_saved_pct": bytes_saved_pct, "overflow_fallbacks": info["overflow_fallbacks"],
        "bytes_h2d_int32": staging.bytes_h2d, "bytes_h2d_int16": info["bytes_h2d"],
        "h2d_gbs": gbs(info["bytes_h2d"], br16["h2d_ms"]),
        "breakdown": {k: br16[k] for k in ("h2d_ms", "kernel_ms", "d2h_ms", "sum_ms")},
        "encode_ms_host_total": info["encode_ms"], "chunks": staging.K, "verification": ver,
        "kernel": "pipeline_features_i16",
        "encoding": ("per side: best price int32 per snapshot plus int16 offsets px - best per level "
                     f"({INT16_MISSING} marks a missing level), decoded on the device inside the kernel; a chunk "
                     "with any |offset| > 32767 is sent as int32 (overflow_fallbacks). Encoding is done once "
                     "on the host like the pinned int32 staging; encode_ms_host_total is its cost."),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 7 pipeline benchmark (three streams, pinned memory, chunks)")
    parser.add_argument("--archive", required=True, help="archive directory")
    parser.add_argument("--out", help="results/<run_id> directory")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--quick", action="store_true", help="few iterations on a subset")
    parser.add_argument("--chunks", type=int, default=None, help=f"number of chunks K (default {DEFAULT_CHUNKS})")
    parser.add_argument("--subset", type=int, default=None, help="use only the first N snapshots")
    parser.add_argument("--int16", action="store_true", help="also measure the int16 offset encoding (stretch)")
    parser.add_argument("--no-nsys", action="store_true", help="skip the Nsight Systems child")
    parser.add_argument("--nsys-child", action="store_true", help="internal: one profiled pass with NVTX ranges")
    parser.add_argument("--warmup", type=int, default=2, help="nsys child warmup passes")
    args = parser.parse_args(argv)
    if args.nsys_child:
        nsys_child(Path(args.archive), args.chunks or DEFAULT_CHUNKS, args.subset, args.warmup)
        return
    if not args.out:
        parser.error("--out is required")
    arch = load_archive(Path(args.archive))
    bench = Bench(Path(args.out), args.source, arch["meta"], n_iters=QUICK_ITERS if args.quick else 30,
                  warmup=1 if args.quick else 3)
    run(bench, arch, quick=args.quick, int16=True if args.int16 else None, chunks=args.chunks,
        nsys=not args.no_nsys, subset=args.subset, archive_dir=Path(args.archive))


if __name__ == "__main__":
    main()
