"""Extra ladder rung: the fused feature kernel written in Triton (when Triton is available).

The kernel mirrors ``v2_fused`` in ``cuda/features.cu``: one program per block of snapshots, the
50 bid and ask sizes summed per snapshot in level order (the first ten also feed the imbalance),
spread and microprice from the level-0 prices and sizes, IEEE division (``div_rn``) as NVRTC uses
by default. It reads the ladder's own device arrays (structure-of-arrays, row stride
``n_stride``) through zero-copy torch views when torch is installed, or through a minimal pointer
adapter otherwise, so it competes on exactly the same data as the CUDA rungs.

Everything here is optional: :func:`load` returns ``(None, reason)`` when Triton cannot be
imported or installed, and the ladder records that reason instead of a rung.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Callable, Optional, Tuple

CONFIGS = ((256, 4), (512, 4), (1024, 4), (1024, 8))  # (snapshots per program, warps per program)
QUICK_CONFIGS = ((512, 4), (1024, 4))


def load(install: bool = True) -> Tuple[Optional[object], str]:
    """Import Triton (installing it with pip once if missing); returns ``(module, version or reason)``."""
    try:
        import triton  # noqa: F401 - only probing whether it is installed
    except ImportError:
        if not install:
            return None, "triton is not installed"
        proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "triton"], capture_output=True, text=True,
                              timeout=900)
        if proc.returncode != 0:
            return None, f"pip install triton failed: {(proc.stderr or proc.stdout)[-400:]}"
    try:
        import triton  # noqa: F811 - imported again after a possible install

        return triton, str(triton.__version__)
    except Exception as exc:  # noqa: BLE001
        return None, f"triton import failed: {exc!r}"[:400]


def build_kernel():
    import triton
    import triton.language as tl

    try:
        div_rn = tl.math.div_rn
    except AttributeError:  # older layouts kept it under libdevice
        from triton.language.extra import libdevice

        div_rn = libdevice.div_rn

    @triton.jit
    def fused_features(bid_px, bid_sz, ask_px, ask_sz, product, tick_usd, n, n_stride,
                       spread_ticks, spread_usd, depth_bid, depth_ask, imbalance10, micro,
                       BLOCK: tl.constexpr, LEVELS: tl.constexpr, TOP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        db = tl.zeros([BLOCK], dtype=tl.float32)
        da = tl.zeros([BLOCK], dtype=tl.float32)
        b10 = tl.zeros([BLOCK], dtype=tl.float32)
        a10 = tl.zeros([BLOCK], dtype=tl.float32)
        bs0 = tl.load(bid_sz + offs, mask=mask, other=0.0)
        as0 = tl.load(ask_sz + offs, mask=mask, other=0.0)
        for lvl in tl.static_range(LEVELS):
            b = tl.load(bid_sz + lvl * n_stride + offs, mask=mask, other=0.0)
            a = tl.load(ask_sz + lvl * n_stride + offs, mask=mask, other=0.0)
            if lvl < TOP:
                b10 += b
                a10 += a
            db += b
            da += a
        bb = tl.load(bid_px + offs, mask=mask, other=0)
        ba = tl.load(ask_px + offs, mask=mask, other=0)
        both = (bb > 0) & (ba > 0)
        s = tl.where(both, ba - bb, 0)
        pr = tl.load(product + offs, mask=mask, other=0).to(tl.int32)
        tick = tl.load(tick_usd + pr, mask=mask, other=0.0)
        tot = b10 + a10
        im = tl.where(tot > 0.0, div_rn(b10 - a10, tl.where(tot > 0.0, tot, 1.0)), 0.0)
        tot0 = bs0 + as0
        ratio = div_rn(bs0 - as0, tl.where(tot0 > 0.0, tot0, 1.0))
        mp = tl.where(both & (tot0 > 0.0), (0.5 * (ba - bb).to(tl.float32)) * ratio, 0.0)
        tl.store(spread_ticks + offs, s, mask=mask)
        tl.store(spread_usd + offs, s.to(tl.float32) * tick, mask=mask)
        tl.store(depth_bid + offs, db, mask=mask)
        tl.store(depth_ask + offs, da, mask=mask)
        tl.store(imbalance10 + offs, im, mask=mask)
        tl.store(micro + offs, mp, mask=mask)

    return fused_features


class _Pointer:
    """What Triton's launcher needs from an argument: a device address and a NumPy-style dtype."""

    def __init__(self, arr) -> None:
        self._arr = arr
        self.dtype = arr.dtype

    def data_ptr(self) -> int:
        return int(self._arr.data.ptr)


def as_device_args(arrays):
    """Zero-copy torch views of CuPy arrays when torch is present, pointer adapters otherwise."""
    try:
        import torch

        return [torch.as_tensor(a, device="cuda") for a in arrays], "torch.as_tensor (__cuda_array_interface__)"
    except Exception:  # noqa: BLE001
        return [_Pointer(a) for a in arrays], "pointer adapter"


class TritonRung:
    """Launches the Triton kernel on the ladder's arrays; ``config`` is (snapshots per program, warps)."""

    def __init__(self, lad) -> None:
        self.kernel = build_kernel()
        ins = [lad.d["bid_px"], lad.d["bid_sz"], lad.d["ask_px"], lad.d["ask_sz"], lad.d_prod, lad.d_tick]
        outs = list(lad.outs())
        args, self.interop = as_device_args(ins + outs)
        self.ins, self.outs = args[:6], args[6:]
        self.n, self.n_stride, self.levels = int(lad.n), int(lad.n_stride), int(lad.levels)
        self.config = CONFIGS[1]
        # Triton launches on the stream its driver reports as current; the ladder's CUDA events are
        # recorded on lad.stream, so the timings only mean something when the two are the same stream.
        import triton

        drv = triton.runtime.driver.active
        self.stream_ptr = int(drv.get_current_stream(drv.get_current_device()))
        self.ladder_stream_ptr = int(lad.stream.ptr)
        if self.stream_ptr != self.ladder_stream_ptr:
            raise RuntimeError(f"Triton launches on stream {self.stream_ptr:#x}, the ladder times stream "
                               f"{self.ladder_stream_ptr:#x}; refusing to time it")

    def launch(self, config: Optional[Tuple[int, int]] = None) -> None:
        block, warps = config or self.config
        grid = ((self.n + block - 1) // block,)
        self.kernel[grid](*self.ins, self.n, self.n_stride, *self.outs, BLOCK=block, LEVELS=self.levels, TOP=10,
                          num_warps=warps)

    def launcher(self, config: Optional[Tuple[int, int]] = None) -> Callable[[], None]:
        return lambda: self.launch(config)
