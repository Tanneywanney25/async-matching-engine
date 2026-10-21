"""Host side of ``cuda/ddscan.cu``: a double-double inclusive prefix scan for any length."""

from __future__ import annotations

import math

import numpy as np

from bench.common import cupy, raw_module

DD_BLOCK = 1024
DD_THREADS = 256


class DDScan:
    """``scan(x, hi, lo)`` writes the compensated inclusive prefix of float64 ``x`` as ``hi + lo``."""

    def __init__(self) -> None:
        self.cp = cupy()
        mod = raw_module(["ddscan.cu"])
        self.k_block = mod.get_function("dd_block_scan")
        self.k_offsets = mod.get_function("dd_block_offsets")
        self.k_add = mod.get_function("dd_add_offsets")
        self.k_window = mod.get_function("window_diff_dd")
        self.bufs: dict = {}

    def scan(self, x, hi, lo) -> None:
        n = int(x.size)
        nb = max(1, math.ceil(n / DD_BLOCK))
        if nb not in self.bufs:
            self.bufs[nb] = (self.cp.empty(nb, self.cp.float64), self.cp.empty(nb, self.cp.float64))
        bh, bl = self.bufs[nb]
        self.k_block((nb,), (DD_THREADS,), (x, hi, lo, bh, bl, np.int64(n)))
        if nb > 1:
            self.k_offsets((1,), (1,), (bh, bl, np.int64(nb)))
            self.k_add((max(1, math.ceil(n / 256)),), (256,), (hi, lo, bh, bl, np.int64(n)))

    def window(self, hi, lo, window: int, out) -> None:
        n = int(hi.size)
        self.k_window((max(1, math.ceil(n / 256)),), (256,), (hi, lo, np.int64(n), np.int32(window), out))
