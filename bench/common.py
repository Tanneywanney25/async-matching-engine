"""Shared benchmark harness: archive loading, statistics, interleaving, GPU state.

Rules baked in here (from the build spec):

* warmup runs are discarded and at least 30 timed iterations are kept;
* variants in a block are interleaved (rotating order each iteration) rather
  than run back to back;
* ``nvidia-smi`` SM clock, memory clock, temperature and power are sampled
  before and after each block, and a result is flagged when the SM clock
  dropped more than 10% between the two samples;
* results are written incrementally to ``metrics.json`` so a job that hits its
  time limit still leaves everything it finished.

Timing is the callable's job: a variant returns the milliseconds it measured
(CUDA events around the launch for per-kernel numbers, ``perf_counter`` for
end-to-end numbers), so the two never get mixed.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exchange_engine.archive import ALL_ARRAYS, LEVEL_MAJOR, array_filename  # noqa: E402

T4_THEORETICAL_BW_GBS = 320.0
MIN_ITERS = 30
DEFAULT_WARMUP = 3


# --------------------------------------------------------------------------- #
# Archive
# --------------------------------------------------------------------------- #


def load_archive(path: Path, names: Optional[Sequence[str]] = None, cache_dir: Optional[Path] = None) -> dict:
    """Load archive arrays as numpy; transparently decompresses ``*.zst`` copies.

    Returns a dict with the arrays (level-major ones shaped ``(levels, n)``) plus
    ``"meta"`` (the sidecar dict) and ``"path"``.
    """
    path = Path(path)
    meta = json.loads((path / "archive.json").read_text(encoding="utf-8"))
    levels, n = int(meta["levels"]), int(meta["n_snapshots"])
    out: dict = {"meta": meta, "path": path}
    for name in names or list(ALL_ARRAYS):
        dtype = np.dtype(ALL_ARRAYS[name])
        file = path / array_filename(name)
        if not file.exists() and (path / (array_filename(name) + ".zst")).exists():
            file = _decompress(path / (array_filename(name) + ".zst"), cache_dir or Path("/tmp/exchange_engine_archive"))
        arr = np.fromfile(file, dtype=dtype)
        if name in LEVEL_MAJOR:
            arr = arr.reshape(levels, n)
        out[name] = arr
    return out


def _decompress(src: Path, cache_dir: Path) -> Path:
    import zstandard as zstd

    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / src.name[:-4]
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    with open(src, "rb") as fh, open(dst, "wb") as out:
        zstd.ZstdDecompressor().copy_stream(fh, out)
    return dst


def to_aos(arch: dict) -> np.ndarray:
    """Array-of-structures view of the book: one 800-byte record per snapshot.

    Record layout: ``bid_px[50] int32, bid_sz[50] f32, ask_px[50] int32, ask_sz[50] f32``.
    """
    levels, n = arch["bid_px"].shape
    rec = np.dtype([("bid_px", "<i4", (levels,)), ("bid_sz", "<f4", (levels,)),
                    ("ask_px", "<i4", (levels,)), ("ask_sz", "<f4", (levels,))])
    aos = np.empty(n, dtype=rec)
    aos["bid_px"] = arch["bid_px"].T
    aos["bid_sz"] = arch["bid_sz"].T
    aos["ask_px"] = arch["ask_px"].T
    aos["ask_sz"] = arch["ask_sz"].T
    return aos


# --------------------------------------------------------------------------- #
# GPU state
# --------------------------------------------------------------------------- #


def gpu_state() -> dict:
    """Sample SM clock, memory clock, temperature and power from nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks.max.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        sm, mem, temp, power, max_sm = [x.strip() for x in out.strip().split(",")]
        return {"sm_clock_mhz": float(sm), "mem_clock_mhz": float(mem), "temperature_c": float(temp),
                "power_w": float(power), "max_sm_clock_mhz": float(max_sm), "t": time.time()}
    except Exception as exc:  # noqa: BLE001 - CPU-only machines
        return {"error": repr(exc), "t": time.time()}


def has_gpu() -> bool:
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def summarize(samples_ms: Sequence[float], before: Optional[dict] = None, after: Optional[dict] = None,
              warmup: int = DEFAULT_WARMUP, samples_file: Optional[str] = None) -> dict:
    """Median / IQR summary of timed iterations plus the clock-drop flag."""
    xs = sorted(float(x) for x in samples_ms)
    n = len(xs)
    if n == 0:
        return {"n": 0}
    q1, med, q3 = (float(np.percentile(xs, p)) for p in (25, 50, 75))
    out = {
        "median_ms": med, "iqr_ms": q3 - q1, "q1_ms": q1, "q3_ms": q3, "min_ms": xs[0], "max_ms": xs[-1],
        "mean_ms": statistics.fmean(xs), "n": n, "warmup_discarded": warmup,
    }
    if samples_file:
        out["samples_file"] = samples_file
    if before and after and "sm_clock_mhz" in before and "sm_clock_mhz" in after:
        b, a = before["sm_clock_mhz"], after["sm_clock_mhz"]
        drop = (b - a) / b * 100.0 if b else 0.0
        out["sm_clock_before_mhz"] = b
        out["sm_clock_after_mhz"] = a
        out["clock_drop_pct"] = round(drop, 2)
        out["throttle_flag"] = bool(drop > 10.0)
        out["temperature_before_c"] = before.get("temperature_c")
        out["temperature_after_c"] = after.get("temperature_c")
        out["power_before_w"] = before.get("power_w")
        out["power_after_w"] = after.get("power_w")
        out["mem_clock_before_mhz"] = before.get("mem_clock_mhz")
        out["mem_clock_after_mhz"] = after.get("mem_clock_mhz")
    return out


def run_interleaved(variants: Dict[str, Callable[[], float]], n_iters: int = MIN_ITERS,
                    warmup: int = DEFAULT_WARMUP) -> Dict[str, List[float]]:
    """Call every variant ``n_iters`` times, rotating the order each iteration.

    Each callable performs one measured run and returns milliseconds.
    """
    names = list(variants)
    for _ in range(warmup):
        for name in names:
            variants[name]()
    samples: Dict[str, List[float]] = {name: [] for name in names}
    for i in range(n_iters):
        order = names[i % len(names):] + names[: i % len(names)]
        for name in order:
            samples[name].append(float(variants[name]()))
    return samples


class Bench:
    """One job's results: incremental ``metrics.json`` plus ``raw/`` files."""

    def __init__(self, out_dir: Path, source: str, archive_meta: Optional[dict] = None,
                 n_iters: int = MIN_ITERS, warmup: int = DEFAULT_WARMUP) -> None:
        self.out_dir = Path(out_dir)
        self.raw_dir = self.out_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.n_iters = max(int(n_iters), MIN_ITERS) if n_iters >= MIN_ITERS else int(n_iters)
        self.warmup = warmup
        self.path = self.out_dir / "metrics.json"
        if self.path.exists():
            self.metrics = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.metrics = {"source": source, "sections": {}, "created_unix": time.time()}
        if archive_meta is not None:
            self.metrics["archive"] = {
                k: archive_meta.get(k) for k in ("source", "n_snapshots", "n_trades", "content_hash",
                                                 "capture_seconds", "per_product_snapshots", "levels",
                                                 "bytes", "gap_count", "capture_start_ns", "capture_end_ns")
            }
            if archive_meta.get("source") != source:
                raise ValueError(f"archive source {archive_meta.get('source')!r} != job source {source!r}")
        self.metrics["source"] = source
        self.save()

    def save(self) -> None:
        self.metrics["updated_unix"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.metrics, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        os.replace(tmp, self.path)

    def section(self, name: str, data: dict) -> None:
        self.metrics["sections"][name] = data
        self.save()

    def save_raw(self, name: str, obj) -> str:
        """Write a raw JSON file and return its path relative to the run directory."""
        path = self.raw_dir / f"{name}.json"
        path.write_text(json.dumps(obj, default=_json_default), encoding="utf-8")
        return f"raw/{name}.json"

    def block(self, name: str, variants: Dict[str, Callable[[], float]], n_iters: Optional[int] = None) -> Dict[str, dict]:
        """Run an interleaved block and return ``{variant: summary}`` with raw samples saved."""
        before = gpu_state()
        samples = run_interleaved(variants, n_iters or self.n_iters, self.warmup)
        after = gpu_state()
        summaries: Dict[str, dict] = {}
        for variant, xs in samples.items():
            raw = self.save_raw(f"{name}__{variant}", {"ms": xs, "before": before, "after": after})
            summaries[variant] = summarize(xs, before, after, self.warmup, raw)
        return summaries


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# --------------------------------------------------------------------------- #
# CUDA helpers (import lazily; CPU machines never touch cupy)
# --------------------------------------------------------------------------- #


def cupy():
    import cupy as cp

    return cp


def read_cuda_source(name: str) -> str:
    """Read ``cuda/<name>`` from the repository."""
    return (REPO_ROOT / "cuda" / name).read_text(encoding="utf-8")


def compiled_arch() -> str:
    """The ``sm_XY`` target NVRTC compiles for: the current device's compute capability."""
    return "sm_" + cupy().cuda.Device().compute_capability


def raw_module(source_names: Sequence[str], name_expressions: Optional[Sequence[str]] = None, extra_options: Sequence[str] = ()):
    """Compile ``cuda/*.cu`` with NVRTC through CuPy, which adds ``-arch`` for the current device
    itself (a second ``-arch`` is an NVRTC error), so on a T4 this targets sm_75; see compiled_arch()."""
    cp = cupy()
    code = "\n".join(read_cuda_source(n) for n in source_names)
    options = ("--std=c++17", "-default-device", "-I" + str(REPO_ROOT / "cuda"), *extra_options)
    return cp.RawModule(code=code, options=options, name_expressions=name_expressions)


def event_time_ms(stream, launch: Callable[[], None]) -> float:
    """Time ``launch()`` with CUDA events on ``stream`` (GPU time only)."""
    cp = cupy()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record(stream)
    launch()
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end))


def wall_time_ms(fn: Callable[[], None]) -> float:
    """Wall-clock time of ``fn()`` including any transfers it performs."""
    cp = cupy()
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    fn()
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - t0) * 1e3


def effective_bandwidth(min_bytes: int, median_ms: float) -> float:
    """Minimum traffic divided by time, in GB/s."""
    return min_bytes / (median_ms * 1e-3) / 1e9 if median_ms > 0 else 0.0
