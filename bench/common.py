"""Shared benchmark harness: archive loading, statistics, interleaving, GPU state.

Rules baked in here (from the build spec):

* warmup runs are discarded and at least 30 timed iterations are kept;
* variants in a block are interleaved (rotating order each iteration) rather
  than run back to back;
* ``nvidia-smi`` SM clock, memory clock, temperature and power are sampled
  before and after each block, for every card the block uses, and a result is
  flagged when any of those cards' SM clock dropped more than 10% between the
  two samples;
* results are written incrementally to ``metrics.json`` so a job that hits its
  time limit still leaves everything it finished.

Timing is the callable's job: a variant returns the milliseconds it measured
(CUDA events around the launch for per-kernel numbers, ``perf_counter`` for
end-to-end numbers), so the two never get mixed.
"""

from __future__ import annotations

import atexit
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


GPU_QUERY = "index,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks.max.sm"
GPU_KEYS = ("sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_w", "max_sm_clock_mhz")


def parse_gpu_query(text: str) -> Dict[int, dict]:
    """Parse ``nvidia-smi --query-gpu=<GPU_QUERY> --format=csv,noheader,nounits``: one line per card.

    Kaggle's T4 sessions have two cards, so the output has two lines; splitting the whole output
    on commas (what 0.2.0 did) raised, and every block recorded an error instead of its clocks.
    """
    cards: Dict[int, dict] = {}
    for line in text.strip().splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 1 + len(GPU_KEYS):
            raise ValueError(f"unexpected nvidia-smi line {line!r}")
        values = {}
        for key, raw in zip(GPU_KEYS, fields[1:]):
            try:
                values[key] = float(raw)
            except ValueError:  # "[N/A]" on some fields of some boards
                values[key] = None
        cards[int(fields[0])] = values
    return cards


def gpu_state(devices: Sequence[int] = (0,)) -> dict:
    """Sample SM clock, memory clock, temperature and power from nvidia-smi.

    The first device's values sit at the top level (the shape every section and the renderer
    read); ``per_gpu`` holds every requested device, keyed by its index as a string.
    """
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        cards = parse_gpu_query(out)
        state: dict = dict(cards[devices[0]])
        state["per_gpu"] = {str(d): cards[d] for d in devices}
        state["t"] = time.time()
        return state
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
    if before and after and before.get("sm_clock_mhz") and after.get("sm_clock_mhz") is not None:
        b, a = before["sm_clock_mhz"], after["sm_clock_mhz"]
        drop = (b - a) / b * 100.0 if b else 0.0
        out["sm_clock_before_mhz"] = b
        out["sm_clock_after_mhz"] = a
        out["clock_drop_pct"] = round(drop, 2)
        out["throttle_flag"] = bool(drop > 10.0)
        per_b, per_a = before.get("per_gpu") or {}, after.get("per_gpu") or {}
        if len(per_b) > 1:
            drops = {d: round((v["sm_clock_mhz"] - per_a[d]["sm_clock_mhz"]) / v["sm_clock_mhz"] * 100.0, 2)
                     for d, v in per_b.items() if v.get("sm_clock_mhz") and (per_a.get(d) or {}).get("sm_clock_mhz") is not None}
            out["clock_drop_pct_per_gpu"] = drops
            out["clock_drop_pct"] = max([out["clock_drop_pct"], *drops.values()])
            out["throttle_flag"] = bool(out["clock_drop_pct"] > 10.0)
    elif before is not None or after is not None:
        out["clocks_missing"] = (before or {}).get("error") or (after or {}).get("error") or "no SM clock sample"
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

    def block(self, name: str, variants: Dict[str, Callable[[], float]], n_iters: Optional[int] = None,
              devices: Sequence[int] = (0,)) -> Dict[str, dict]:
        """Run an interleaved block and return ``{variant: summary}`` with raw samples saved.

        ``devices`` lists the cards the block runs on; their clocks are sampled before and after.
        """
        before = gpu_state(devices)
        samples = run_interleaved(variants, n_iters or self.n_iters, self.warmup)
        after = gpu_state(devices)
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
    itself, as it does -default-device (a duplicate is an NVRTC error; NVRTC defaults to C++17), so on a T4 this targets sm_75; see compiled_arch().

    Kernels fetched with ``get_function`` record every distinct launch configuration they are
    called with (see :class:`LaunchRecorder`), which the ``resources`` section turns into
    theoretical occupancy for exactly the configurations that were benchmarked."""
    cp = cupy()
    code = "\n".join(read_cuda_source(n) for n in source_names)
    options = ("-I" + str(REPO_ROOT / "cuda"), *extra_options)  # CuPy adds -arch and -default-device itself
    return RecordingModule(cp.RawModule(code=code, options=options, name_expressions=name_expressions),
                           tuple(source_names), tuple(options))


# --------------------------------------------------------------------------- #
# Launch recorder
# --------------------------------------------------------------------------- #


def _dim3(value) -> tuple:
    dims = tuple(int(v) for v in value) if isinstance(value, (tuple, list)) else (int(value),)
    return dims + (1,) * (3 - len(dims))


class LaunchRecorder:
    """Distinct ``(grid, block, dynamic shared memory)`` configurations per kernel, with launch counts.

    One instance per process (:data:`LAUNCHES`). When ``BENCH_LAUNCH_LOG_DIR`` is set (``run_all``
    sets it to ``<out>/raw/launches``) the registry is written there at exit, so kernels launched in
    the Nsight Systems child processes are counted too.
    """

    def __init__(self) -> None:
        self.kernels: Dict[str, dict] = {}

    def note(self, name: str, sources: tuple, options: tuple, grid, block, shared_mem: int) -> None:
        entry = self.kernels.get(name)
        if entry is None:
            entry = self.kernels[name] = {"sources": list(sources), "options": list(options), "configs": {}}
        key = f"{_dim3(grid)}|{_dim3(block)}|{int(shared_mem)}"
        cfg = entry["configs"].get(key)
        if cfg is None:
            cfg = entry["configs"][key] = {"grid": list(_dim3(grid)), "block": list(_dim3(block)),
                                           "shared_mem": int(shared_mem), "launches": 0}
        cfg["launches"] += 1

    def snapshot(self) -> Dict[str, dict]:
        return {name: {"sources": e["sources"], "options": e["options"], "configs": list(e["configs"].values())}
                for name, e in self.kernels.items()}

    def dump(self, directory: Optional[str] = None) -> Optional[Path]:
        directory = directory or os.environ.get("BENCH_LAUNCH_LOG_DIR")
        if not directory or not self.kernels:
            return None
        path = Path(directory) / f"launches_{os.getpid()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot()), encoding="utf-8")
        return path


LAUNCHES = LaunchRecorder()
atexit.register(LAUNCHES.dump)


def merge_launch_logs(snapshots: Sequence[Dict[str, dict]]) -> Dict[str, dict]:
    """Combine registries from several processes, summing launch counts per configuration."""
    merged: Dict[str, dict] = {}
    for snap in snapshots:
        for name, entry in snap.items():
            dst = merged.setdefault(name, {"sources": entry["sources"], "options": entry["options"], "configs": {}})
            for cfg in entry["configs"]:
                key = f"{tuple(cfg['grid'])}|{tuple(cfg['block'])}|{cfg['shared_mem']}"
                if key in dst["configs"]:
                    dst["configs"][key]["launches"] += cfg["launches"]
                else:
                    dst["configs"][key] = dict(cfg)
    return {name: {**e, "configs": list(e["configs"].values())} for name, e in merged.items()}


class RecordingKernel:
    """A CuPy ``RawKernel`` that notes its launch configuration; everything else is forwarded."""

    __slots__ = ("_kernel", "_name", "_sources", "_options")

    def __init__(self, kernel, name: str, sources: tuple, options: tuple) -> None:
        self._kernel, self._name, self._sources, self._options = kernel, name, sources, options

    def __call__(self, grid, block, args, **kwargs):
        LAUNCHES.note(self._name, self._sources, self._options, grid, block, kwargs.get("shared_mem", 0))
        return self._kernel(grid, block, args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._kernel, attr)

    @property
    def raw(self):
        return self._kernel


class RecordingModule:
    """A CuPy ``RawModule`` whose ``get_function`` returns :class:`RecordingKernel` objects."""

    def __init__(self, module, sources: tuple, options: tuple) -> None:
        self._module, self.sources, self.options = module, sources, options

    def get_function(self, name: str) -> RecordingKernel:
        return RecordingKernel(self._module.get_function(name), name, self.sources, self.options)

    def __getattr__(self, attr):
        return getattr(self._module, attr)


def event_time_ms(stream, launch: Callable[[], None]) -> float:
    """Time ``launch()`` with CUDA events on ``stream`` (GPU time only)."""
    cp = cupy()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record(stream)
    launch()
    end.record(stream)
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end))


SPIN_US = 300.0  # busy-wait ahead of a queued timing: far longer than any host dispatch measured here
SPIN_SOURCE = r"""
extern "C" __global__ void spin_wait(long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) { }
}"""
_SPIN_KERNEL = None


def spin_cycles(spin_us: float = SPIN_US) -> int:
    """SM cycles for ``spin_us`` at the device's rated clock (the wait only lengthens if the clock is lower)."""
    cp = cupy()
    khz = int(cp.cuda.Device().attributes["ClockRate"])
    return int(spin_us * khz / 1000.0)


def queued_time_ms(stream, launch: Callable[[], None], cycles: int, host_us: Optional[List[float]] = None) -> float:
    """GPU time of ``launch()`` with its work already queued when the start event fires.

    :func:`event_time_ms` records its start event on an idle GPU, so the time the host takes to
    dispatch the launch lands inside the measurement. Here a one-thread busy-wait kernel runs first;
    the start event, the launch and the end event are enqueued while it spins, so the events bracket
    only GPU execution. ``host_us`` collects the host-side enqueue time of each ``launch()``.
    """
    global _SPIN_KERNEL
    cp = cupy()
    if _SPIN_KERNEL is None:
        _SPIN_KERNEL = cp.RawKernel(SPIN_SOURCE, "spin_wait")
    start, end = cp.cuda.Event(), cp.cuda.Event()
    _SPIN_KERNEL((1,), (1,), (np.int64(cycles),), stream=stream)
    start.record(stream)
    t0 = time.perf_counter()
    launch()
    if host_us is not None:
        host_us.append((time.perf_counter() - t0) * 1e6)
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
