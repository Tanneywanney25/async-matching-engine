"""Render every document that contains a measured number, from committed results only.

    python scripts/render_docs.py [--out-dir DIR] [--allow-synthetic]

Reads ``results/*/manifest.json`` and ``results/*/metrics.json``. Only runs whose metrics and
manifest both say ``"source": "real"``, whose manifest GPU name contains ``T4`` and which were
full (not ``--quick``) runs feed the GPU documents; sections are merged across such runs with the
latest run winning per section. Synthetic input is refused: a synthetic run is never selected, and
if no real T4 run exists the renderer exits non-zero instead of rendering.

Every number goes through :meth:`Renderer.fmt`, which records ``{rendered token: [{run_id, path,
value}]}`` in ``results/provenance.json``; ``scripts/check_numbers.py`` checks the documents
against it. Writes ``BENCHMARKS.md``, the README GPU section (between the gpu-lab markers) and its
local Benchmark table, ``DESIGN.md``, ``INTERVIEW_NOTES.md``, ``BULLETS.md``, the ``CHANGELOG.md``
entry, ``public/data/gpu_lab.json`` and ``public/data/pipeline_timeline.svg``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import check_bullets  # noqa: E402

NUM_RE = re.compile(r"(?<![A-Za-z_\d.])\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?")
# Raw links on the dashboard: the results are merged into main, which keeps every run directory.
RAW_BASE = "https://github.com/Tanneywanney25/async-matching-engine/blob/main/"
MARK_START, MARK_END = "<!-- gpu-lab:start -->", "<!-- gpu-lab:end -->"
DATA_LIMIT_BYTES = 5 * 1024 * 1024
LADDER_ORDER = ["v0_aos", "v1_soa", "v2_soa_fused", "v3_soa_fused_vec4", "v4_soa_fused_vec4_sweep", "triton_fused"]
README_DOC = "README.md (GPU section)"
# Markdown prose never needs a brace, so one outside code means a template expression or a
# placeholder that was written out instead of evaluated (INTERVIEW_NOTES.md question 2 in 0.2.0).
LEFTOVER_RE = re.compile(r"[{}]|<\?|\$\(|%\(\w+\)")


def template_leftovers(text: str) -> List[str]:
    """Lines of ``text`` that still hold template syntax outside fenced code and inline code."""
    prose = re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    prose = re.sub(r"`[^`\n]*`", "``", prose)
    return [ln.strip() for ln in prose.splitlines() if LEFTOVER_RE.search(ln)]


class RefuseError(SystemExit):
    pass


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


class Run:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.id = path.name
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))

    @property
    def source(self) -> str:
        return str(self.metrics.get("source"))

    @property
    def real(self) -> bool:
        return self.source == "real" and self.manifest.get("source") == "real"

    @property
    def t4(self) -> bool:
        return "T4" in str(self.manifest.get("gpu_name") or "")

    @property
    def quick(self) -> bool:
        return bool((self.metrics.get("run_all") or {}).get("quick"))

    @property
    def started(self) -> float:
        return float(self.manifest.get("started_unix") or 0.0)


def load_runs(results: Path) -> List[Run]:
    runs = []
    for d in sorted(results.iterdir()) if results.exists() else []:
        if d.is_dir() and (d / "manifest.json").exists() and (d / "metrics.json").exists():
            try:
                runs.append(Run(d))
            except ValueError:
                continue
    return runs


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #


class Renderer:
    def __init__(self, runs: Sequence[Run], allow_synthetic: bool = False) -> None:
        self.runs = list(runs)
        gpu = [r for r in self.runs if r.t4 and (allow_synthetic or not r.quick) and (r.real or allow_synthetic)]
        self.refused = [r.id for r in self.runs if r.t4 and not r.real]
        if not gpu:
            raise RefuseError("no full real-data T4 run in results/: refusing to render (synthetic input is never rendered)")
        self.sections: Dict[str, Tuple[Run, dict]] = {}
        for run in sorted(gpu, key=lambda r: r.started):
            for name, data in (run.metrics.get("sections") or {}).items():
                self.sections[name] = (run, data)
        for name, (run, _data) in self.sections.items():
            if not run.real and not allow_synthetic:
                raise RefuseError(f"section {name} would come from non-real run {run.id}")
        self.latest = max(gpu, key=lambda r: r.started)
        local = [r for r in self.runs if r.id.startswith("engine-local-") and r.real]
        self.local_engine = max(local, key=lambda r: r.started) if local else None
        self.prov: Dict[str, List[dict]] = {}
        self.doc = ""  # the document being rendered; check_numbers.py matches tokens per document

    # -- access ---------------------------------------------------------------- #

    def has(self, section: str) -> bool:
        return section in self.sections

    def get(self, section: str, path: str, default: Any = None) -> Any:
        if section not in self.sections:
            return default
        cur: Any = self.sections[section][1]
        for part in path.split("."):
            if isinstance(cur, list):
                try:
                    cur = cur[int(part)]
                except (ValueError, IndexError):
                    return default
            elif isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def run_of(self, section: str) -> Run:
        return self.sections[section][0]

    def raw_link(self, section: str, rel: Optional[str], label: str = "raw") -> str:
        if not rel or section not in self.sections:
            return ""
        return f"[{label}](results/{self.run_of(section).id}/{rel})"

    # -- formatting ------------------------------------------------------------- #

    def record(self, text: str, run_id: str, path: str, value: Any) -> str:
        for tok in NUM_RE.findall(text):
            self.prov.setdefault(tok, []).append({"run_id": run_id, "path": path, "value": value, "doc": self.doc})
        return text

    def fmt(self, section: str, path: str, kind: str = "num", scale: float = 1.0, value: Any = None) -> str:
        v = self.get(section, path) if value is None else value
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "n/a"
        run = self.run_of(section) if section in self.sections else self.latest
        x = float(v) * scale if kind not in ("str",) else v
        text = format_value(x, kind)
        return self.record(text, run.id, f"{section}.{path}" + (f"*{scale:g}" if scale != 1.0 else ""), v)

    def fmt_archive(self, path: str, kind: str = "int", scale: float = 1.0) -> str:
        v = (self.latest.metrics.get("archive") or {}).get(path)
        if v is None:
            return "n/a"
        return self.record(format_value(float(v) * scale, kind), self.latest.id, f"archive.{path}", v)

    def fmt_m(self, section: str, path: str, kind: str = "ms") -> str:
        """``median (IQR x)`` of a measurement dict."""
        med = self.fmt(section, f"{path}.median_ms", kind)
        iqr = self.fmt(section, f"{path}.iqr_ms", kind)
        return f"{med} (IQR {iqr})"


def format_value(x: float, kind: str) -> str:
    if kind == "int":
        return f"{x:,.0f}"
    if kind == "pct":
        return f"{x:.1f}" if abs(x) < 1000 else f"{x:,.0f}"
    if kind == "x":
        return f"{x:.1f}" if x < 100 else f"{x:,.0f}"
    if kind == "gbs":
        return f"{x:.1f}"
    if kind == "sci":
        return f"{x:.1e}"
    if kind == "us":
        return f"{x:.2f}"
    if kind == "us3":  # sub-microsecond latencies of the C++ core
        return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"
    if kind == "r2":
        return f"{x:.4f}"
    if kind == "billions":
        return f"{x / 1e9:.2f}"
    if kind == "millions":
        return f"{x / 1e6:.1f}"
    if kind in ("ms", "num"):
        if x == 0:
            return "0"
        digits = max(0, 2 - int(math.floor(math.log10(abs(x)))))
        return f"{x:,.{min(digits, 6)}f}"
    return str(x)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def ceiling_words(r: "Renderer") -> str:
    kind = r.get("bandwidth", "ceiling_kind")
    return {"read_kernel": "measured read-bandwidth ceiling", "cudamemcpy": "measured cudaMemcpy ceiling",
            "copy_kernel": "measured copy ceiling"}.get(kind, "measured bandwidth ceiling")


def split_regimes(xs: Sequence[float]) -> Optional[float]:
    """The threshold between two levels of a sample set, when its sorted values jump by half or more and each side
    holds at least three samples; ``None`` for a single level."""
    ys = sorted(xs)
    best = None
    for k in range(3, len(ys) - 2):
        if ys[k - 1] > 0 and ys[k] / ys[k - 1] >= 1.5 and (best is None or ys[k] / ys[k - 1] > best[0]):
            best = (ys[k] / ys[k - 1], (ys[k] + ys[k - 1]) / 2.0)
    return best[1] if best else None


def monitor_timeline(run_dir: Path, block: str) -> Tuple[List[dict], List[dict]]:
    """A block's monitor rows, each with an ``epoch`` time aligned to the calls' ``time.time()`` stamps, and its calls."""
    sys.path.insert(0, str(REPO))
    import calendar

    from bench.common import parse_monitor

    try:
        rows = parse_monitor((run_dir / "raw" / f"{block}__gpu_monitor.csv").read_text(encoding="utf-8"))
        calls = json.loads((run_dir / "raw" / f"{block}__iterations.json").read_text(encoding="utf-8"))["calls"]
    except (OSError, ValueError, KeyError, TypeError):
        return [], []
    if not rows or not calls:
        return [], calls or []
    for x in rows:
        whole, _, frac = x["timestamp"].partition(".")
        x["epoch"] = calendar.timegm(time.strptime(whole, "%Y/%m/%d %H:%M:%S")) + float("0." + (frac or "0"))
    # nvidia-smi prints the container's local time; the monitor starts just before the first call,
    # so the offset to the calls' epoch times is that gap rounded to whole quarter hours
    offset = round((rows[0]["epoch"] - calls[0]["t0"]) / 900.0) * 900.0
    for x in rows:
        x["epoch"] -= offset
    return rows, calls


def block_clock(run_dir: Path, block: str) -> Optional[dict]:
    """A block's clock as seen during its timed iterations: monitor samples inside the span, plus its raw clock samples."""
    rows, calls = monitor_timeline(run_dir, block)
    b = a = {}
    for f in sorted((run_dir / "raw").glob(f"{block}__*.json")):  # the first of the block's sample files with clock samples
        if f.name.endswith("__iterations.json"):
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("before"), dict) and isinstance(d.get("after"), dict) \
                and d["before"].get("sm_clock_mhz") is not None and d["after"].get("sm_clock_mhz") is not None:
            b, a = d["before"], d["after"]
            break
    if not b:
        return None
    out = {"before": b["sm_clock_mhz"], "after": a["sm_clock_mhz"], "max": b.get("max_sm_clock_mhz"),
           "polled": (run_dir / "raw" / f"{block}__gpu_monitor.csv").exists(), "in_span": []}
    if rows and calls:
        lo, hi = min(c["t0"] for c in calls), max(c["t1"] for c in calls)
        out["in_span"] = [x for x in rows if lo <= x["epoch"] <= hi]
        if len(rows) > 1:
            gaps = sorted(rows[i + 1]["epoch"] - rows[i]["epoch"] for i in range(len(rows) - 1))
            out["interval_ms"] = gaps[len(gaps) // 2] * 1000.0
    span = out["in_span"]
    clocks = sorted([x["sm_clock_mhz"] for x in span] + [out["before"], out["after"]])  # timed span plus its two edges
    out["clock_min"], out["clock_max"], out["clock_median"] = clocks[0], clocks[-1], clocks[len(clocks) // 2]
    reasons: Dict[str, int] = {}
    for x in span:
        for n in x.get("reasons") or []:
            reasons[n] = reasons.get(n, 0) + 1
    out["reasons"] = reasons
    return out


def monitor_clock_at(r: Renderer, run_id: str, block: str, variant: str) -> List[Tuple[float, float, list]]:
    """``(ms, SM clock, active reasons)`` for every timed call of ``variant``, from the nearest monitor sample."""
    run_dir = next(run.path for run in r.runs if run.id == run_id)
    rows, calls = monitor_timeline(run_dir, block)
    if not rows:
        return []
    out = []
    for c in calls:
        if c["variant"] != variant:
            continue
        mid = (c["t0"] + c["t1"]) / 2.0
        x = min(rows, key=lambda row: abs(row["epoch"] - mid))
        out.append((c["ms"], x["sm_clock_mhz"], x.get("reasons") or []))
    return out


def v0_regimes_md(r: Renderer) -> List[str]:
    """v0's samples in two levels, set against the card's clock and power state when each was taken."""
    rungs = r.get("ladder", "rungs", []) or []
    idx = next((i for i, x in enumerate(rungs) if x["name"] == "v0_aos"), None)
    if idx is None:
        return []
    km = rungs[idx]["kernel_ms"]
    run = r.run_of("ladder")
    try:
        xs = json.loads((run.path / km["samples_file"]).read_text(encoding="utf-8"))["ms"]
    except (OSError, ValueError, KeyError):
        return []
    cut = split_regimes(xs)
    if cut is None:
        return []
    fast = [x for x in xs if x < cut]
    slow = [x for x in xs if x >= cut]
    first_fast = all(x < cut for x in xs[:len(fast)])
    rec = lambda v, kind, what: r.record(format_value(v, kind), run.id, f"ladder v0 samples ({km['samples_file']}): {what}", v)  # noqa: E731
    text = (f"v0 is the one rung whose time changes with the card's state. Its {rec(len(xs), 'int', 'samples')} interleaved samples "
            f"fall in two levels: {rec(len(fast), 'int', 'fast samples')}"
            + (" (the first ones)" if first_fast else "")
            + f" near {rec(sorted(fast)[len(fast) // 2], 'ms', 'fast median')} ms and {rec(len(slow), 'int', 'slow samples')} near "
              f"{rec(sorted(slow)[len(slow) // 2], 'ms', 'slow median')} ms.")
    mon = km.get("monitor") or {}
    at = monitor_clock_at(r, run.id, "ladder_kernel", "v0_aos") if mon else []
    if at:
        fc = [c for ms, c, _rs in at if ms < cut]
        sc = [c for ms, c, _rs in at if ms >= cut]
        capped = sum(1 for ms, _c, rs in at if ms >= cut and "sw_power_cap" in rs)
        if fc and sc:
            text += (f" In the background log, sampled every 100 ms, the fast ones fall at an SM clock of at most "
                     f"{rec(max(fc), 'int', 'fast clock max')} MHz, before the card boosted, and the slow ones at a median of "
                     f"{rec(sorted(sc)[len(sc) // 2], 'int', 'slow clock median')} MHz"
                     + (f", {rec(capped, 'int', 'slow samples under the power cap')} of them with the software power cap active"
                        if capped else "") + ".")
    alone = r.get("ladder", "v0_alone_ms") or {}
    if alone.get("median_ms") is not None:
        ac = block_clock(run.path, "ladder_v0_alone") or {}
        span = ac.get("in_span") or []
        text += (f" Timed alone, v0 took {r.fmt('ladder', 'v0_alone_ms.median_ms', 'ms')} ms at the median (IQR "
                 f"{r.fmt('ladder', 'v0_alone_ms.iqr_ms', 'ms')})"
                 + (f" with the card at {rec(ac['clock_min'], 'int', 'v0 alone in-span clock min')} to "
                    f"{rec(ac['clock_max'], 'int', 'v0 alone in-span clock max')} MHz during its timed iterations" if span else "")
                 + ", so the slow level is its steady state once the card is under sustained load, the state the table's median reflects.")
    fast_iters = [i for i, x in enumerate(xs) if x < cut]
    slow_iters = [i for i, x in enumerate(xs) if x >= cut]
    changes = {}
    for f in sorted((run.path / "raw").glob("ladder_kernel__*.json")):
        variant = f.name[len("ladder_kernel__"):-len(".json")]
        if variant in ("v0_aos", "iterations"):
            continue
        try:
            ys = json.loads(f.read_text(encoding="utf-8"))["ms"]
        except (OSError, ValueError, KeyError):
            continue
        a = sorted(ys[i] for i in fast_iters if i < len(ys))
        b = sorted(ys[i] for i in slow_iters if i < len(ys))
        if a and b:
            changes[variant] = (b[len(b) // 2] / a[len(a) // 2] - 1.0) * 100.0
    if changes:
        lo, hi = min(changes, key=changes.get), max(changes, key=changes.get)

        def change(v: str) -> str:
            c = changes[v]
            return f"{rec(abs(c), 'pct', f'{v} change across the switch')}% {'faster' if c < 0 else 'slower'} (`{v}`)"
        text += f" Over the same iterations the other variants changed between {change(lo)} and {change(hi)}."
    text += " Why the uncoalesced array-of-structures kernels slow down at the higher clock is not established by these measurements."
    return ["### v0 and the card's clock", "", text, ""]


def queued_md(r: Renderer) -> List[str]:
    """Every rung's GPU-only time (launches queued behind a busy-wait) beside its event-timed figure."""
    q = r.get("ladder", "queued") or {}
    if not q.get("kernel_ms"):
        return []
    out = ["### GPU time with the launches queued ahead", "",
           "The per-kernel times above record the start event on an idle GPU, so each includes the host's dispatch of the "
           f"launch. This block, interleaved like the others, queues each pass behind a {r.fmt('ladder', 'queued.spin_us', 'int')} µs "
           "busy-wait kernel, so the events bracket GPU execution only, and times how long the host took to enqueue the pass. "
           + (f"The slowest enqueue took {r.fmt('ladder', 'queued.host_enqueue_max_us', 'us')} µs, well inside the wait."
              if q.get("dispatch_hidden") else
              f"The slowest enqueue took {r.fmt('ladder', 'queued.host_enqueue_max_us', 'us')} µs, more than half the wait, so some "
              "dispatch may remain inside these figures."), "",
           "| Rung | Events around the launch, median ms | Launches queued, median ms (IQR) | Host enqueue, median µs |",
           "|---|---:|---:|---:|"]
    above = []
    for i, rung in ladder_rows(r):
        name = rung["name"]
        if name not in q["kernel_ms"]:
            continue
        out.append(f"| {rung_label(name)} | {r.fmt('ladder', f'rungs.{i}.kernel_ms.median_ms', 'ms')} | "
                   f"{r.fmt_m('ladder', f'queued.kernel_ms.{name}')} | {r.fmt('ladder', f'queued.host_enqueue_us.{name}.median_us', 'us')} |")
        if q["kernel_ms"][name]["median_ms"] > rung["kernel_ms"]["median_ms"]:
            above.append(name)
    note = ([f"Where the queued median sits above the event-timed one ({', '.join(f'`{n}`' for n in above)}), the difference "
             "cannot be dispatch, which only the event-timed figure contains. These runs do not separate the two blocks' "
             "different moments of the card's clock and power state from the effect of the near-idle busy-wait itself on that "
             "state" + (", and v0 is the rung most sensitive to it (see above)." if "v0_aos" in above else ".")] if above else [])
    return out + [""] + (note + [""] if note else [])


def triton_md(r: Renderer) -> List[str]:
    """The extra Triton rung, compared with v2 (the CUDA kernel it mirrors) and the best CUDA rung."""
    t = r.get("ladder", "triton") or {}
    if not t.get("attempted"):
        return []
    if not t.get("ok"):
        why = str(t.get("reason") or t.get("error") or "unknown")[:200].replace("`", "'")
        failed = [k for k, v in (t.get("verification_by_config") or {}).items() if not (v.get("verification") or {}).get("verified")]
        detail = (f" Configurations that failed the tolerance check against the NumPy reference: {', '.join(f'`{k}`' for k in failed)}."
                  if failed else "")
        return ["### Extra rung: the fused kernel in Triton", "",
                f"Triton `{t.get('version', '?')}` was not timed in this run: `{why}`.{detail}", ""]
    idx = {rung["name"]: i for i, rung in enumerate(r.get("ladder", "rungs", []) or [])}
    if "triton_fused" not in idx or "v2_soa_fused" not in idx:
        return []
    best_name = r.get("ladder", "speedups.best_rung")
    run_id = r.run_of("ladder").id

    def ratio(path_of, label: str) -> Optional[str]:
        a, b, c = (r.get("ladder", path_of(n)) for n in ("triton_fused", "v2_soa_fused", best_name))
        if None in (a, b, c):
            return None
        return (r.record(f"{a / b:.2f}", run_id, f"ladder Triton / v2, {label}", a / b),
                r.record(f"{a / c:.2f}", run_id, f"ladder Triton / best CUDA rung, {label}", a / c))

    ev = ratio(lambda n: f"rungs.{idx[n]}.kernel_ms.median_ms", "events around the launch")
    qd = ratio(lambda n: f"queued.kernel_ms.{n}.median_ms", "launches queued")
    same = t.get("identical_to_v2_all")
    sweep = [f"{r.fmt('ladder', f'triton.sweep.{i}.block_size', 'int')} snapshots and {r.fmt('ladder', f'triton.sweep.{i}.num_warps', 'int')} "
             f"warps per program: {r.fmt('ladder', f'triton.sweep.{i}.median_ms', 'ms')} ms" for i, _ in enumerate(t.get("sweep") or [])]
    best_label = rung_label(best_name)
    if qd:
        timing = (f"Measured with its launch queued ahead, the best one takes {qd[0]} times the GPU time of the CUDA v2 kernel "
                  f"it mirrors and {qd[1]} times that of the best CUDA rung ({best_label}). With the events around the launch, as "
                  f"in the ladder table, the ratios are {ev[0]} and {ev[1]}: enqueuing one Triton launch took a median "
                  f"{r.fmt('ladder', 'queued.host_enqueue_us.triton_fused.median_us', 'us')} µs on the host, against "
                  f"{r.fmt('ladder', 'queued.host_enqueue_us.v2_soa_fused.median_us', 'us')} µs for v2 through CuPy.")
    else:
        timing = (f"The best one runs in {ev[0]} times the time of the CUDA v2 kernel it mirrors and {ev[1]} times that of the "
                  f"best CUDA rung ({best_label}), events around the launch.")
    return ["### Extra rung: the fused kernel in Triton", "",
            f"Triton `{t.get('version')}` compiles the same fused computation as v2 (level sums in level order, IEEE division) and "
            "reads the ladder's own device arrays, so it competes on identical data. Every launch configuration was checked "
            "against the NumPy reference before any was timed; the ones that passed, timed "
            + ("with launches queued ahead, GPU time only, " if t.get("sweep_timing") else "with events around the launch, ")
            + f"in an interleaved block of their own: {'; '.join(sweep)}. {timing} It sits in the ladder table (its Block column is "
            "snapshots per program) and is never counted as the best rung. Its outputs "
            + ("are bit-for-bit identical to v2's." if same else
               "differ from v2's in at least one output, though they pass the same tolerance check against the NumPy reference."
               if r.get("ladder", f"rungs.{idx['triton_fused']}.verified") else "FAIL the tolerance check against the NumPy reference."), ""]


def rung_label(name: str) -> str:
    return {"v0_aos": "v0 array-of-structures, one kernel per metric", "v1_soa": "v1 structure-of-arrays, one kernel per metric",
            "v2_soa_fused": "v2 structure-of-arrays, fused", "v3_soa_fused_vec4": "v3 fused, 128-bit loads",
            "v4_soa_fused_vec4_sweep": "v4 v3 + grid-stride, best launch configuration",
            "triton_fused": "Triton, fused like v2 (extra rung, best of a small sweep)"}.get(name, name)


def hardware_block(r: Renderer) -> str:
    man = r.latest.manifest
    gpu = r.latest.metrics.get("gpu") or {}
    a = r.latest.metrics.get("archive") or {}
    start = datetime.fromtimestamp((a.get("capture_start_ns") or 0) / 1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = datetime.fromtimestamp((a.get("capture_end_ns") or 0) / 1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sm = r.record(format_value(gpu.get("sm_count") or 0, "int"), r.latest.id, "gpu.sm_count", gpu.get("sm_count"))
    mem = r.record(format_value(gpu.get("memory_gb") or 0, "num"), r.latest.id, "gpu.memory_gb", gpu.get("memory_gb"))
    bw = r.record(format_value(gpu.get("theoretical_bw_gbs") or 0, "gbs"), r.latest.id, "gpu.theoretical_bw_gbs",
                  gpu.get("theoretical_bw_gbs"))
    hours = r.fmt_archive("capture_seconds", "num", 1 / 3600)
    return "\n".join([
        "| Item | Value |", "|---|---|",
        f"| GPU | {man.get('gpu_name')}, {sm} streaming multiprocessors, {mem} GB, compute capability `{gpu.get('compute_capability')}` |",
        f"| Theoretical memory bandwidth (from device properties) | {bw} GB/s |",
        f"| Driver / CUDA runtime / nvcc | `{man.get('driver')}` / `{gpu.get('cuda_runtime')}` / `{str(man.get('nvcc', '')).splitlines()[0] if man.get('nvcc') else ''}` |",
        f"| Nsight Systems | `{man.get('nsys')}` |",
        f"| Host CPU | `{man.get('cpu_model')}`, {r.record(format_value(man.get('cpu_cores') or 0, 'int'), r.latest.id, 'manifest.cpu_cores', man.get('cpu_cores'))} logical cores |",
        f"| Run | `{r.latest.id}` at git `{str(man.get('git_sha', ''))[:12]}`, NVRTC target `{gpu.get('nvrtc_arch')}` |",
        f"| Archive | {r.fmt_archive('n_snapshots')} snapshots, {r.fmt_archive('n_trades')} trades, source `{a.get('source')}` |",
        f"| Capture window | `{start}` to `{end}` ({hours} hours), sequence gaps: {r.fmt_archive('gap_count')} |",
        f"| Archive content hash | `{a.get('content_hash')}` |",
    ])


def clock_coverage(r: Renderer) -> Tuple[int, int, List[str]]:
    """``(blocks with SM clock samples, timed blocks, runs whose blocks lack them)`` over rendered sections."""
    sampled = total = 0
    missing_runs = set()
    for _name, (run, data) in r.sections.items():
        for _path, m in walk_measurements(data):
            total += 1
            if m.get("sm_clock_before_mhz") is not None and m.get("sm_clock_after_mhz") is not None:
                sampled += 1
            else:
                missing_runs.add(run.id)
    return sampled, total, sorted(missing_runs)


def clock_sentence(r: Renderer) -> str:
    sampled, total, missing = clock_coverage(r)
    if total and sampled == total:
        return "the SM clock, memory clock, temperature and power were sampled before and after every block."
    run_id = missing[0] if missing else r.latest.id
    of = r.record(format_value(total, "int"), run_id, "derived: timed blocks rendered", total)
    if not sampled:
        return (f"no SM clock, memory clock, temperature or power samples exist for the {of} timed blocks on this page, "
                "because the runs' sampler failed on the two-card session (see Clock throttling).")
    got = r.record(format_value(sampled, "int"), run_id, "derived: timed blocks with SM clock samples", sampled)
    return (f"SM clock, memory clock, temperature and power samples exist for {got} of the {of} timed blocks on this page; "
            "the rest come from runs whose sampler failed on the two-card session (see Clock throttling).")


def throttle_notes(r: Renderer) -> List[str]:
    notes = []
    for name, (run, data) in sorted(r.sections.items()):
        for path, m in walk_measurements(data):
            if m.get("throttle_flag"):
                drop = r.record(format_value(m.get("clock_drop_pct") or 0, "pct"), run.id, f"{name}.{path}.clock_drop_pct",
                                m.get("clock_drop_pct"))
                notes.append(f"- `{name}.{path}`: SM clock dropped {drop}% during the block.")
    return notes


SLOWDOWN_REASONS = ("hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake")


def monitored_blocks(r: Renderer) -> Dict[str, Tuple[str, dict]]:
    """Every timed block of the rendered runs that left clock samples, keyed by block name: ``(run id, block_clock)``.

    Only runs that used the background monitor are included; older runs are covered by the before/after flag.
    """
    out: Dict[str, Tuple[str, dict]] = {}
    for run in sorted({run for run, _ in r.sections.values()}, key=lambda x: x.started):
        raw = run.path / "raw"
        if not any(raw.glob("*__gpu_monitor.csv")):
            continue
        for block in sorted({f.name.split("__")[0] for f in raw.glob("*__*.json")}):
            info = block_clock(run.path, block)
            if info is not None:
                out[block] = (run.id, info)
    return out


def monitor_lines(r: Renderer) -> List[str]:
    blocks = monitored_blocks(r)
    if not blocks:
        return []
    run_id = next(iter(blocks.values()))[0]
    rec = lambda v, kind, what: r.record(format_value(v, kind), run_id, f"derived from raw/*__gpu_monitor.csv: {what}", v)  # noqa: E731
    names = lambda bs: ", ".join(f"`{b}`" for b in bs)  # noqa: E731
    polled = {b: i for b, (_r, i) in blocks.items() if i["polled"]}
    sampled = {b: i for b, i in polled.items() if i["in_span"]}
    thin = sorted(b for b, i in polled.items() if not i["in_span"])
    unpolled = {b: i for b, (_r, i) in blocks.items() if not i["polled"]}
    capped = sorted(b for b, i in sampled.items() if i["reasons"].get("sw_power_cap"))
    slowed = sorted(b for b, i in sampled.items() if any(i["reasons"].get(x) for x in SLOWDOWN_REASONS))
    top = max((i["max"] for _b, (_r, i) in blocks.items() if i.get("max")), default=None)
    # power-capped blocks boosted and were held down, which the sentence above covers; the rest ran mostly unboosted
    reduced = sorted(b for b, (_r, i) in blocks.items()
                     if top and i["clock_median"] < 0.9 * top and b not in capped)
    intervals = sorted(i["interval_ms"] for i in polled.values() if i.get("interval_ms"))
    lines = [f"`nvidia-smi` polled the SM clock and the driver's clock-event reasons in the background during the "
             f"{rec(len(polled), 'int', 'polled blocks')} blocks that time GPU work alone, at a requested 100 ms"
             + (f" (about every {rec(intervals[len(intervals) // 2], 'int', 'median polling interval ms')} ms in practice)" if intervals else "")
             + f". {rec(len(sampled), 'int', 'polled blocks with samples inside their timed iterations')} of them lasted long enough "
               "to be sampled during their timed iterations"
             + (f"; {names(thin)} did not, and for them only the samples taken after the warmup and after the last iteration "
                "exist." if thin else ".")
             + (f" In {rec(len(capped), 'int', 'sampled blocks with the power cap active')} of the sampled blocks the card reached "
                f"its software power cap ({names(capped)}), so their figures were measured at a power-limited clock"
                if capped else " None of the sampled blocks reached the card's software power cap")
             + ("; none reported a thermal or hardware slowdown." if not slowed else
                f"; {names(slowed)} also reported a thermal or hardware slowdown.")]
    if reduced:
        parts = []
        for b in reduced:
            i = blocks[b][1]
            lo, hi = rec(i["clock_min"], "int", f"{b} lowest clock"), rec(i["clock_max"], "int", f"{b} highest clock")
            parts.append(f"`{b}` ({lo} MHz)" if i["clock_min"] == i["clock_max"] else f"`{b}` ({lo} to {hi} MHz)")
        polled_red = [b for b in reduced if blocks[b][1]["polled"]]
        lines.append(f"For most of the timing of {rec(len(reduced), 'int', 'blocks mostly below 90% of the maximum clock')} other "
                     f"blocks the card's clock stayed below nine tenths of its {rec(top, 'int', 'maximum SM clock')} MHz maximum "
                     "(clock samples during and at the edges of the timing"
                     + ("; none of the polled ones showed a power-cap or slowdown reason, so the card had simply not boosted"
                        if polled_red and not any(blocks[b][1]["reasons"] for b in polled_red) else "")
                     + f"): {', '.join(parts)}. Their variants are interleaved, so the comparisons inside each block are like for "
                       "like, but their absolute figures, the bandwidths and rates on this page, were measured at those clocks.")
    if unpolled:
        flagged = sorted(b for b, i in unpolled.items() if i["before"] and (i["before"] - i["after"]) / i["before"] > 0.10)
        lines.append(f"The {rec(len(unpolled), 'int', 'unpolled blocks')} blocks that time host code or wall-clock transfers "
                     "were not polled, to keep the poller's own CPU use out of their host timings; the polled blocks are not free "
                     "of host work either, since their event-timed figures include host dispatch. For the unpolled blocks the SM "
                     "clock was sampled after the warmup and after the last iteration, and "
                     + (f"{names(flagged)} lost more than a tenth of it between the two." if flagged else
                        "none lost more than a tenth of it between the two."))
    return lines


def clock_section(r: Renderer) -> List[str]:
    sampled, total, missing = clock_coverage(r)
    notes = throttle_notes(r)
    lines = []
    mon = monitor_lines(r)
    if mon:
        lines += mon
    elif sampled:
        lines += notes or [("No timed block lost more than a tenth of its SM clock between the samples taken before and after it"
                            + (", on every card the block used." if sampled == total else
                               " among the blocks that have samples."))]
    if sampled < total:
        lines.append(f"Runs {', '.join(f'`{m}`' for m in missing)} recorded no clocks: on Kaggle's two-card T4 sessions their "
                     "sampler parsed both cards' `nvidia-smi` lines as one and failed on every block (see `DECISIONS.md`), "
                     "so nothing is claimed about throttling for their numbers.")
    return lines


def walk_measurements(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        if "median_ms" in obj and "n" in obj:
            yield prefix, obj
            return
        for k, v in obj.items():
            yield from walk_measurements(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_measurements(v, f"{prefix}.{i}")


def ladder_rows(r: Renderer) -> List[Tuple[int, dict]]:
    rungs = r.get("ladder", "rungs", []) or []
    return sorted(enumerate(rungs), key=lambda t: LADDER_ORDER.index(t[1]["name"]) if t[1]["name"] in LADDER_ORDER else 99)


def ladder_table(r: Renderer, with_raw: bool = True) -> str:
    rows = ["| Rung | Per-kernel median ms (IQR) | End to end ms" + (" | End to end, pageable ms" if with_raw else "")
            + " | Effective GB/s | % of measured ceiling | % of theoretical | Block | Verified |" + (" Raw |" if with_raw else ""),
            "|---|---:|---:|" + ("---:|" if with_raw else "") + "---:|---:|---:|---:|:---:|" + ("---|" if with_raw else "")]
    for i, rung in ladder_rows(r):
        p = f"rungs.{i}"
        pageable = f"{r.fmt('ladder', f'{p}.e2e_pageable_ms.median_ms', 'ms')} | " if with_raw else ""
        line = (f"| {rung_label(rung['name'])} | {r.fmt_m('ladder', f'{p}.kernel_ms')} | {r.fmt('ladder', f'{p}.e2e_ms.median_ms', 'ms')} | {pageable}"
                f"{r.fmt('ladder', f'{p}.effective_bw_gbs', 'gbs')} | {r.fmt('ladder', f'{p}.pct_of_ceiling', 'pct')} | "
                f"{r.fmt('ladder', f'{p}.pct_of_theoretical', 'pct')} | {r.fmt('ladder', f'{p}.block_size', 'int')} | "
                f"{'yes' if rung.get('verified') else 'NO'} |")
        if with_raw:
            line += f" {r.raw_link('ladder', (rung.get('kernel_ms') or {}).get('samples_file'))} |"
        rows.append(line)
    return "\n".join(rows)


NATIVE_PATHS = (("native", "C++ core, native harness (no Python in the loop)"),
                ("binding", "C++ core through the pybind11 binding, integer ticks and lots"),
                ("binding_decimal", "C++ core through `NativeOrderBook`, Decimal orders"),
                ("python_engine", "Python engine (the reference implementation)"))


def native_engine_md(r: Renderer) -> List[str]:
    """The C++ core's build checks and its latency table on the synthetic stream."""
    sec = "native_engine"
    base = r.sections[sec][1]
    b = "build."
    out = ["## C++20 order book core (CPU)", "",
           f"`native/`, built with CMake in the same Kaggle job (`{r.get(sec, 'build.compiler')}`, {r.get(sec, 'build.flags')}) and run on its host "
           f"CPU (`{r.get(sec, 'cpu_model')}`). Its native tests passed {r.fmt(sec, b + 'test_order_book.checks', 'int')} checks, among them a "
           "randomized differential test against a `std::map` and `std::deque` reference book, and a counting allocator saw "
           f"{r.fmt(sec, b + 'test_no_alloc.allocations', 'int')} heap allocations over {r.fmt(sec, b + 'test_no_alloc.operations', 'int')} "
           "operations once the book was built. "
           + (f"The differential fuzzer ran {r.fmt(sec, 'fuzz.streams', 'int')} seeded streams ({r.fmt(sec, 'fuzz.operations', 'int')} "
              f"operations) against the reference book with {r.fmt(sec, 'fuzz.divergences', 'int')} divergences, comparing fills, "
              "statuses, best prices and depth after every operation." if r.get(sec, "fuzz") else ""), ""]
    stream = r.record(str(base.get("stream", "")), r.run_of(sec).id, f"{sec}.stream (the run's own description)", base.get("stream"))
    out += [f"Stream: {stream}. Every path below replays the same operations in one process; they agreed on fills, "
            f"resting orders and cancels found: {'yes' if r.get(sec, 'consistent') else 'NO'}.", "",
            "| Path | add p50 µs | add p99 µs | add p99.9 µs | cancel p50 µs | cancel p99 µs | cancel p99.9 µs |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for path, label in NATIVE_PATHS:
        p = f"{path}."
        if r.get(sec, p + "add_order") is None:
            continue
        out.append(f"| {label} | " + " | ".join(r.fmt(sec, f"{p}{op}.{stat}", "us3") for op in ("add_order", "cancel_order")
                                                 for stat in ("p50_us", "p99_us", "p999_us")) + " |")
    found, cancels = r.get(sec, "counts.native.cancels_found"), r.get(sec, "cancels")
    found_txt = (f"Of the {r.fmt(sec, 'cancels', 'int')} cancels, {r.fmt(sec, 'counts.native.cancels_found', 'int')} found a resting "
                 "order; the others name orders the stream had already filled, so their latency is a lookup that finds nothing. "
                 if found is not None and cancels and found < cancels else "")
    engine_txt = ""
    if r.has("engine") and r.get("engine", "add_order.p99_us") is not None:
        where = ("in this job" if r.run_of("engine").id == r.run_of(sec).id else f"in run `{r.run_of('engine').id}`")
        engine_txt = (f" (`scripts/benchmark.py`, run separately {where}, reported a p99 `add_order` of "
                      f"{r.fmt('engine', 'add_order.p99_us', 'us')} µs)")
    out += ["", found_txt + "Every ratio here uses the Python engine row of this table, measured in the same process on the same "
            f"stream{engine_txt}. At p99 the native `add` is {faster_slower(r, sec, 'p99_add_python_engine_over_native')} than the "
            f"Python engine's, {faster_slower(r, sec, 'p99_add_python_engine_over_binding')} through the binding with integers and "
            f"{faster_slower(r, sec, 'p99_add_python_engine_over_binding_decimal')} with Decimal orders. The integer binding row adds "
            "the call from Python to the native work; the Decimal row adds, in `NativeOrderBook`, the conversion to ticks and lots "
            "and the Python `Fill` objects built for every fill. "
            "The native harness reads `std::chrono::steady_clock` twice per operation; a pair of reads costs "
            f"{r.fmt(sec, 'native.timer_overhead_us.p50', 'us3')} µs at the median, included above, not subtracted. "
            f"Raw: {r.raw_link(sec, r.get(sec, 'raw_file'))}.", ""]
    return out


REALISTIC_KINDS = (("limit", "limit"), ("market", "market"), ("cancel", "cancel"))


def faster_slower(r: Renderer, sec: str, path: str) -> str:
    """``"X times faster"`` for a ratio of at least 1, else ``"Y times slower"`` with Y its inverse."""
    v = r.get(sec, path)
    if v is None or not isinstance(v, (int, float)) or v <= 0:
        return "n/a"
    if v >= 1.0:
        return f"{r.fmt(sec, path, 'x')} times faster"
    inv = 1.0 / v
    return f"{r.record(format_value(inv, 'x'), r.run_of(sec).id, f'{sec}.{path} (inverted)', inv)} times slower"


def realistic_engine_md(r: Renderer) -> List[str]:
    """The same four paths on the operation stream derived from the recorded book (``native_engine.realistic``)."""
    sec, q = "native_engine", "realistic."
    real = r.get(sec, "realistic")
    if not real:
        return []
    stream = r.record(str(real.get("stream", "")), r.run_of(sec).id, f"{sec}.realistic.stream (the run's own description)",
                      real.get("stream"))
    out = ["### The same paths on a stream derived from the recorded book", "",
           f"Stream: {stream} ({r.fmt(sec, q + 'ops', 'int')} operations: {r.fmt(sec, q + 'op_counts.limit', 'int')} limit orders, "
           f"{r.fmt(sec, q + 'op_counts.market', 'int')} market orders, {r.fmt(sec, q + 'op_counts.cancel', 'int')} cancels). "
           + limit_match_sentence(r)
           + "The paths agreed on fills, resting orders and "
           f"cancels found: {'yes' if r.get(sec, q + 'consistent') else 'NO'}.", "",
           "| Path | limit p50 µs | limit p99 µs | market p50 µs | market p99 µs | cancel p50 µs | cancel p99 µs |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for path, label in NATIVE_PATHS:
        p = f"{q}{path}."
        if r.get(sec, p + "limit") is None:
            continue
        out.append(f"| {label} | " + " | ".join(r.fmt(sec, f"{p}{kind}.{stat}", "us3") for kind, _ in REALISTIC_KINDS
                                                 for stat in ("p50_us", "p99_us")) + " |")
    timer = (f" A pair of clock reads in this mode costs {r.fmt(sec, q + 'native.timer_overhead_us.p50', 'us3')} µs at the median, "
             "included above." if r.get(sec, q + "native.timer_overhead_us.p50") is not None else "")
    out += ["", f"Against the Python engine row, measured in the same process on the same operations, the native core is "
            f"{faster_slower(r, sec, q + 'p99_limit_python_engine_over_native')} at p99 on limit orders and "
            f"{faster_slower(r, sec, q + 'p99_market_python_engine_over_native')} on market orders; on cancels it is "
            f"{faster_slower(r, sec, q + 'p99_cancel_python_engine_over_native')}. On the synthetic stream above the `add` ratio "
            f"is {r.fmt(sec, 'p99_add_python_engine_over_native', 'x')}.{timer} "
            f"Raw: {r.raw_link(sec, r.get(sec, q + 'raw_file'))}.", ""]
    return out


def limit_match_sentence(r: Renderer) -> str:
    """Whether any derived limit order matched on arrival, as counted by the generator."""
    sec, q = "native_engine", "realistic."
    matched = r.get(sec, q + "limit_orders_matched")
    if matched is None:
        return ""
    if matched == 0:
        return ("Removals come before additions within each snapshot, and no limit order matched on arrival, so every fill "
                "comes from a market order. ")
    return (f"{r.fmt(sec, q + 'limit_orders_matched', 'int')} limit orders crossed a stale level outside a snapshot's reconciled "
            f"range and matched on arrival ({r.fmt(sec, q + 'limit_fills', 'int')} fills); every other limit order rested. ")


def int16_md(r: Renderer) -> List[str]:
    e = r.get("pipeline", "int16_encoding") or {}
    if not e.get("attempted") or "e2e_ms" not in e:
        return []
    q = "int16_encoding"
    change = (e["e2e_ms"]["median_ms"] / e["int32_e2e_ms"]["median_ms"] - 1) * 100.0
    run_id = r.run_of("pipeline").id
    change_txt = r.record(format_value(abs(change), "pct"), run_id, "pipeline.int16_encoding e2e change %", change)
    same = (e.get("identical_to_int32") or {}).get("identical")
    return ["### Narrower prices on the wire: int16 tick offsets", "",
            "Each side's best price travels as int32 per snapshot and every level as an int16 offset from it, decoded "
            "inside the kernel; a chunk holding any offset beyond 32,767 ticks is sent as int32 instead.", "",
            "| Measurement | int32 | int16 offsets |", "|---|---:|---:|",
            f"| Bytes host to device per pass | {r.fmt('pipeline', q + '.bytes_h2d_int32', 'int')} | {r.fmt('pipeline', q + '.bytes_h2d_int16', 'int')} |",
            f"| End to end, overlapped (median ms, IQR) | {r.fmt_m('pipeline', q + '.int32_e2e_ms')} | {r.fmt_m('pipeline', q + '.e2e_ms')} |",
            f"| Host to device, summed over chunks (ms) | {r.fmt('pipeline', 'breakdown.h2d_ms', 'ms')} | {r.fmt('pipeline', q + '.breakdown.h2d_ms', 'ms')} |",
            "",
            f"The encoding moves {r.fmt('pipeline', q + '.bytes_saved_pct', 'pct')}% fewer bytes; "
            + (f"{r.fmt('pipeline', q + '.overflow_fallbacks', 'int')} of {r.fmt('pipeline', q + '.chunks', 'int')} chunks "
               f"({r.fmt('pipeline', q + '.snapshots_in_fallback_chunks', 'int')} snapshots) fell back to int32 because a level sat more than "
               "32,767 ticks from the best price." if e.get("overflow_fallbacks") else
               f"none of the {r.fmt('pipeline', q + '.chunks', 'int')} chunks needed the int32 fallback, because every recorded level sat "
               "within 32,767 ticks of its side's best price.")
            + f" End to end the pass is {change_txt}% {'faster' if change < 0 else 'slower'}, and its "
            f"outputs are {'bit-for-bit identical to' if same else 'NOT identical to'} the int32 pass's "
            f"({'verified' if e.get('ok') else 'check failed'} against the NumPy reference too).", ""]


LINK_MEANING = {"SYS": "through the CPU interconnect between NUMA nodes", "NODE": "through PCIe host bridges within one NUMA node",
                "PHB": "through a PCIe host bridge, that is the CPU", "PXB": "through more than one PCIe switch",
                "PIX": "through a single PCIe switch"}


def two_gpu_link(t: dict) -> Tuple[str, str]:
    link = (t.get("topology") or {}).get("links", {}).get("GPU0-GPU1", "unknown")
    return link, LINK_MEANING.get(link, "by a path `nvidia-smi` does not name")


def two_gpu_kind(t: dict) -> str:
    """How much copy bandwidth the second card added: ``scales``, ``partial``, ``barely`` or ``none``."""
    ratio = t.get("h2d_aggregate_vs_one_gpu") or 0.0
    return "scales" if ratio >= 1.7 else "none" if ratio <= 1.05 else "barely" if ratio <= 1.25 else "partial"


KIND_FOUND = {"none": "that two cards copied no faster than one", "barely": "that two cards copied barely faster than one",
              "partial": "that the second card added only part of a card's copy bandwidth",
              "scales": "that two cards copied close to twice as fast as one"}
KIND_SHORT = {"none": "showed no gain", "barely": "showed almost none", "partial": "showed less", "scales": "showed close to double"}


def same_topology(a: dict, b: dict) -> bool:
    """Whether two runs' ``nvidia-smi`` views of the cards match: topology links and each card's bus id and rated link."""
    return bool(a.get("pcie")) and (a.get("topology") or {}).get("links") == (b.get("topology") or {}).get("links") \
        and a.get("pcie") == b.get("pcie")


def two_gpu_history(r: Renderer) -> List[Tuple[str, str, bool]]:
    """``(run id, kind, same topology as the rendered run)`` for every full real T4 run that measured two cards, oldest first."""
    if not r.has("pipeline"):
        return []
    t = r.sections["pipeline"][1].get("two_gpu") or {}
    out = []
    for run in sorted(r.runs, key=lambda x: x.started):
        if not (run.t4 and run.real and not run.quick):
            continue
        other = ((run.metrics.get("sections") or {}).get("pipeline") or {}).get("two_gpu") or {}
        if "two_gpu_ms" in other:
            out.append((run.id, two_gpu_kind(other), same_topology(t, other)))
    return out


def earlier_two_gpu_sentence(r: Renderer) -> str:
    """When full runs in different sessions disagree about the second card, every one of them, named."""
    hist = two_gpu_history(r)
    if len({k for _i, k, _s in hist}) < 2:
        return ""
    cur = r.run_of("pipeline").id
    found = "; ".join(f"`{rid}` found {KIND_FOUND[k]}" for rid, k, _s in hist)
    same = all(s for _i, _k, s in hist)
    total = r.record(format_value(len(hist), "int"), cur, "derived: full runs that measured two cards", len(hist))
    return (f" The result is not stable across Kaggle sessions. Of the {total} full runs that measured it, each in its own "
            f"session, {found}" + (", and their `nvidia-smi` topology, bus ids and rated links read the same" if same else "")
            + f". These runs do not show why; this page reports `{cur}`, the latest.")


def other_sessions_short(r: Renderer) -> str:
    """``" in this session, while 1 of the 2 other full runs ... showed close to double"`` when runs disagree, else ``""``."""
    hist = two_gpu_history(r)
    if len({k for _i, k, _s in hist}) < 2:
        return ""
    cur = r.run_of("pipeline").id
    mine = next((k for i, k, _s in hist if i == cur), None)
    others = [(i, k) for i, k, _s in hist if i != cur]
    differ = [k for _i, k in others if k != mine]
    same = all(s for _i, _k, s in hist)
    n_diff = r.record(format_value(len(differ), "int"), cur, "derived: other full runs with a different second-card verdict", len(differ))
    n_other = r.record(format_value(len(others), "int"), cur, "derived: other full runs that measured two cards", len(others))
    who = ("the other full run, in its own Kaggle session" if len(others) == 1 else
           f"{n_diff} of the {n_other} other full runs, each in its own Kaggle session")
    return (f" in this session, while {who}" + (" reporting the same topology," if same else ",") + " "
            + " or ".join(KIND_SHORT[k] for k in sorted(set(differ))))


def transfer_bound(r: Renderer) -> bool:
    share = r.get("pipeline", "transfer_share_pct")
    return share is not None and share >= 50.0


def two_gpu_md(r: Renderer) -> List[str]:
    t = r.get("pipeline", "two_gpu") or {}
    if not t.get("attempted"):
        return []
    if "two_gpu_ms" not in t:
        return ["### One card against two", "", f"Not measured in this run: `{str(t.get('reason') or t.get('error'))[:160]}`.", ""]
    q = "two_gpu"
    link, meaning = two_gpu_link(t)
    kind = two_gpu_kind(t)
    if kind == "scales":
        verdict = ("the two cards' copies run side by side at close to each card's own rate, so in this session each T4 had "
                   "its own path to host memory"
                   + (", and the whole pipeline scales with the second card" if (t.get("speedup") or 0.0) >= 1.5 else
                      ", yet the whole pipeline's speedup stays well short of that, so something besides the copies limits it"))
    elif kind in ("none", "barely"):
        verdict = (("two cards together copy no faster than one" if kind == "none" else "two cards together copy barely faster than one")
                   + ", so in this session the limit sat upstream of the cards' own links, on a path both shared"
                   + (", and the second card only added compute this transfer-bound pipeline does not need" if transfer_bound(r) else ""))
    else:
        verdict = ("the second card adds copy bandwidth but not a full card's worth, so in this session part of the path to "
                   "host memory was shared between them")
    now = {}
    for line in t.get("pcie_current_after_copies") or []:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3:
            now[parts[0]] = f"Gen{parts[1]} x{parts[2]}"
    cards = []
    for line in t.get("pcie") or []:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 4:
            cards.append(f"`GPU{parts[0]}` at `{parts[1]}`, rated `Gen{parts[2]} x{parts[3]}`"
                         + (f" and running at `{now[parts[0]]}` just after the copies" if parts[0] in now else ""))
    return ["### One card against two", "",
            f"`nvidia-smi topo -m` reports the two cards as `{link}`, connected {meaning}; each has its own PCIe link "
            f"({', '.join(cards)}). The archive is split in two contiguous halves of "
            f"{r.fmt('pipeline', q + '.chunks_per_device', 'int')} chunks, each card with its own pinned host buffer, module, "
            "streams and device buffers, and the two cards' chunks are issued alternately.", "",
            "| Measurement | One card | Two cards |", "|---|---:|---:|",
            f"| Whole pipeline, overlapped (median ms, IQR) | {r.fmt_m('pipeline', q + '.one_gpu_ms')} | {r.fmt_m('pipeline', q + '.two_gpu_ms')} |",
            f"| Copy-only pass, host to device (median ms, IQR) | {r.fmt_m('pipeline', q + '.h2d_only_one_gpu_ms')} | {r.fmt_m('pipeline', q + '.h2d_only_two_gpu_ms')} |",
            f"| Host-to-device bandwidth (GB/s, aggregate for two) | {r.fmt('pipeline', q + '.h2d_gbs_one_gpu', 'gbs')} | {r.fmt('pipeline', q + '.h2d_gbs_two_gpu_aggregate', 'gbs')} |",
            "",
            f"Two cards take {r.fmt('pipeline', q + '.two_gpu_ms.median_ms', 'ms')} ms against {r.fmt('pipeline', q + '.one_gpu_ms.median_ms', 'ms')} ms "
            f"on one, a speedup of {r.fmt('pipeline', q + '.speedup', 'num')} ({r.fmt('pipeline', q + '.scaling_efficiency_pct', 'pct')}% of "
            f"perfect scaling), and copy {r.fmt('pipeline', q + '.h2d_aggregate_vs_one_gpu', 'num')} times as many bytes per second as one "
            f"card: {verdict}. A single-threaded "
            f"host memcpy on this machine runs at {r.fmt('pipeline', q + '.host_memcpy_gbs', 'gbs')} GB/s for comparison. The "
            f"two-card outputs are {'bit-for-bit identical to' if (t.get('identical_to_one_gpu') or {}).get('identical') else 'NOT identical to'} "
            f"the one-card outputs and {'match' if (t.get('verification') or {}).get('verified') else 'do NOT match'} the NumPy reference."
            + earlier_two_gpu_sentence(r), ""]


def lookback_check_lines(r: Renderer) -> List[str]:
    """What the decoupled look-back scan passed (in a child process with a timeout) before it was timed."""
    chk = r.get("scans", "lookback_check")
    if not chk:
        return []
    if not chk.get("passed"):
        return [f"The single-pass decoupled look-back scan did not pass its pre-timing check in this run "
                f"(`{str(chk.get('error', 'a case failed'))[:160]}`), so it is not timed below.", ""]
    cases = chk.get("cases") or []
    ints = [c for c in cases if c.get("input") == "integer"]
    n_min, n_max = min(c["n"] for c in cases), max(c["n"] for c in cases)
    run_id = r.run_of("scans").id
    return [f"Before it was timed, the single-pass decoupled look-back scan (Merrill and Garland, 2016) passed "
            f"{r.record(format_value(len(cases), 'int'), run_id, 'scans.lookback_check.cases (count)', len(cases))} correctness cases "
            f"in a child process with a timeout, at sizes from {r.record(format_value(n_min, 'int'), run_id, 'scans.lookback_check.cases.n (min)', n_min)} "
            f"to {r.record(format_value(n_max, 'int'), run_id, 'scans.lookback_check.cases.n (max)', n_max)} elements: "
            f"{r.record(format_value(len(ints), 'int'), run_id, 'scans.lookback_check.cases (integer inputs)', len(ints))} of them on integer-valued "
            "inputs, whose partial sums are exact in float64, so they must equal `numpy.cumsum` bit for bit whatever order the "
            f"look-back adds the tiles in, and {r.fmt('scans', 'lookback_check.repeats', 'int')} repeated runs at "
            f"{r.fmt('scans', 'lookback_check.repeat_n', 'int')} elements with {r.fmt('scans', 'lookback_check.repeat_failures', 'int')} "
            "mismatches.", ""]


def lookback_gap_sentence(r: Renderer, key: str) -> str:
    gap = r.get("scans", key + "lookback_vs_cub")
    if not gap:
        return ""
    p = key + "lookback_vs_cub"
    if gap["lookback_ms"] > gap["cub_ms"]:
        return (f" The decoupled look-back scan ran in {r.fmt('scans', p + '.lookback_ms', 'ms')} ms against CUB's "
                f"{r.fmt('scans', p + '.cub_ms', 'ms')} ms: CUB is still faster, and the hand-written single pass is "
                f"{r.fmt('scans', p + '.pct_slower_than_cub', 'pct')}% slower.")
    return (f" The decoupled look-back scan ran in {r.fmt('scans', p + '.lookback_ms', 'ms')} ms against CUB's "
            f"{r.fmt('scans', p + '.cub_ms', 'ms')} ms, {r.fmt('scans', p + '.pct_slower_than_cub', 'pct', -1.0)}% faster.")


def benchmarks_md(r: Renderer) -> str:
    out = ["# Benchmarks", "",
           "Every figure on this page is rendered by `scripts/render_docs.py` from `results/*/metrics.json` of real-data "
           "runs on a Kaggle Tesla T4 and links to the raw samples it came from. Times are medians of interleaved timed "
           "iterations after discarded warmup runs, with the interquartile range (IQR) beside them; " + clock_sentence(r), "",
           "## Hardware and data", "", hardware_block(r), ""]
    if r.has("bandwidth"):
        out += ["## Memory bandwidth ceiling", "",
                "| Measurement | Median ms (IQR) | GB/s (read + write) | Raw |", "|---|---:|---:|---|",
                f"| Device-to-device copy kernel, 128-bit loads | {r.fmt_m('bandwidth', 'copy_kernel')} | {r.fmt('bandwidth', 'copy_kernel_gbs', 'gbs')} | {r.raw_link('bandwidth', r.get('bandwidth', 'copy_kernel.samples_file'))} |",
                f"| `cudaMemcpy` device to device | {r.fmt_m('bandwidth', 'cudamemcpy_d2d')} | {r.fmt('bandwidth', 'cudamemcpy_gbs', 'gbs')} | {r.raw_link('bandwidth', r.get('bandwidth', 'cudamemcpy_d2d.samples_file'))} |",
                *([f"| Read-only streaming kernel, 128-bit loads (bytes read) | {r.fmt_m('bandwidth', 'read_kernel')} | {r.fmt('bandwidth', 'read_kernel_gbs', 'gbs')} | {r.raw_link('bandwidth', r.get('bandwidth', 'read_kernel.samples_file'))} |"]
                  if r.get('bandwidth', 'read_kernel') else []),
                "", "A copy spends half its DRAM time writing and pays the read/write turnaround; the feature kernels read about "
                "seventeen bytes for every one they write, so the read-only figure is the fair ceiling for them and the highest "
                "measured figure is used.", "",
                f"Buffer size {r.fmt('bandwidth', 'bytes', 'int')} bytes. Measured ceiling {r.fmt('bandwidth', 'ceiling_gbs', 'gbs')} GB/s, "
                f"{r.fmt('bandwidth', 'ceiling_gbs', 'pct', 100.0 / (r.get('bandwidth', 'theoretical_gbs') or 1))}% of the "
                f"{r.fmt('bandwidth', 'theoretical_gbs', 'gbs')} GB/s theoretical figure.", ""]
    if r.has("ladder"):
        out += ["## Feature kernels: the optimization ladder", "",
                f"{r.fmt('ladder', 'n_snapshots', 'int')} snapshots x {r.fmt('ladder', 'levels', 'int')} levels. Effective bandwidth counts the "
                f"minimum traffic, {r.fmt('ladder', 'bytes_min_io_per_snapshot', 'int')} bytes per snapshot (every level's sizes, the top-of-book "
                "prices, the product id and six outputs). Per-kernel time is CUDA events around the launches only; end to end "
                "is wall clock including the host-to-device copy of exactly the inputs the kernels read and the copy back of the "
                "six outputs, from pinned host buffers, with the same from pageable NumPy memory beside it.", "",
                ladder_table(r), ""]
        sweep = r.get("ladder", "sweep.table", []) or []
        if sweep:
            out += ["### Launch configuration sweep (v4, grid-stride)", "", "| Block | Grid | Median ms | IQR ms | Effective GB/s |", "|---:|---:|---:|---:|---:|"]
            for i, _row in enumerate(sweep):
                p = f"sweep.table.{i}"
                out.append(f"| {r.fmt('ladder', p + '.block_size', 'int')} | {r.fmt('ladder', p + '.grid', 'int')} | "
                           f"{r.fmt('ladder', p + '.median_ms', 'ms')} | {r.fmt('ladder', p + '.iqr_ms', 'ms')} | "
                           f"{r.fmt('ladder', p + '.effective_bw_gbs', 'gbs')} |")
            out.append("")
        out += v0_regimes_md(r) + queued_md(r) + triton_md(r)
    if r.has("baselines"):
        out += ["## Baselines (same machine, same job, same archive)", "",
                "| Implementation | Median ms (IQR) | Verified | Raw |", "|---|---:|:---:|---|"]
        for key, label, path in (("cpp_single", "C++ single thread, `g++ -O3 -march=native`", "cpp_single.ms"),
                                 ("cpp_omp", "C++ with OpenMP on every core", "cpp_omp.ms"),
                                 ("numpy", "Vectorized NumPy", "numpy.ms"),
                                 ("cupy_k", "CuPy expression chain, per kernel", "cupy.kernel_ms"),
                                 ("cupy_e", "CuPy expression chain, end to end", "cupy.e2e_ms")):
            if r.get("baselines", path) is None:
                continue
            ver = r.get("baselines", path.split(".")[0] + ".verified")
            out.append(f"| {label} | {r.fmt_m('baselines', path)} | {'yes' if ver else 'NO'} | {r.raw_link('baselines', r.get('baselines', path + '.samples_file'))} |")
        sp = "speedups"
        out += ["", f"Best rung: `{r.get('ladder', sp + '.best_rung')}`. Per kernel it is {r.fmt('ladder', sp + '.kernel_vs_cpp_single', 'x')} times faster "
                f"than single-threaded C++, {r.fmt('ladder', sp + '.kernel_vs_cpp_omp', 'x')} times faster than C++ with OpenMP on every core and "
                f"{r.fmt('ladder', sp + '.kernel_vs_cupy', 'x')} times faster than the CuPy chain. End to end, transfers included, it is "
                f"{r.fmt('ladder', sp + '.e2e_vs_numpy', 'x')} times faster than NumPy and {r.fmt('ladder', sp + '.e2e_vs_cupy', 'x')} times "
                "faster than the CuPy chain with its own transfers. From pageable memory the end-to-end advantage over NumPy "
                f"shrinks to {r.fmt('ladder', sp + '.e2e_vs_numpy_pageable', 'x')} times.", ""]
    if r.has("scans"):
        same_job = bool((r.get("scans", "ceilings_gbs") or {}).get("read_ceiling"))
        ceiling_src = ("the read-only streaming ceiling measured in the same job" if same_job else
                       f"the bandwidth ceiling of run `{r.run_of('bandwidth').id}`, a different job, since this run measured none"
                       if r.has("bandwidth") else "a ceiling this run did not measure")
        out += ["## Prefix scans", "",
                f"GB/s counts every float64 element twice, once read and once written; the last column compares that with {ceiling_src}.", ""]
        out += lookback_check_lines(r)
        for key, title in (("", "Archive-length series"), ("large.", "Large series")):
            n = r.fmt("scans", key + "n_elements", "int")
            out += [f"### {title} ({n} float64 elements)", "",
                    "| Implementation | Median ms (IQR) | Elements per second (billions) | GB/s | % of read ceiling | Verified | Raw |",
                    "|---|---:|---:|---:|---:|:---:|---|"]
            for i, impl in enumerate(r.get("scans", key + "implementations", []) or []):
                p = f"{key}implementations.{i}"
                if not impl.get("ms"):
                    out.append(f"| {impl['name']} | not timed: {impl.get('error', '')[:80]} | | | | | |")
                    continue
                pct = (r.fmt("scans", p + ".pct_of_read_ceiling", "pct") if impl.get("pct_of_read_ceiling") is not None
                       else r.fmt("scans", p + ".gbs", "pct", 100.0 / (r.get("bandwidth", "ceiling_gbs") or float("nan"))))
                out.append(f"| {impl['name']} | {r.fmt_m('scans', p + '.ms')} | {r.fmt('scans', p + '.elements_per_s', 'billions')} | "
                           f"{r.fmt('scans', p + '.gbs', 'gbs')} | {pct} | {'yes' if impl.get('verified') else 'NO'} | "
                           f"{r.raw_link('scans', impl['ms'].get('samples_file'))} |")
            out += ["", f"Winner: `{r.get('scans', key + 'winner')}`." + lookback_gap_sentence(r, key), ""]
        out += ["### Rolling windows", "",
                f"- Rolling volume-weighted average price over {r.fmt('scans', 'rolling_vwap.n_trades', 'int')} trades (100-trade window, and "
                f"time windows sampled at {r.fmt('scans', 'rolling_vwap.n_snapshots', 'int')} snapshots): {r.fmt_m('scans', 'rolling_vwap.ms')} ms "
                f"for the whole chain; maximum relative error against `metrics.rolling_vwap` in Decimal "
                f"{r.fmt('scans', 'rolling_vwap.max_rel_err_100', 'sci')} over {r.fmt('scans', 'rolling_vwap.decimal_sample', 'int')} sampled trades. "
                f"Prefixes are double-double; against `math.fsum` window sums at {r.fmt('scans', 'rolling_vwap.time_windows.1s.sample', 'int')} sampled "
                f"snapshots the one-second window's maximum relative error is {r.fmt('scans', 'rolling_vwap.time_windows.1s.max_rel_err', 'sci')} "
                f"(a plain float64 cumulative sum reaches {r.fmt('scans', 'rolling_vwap.time_windows.1s.float64_cumsum_max_rel_err', 'sci')}).",
                f"- Rolling one-second mean spread and imbalance over {r.fmt('scans', 'rolling_book.n_snapshots', 'int')} snapshots: "
                f"{r.fmt_m('scans', 'rolling_book.ms')} ms, {r.fmt('scans', 'rolling_book.snapshots_per_s', 'millions')} million snapshots per second.", ""]
    if r.has("precision"):
        out += ["## Prefix-sum precision", "",
                f"Window sums of {r.fmt('precision', 'window', 'int')} values over {r.fmt('precision', 'n', 'int')} float32 top-of-book notionals, "
                "against an exact integer reference.", "",
                "| Variant | Max relative error | Median relative error | Median ms (IQR) |", "|---|---:|---:|---:|"]
        for i, v in enumerate(r.get("precision", "variants", []) or []):
            p = f"variants.{i}"
            star = " (production)" if v["name"] == r.get("precision", "production_choice") else ""
            out.append(f"| {v['name']}{star} | {r.fmt('precision', p + '.max_rel_err', 'sci')} | {r.fmt('precision', p + '.median_rel_err', 'sci')} | {r.fmt_m('precision', p + '.ms')} |")
        out += ["", f"Production choice `{r.get('precision', 'production_choice')}` costs {r.fmt('precision', 'cost_vs_float32_naive_pct', 'pct')}% "
                "time against float32 naive.", ""]
    if r.has("layout"):
        out += ["## Layout under Nsight Systems", "",
                f"Total GPU time of the four book-level kernels over {r.fmt('layout', 'iters', 'int')} timed passes plus the same warmup "
                "passes in both layouts, from `nsys stats --report cuda_gpu_kern_sum`.", "",
                "| Kernel | Array-of-structures ms | Structure-of-arrays ms | Reduction % |", "|---|---:|---:|---:|"]
        for i, _k in enumerate(r.get("layout", "kernels", []) or []):
            p = f"kernels.{i}"
            out.append(f"| {r.get('layout', p + '.name')} | {r.fmt('layout', p + '.aos_ms', 'ms')} | {r.fmt('layout', p + '.soa_ms', 'ms')} | {r.fmt('layout', p + '.reduction_pct', 'pct')} |")
        out += [f"| **Total** | {r.fmt('layout', 'aos_total_kernel_ms', 'ms')} | {r.fmt('layout', 'soa_total_kernel_ms', 'ms')} | **{r.fmt('layout', 'reduction_pct', 'pct')}** |", "",
                f"Reports: {r.raw_link('layout', r.get('layout', 'reports.aos'), 'aos.nsys-rep')}, {r.raw_link('layout', r.get('layout', 'reports.soa'), 'soa.nsys-rep')}, "
                f"{r.raw_link('layout', r.get('layout', 'reports.aos_csv'), 'aos CSV')}, {r.raw_link('layout', r.get('layout', 'reports.soa_csv'), 'soa CSV')}. "
                f"Nsight Compute: {'ran' if r.get('layout', 'ncu.ok') else 'blocked'} (`{str(r.get('layout', 'ncu.message', ''))[:90]}`).", ""]
    if r.has("slippage"):
        out += ["## Execution-cost surface", "",
                f"{r.fmt('slippage', 'orders', 'int')} simulated orders: {r.fmt('slippage', 'n_snapshots', 'int')} snapshots x both sides x "
                f"{r.fmt('slippage', 'n_sizes', 'int')} order sizes.", "",
                "| Kernel | Median ms (IQR) | Orders per second (billions) | Raw |", "|---|---:|---:|---|",
                f"| Naive, one thread per order | {r.fmt_m('slippage', 'naive.ms')} | {r.fmt('slippage', 'naive.orders_per_s', 'billions')} | {r.raw_link('slippage', r.get('slippage', 'naive.ms.samples_file'))} |",
                f"| Warp-cooperative | {r.fmt_m('slippage', 'warp.ms')} | {r.fmt('slippage', 'warp.orders_per_s', 'billions')} | {r.raw_link('slippage', r.get('slippage', 'warp.ms.samples_file'))} |",
                "", f"Warp-cooperative speedup {r.fmt('slippage', 'speedup_warp_vs_naive', 'x')} times; divergence penalty of the naive walk "
                f"{r.fmt('slippage', 'divergence_penalty_pct', 'pct')}%. Verified against the Decimal walker on {r.fmt('slippage', 'verification.sample', 'int')} "
                f"snapshots: exact ticks `{r.get('slippage', 'verification.exact_ticks')}`, maximum average-price error "
                f"{r.fmt('slippage', 'verification.max_avg_price_tick_err', 'sci')} ticks. Surface: {r.raw_link('slippage', r.get('slippage', 'surface_file'), 'per-minute medians')}.", ""]
    if r.has("ofi"):
        out += ["## Order flow imbalance", "",
                f"{r.fmt('ofi', 'levels', 'int')}-level order flow imbalance (Cont, Kukanov and Stoikov), stencil kernel "
                f"{r.fmt_m('ofi', 'kernel_ms')} ms. Out-of-sample fit on the last 30% of each product's time series:", "",
                "| Product | Horizon | Out-of-sample R squared | In-sample R squared | Test points |", "|---|---|---:|---:|---:|"]
        for prod, horizons in sorted((r.get("ofi", "per_product", {}) or {}).items()):
            for key in sorted(horizons, key=lambda k: int(k[1:-1])):
                p = f"per_product.{prod}.{key}"
                out.append(f"| {prod} | {r.fmt('ofi', p + '.horizon_s', 'int')} s | {r.fmt('ofi', p + '.r2_oos', 'r2')} | "
                           f"{r.fmt('ofi', p + '.r2_in', 'r2')} | {r.fmt('ofi', p + '.n_test', 'int')} |")
        out += ["", "A statistical description of the recorded window only; no trading or profit claim is made.", ""]
    if r.has("pipeline"):
        out += ["## End-to-end pipeline and transfers", "",
                f"{r.fmt('pipeline', 'chunks', 'int')} chunks, pinned host memory, three streams.", "",
                "| Measurement | Value |", "|---|---:|",
                f"| Overlapped pass (wall clock) | {r.fmt_m('pipeline', 'overlapped_ms')} ms |",
                f"| Serial pass, one stream | {r.fmt_m('pipeline', 'serial_ms')} ms |",
                f"| Saved by overlap | {r.fmt('pipeline', 'overlap_saved_pct', 'pct')}% |",
                f"| Host to device | {r.fmt('pipeline', 'breakdown.h2d_ms', 'ms')} ms at {r.fmt('pipeline', 'h2d_gbs', 'gbs')} GB/s |",
                f"| Kernels | {r.fmt('pipeline', 'breakdown.kernel_ms', 'ms')} ms |",
                f"| Device to host | {r.fmt('pipeline', 'breakdown.d2h_ms', 'ms')} ms at {r.fmt('pipeline', 'd2h_gbs', 'gbs')} GB/s |",
                f"| Transfer share of stage time | {r.fmt('pipeline', 'transfer_share_pct', 'pct')}% |",
                "", f"Timeline: {r.raw_link('pipeline', r.get('pipeline', 'timeline_svg'), 'SVG from the Nsight Systems trace')}.", ""]
        out += int16_md(r) + two_gpu_md(r)
    if r.has("engine"):
        out += ["## Matching engine (CPU)", "",
                f"`scripts/benchmark.py`, {r.fmt('engine', 'orders', 'int')} orders and {r.fmt('engine', 'cancels', 'int')} cancels, run on the T4 job's host "
                f"CPU (`{r.get('engine', 'cpu_model')}`, Python `{r.get('engine', 'python')}`). Matching path only: an in-memory book, no feed, no network.", "",
                "| Operation | Mean µs | p50 µs | p95 µs | p99 µs |", "|---|---:|---:|---:|---:|"]
        for op in ("add_order", "cancel_order"):
            out.append(f"| `{op}` | {r.fmt('engine', op + '.mean_us', 'us')} | {r.fmt('engine', op + '.p50_us', 'us')} | "
                       f"{r.fmt('engine', op + '.p95_us', 'us')} | {r.fmt('engine', op + '.p99_us', 'us')} |")
        out += ["", f"Raw latencies: {r.raw_link('engine', r.get('engine', 'raw_file'))}. The laptop's own figures are in the README's Benchmark table.", ""]
    if r.has("native_engine"):
        out += native_engine_md(r) + realistic_engine_md(r)
    if r.has("resources"):
        out += resources_md(r)
    out += ["## Clock throttling", "", *clock_section(r), ""]
    runs = sorted({run.id for run, _ in r.sections.values()})
    out += ["## Runs", "", *[f"- [`{rid}`](results/{rid}/manifest.json)" for rid in runs], ""]
    return "\n".join(out)


# -- kernel resources ---------------------------------------------------------- #

LIMIT_WORDS = {"registers": "registers", "shared_memory": "shared memory", "warps": "warp slots", "blocks": "the 16-block cap"}


def limit_words(names: Sequence[str]) -> str:
    words = [LIMIT_WORDS.get(n, n) for n in names]
    if len(words) <= 2:
        return " and ".join(words) or "nothing"
    return ", ".join(words[:-1]) + " and " + words[-1]


def primary_occupancy(row: dict) -> Tuple[int, Optional[dict]]:
    """Index and entry of the occupancy computed at the kernel's benchmarked block size."""
    for j, occ in enumerate(row.get("occupancy") or []):
        if occ["block"] == row.get("primary_block"):
            return j, occ
    return -1, None


def resource_kernel_rows(r: Renderer) -> List[Tuple[int, dict]]:
    rows = r.get("resources", "kernels", []) or []
    return sorted(enumerate(rows), key=lambda t: (t[1]["file"], t[1]["kernel"]))


def cub_scan_launch(r: Renderer, size: str) -> Tuple[int, Optional[dict]]:
    for j, launch in enumerate(r.get("resources", f"cub.launches.{size}", []) or []):
        if launch.get("kernel") == "scan_kernel":
            return j, launch
    return -1, None


def resources_md(r: Renderer) -> List[str]:
    dev = "device"
    rows = resource_kernel_rows(r)
    n_cfg = sum(len(row.get("occupancy") or []) for _i, row in rows)
    agree = sum(1 for _i, row in rows for o in row.get("occupancy") or [] if o.get("calculator_agrees"))
    match = sum(1 for _i, row in rows if row.get("nvrtc_matches_nvcc"))
    run_id = r.run_of("resources").id
    out = ["## Kernel resources and theoretical occupancy", "",
           f"Registers per thread, spills and static shared memory are `nvcc -cubin -arch=sm_75 -Xptxas -v` output "
           f"(`{(r.get('resources', 'nvcc') or [''])[-1]}`); blocks per streaming multiprocessor come from "
           "`cudaOccupancyMaxActiveBlocksPerMultiprocessor` at the block size each kernel was benchmarked with, recorded as the "
           f"benchmarks ran. The card holds {r.fmt('resources', dev + '.regsPerMultiprocessor', 'int')} registers, "
           f"{r.fmt('resources', dev + '.maxThreadsPerMultiProcessor', 'int')} threads, "
           f"{r.fmt('resources', dev + '.sharedMemPerMultiprocessor', 'int')} bytes of shared memory and "
           f"{r.fmt('resources', dev + '.maxBlocksPerMultiProcessor', 'int')} resident blocks per streaming multiprocessor. "
           "Occupancy is the share of the multiprocessor's warp slots the resident blocks fill, and the last column names the "
           "resource that stops one more block from fitting, from a reimplementation of the occupancy calculator that agreed "
           f"with the API on {r.record(format_value(agree, 'int'), run_id, 'derived: configurations where the calculator agrees with the API', agree)} "
           f"of {r.record(format_value(n_cfg, 'int'), run_id, 'derived: launched configurations', n_cfg)} launched configurations. "
           f"For {r.record(format_value(match, 'int'), run_id, 'derived: kernels whose NVRTC binary matches nvcc', match)} of "
           f"{r.record(format_value(len(rows), 'int'), run_id, 'derived: launched kernels', len(rows))} kernels the NVRTC binary "
           f"that was actually timed (CuPy's NVRTC `{r.get('resources', 'cupy_nvrtc')}`) has the same register and shared-memory "
           "counts as the nvcc build; where it differs the timed binary's count is shown in brackets"
           + (", and for " + ", ".join(f"`{k}`" for k in r.get("resources", "occupancy_differs_for_nvrtc")) + " that changes its blocks per SM."
              if r.get("resources", "occupancy_differs_for_nvrtc") else ", and in no case does that change the blocks per SM.")
           + (" Not listed: the Triton rung's kernel, which Triton compiles rather than nvcc"
              + (", and the one-thread busy-wait the queued timing launches ahead of each pass." if r.get("ladder", "queued") else ".")
              if (r.get("ladder", "triton") or {}).get("ok") else
              " Not listed: the one-thread busy-wait the queued timing launches ahead of each pass." if r.get("ladder", "queued") else ""), "",
           "| Kernel | Source | Registers per thread | Spill stores / loads, bytes | Static shared memory, bytes | Block | Blocks per SM | Occupancy % | Limited by |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for i, row in rows:
        p = f"kernels.{i}"
        j, occ = primary_occupancy(row)
        if occ is None:
            continue
        q = f"{p}.occupancy.{j}"
        bps = r.fmt("resources", q + ".api_blocks_per_sm", "int") if occ.get("api_blocks_per_sm") is not None else r.fmt("resources", q + ".blocks_per_sm", "int")
        regs = r.fmt("resources", p + ".registers", "int")
        if row.get("nvrtc_num_regs") not in (None, row["registers"]):
            regs += f" (timed: {r.fmt('resources', p + '.nvrtc_num_regs', 'int')})"
        out.append(f"| `{row['kernel']}` | `{row['file']}` | {regs} | "
                   f"{r.fmt('resources', p + '.spill_stores', 'int')} / {r.fmt('resources', p + '.spill_loads', 'int')} | "
                   f"{r.fmt('resources', p + '.static_smem_bytes', 'int')} | {r.fmt('resources', p + '.primary_block', 'int')} | {bps} | "
                   f"{r.fmt('resources', q + '.occupancy', 'pct', 100.0)} | {limit_words(occ['limited_by'])} |")
    for size in ("863153", "16777216"):
        j, launch = cub_scan_launch(r, size)
        k = r.get("resources", "cub.kernels.scan_kernel")
        if launch is None or not k or "occupancy" not in launch:
            continue
        q = f"cub.launches.{size}.{j}"
        out.append(f"| CUB scan kernel at {r.fmt('resources', q + '.grid', 'int')} tiles of {r.fmt('resources', q + '.block', 'int')} "
                   f"x {r.fmt('resources', q + '.items_per_thread', 'int')} elements | `libscan_cub.so` | "
                   f"{r.fmt('resources', 'cub.kernels.scan_kernel.registers', 'int')} | stack {r.fmt('resources', 'cub.kernels.scan_kernel.stack_bytes', 'int')} | "
                   f"{r.fmt('resources', 'cub.kernels.scan_kernel.static_smem_bytes', 'int')} | {r.fmt('resources', q + '.block', 'int')} | "
                   f"{r.fmt('resources', q + '.cub_blocks_per_sm', 'int')} | {r.fmt('resources', q + '.occupancy.occupancy', 'pct', 100.0)} | "
                   f"{limit_words(launch['occupancy']['limited_by'])} |")
    out += ["", "CUB's kernels are compiled into `libscan_cub.so`, so their registers and shared memory come from `cuobjdump "
            "--dump-resource-usage` (which reports the stack frame rather than spills) and their launch shape and blocks per SM "
            "from CUB's own `-DCUB_DEBUG_LOG` output, which computes occupancy with the same API. "
            f"Raw: {r.raw_link('resources', 'raw/resources', 'ptxas logs and CUB log')}, "
            f"{r.raw_link('resources', r.get('resources', 'launch_log'), 'launch configurations')}.", ""]
    v4 = next((row for _i, row in rows if row["kernel"] == "v4_fused_vec4_gs"), None)
    sweep = r.get("ladder", "sweep.table", []) or []
    if v4 and len(v4.get("occupancy") or []) > 1 and sweep:
        vi = next(i for i, row in rows if row is v4)
        out += ["### Occupancy across the v4 launch sweep", "",
                "| Block | Blocks per SM | Occupancy % | Limited by | Fastest grid's median ms |", "|---:|---:|---:|---|---:|"]
        for j, occ in enumerate(v4["occupancy"]):
            best = min(((k, t) for k, t in enumerate(sweep) if t["block_size"] == occ["block"]), key=lambda kt: kt[1]["median_ms"], default=None)
            ms = r.fmt("ladder", f"sweep.table.{best[0]}.median_ms", "ms") if best else "not swept"
            q = f"kernels.{vi}.occupancy.{j}"
            out.append(f"| {r.fmt('resources', q + '.block', 'int')} | {r.fmt('resources', q + '.blocks_per_sm', 'int')} | "
                       f"{r.fmt('resources', q + '.occupancy', 'pct', 100.0)} | {limit_words(occ['limited_by'])} | {ms} |")
        if len({round(o["occupancy"], 6) for o in v4["occupancy"]}) == 1:
            out += ["", "Every swept block size reaches the same theoretical occupancy, so occupancy does not explain the "
                        "differences across the sweep; the grid size changes them more (see the launch configuration sweep above)."]
        out.append("")
    return out


# Kernels whose own launch has a measured median, for picking the heaviest ones in DESIGN.md.
def kernel_launch_ms(r: Renderer) -> Dict[str, Tuple[str, str, float]]:
    """``{kernel: (section, metrics path, median ms per launch)}`` for kernels timed on their own."""
    out: Dict[str, Tuple[str, str, float]] = {}
    for kernel, (section, path) in {"slippage_naive": ("slippage", "naive.ms.median_ms"), "slippage_warp": ("slippage", "warp.ms.median_ms"),
                                     "ofi_stencil": ("ofi", "kernel_ms.median_ms")}.items():
        if r.get(section, path) is not None:
            out[kernel] = (section, path, float(r.get(section, path)))
    for i, rung in enumerate(r.get("ladder", "rungs", []) or []):
        kernel = {"v2_soa_fused": "v2_fused", "v3_soa_fused_vec4": "v3_fused_vec4", "v4_soa_fused_vec4_sweep": "v4_fused_vec4_gs"}.get(rung["name"])
        if kernel:
            out[kernel] = ("ladder", f"rungs.{i}.kernel_ms.median_ms", float(rung["kernel_ms"]["median_ms"]))
    for layout in ("aos", "soa"):  # the Nsight Systems children time each layout kernel with events too
        for name, ms in (r.get("layout", f"children.{layout}.event_median_ms", {}) or {}).items():
            out[f"layout_{name}_{layout}"] = ("layout", f"children.{layout}.event_median_ms.{name}", float(ms))
    return out


def slowest_full(r: Renderer) -> Optional[bool]:
    """Whether the three kernels with the longest measured launches all reach full theoretical occupancy."""
    rows = {row["kernel"]: row for _i, row in resource_kernel_rows(r)}
    timed = sorted(((ms, k) for k, (_sec, _path, ms) in kernel_launch_ms(r).items() if k in rows), reverse=True)[:3]
    if not timed:
        return None
    return all((primary_occupancy(rows[k])[1] or {}).get("occupancy", 0) >= 0.999 for _ms, k in timed)


def occupancy_design_note(r: Renderer) -> str:
    rows = {row["kernel"]: (i, row) for i, row in resource_kernel_rows(r)}
    timed = sorted(((ms, k, sec, path) for k, (sec, path, ms) in kernel_launch_ms(r).items() if k in rows), reverse=True)[:3]
    if not timed:
        return ""
    parts = []
    for _ms, kernel, sec, path in timed:
        i, row = rows[kernel]
        j, occ = primary_occupancy(row)
        if occ is None:
            continue
        q = f"kernels.{i}.occupancy.{j}"
        regs = r.fmt("resources", f"kernels.{i}.registers", "int")
        smem = r.fmt("resources", f"kernels.{i}.static_smem_bytes", "int")
        lim = occ["limits"]
        spare = [k for k in ("registers", "shared_memory") if lim.get(k) is not None and lim[k] > occ["blocks_per_sm"]]
        spare_txt = (f"; {limit_words(spare)} would allow "
                     + " and ".join(f"{r.fmt('resources', f'{q}.limits.{k}', 'int')}" for k in spare) + " blocks") if spare else ""
        smem_txt = f"{smem} bytes of static shared memory" if row.get("static_smem_bytes") else "no shared memory"
        parts.append(f"`{kernel}` ({r.fmt(sec, path, 'ms')} ms per launch) uses {regs} registers per thread and {smem_txt}, "
                     f"so at {r.fmt('resources', f'kernels.{i}.primary_block', 'int')} threads per block "
                     f"{r.fmt('resources', q + '.blocks_per_sm', 'int')} blocks fit on a multiprocessor, "
                     f"{r.fmt('resources', q + '.occupancy', 'pct', 100.0)}% occupancy, and {limit_words(occ['limited_by'])} bind first{spare_txt}.")
    full = all((primary_occupancy(rows[k][1])[1] or {}).get("occupancy", 0) >= 0.999 for _ms, k, _s, _p in timed)
    causes = []
    if any(k == "slippage_naive" for _ms, k, _s, _p in timed) and r.has("slippage"):
        causes.append(f"the naive walk loses {r.fmt('slippage', 'divergence_penalty_pct', 'pct')}% of its time to divergence")
    if any(k.startswith("layout_") and k.endswith("_aos") for _ms, k, _s, _p in timed) and r.has("layout"):
        causes.append(f"the array-of-structures kernels read uncoalesced memory, and the same kernels in structure-of-arrays "
                      f"order take {r.fmt('layout', 'reduction_pct', 'pct')}% less time")
    closing = ""
    if full:
        closing = (" All three already fill every warp slot the multiprocessor has, so more occupancy cannot speed them up"
                   + ("; what does is what the other sections measured: " + " and ".join(causes) + "." if causes else "."))
    return ("The three kernels with the longest measured launches, from the resource table in BENCHMARKS.md: " + " ".join(parts)
            + closing + " These are theoretical figures: without counters there is no achieved occupancy, so the table says "
              "what the compiler and launch shape allow, not what the scheduler reached.")


def resources_lab(r: Renderer) -> Optional[dict]:
    if not r.has("resources"):
        return None
    out = []
    for _i, row in resource_kernel_rows(r):
        _j, occ = primary_occupancy(row)
        if occ is None:
            continue
        out.append({"kernel": row["kernel"], "file": row["file"], "registers": row["registers"], "spill_stores": row["spill_stores"],
                    "spill_loads": row["spill_loads"], "static_smem_bytes": row["static_smem_bytes"], "block": row["primary_block"],
                    "blocks_per_sm": occ.get("api_blocks_per_sm", occ["blocks_per_sm"]), "occupancy": occ["occupancy"],
                    "limited_by": occ["limited_by"], "calculator_agrees": occ.get("calculator_agrees")})
    for size in ("16777216",):
        _j, launch = cub_scan_launch(r, size)
        k = r.get("resources", "cub.kernels.scan_kernel") or {}
        if launch and "occupancy" in launch:
            out.append({"kernel": "cub scan_kernel", "file": "libscan_cub.so", "registers": k.get("registers"), "spill_stores": None,
                        "spill_loads": None, "static_smem_bytes": k.get("static_smem_bytes"), "block": launch["block"],
                        "blocks_per_sm": launch["cub_blocks_per_sm"], "occupancy": launch["occupancy"]["occupancy"],
                        "limited_by": launch["occupancy"]["limited_by"], "calculator_agrees": launch.get("calculator_agrees")})
    return {"kernels": out, "device": r.get("resources", "device"), "run_id": r.run_of("resources").id}


# -- README ------------------------------------------------------------------- #


def readme_gpu_section(r: Renderer) -> str:
    sp = "speedups"
    lines = [MARK_START, "## GPU analytics layer", "",
             "An offline layer that replays the recorded Coinbase book on an NVIDIA Tesla T4. Every number below is rendered "
             "from committed raw results by `scripts/render_docs.py`; the full set with raw links is in [BENCHMARKS.md](BENCHMARKS.md), "
             "the reasoning in [DESIGN.md](DESIGN.md).", "",
             "```mermaid", "flowchart LR", "  A[Coinbase WebSocket] --> B[recorder.py<br/>zstd JSON lines]",
             "  B --> C[build_archive.py<br/>tick-encoded level-major archive]", "  C --> D[(Kaggle private dataset)]",
             "  D --> E[Kaggle T4 job<br/>bench.run_all]", "  E --> F[results/run/metrics.json]",
             "  F --> G[render_docs.py]", "  G --> H[BENCHMARKS.md, README, BULLETS.md]", "  G --> I[public/data/gpu_lab.json<br/>GPU Lab dashboard]", "```", "",
             f"Archive: {r.fmt_archive('n_snapshots')} book snapshots and {r.fmt_archive('n_trades')} trades of BTC-USD and ETH-USD, "
             f"{r.fmt_archive('capture_seconds', 'num', 1 / 3600)} hours of capture, {r.fmt_archive('gap_count')} sequence gaps.", ""]
    if r.has("ladder"):
        lines += ["### The optimization ladder", "", ladder_table(r, with_raw=False), ""]
    lines += ["### Key findings", ""]
    if r.has("ladder"):
        lines.append(f"- The fused, vectorized feature kernel reaches {r.fmt('ladder', 'rungs.' + str(best_index(r)) + '.pct_of_ceiling', 'pct')}% of the "
                     f"{ceiling_words(r)} and is {r.fmt('ladder', sp + '.kernel_vs_cpp_single', 'x')} times faster than single-threaded C++ per "
                     f"kernel, but only {r.fmt('ladder', sp + '.e2e_vs_numpy', 'x')} times faster than NumPy once transfers are counted.")
    if r.has("layout"):
        lines.append(f"- Moving the book levels from array-of-structures to structure-of-arrays order cut total kernel time by "
                     f"{r.fmt('layout', 'reduction_pct', 'pct')}% in the Nsight Systems profiles.")
    if r.has("precision"):
        lines.append(f"- Rolling windows built from prefix sums cancel: the worst float32 window was off by "
                     f"{r.fmt('precision', 'variants.0.max_rel_err', 'sci')} of its value and even float64 reached "
                     f"{r.fmt('precision', 'variants.1.max_rel_err', 'sci')}, above a tenth of a tick, so production uses a double-double scan.")
    if r.has("slippage"):
        lines.append(f"- The warp-cooperative execution-cost kernel prices {r.fmt('slippage', 'warp.orders_per_s', 'billions')} billion simulated "
                     f"orders per second, {r.fmt('slippage', 'speedup_warp_vs_naive', 'x')} times the divergent thread-per-order walk.")
    if r.has("ofi"):
        lines.append(f"- Ten-level order flow imbalance has essentially no out-of-sample power for the next second's mid move "
                     f"(R squared {r.fmt('ofi', 'per_product.BTC-USD.h1s.r2_oos', 'r2')} for BTC-USD), as expected of a liquid market; "
                     "it is reported as a description, not a signal.")
    if r.has("pipeline"):
        lines.append(f"- Transfers take {r.fmt('pipeline', 'transfer_share_pct', 'pct')}% of the pipeline's stage time even with three overlapping "
                     "streams, so the archive should stay resident on the card and every analysis should run in one residency.")
    lines += readme_phase2_findings(r)
    lines += ["", "**The GPU is not in the live matching path.** A PCIe round trip alone takes longer than the engine's matching latency, so "
              "the GPU only ever analyses recorded history.", "",
              "### Limitations", "",
              "- Kaggle T4 sessions and their four-vCPU host; numbers move with the cards' clocks, which are recorded per block and card.",
              "- Nsight Compute is blocked in Kaggle's container, so there are no per-kernel hardware counters: occupancy is theoretical, "
              "from the compiler and the occupancy API, and timelines come from Nsight Systems.",
              "- The archive covers one recording window of two products; the order flow imbalance fit describes that window only.",
              ("- The C++20 order book core (`native/`) is checked against the reference book by the differential fuzzer; the Python "
               "engine remains the reference implementation." if r.has("native_engine") else
               "- The compiled C++ order book core is not measured in these results; the Python engine remains the reference implementation."),
              "", MARK_END]
    return "\n".join(lines)


def readme_phase2_findings(r: Renderer) -> List[str]:
    out = []
    gap = r.get("scans", "large.lookback_vs_cub")
    if gap:
        out.append(f"- A hand-written single-pass decoupled look-back scan reaches {r.fmt('scans', 'large.implementations.' + str(scan_index(r, 'decoupled_lookback')) + '.gbs', 'gbs')} GB/s "
                   f"on {r.fmt('scans', 'large.n_elements', 'int')} elements, "
                   + (f"{r.fmt('scans', 'large.lookback_vs_cub.pct_slower_than_cub', 'pct')}% behind CUB." if gap["lookback_ms"] > gap["cub_ms"]
                      else "ahead of CUB."))
    e = r.get("pipeline", "int16_encoding") or {}
    if e.get("ok") and "e2e_ms" in e:
        change = (e["e2e_ms"]["median_ms"] / e["int32_e2e_ms"]["median_ms"] - 1) * 100.0
        run_id = r.run_of("pipeline").id
        change_txt = r.record(format_value(abs(change), "pct"), run_id, "pipeline.int16_encoding e2e change %", change)
        same = (e.get("identical_to_int32") or {}).get("identical")
        line = (f"- Sending prices as int16 offsets from the best price moves {r.fmt('pipeline', 'int16_encoding.bytes_saved_pct', 'pct')}% fewer "
                f"bytes and makes the transfer-bound pipeline {change_txt}% {'faster' if change < 0 else 'slower'} end to end, "
                f"{'with bit-identical outputs' if same else 'but its outputs are NOT identical to the int32 pass'}")
        t = r.get("pipeline", "two_gpu") or {}
        if t.get("h2d_gbs_one_gpu") and t.get("h2d_gbs_two_gpu_aggregate"):
            verdict = {"none": "adds no copy bandwidth", "barely": "adds almost no copy bandwidth",
                       "partial": "adds some copy bandwidth", "scales": "nearly doubles the copy bandwidth"}[two_gpu_kind(t)]
            line += (f"; a second T4 {verdict}: {r.fmt('pipeline', 'two_gpu.h2d_gbs_two_gpu_aggregate', 'gbs')} GB/s on two cards "
                     f"against {r.fmt('pipeline', 'two_gpu.h2d_gbs_one_gpu', 'gbs')} GB/s on one" + other_sessions_short(r) + ".")
        else:
            line += "."
        out.append(line)
    t = r.get("ladder", "triton") or {}
    q = r.get("ladder", "queued.kernel_ms") or {}
    if t.get("ok") and "triton_fused" in q and "v2_soa_fused" in q:
        ratio = q["triton_fused"]["median_ms"] / q["v2_soa_fused"]["median_ms"]
        ratio_txt = r.record(f"{ratio:.2f}", r.run_of("ladder").id, "ladder Triton / v2, launches queued", ratio)
        out.append(f"- Written in Triton, the fused feature kernel needs {ratio_txt} times the GPU time of the CUDA kernel it mirrors, "
                   + ("with bit-identical outputs." if t.get("identical_to_v2_all") else "and passes the same tolerance check."))
    if r.has("native_engine") and r.get("native_engine", "native.add_order"):
        div = r.get("native_engine", "fuzz.divergences")
        fuzz = ("" if div is None else ", and matches the reference book on every fuzzed stream" if div == 0 else
                f", but diverges from the reference book on {r.fmt('native_engine', 'fuzz.divergences', 'int')} fuzzed streams")
        out.append(f"- The C++20 core adds an order in {r.fmt('native_engine', 'native.add_order.p99_us', 'us3')} microseconds at p99 natively, "
                   f"{faster_slower(r, 'native_engine', 'p99_add_python_engine_over_native')} than the Python engine on the same stream in the same "
                   f"process{fuzz}."
                   + (f" On a stream derived from the recorded book its p99 limit order is "
                      f"{faster_slower(r, 'native_engine', 'realistic.p99_limit_python_engine_over_native')} than the Python engine's."
                      if r.get("native_engine", "realistic.p99_limit_python_engine_over_native") else ""))
    if r.has("resources"):
        rows = r.get("resources", "kernels", []) or []
        full = sum(1 for row in rows if (primary_occupancy(row)[1] or {}).get("occupancy", 0) >= 0.999)
        run_id = r.run_of("resources").id
        out.append(f"- {r.record(format_value(full, 'int'), run_id, 'derived: kernels at full theoretical occupancy', full)} of "
                   f"{r.record(format_value(len(rows), 'int'), run_id, 'derived: launched kernels', len(rows))} launched kernels reach full "
                   "theoretical occupancy; the others are listed in BENCHMARKS.md with the resource that limits them.")
    return out


def scan_index(r: Renderer, name: str) -> int:
    for i, impl in enumerate(r.get("scans", "large.implementations", []) or []):
        if impl.get("name") == name:
            return i
    return -1


def design_phase2(r: Renderer) -> List[str]:
    out: List[str] = []
    t = r.get("pipeline", "two_gpu") or {}
    if "two_gpu_ms" in t:
        kind = two_gpu_kind(t)
        link, meaning = two_gpu_link(t)
        rates = (f"one card copies at {r.fmt('pipeline', 'two_gpu.h2d_gbs_one_gpu', 'gbs')} GB/s and two together at "
                 f"{r.fmt('pipeline', 'two_gpu.h2d_gbs_two_gpu_aggregate', 'gbs')} GB/s, and `nvidia-smi topo -m` reports their "
                 f"connection as `{link}`, {meaning}")
        why = (f"Transfers take {r.fmt('pipeline', 'transfer_share_pct', 'pct')}% of the pipeline's stage time, so a second T4 with "
               "its own PCIe link looked like a way to double the copy rate. ")
        e = r.get("pipeline", "int16_encoding") or {}
        lever = ""
        if e.get("ok") and e.get("bytes_saved_pct") is not None:
            same = (e.get("identical_to_int32") or {}).get("identical")
            lever = ((" The lever that works here is sending fewer bytes: the int16 encoding cuts " if kind in ("none", "barely") else
                      " Sending fewer bytes helps either way: the int16 encoding cuts ")
                     + f"{r.fmt('pipeline', 'int16_encoding.bytes_saved_pct', 'pct')}% of them"
                     + (" with identical outputs." if same else ", though its outputs differ from the int32 pass."))
        earlier = earlier_two_gpu_sentence(r)
        if kind in ("none", "barely"):
            out += ["## Two cards", "",
                    why + f"Measured in this session, it does not: {rates}. Here, whatever they share upstream of their own links "
                    "set the ceiling." + earlier + lever, ""]
        else:
            out += ["## Two cards", "",
                    why + f"Measured in this session, {'it nearly does' if kind == 'scales' else 'it partly does'}: {rates}. Two cards ran the whole "
                    f"pipeline {r.fmt('pipeline', 'two_gpu.speedup', 'num')} times as fast as one." + earlier + lever, ""]
    if r.has("native_engine"):
        out += ["## A C++ core beside the Python engine", "",
                "The Python engine stays the reference: it is clear, exact in Decimal, and the fuzzer's reference book checks it. The C++20 "
                "core exists to show what the same semantics cost without an interpreter. Its price levels sit in a tick-indexed window "
                "around the best price with a two-level bitmap to find the next level, far prices go to a sorted fallback with reserved "
                "capacity, each level keeps an intrusive FIFO list, orders and levels come from pools with free lists, and an "
                "open-addressing map finds an order to cancel. Fills are reported through a callback, so after construction the hot path "
                f"allocates nothing (a counting allocator saw {r.fmt('native_engine', 'build.test_no_alloc.allocations', 'int')} allocations over "
                f"{r.fmt('native_engine', 'build.test_no_alloc.operations', 'int')} operations). Decimal becomes integer ticks and lots only in "
                "`NativeOrderBook`, at the boundary: a call with Decimal orders costs "
                f"{r.fmt('native_engine', 'binding_decimal.add_order.p50_us', 'us3')} microseconds at the median, one with integers through "
                f"the binding {r.fmt('native_engine', 'binding.add_order.p50_us', 'us3')}, and the core itself "
                f"{r.fmt('native_engine', 'native.add_order.p50_us', 'us3')}.", ""]
    return out


def best_index(r: Renderer) -> int:
    best = r.get("ladder", "speedups.best_rung")
    for i, rung in enumerate(r.get("ladder", "rungs", []) or []):
        if rung["name"] == best:
            return i
    return 0


def readme_benchmark_table(r: Renderer) -> Optional[str]:
    run = r.local_engine
    if run is None:
        return None
    e = (run.metrics.get("sections") or {}).get("engine") or {}
    man = run.manifest

    def f(op, key):
        return format_value(e[op][key], "us") + " µs"
    return "\n".join([
        f"Measured on this laptop (`{man.get('cpu_model')}`, Python `{e.get('python')}`, `{man.get('os', '')}`), single core, matching path "
        f"only; raw latencies in [`results/{run.id}`](results/{run.id}):", "",
        "| Operation | mean | p50 | p95 | p99 |", "|---|---:|---:|---:|---:|",
        f"| `add_order` (insert + match) | {f('add_order', 'mean_us')} | {f('add_order', 'p50_us')} | {f('add_order', 'p95_us')} | {f('add_order', 'p99_us')} |",
        f"| `cancel_order` | {f('cancel_order', 'mean_us')} | {f('cancel_order', 'p50_us')} | {f('cancel_order', 'p95_us')} | {f('cancel_order', 'p99_us')} |",
    ])


def update_readme(text: str, r: Renderer) -> str:
    section = readme_gpu_section(r)
    if MARK_START in text and MARK_END in text:
        text = text[:text.index(MARK_START)] + section + text[text.index(MARK_END) + len(MARK_END):]
    elif "## License" in text:
        text = text.replace("## License", section + "\n\n## License", 1)
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    table = readme_benchmark_table(r)
    if table and "## Benchmark" in text:
        start = text.index("## Benchmark")
        nxt = text.find("\n## ", start + 5)
        body = text[start:nxt if nxt != -1 else len(text)]
        new_body = ("## Benchmark\n\n```bash\npython scripts/benchmark.py --orders 100000 --cancels 10000 --results-dir results/engine-local-<date>\n```\n\n"
                    + table + "\n\nNumbers vary by machine; re-run the command to reproduce on yours. The GPU job runs the same "
                    "benchmark on its own host CPU (see [BENCHMARKS.md](BENCHMARKS.md)).\n\n")
        text = text.replace(body, new_body, 1)
    return text


# -- DESIGN ------------------------------------------------------------------- #


def float32_spacing(r: Renderer) -> Optional[Tuple[str, str, str]]:
    """``(price, spacing in cents, spacing as % of a tick)`` at the archive's highest top-of-book price.

    ``tolerance_rel`` is a tenth of a tick over the highest price in ticks, so that price is
    ``0.1 / tolerance_rel`` ticks of one cent; ``numpy.spacing`` gives the float32 step there.
    """
    tol = r.get("precision", "tolerance_rel")
    if not tol:
        return None
    import numpy as np
    ticks = round(0.1 / float(tol))
    dollars = ticks * 0.01
    cents = float(np.spacing(np.float32(dollars))) * 100
    run_id, path = r.run_of("precision").id, "precision.tolerance_rel"
    price = "$" + r.record(f"{dollars:,.2f}", run_id, path + " -> 0.1 / tolerance_rel ticks of $0.01", tol)
    step = r.record(f"{cents:.2f}", run_id, path + " -> numpy.spacing(float32(price)) in cents", tol)
    share = r.record(f"{cents * 100:.0f}", run_id, path + " -> spacing as % of the one-cent tick", tol)
    return price, step, share


def float32_spacing_sentence(r: Renderer) -> str:
    sp = float32_spacing(r)
    if sp is None:
        return "Near the top of the recorded prices a float32 cannot hold every cent,"
    price, step, share = sp
    return f"At the archive's highest top-of-book price, {price}, a float32 has a spacing of {step} cents, {share}% of the one-cent tick,"


def design_md(r: Renderer) -> str:
    sp = "speedups"
    return "\n".join([
        "# Design decisions", "",
        "Rendered by `scripts/render_docs.py`; every number comes from a committed real-data T4 run.", "",
        "## Offline GPU, never the live path", "",
        "The matching engine answers in microseconds on one CPU core. A PCIe round trip to any discrete GPU costs more than "
        "that before a kernel even starts, so putting the card in the order path could only make it slower. The GPU instead "
        "replays the recorded history, where the work is wide (every snapshot, every order size) and latency does not matter. "
        + (f"The pipeline measurements make the same point from the other side: transfers are {r.fmt('pipeline', 'transfer_share_pct', 'pct')}% "
           "of stage time, so analyses run in one residency on the card." if r.has("pipeline") else ""), "",
        "## Integer ticks, not float32 prices", "",
        float32_spacing_sentence(r) + " so a float32 "
        "spread is mostly rounding error (`tests/test_archive.py` shows the failure on a real price). Prices are int32 tick "
        "counts; spreads and level indices are exact integers on the device, sizes stay float32, and the microprice is kept as "
        "an offset from the mid so float32 does not lose it.", "",
        "## Scan precision", "",
        (f"A rolling window formed as the difference of two prefix sums cancels: the prefix grows with the stream while the "
         f"window does not. Over {r.fmt('precision', 'n', 'int')} top-of-book notionals the worst float32 window was off by "
         f"{r.fmt('precision', 'variants.0.max_rel_err', 'sci')} of its own value, Kahan-compensated float32 by "
         f"{r.fmt('precision', 'variants.2.max_rel_err', 'sci')}, block-rebased float32 by {r.fmt('precision', 'variants.3.max_rel_err', 'sci')} "
         f"and even float64 by {r.fmt('precision', 'variants.1.max_rel_err', 'sci')}, above the budget of a tenth of a tick at the largest "
         f"price ({r.fmt('precision', 'tolerance_rel', 'sci')} relative). A double-double prefix (an unevaluated hi + lo pair built with "
         "TwoSum, exclusive prefixes shuffled rather than subtracted) is the only design inside it: its windows match the exact "
         "integer reference after rounding. It costs "
         f"{r.fmt('precision', 'variants.4.ms.median_ms', 'ms')} ms against {r.fmt('precision', 'variants.0.ms.median_ms', 'ms')} ms for "
         "float32 per pass, negligible next to moving the data, so the rolling VWAP uses it; its one-second windows match "
         f"`math.fsum` exactly where a float64 cumulative sum is off by up to "
         f"{r.fmt('scans', 'rolling_vwap.time_windows.1s.float64_cumsum_max_rel_err', 'sci')}." if r.has("precision") else ""), "",
        "## Structure-of-arrays book levels", "",
        (f"Level-major storage lets a warp read 32 consecutive snapshots of one level in one transaction. In Nsight Systems the "
         f"same four kernels took {r.fmt('layout', 'aos_total_kernel_ms', 'ms')} ms in array-of-structures order and "
         f"{r.fmt('layout', 'soa_total_kernel_ms', 'ms')} ms in structure-of-arrays order, a cut of {r.fmt('layout', 'reduction_pct', 'pct')}%." if r.has("layout") else ""), "",
        "## Warp-cooperative slippage", "",
        (f"A thread per (snapshot, side, size) walking the book diverges: lanes of one warp stop at different depths and the warp "
         f"runs as long as its deepest order. One warp per snapshot instead builds the cumulative quantity and notional of all "
         f"levels with shuffles and lets each lane binary-search two sizes, so every lane does the same work. Measured: "
         f"{r.fmt('slippage', 'speedup_warp_vs_naive', 'x')} times faster, {r.fmt('slippage', 'divergence_penalty_pct', 'pct')}% of the "
         "naive kernel's time was divergence." if r.has("slippage") else ""), "",
        *(["## What limits occupancy", "", occupancy_design_note(r), ""] if r.has("resources") else []),
        *design_phase2(r),
        "## Nsight Systems rather than Nsight Compute", "",
        "Kaggle's container denies access to the GPU performance counters (`ERR_NVGPUCTRPERM`), so Nsight Compute cannot run "
        "there (it was tried, see `DECISIONS.md`). Nsight Systems needs no counters and answers what this project asks: total "
        "kernel time per layout, stream overlap and transfer share. An SSH backend for a rented T4 with counters enabled is "
        "designed in `remote/README.md` for the day kernel-level metrics are needed.", "",
        "## Honest baselines", "",
        (f"The single-threaded C++ baseline is the one bullet 1 quotes, and it is compiled with `-O3 -march=native` and a "
         f"cache-friendly level-outer loop. The same code with OpenMP on every core is reported too: the best kernel is "
         f"{r.fmt('ladder', sp + '.kernel_vs_cpp_omp', 'x')} times faster than that, a smaller and fairer figure." if r.has("ladder") else ""), "",
    ])


# -- INTERVIEW NOTES -------------------------------------------------------- #


def went_wrong(decisions: str) -> List[str]:
    lines = [ln[2:].strip() for ln in decisions.splitlines() if ln.startswith("- ")]

    def tags(ln: str) -> set:  # tags lead an entry, right after its timestamp; a tag named in prose is not one
        m = re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} ((?:\[[a-z-]+\] ?)*)", ln)
        return set(re.findall(r"\[([a-z-]+)\]", m.group(1))) if m else set()

    # newest first: the log is chronological, and the latest phase's biggest problems lead
    tagged = ([ln for ln in reversed(lines) if "biggest" in tags(ln)]
              + [ln for ln in lines if "went-wrong" in tags(ln) and "biggest" not in tags(ln)])
    if len(tagged) < 3:
        tagged += [ln for ln in reversed(lines) if ln not in tagged and re.search(r"failed|fix|fallback", ln, re.I)]
    out = []
    for ln in tagged[:3]:
        ln = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} (?:\[[a-z-]+\] ?)*", "", ln)
        parts = ln.split("`")  # even indices are prose, odd ones are existing code spans
        out.append("`".join(NUM_RE.sub(lambda m: f"`{m.group(0)}`", part) if i % 2 == 0 else part
                            for i, part in enumerate(parts)))
    return out


def phase2_questions(r: Renderer) -> List[Tuple[str, str]]:
    """Interview questions the 0.3.0 work invites, each only when its measurement is in a real run."""
    qa: List[Tuple[str, str]] = []
    if r.has("resources"):
        rows = r.get("resources", "kernels", []) or []
        full = sum(1 for row in rows if (primary_occupancy(row)[1] or {}).get("occupancy", 0) >= 0.999)
        run_id = r.run_of("resources").id
        qa.append(("Without hardware counters, how do you know what limits a kernel?",
                   f"From the compiler and the occupancy API: `nvcc -Xptxas -v` gives registers, spills and shared memory, and "
                   f"`cudaOccupancyMaxActiveBlocksPerMultiprocessor` gives resident blocks at the benchmarked block size. "
                   f"{r.record(format_value(full, 'int'), run_id, 'derived: kernels at full theoretical occupancy', full)} of "
                   f"{r.record(format_value(len(rows), 'int'), run_id, 'derived: launched kernels', len(rows))} launched kernels reach full "
                   "theoretical occupancy"
                   + ({True: ", the three slowest among them, so for those the answer lies elsewhere: divergence and uncoalesced "
                             "reads, which the slippage and layout sections measure.",
                       False: ", but not all of the three slowest, so for those occupancy is the first thing to check; DESIGN.md "
                              "names what binds each.",
                       None: "."}[slowest_full(r)])
                   + " It is a theoretical figure, not achieved occupancy."))
    chk = r.get("scans", "lookback_check") or {}
    if chk.get("passed") and r.get("scans", "large.lookback_vs_cub"):
        qa.append(("How does the single-pass scan avoid deadlock, and how did you test it?",
                   "Blocks take tile numbers from an atomic counter instead of trusting the launch order, so every tile a block waits "
                   "on belongs to a block that is already running; statuses are published with release stores and read with acquire "
                   f"loads. Before timing, it ran in a child process with a timeout on integer inputs that must match `numpy.cumsum` bit "
                   f"for bit at any summation order, {r.fmt('scans', 'lookback_check.repeats', 'int')} times at "
                   f"{r.fmt('scans', 'lookback_check.repeat_n', 'int')} elements with {r.fmt('scans', 'lookback_check.repeat_failures', 'int')} "
                   "mismatches."))
    t = r.get("pipeline", "two_gpu") or {}
    if "two_gpu_ms" in t:
        link, meaning = two_gpu_link(t)
        moved = (f"the copy-only pass moved {r.fmt('pipeline', 'two_gpu.h2d_gbs_one_gpu', 'gbs')} GB/s on one card and "
                 f"{r.fmt('pipeline', 'two_gpu.h2d_gbs_two_gpu_aggregate', 'gbs')} GB/s on two together, and `nvidia-smi topo -m` reports "
                 f"their connection as `{link}`, {meaning}")
        if two_gpu_kind(t) in ("none", "barely"):
            qa.append(("Why did the second T4 not speed up the pipeline?",
                       ("Because the pipeline is transfer-bound and, in this session, the copies hit a limit both cards shared: "
                        if transfer_bound(r) else "Because, in this session, the copies hit a limit both cards shared: ") + moved + "."
                       + (" Here the second card added only compute, which this pipeline barely uses; sending fewer bytes helps "
                          "either way." if transfer_bound(r) else "")
                       + (f" That holds{other_sessions_short(r)}, so the result varies between sessions, and the runs do not "
                          "show why." if other_sessions_short(r) else "")))
        else:
            qa.append(("What did the second T4 buy?",
                       f"Copy bandwidth: {moved}. The whole pipeline ran {r.fmt('pipeline', 'two_gpu.speedup', 'num')} times as fast "
                       "on two cards as on one."
                       + (f" That holds{other_sessions_short(r)}, so the result varies between sessions, and the runs do not "
                          "show why." if other_sessions_short(r) else "")))
    e = r.get("pipeline", "int16_encoding") or {}
    if e.get("ok") and "e2e_ms" in e:
        qa.append(("What did narrowing prices to int16 buy?",
                   f"{r.fmt('pipeline', 'int16_encoding.bytes_saved_pct', 'pct')}% fewer bytes over PCIe and "
                   f"{r.fmt('pipeline', 'int16_encoding.e2e_ms.median_ms', 'ms')} ms instead of "
                   f"{r.fmt('pipeline', 'int16_encoding.int32_e2e_ms.median_ms', 'ms')} ms end to end, "
                   + ("with outputs identical bit for bit, " if (e.get("identical_to_int32") or {}).get("identical") else
                      "though its outputs are NOT identical to the int32 pass, ")
                   + f"because each level is sent as an offset from its side's best price and {r.fmt('pipeline', 'int16_encoding.overflow_fallbacks', 'int')} "
                   "chunks needed the int32 fallback."))
    if r.has("native_engine") and r.get("native_engine", "fuzz.divergences") == 0:
        agree = (" The benchmark also checks that all four replay paths, the Python engine among them, agree on fills, resting "
                 "orders and cancels found." if r.get("native_engine", "consistent") else "")
        qa += [("How do you know the C++ core has the Python engine's semantics?",
                "The differential fuzzer checks it against the naive list-based reference book that the test suite also checks the "
                f"Python engine against (`tests/test_differential.py`): {r.fmt('native_engine', 'fuzz.streams', 'int')} seeded streams, "
                f"{r.fmt('native_engine', 'fuzz.operations', 'int')} operations, {r.fmt('native_engine', 'fuzz.divergences', 'int')} "
                "divergences, comparing fills, statuses, best prices and depth after every operation." + agree),
               ("Where does the time go when the C++ core is called from Python?",
                f"Natively an `add` costs {r.fmt('native_engine', 'native.add_order.p50_us', 'us3')} microseconds at the median; through the "
                f"binding with integers {r.fmt('native_engine', 'binding.add_order.p50_us', 'us3')}, and with Decimal orders "
                f"{r.fmt('native_engine', 'binding_decimal.add_order.p50_us', 'us3')}"
                + (", so the Python call costs more than the matching itself, and the Decimal path adds the conversion and the Python "
                   "fill objects on top."
                   if (r.get("native_engine", "binding.add_order.p50_us") or 0) > 2 * (r.get("native_engine", "native.add_order.p50_us") or 0)
                   else ".")
                + " Decimal is converted only at that boundary, so the core works in integer ticks.")]
    return qa


def interview_md(r: Renderer, decisions: str) -> str:
    sp = "speedups"
    qa = []
    if r.has("ladder"):
        qa += [
            ("Why is the end-to-end speedup so much smaller than the per-kernel one?",
             f"The kernel is {r.fmt('ladder', sp + '.kernel_vs_cpp_single', 'x')} times faster than single-threaded C++, but end to end it is "
             f"{r.fmt('ladder', sp + '.e2e_vs_numpy', 'x')} times faster than NumPy because the inputs cross PCIe first. The work is a few "
             "adds per byte, so the copy dominates; that is why analyses run in one residency."),
            ("How close to the hardware limit is the best feature kernel?",
             f"{r.fmt('ladder', 'rungs.' + str(best_index(r)) + '.effective_bw_gbs', 'gbs')} GB/s of minimum traffic, "
             f"{r.fmt('ladder', 'rungs.' + str(best_index(r)) + '.pct_of_ceiling', 'pct')}% of the {r.fmt('bandwidth', 'ceiling_gbs', 'gbs')} GB/s "
             f"the read-only streaming kernel achieves (a copy is slower because half its traffic is writes) and {r.fmt('ladder', 'rungs.' + str(best_index(r)) + '.pct_of_theoretical', 'pct')}% of the theoretical figure."),
            ("What did each rung of the ladder buy?",
             "Array-of-structures to structure-of-arrays fixes coalescing, fusion removes repeated passes over the same sizes, "
             "128-bit loads cut instruction count, and the sweep picks the launch shape; the table in BENCHMARKS.md has each rung's median."),
            ("Is the arithmetic intensity high enough for the GPU to be compute bound?",
             f"No: {r.fmt('ladder', 'flops_per_snapshot', 'int')} floating-point operations per {r.fmt('ladder', 'bytes_min_io_per_snapshot', 'int')} bytes, "
             "far left of the roofline's ridge point, so memory bandwidth is the only ceiling that matters."),
            ("Why compare against OpenMP when the bullet quotes single-threaded C++?",
             f"Because an honest comparison beats a flattering one: with every core the CPU closes the gap to "
             f"{r.fmt('ladder', sp + '.kernel_vs_cpp_omp', 'x')} times."),
        ]
    if r.has("layout"):
        qa.append(("How did you measure the layout effect?",
                   f"The same four kernels in both layouts under `nsys profile`, total kernel time from `cuda_gpu_kern_sum`: "
                   f"{r.fmt('layout', 'aos_total_kernel_ms', 'ms')} ms versus {r.fmt('layout', 'soa_total_kernel_ms', 'ms')} ms, "
                   f"{r.fmt('layout', 'reduction_pct', 'pct')}% less."))
    if r.has("scans"):
        gap = r.get("scans", "large.lookback_vs_cub")
        won = (f"`{r.get('scans', 'large.winner')}` on {r.fmt('scans', 'large.n_elements', 'int')} elements. "
               + (f"The hand-written single-pass decoupled look-back scan came within "
                  f"{r.fmt('scans', 'large.lookback_vs_cub.pct_slower_than_cub', 'pct')}% of it "
                  f"({r.fmt('scans', 'large.lookback_vs_cub.lookback_ms', 'ms')} ms against "
                  f"{r.fmt('scans', 'large.lookback_vs_cub.cub_ms', 'ms')} ms); both read and write each element once, "
                  "while the Blelloch and warp-shuffle scans need a second pass for the block carries."
                  if gap and gap["lookback_ms"] > gap["cub_ms"] else
                  "CUB's single-pass decoupled look-back reads and writes each element once; the multi-pass scans need a "
                  "second pass for the block carries."))
        qa += [("Which scan won?", won),
               ("How do you know the rolling price is right?",
                f"Against `metrics.rolling_vwap` in Decimal on sampled trades: maximum relative error {r.fmt('scans', 'rolling_vwap.max_rel_err_100', 'sci')}.")]
    if r.has("precision"):
        qa.append(("Why not float32, or even float64, for the prefix sums?",
                   f"Window error reached {r.fmt('precision', 'variants.0.max_rel_err', 'sci')} relative in float32 and "
                   f"{r.fmt('precision', 'variants.1.max_rel_err', 'sci')} in float64, both above a tenth of a tick; the double-double "
                   f"scan stays inside it at {r.fmt('precision', 'cost_vs_float32_naive_pct', 'pct')}% extra time over float32, "
                   "which is still a few milliseconds."))
    if r.has("slippage"):
        qa += [("What is divergence and how big was it here?",
                f"Lanes of a warp taking different branch paths serialize. The thread-per-order walk lost "
                f"{r.fmt('slippage', 'divergence_penalty_pct', 'pct')}% of its time to it; the warp-cooperative kernel runs "
                f"{r.fmt('slippage', 'speedup_warp_vs_naive', 'x')} times faster."),
               ("How was the execution-cost kernel verified?",
                f"On {r.fmt('slippage', 'verification.sample', 'int')} random snapshots against a Decimal book walker, exact in ticks for the "
                "levels consumed.")]
    if r.has("pipeline"):
        qa.append(("Did the three streams actually overlap?",
                   f"Yes, the Nsight Systems timeline shows copy and compute lanes active together; overlap saved "
                   f"{r.fmt('pipeline', 'overlap_saved_pct', 'pct')}% of the serial wall time, and transfers were still "
                   f"{r.fmt('pipeline', 'transfer_share_pct', 'pct')}% of stage time."))
    if r.has("ofi"):
        qa.append(("Does order flow imbalance predict anything?",
                   f"Not the future mid at these horizons: out of sample the fit is no better than the mean (R squared "
                   f"{r.fmt('ofi', 'per_product.BTC-USD.h1s.r2_oos', 'r2')} for BTC-USD at one second, "
                   f"{r.fmt('ofi', 'per_product.BTC-USD.h10s.r2_oos', 'r2')} at ten), which is what a liquid market should show; the "
                   "study is a description, not a strategy."))
    if r.has("engine"):
        qa.append(("Why is the matching engine in Python, and how fast is it?",
                   f"Clarity and Decimal exactness. On the job's host CPU `add_order` p99 was {r.fmt('engine', 'add_order.p99_us', 'us')} "
                   "microseconds for the matching path alone; the GPU would add a PCIe round trip on top."))
    sp = float32_spacing(r)
    qa += phase2_questions(r)
    qa.append(("Why integer ticks?",
               (f"At the archive's highest price, {sp[0]}, neighbouring float32 values are {sp[1]} cents apart, so a one-cent "
                "spread cannot be represented exactly" if sp else "A float32 price near the top of the archive cannot represent a "
                "one-cent spread exactly") + "; int32 ticks make every price comparison exact.")),
    qa = qa[:25]
    out = ["# Interview notes", "", "Rendered from the measured numbers by `scripts/render_docs.py`.", "", "## Questions", ""]
    for i, (q, a) in enumerate(qa, 1):
        out += [f"### {r.record(str(i), r.latest.id, 'question_index', i)}. {q}", "", a, ""]
    out += ["## Three things that went wrong", "", *[f"- {w}" for w in went_wrong(decisions)], ""]
    return "\n".join(out)


# -- BULLETS ------------------------------------------------------------------ #


def bullet_specs(r: Renderer) -> List[Tuple[str, List[List[str]]]]:
    N = r.fmt_archive("n_snapshots")
    A = r.fmt("ladder", "speedups.kernel_vs_cpp_single", "x")
    B = r.fmt("ladder", "speedups.e2e_vs_numpy", "x")
    C = r.fmt("layout", "reduction_pct", "pct")
    D = r.fmt("engine", "add_order.p99_us", "us")
    orders = r.fmt("engine", "orders", "int")
    bi = str(best_index(r))
    pct = r.fmt("ladder", f"rungs.{bi}.pct_of_ceiling", "pct")
    gbs = r.fmt("ladder", f"rungs.{bi}.effective_bw_gbs", "gbs")
    v0 = r.fmt("ladder", "rungs.0.effective_bw_gbs", "gbs")
    rate = r.fmt("slippage", "warp.orders_per_s", "billions")
    ws = r.fmt("slippage", "speedup_warp_vs_naive", "x")
    sizes = r.fmt("slippage", "n_sizes", "int")
    return [
        ("1", [["", "Built ", "Wrote "], ["CUDA C++ kernels on an NVIDIA T4 for spread, depth and order imbalance over "],
               [f"{N} recorded book snapshots, ", f"{N} recorded Coinbase book snapshots, ", f"{N} recorded order book snapshots, "],
               [f"{A} times faster than single-threaded C++ per kernel and {B} times faster than NumPy end to end."]]),
        ("2", [["", "Built a ", "Wrote a ", "Implemented a "],
               ["shared-memory parallel prefix scan for rolling volume-weighted average price with book levels stored in ",
                "shared-memory parallel prefix scan for a rolling volume-weighted average price with book levels stored in ",
                "shared-memory parallel prefix scans for rolling volume-weighted average price with book levels stored in ",
                "shared-memory parallel prefix scan for rolling volume-weighted average price, with book levels stored in "],
               ["structure-of-arrays order for coalesced reads, ", "structure-of-arrays order for coalesced memory reads, "],
               [f"cutting measured kernel time {C}% in Nsight Systems profiles.",
                f"cutting measured kernel time by {C}% in Nsight Systems profiles.",
                f"cutting total measured kernel time {C}% in Nsight Systems profiles."]]),
        ("3", [["", "Built an ", "Wrote an "],
               ["asynchronous Python matching engine with price-time priority, partial fills and cancellations on "],
               ["Coinbase's live WebSocket order book feed, ", "Coinbase's live WebSocket feed, ",
                "Coinbase's live WebSocket level 2 feed, "],
               [f"benchmarked across {orders} orders at {D} microseconds 99th-percentile latency.",
                f"benchmarked across {orders} orders at {D} microseconds of 99th-percentile latency.",
                f"benchmarked over {orders} orders at {D} microseconds 99th-percentile latency."]]),
        ("4", [["Took ", "Tuned "], ["CUDA order book feature kernels on an NVIDIA T4 ", "CUDA book feature kernels on an NVIDIA T4 "],
               [f"from {v0} to {gbs} gigabytes per second of effective bandwidth, ",
                f"from {v0} to {gbs} gigabytes per second of effective memory bandwidth, "],
               [f"{pct}% of the measured memory ceiling, ", f"or {pct}% of the measured memory ceiling, ",
                f"{pct}% of the measured bandwidth ceiling, "],
               ["with structure-of-arrays levels and 128-bit loads.", "using structure-of-arrays levels and 128-bit loads.",
                "with structure-of-arrays book levels and 128-bit loads.", "by fusing kernels over structure-of-arrays book levels.",
                "with fused kernels over structure-of-arrays book levels.", "using fused kernels on structure-of-arrays levels.",
                "with structure-of-arrays book levels and vectorized 128-bit loads."]]),
        ("5", [["Wrote ", "Built "], ["a warp-cooperative CUDA kernel that prices ", "a warp-cooperative CUDA slippage kernel that prices "],
               [f"orders of {sizes} sizes against every recorded book snapshot ", f"market orders of {sizes} sizes against every recorded book snapshot ",
                f"orders of {sizes} notional sizes against every recorded snapshot "],
               [f"at {rate} billion simulated orders per second, ", f"at {rate} billion orders per second, "],
               [f"{ws} times faster than a thread-per-order walk.", f"{ws} times faster than a naive thread-per-order walk.",
                f"{ws} times faster than a divergent thread-per-order walk."]]),
    ] + phase2_bullet_specs(r)


def phase2_bullet_specs(r: Renderer) -> List[Tuple[str, List[List[str]]]]:
    """Alternates for 0.3.0 results, each only when its measurement exists in a real run."""
    specs: List[Tuple[str, List[List[str]]]] = []
    if (r.has("native_engine") and r.get("native_engine", "fuzz.divergences") == 0 and r.get("native_engine", "consistent")
            and (r.get("native_engine", "p99_add_python_engine_over_native") or 0) > 1):
        streams = r.fmt("native_engine", "fuzz.streams", "int")
        p99 = r.fmt("native_engine", "native.add_order.p99_us", "us")
        speed = r.fmt("native_engine", "p99_add_python_engine_over_native", "x")
        specs.append(("6", [["Wrote ", "Built "],
                            ["a C++20 order book core for Python with no heap allocation per order ",
                             "a C++20 order book core with pybind11 bindings and no heap allocation per order ",
                             "a C++20 order book core for Python with no heap allocation on the order path "],
                            [f"that matched a reference order book on {streams} fuzzed streams, ",
                             f"that matched a reference book on {streams} fuzzed order streams, ",
                             f"matching a reference order book on {streams} fuzzed streams, "],
                            [f"at {p99} microseconds 99th-percentile latency, {speed} times faster than Python.",
                             f"with {p99} microseconds 99th-percentile latency, {speed} times faster than Python.",
                             f"at a 99th-percentile latency of {p99} microseconds, {speed} times faster than Python."]]))
    e = r.get("pipeline", "int16_encoding") or {}
    if e.get("ok") and "e2e_ms" in e and e["e2e_ms"]["median_ms"] < e["int32_e2e_ms"]["median_ms"]:
        n = r.fmt_archive("n_snapshots")
        saved = r.fmt("pipeline", "int16_encoding.bytes_saved_pct", "pct")
        change = (1 - e["e2e_ms"]["median_ms"] / e["int32_e2e_ms"]["median_ms"]) * 100.0
        faster = r.record(format_value(change, "pct"), r.run_of("pipeline").id, "pipeline.int16_encoding e2e change %", change)
        specs.append(("7", [["Cut ", "Reduced "],
                            [f"host-to-device traffic for {n} recorded order book snapshots by {saved}% ",
                             f"the host-to-device traffic of {n} order book snapshots by {saved}% ",
                             f"host-to-device copies of {n} recorded book snapshots by {saved}% "],
                            ["with int16 tick offsets decoded on an NVIDIA T4, ", "by sending int16 tick offsets decoded inside an NVIDIA T4 kernel, ",
                             "with int16 price offsets decoded on an NVIDIA T4, "],
                            [f"making the transfer-bound pipeline {faster}% faster with bit-identical outputs.",
                             f"so the transfer-bound pipeline ran {faster}% faster with bit-identical outputs.",
                             f"speeding the transfer-bound pipeline up {faster}% with identical outputs."]]))
    return specs


def fit_bullets(r: Renderer) -> List[Tuple[str, str]]:
    out = []
    for key, parts in bullet_specs(r):
        chosen = None
        for combo in itertools.product(*parts):
            text = "".join(combo)
            text = text[0].upper() + text[1:]
            lines = check_bullets.split_two_lines(text)
            if lines and not check_bullets.problems(lines, alternate=key not in check_bullets.TEMPLATE):
                chosen = lines
                break
        if chosen is None:
            raise SystemExit(f"bullet {key}: no phrasing fits two lines of 98 to 110 characters")
        out.append((key, "\n".join(chosen)))
    return out


def bullets_md(r: Renderer) -> str:
    bullets = fit_bullets(r)
    titles = {"1": "Template bullet 1 (feature kernels)", "2": "Template bullet 2 (prefix scan and layout)",
              "3": "Template bullet 3 (matching engine)", "4": "Alternate: GPU performance (percentage of peak bandwidth)",
              "5": "Alternate: execution cost (the slippage kernel)", "6": "Alternate: the C++20 order book core",
              "7": "Alternate: transfer engineering (int16 tick offsets)"}
    out = ["# Resume bullets", "",
           "Rendered by `scripts/render_docs.py` from real-data T4 results; `scripts/check_bullets.py` checks the two-line "
           "length rule and the style rules. Sources: [BENCHMARKS.md](BENCHMARKS.md).", ""]
    for key, text in bullets:
        out += [f"## {r.record(titles[key], r.latest.id, 'bullet_index', key)}", "", f"<!-- bullet:{key} -->", text, ""]
    return "\n".join(out)


# -- CHANGELOG ---------------------------------------------------------------- #


CHANGELOG_030 = [
    "### Added",
    "- Kernel resource table: registers, spills and shared memory from `nvcc -Xptxas -v`, theoretical occupancy from the",
    "  occupancy API at every benchmarked launch configuration (recorded as the benchmarks run), and the binding resource.",
    "- Single-pass decoupled look-back scan (Merrill and Garland, 2016) with an atomic tile counter and release/acquire status",
    "  words, checked in a child process with a timeout before it is timed against Blelloch, warp shuffle and CUB.",
    "- int16 tick-offset encoding measured on the real archive, and the pipeline split across both T4s with its own streams and",
    "  pinned buffers per card; outputs are compared bit for bit with the one-card, int32 pass.",
    "- C++20 order book core (`native/`, CMake): tick-indexed price window with an ordered fallback, intrusive FIFO levels,",
    "  pooled orders, open-addressing id map, no heap allocation on the hot path; pybind11 binding, `NativeOrderBook`,",
    "  `fuzz_engine.py --engine native`, a native latency harness and `benchmark_native.py`.",
    "- GPU Lab panels for kernel resources, transfers and scaling, and the C++ core's latency; new interview questions and bullets.",
    "- CI builds the C++ core, runs its native tests, fuzzes it against the reference and compile-checks every CUDA kernel.",
    "",
    "### Fixed",
    "- A raw template expression in `INTERVIEW_NOTES.md`; the renderer now refuses to write unrendered template syntax.",
    "- `check_numbers.py` matched numbers across documents; it now requires provenance in the same document and covers",
    "  `DESIGN.md` and `INTERVIEW_NOTES.md`.",
    "- GPU clocks were never sampled on Kaggle's two-card sessions; every block now samples the cards it uses.",
    "- Remote jobs ran a pinned branch; they now run the pushed SHA of the current branch under a phase budget.",
]


def changelog(text: str) -> str:
    if "## [0.2.0]" in text:
        if "## [0.3.0]" not in text:
            entry = "\n".join([f"## [0.3.0] - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", "", *CHANGELOG_030, "", ""])
            text = text.replace("## [0.2.0]", entry + "## [0.2.0]", 1)
        return text
    entry = "\n".join([
        "## [0.2.0] - 2026-10-21", "", "### Added",
        "- Raw Coinbase recorder (`recorder.py`, `scripts/record.py`) and a deterministic tick-encoded level-major snapshot archive.",
        "- Kaggle T4 runner (`remote/`), benchmark harness (`bench/`) and CUDA kernels (`cuda/`): feature ladder, scans, precision study,",
        "  layout profiling, execution-cost surface, order flow imbalance and a three-stream pipeline.",
        "- Hypothesis invariants and a differential fuzzer for the matching engine.",
        "- GPU Lab dashboard section, `/analytics` routes, documentation rendered from committed results.", "", ""])
    marker = "## [0.1.0]"
    return text.replace(marker, entry + marker, 1) if marker in text else text + "\n" + entry


# -- dashboard data -------------------------------------------------------------- #


def gpu_lab_json(r: Renderer) -> dict:
    g = r.get
    out: dict = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "raw_base": RAW_BASE,
                 "statement": "The GPU is not in the live matching path: a PCIe round trip alone takes longer than the engine's matching latency.",
                 "provenance": {"runs": [{"run_id": run.id, "git_sha": run.manifest.get("git_sha"), "gpu_name": run.manifest.get("gpu_name"),
                                          "source": run.source, "date": datetime.fromtimestamp(run.started, tz=timezone.utc).strftime("%Y-%m-%d")}
                                         for run in sorted({run for run, _ in r.sections.values()}, key=lambda x: x.id)]},
                 "archive": {k: (r.latest.metrics.get("archive") or {}).get(k) for k in ("n_snapshots", "n_trades", "capture_seconds", "gap_count", "content_hash")}}
    if r.has("ladder"):
        rungs = [{"name": x["name"], "label": rung_label(x["name"]), "kernel_median_ms": x["kernel_ms"]["median_ms"],
                  "kernel_iqr_ms": x["kernel_ms"]["iqr_ms"], "e2e_median_ms": x["e2e_ms"]["median_ms"],
                  "effective_bw_gbs": x["effective_bw_gbs"], "pct_of_ceiling": x["pct_of_ceiling"],
                  "pct_of_theoretical": x["pct_of_theoretical"], "block_size": x["block_size"], "extra": bool(x.get("extra")),
                  "kernel_queued_median_ms": (x.get("kernel_queued_ms") or {}).get("median_ms")} for _i, x in ladder_rows(r)]
        b = lambda p: (g("baselines", p) or {}).get("median_ms")  # noqa: E731
        out["ladder"] = {"rungs": rungs, "ceiling_gbs": g("bandwidth", "ceiling_gbs"), "theoretical_gbs": g("bandwidth", "theoretical_gbs"),
                         "baselines": {"cpp_single_ms": b("cpp_single.ms"), "cpp_omp_ms": b("cpp_omp.ms"), "numpy_ms": b("numpy.ms"),
                                       "cupy_kernel_ms": b("cupy.kernel_ms"), "cupy_e2e_ms": b("cupy.e2e_ms")},
                         "speedups": {k: g("ladder", f"speedups.{k}") for k in ("kernel_vs_cpp_single", "e2e_vs_numpy", "kernel_vs_cpp_omp", "kernel_vs_cupy")}}
        flops_snap = g("ladder", "flops_per_snapshot") or 0
        n = g("ladder", "n_snapshots") or 0
        kernels = [{"name": x["name"], "intensity_flop_per_byte": x.get("arithmetic_intensity_flop_per_byte"),
                    "gflops": flops_snap * n / (x["kernel_ms"]["median_ms"] * 1e-3) / 1e9} for _i, x in ladder_rows(r)]
        out["roofline"] = {"peak_gflops": 8100, "peak_note": "T4 single-precision peak at boost clock (NVIDIA specification)",
                           "bw_ceiling_gbs": g("bandwidth", "ceiling_gbs"), "bw_theoretical_gbs": g("bandwidth", "theoretical_gbs"), "kernels": kernels}
    if r.has("pipeline"):
        out["timeline_svg"] = "/data/pipeline_timeline.svg"
        out["pipeline"] = {"h2d_ms": g("pipeline", "breakdown.h2d_ms"), "kernel_ms": g("pipeline", "breakdown.kernel_ms"),
                           "d2h_ms": g("pipeline", "breakdown.d2h_ms"), "overlapped_ms": g("pipeline", "overlapped_ms.median_ms"),
                           "serial_ms": g("pipeline", "serial_ms.median_ms"), "overlap_saved_pct": g("pipeline", "overlap_saved_pct"),
                           "transfer_share_pct": g("pipeline", "transfer_share_pct"), "consequence": g("pipeline", "consequence")}
    if r.has("precision"):
        out["precision"] = {"variants": [{"name": v["name"], "max_rel_err": v["max_rel_err"], "median_rel_err": v["median_rel_err"],
                                          "median_ms": v["ms"]["median_ms"]} for v in g("precision", "variants", [])],
                            "production_choice": g("precision", "production_choice")}
    if r.has("scans"):
        out["scans"] = {"implementations": [{"name": i["name"], "median_ms": i["ms"]["median_ms"], "elements_per_s": i["elements_per_s"],
                                             "gbs": i["gbs"], "pct_of_read_ceiling": i.get("pct_of_read_ceiling")}
                                            for i in g("scans", "large.implementations", []) if i.get("ms")],
                        "lookback_vs_cub": g("scans", "large.lookback_vs_cub"),
                        "winner": g("scans", "large.winner"), "n_elements": g("scans", "large.n_elements")}
    if r.has("slippage"):
        run = r.run_of("slippage")
        surf_path = run.path / (g("slippage", "surface_file") or "raw/slippage_surface.json")
        surface = json.loads(surf_path.read_text(encoding="utf-8")) if surf_path.exists() else {}
        out["slippage"] = {"sizes_usd": surface.get("sizes_usd") or g("slippage", "sizes_usd"), "products": surface.get("products", {}),
                           "orders_per_s_warp": g("slippage", "warp.orders_per_s"), "orders_per_s_naive": g("slippage", "naive.orders_per_s"),
                           "speedup": g("slippage", "speedup_warp_vs_naive")}
    if r.has("ofi"):
        run = r.run_of("ofi")
        sc_path = run.path / (g("ofi", "scatter_file") or "raw/ofi_scatter.json")
        scatter = json.loads(sc_path.read_text(encoding="utf-8")) if sc_path.exists() else {}
        out["ofi"] = {"per_product": {p: {h: {k: f.get(k) for k in ("r2_oos", "r2_in", "beta", "alpha", "n_test")} for h, f in hs.items()}
                                      for p, hs in (g("ofi", "per_product", {}) or {}).items()}, "scatter": scatter}
    bench_rows = []
    for name, (run, data) in sorted(r.sections.items()):
        for path, m in walk_measurements(data):
            bench_rows.append({"section": name, "name": path, "metric": "median", "value": m["median_ms"], "unit": "ms",
                               "run_id": run.id, "raw": f"results/{run.id}/{m.get('samples_file', 'metrics.json')}"})
    out["benchmarks"] = bench_rows
    if r.has("resources"):
        out["resources"] = resources_lab(r)
    transfer = transfer_lab(r)
    if transfer:
        out["transfer"] = transfer
    if r.has("native_engine"):
        out["native_engine"] = native_lab(r)
    if r.has("engine"):
        out["engine"] = {"p99_us": g("engine", "add_order.p99_us"), "p50_us": g("engine", "add_order.p50_us"),
                         "orders": g("engine", "orders"), "cpu_model": g("engine", "cpu_model"), "run_id": r.run_of("engine").id}
    return out


def transfer_lab(r: Renderer) -> Optional[dict]:
    e = r.get("pipeline", "int16_encoding") or {}
    t = r.get("pipeline", "two_gpu") or {}
    out: dict = {}
    if "e2e_ms" in e:
        out["int16"] = {"bytes_int32": e.get("bytes_h2d_int32"), "bytes_int16": e.get("bytes_h2d_int16"),
                        "bytes_saved_pct": e.get("bytes_saved_pct"), "fallbacks": e.get("overflow_fallbacks"), "chunks": e.get("chunks"),
                        "e2e_int32_ms": e["int32_e2e_ms"]["median_ms"], "e2e_int16_ms": e["e2e_ms"]["median_ms"],
                        "identical": (e.get("identical_to_int32") or {}).get("identical"), "ok": e.get("ok")}
    if "two_gpu_ms" in t:
        out["two_gpu"] = {"one_gpu_ms": t["one_gpu_ms"]["median_ms"], "two_gpu_ms": t["two_gpu_ms"]["median_ms"],
                          "speedup": t.get("speedup"), "h2d_gbs_one_gpu": t.get("h2d_gbs_one_gpu"),
                          "h2d_gbs_two_gpu": t.get("h2d_gbs_two_gpu_aggregate"), "host_memcpy_gbs": t.get("host_memcpy_gbs"),
                          "link": (t.get("topology") or {}).get("links", {}).get("GPU0-GPU1"),
                          "identical": (t.get("identical_to_one_gpu") or {}).get("identical")}
    if out:
        out["run_id"] = r.run_of("pipeline").id
    return out or None


def native_lab(r: Renderer) -> dict:
    sec = r.sections["native_engine"][1]
    paths = []
    for key, label in NATIVE_PATHS:
        if key in sec:
            paths.append({"key": key, "label": label.replace("`", ""),
                          "add": {k: sec[key]["add_order"][k] for k in ("p50_us", "p99_us", "p999_us")},
                          "cancel": {k: sec[key]["cancel_order"][k] for k in ("p50_us", "p99_us", "p999_us")}})
    fuzz = sec.get("fuzz") or {}
    alloc = (sec.get("build") or {}).get("test_no_alloc") or {}
    return {"paths": paths, "orders": sec.get("orders"), "cancels": sec.get("cancels"), "consistent": sec.get("consistent"),
            "fuzz": {k: fuzz.get(k) for k in ("streams", "operations", "divergences")},
            "allocations": alloc.get("allocations"), "alloc_operations": alloc.get("operations"),
            "cpu_model": sec.get("cpu_model"), "run_id": r.run_of("native_engine").id,
            "speedup_p99_vs_python": sec.get("p99_add_python_engine_over_native"),
            "realistic": realistic_lab(sec.get("realistic"))}


def realistic_lab(real: Optional[dict]) -> Optional[dict]:
    if not real:
        return None
    paths = [{"key": key, "label": label.replace("`", ""),
              **{kind: {k: real[key][kind][k] for k in ("p50_us", "p99_us")} for kind, _ in REALISTIC_KINDS}}
             for key, label in NATIVE_PATHS if key in real]
    return {"paths": paths, "ops": real.get("ops"), "op_counts": real.get("op_counts"), "consistent": real.get("consistent"),
            "stream": real.get("stream")}


def finite(obj: Any) -> Any:
    """Replace NaN and infinities with None so the dashboard JSON is strict."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render documents from committed results")
    parser.add_argument("--results", default=str(REPO / "results"))
    parser.add_argument("--out-dir", default=None, help="write here instead of the repository (dry runs, tests)")
    parser.add_argument("--allow-synthetic", action="store_true", help="dry runs into --out-dir only")
    args = parser.parse_args(argv)
    if args.allow_synthetic and not args.out_dir:
        print("--allow-synthetic requires --out-dir: synthetic numbers never reach the repository's documents", file=sys.stderr)
        return 2
    try:
        r = Renderer(load_runs(Path(args.results)), allow_synthetic=args.allow_synthetic)
    except RefuseError as exc:
        print(f"render_docs: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out_dir) if args.out_dir else REPO
    out.mkdir(parents=True, exist_ok=True)
    decisions = (REPO / "DECISIONS.md").read_text(encoding="utf-8") if (REPO / "DECISIONS.md").exists() else ""
    docs = {}
    for name, build in (("BENCHMARKS.md", benchmarks_md), ("DESIGN.md", design_md),
                        ("INTERVIEW_NOTES.md", lambda rr: interview_md(rr, decisions)), ("BULLETS.md", bullets_md)):
        r.doc = name
        docs[name] = build(r)
    readme_src = (out / "README.md") if (out / "README.md").exists() else REPO / "README.md"
    r.doc = README_DOC
    readme = update_readme(readme_src.read_text(encoding="utf-8"), r)
    section = readme[readme.index(MARK_START):readme.index(MARK_END)] if MARK_START in readme else readme
    leftovers = [(name, ln) for name, text in [*docs.items(), (README_DOC, section)]
                 for ln in template_leftovers(text)]
    if leftovers:
        for name, ln in leftovers:
            print(f"render_docs: unrendered template syntax in {name}: {ln[:120]}", file=sys.stderr)
        return 4
    for name, text in docs.items():
        (out / name).write_text(text.rstrip() + "\n", encoding="utf-8")
    (out / "README.md").write_text(readme, encoding="utf-8")
    cl_src = (out / "CHANGELOG.md") if (out / "CHANGELOG.md").exists() else REPO / "CHANGELOG.md"
    (out / "CHANGELOG.md").write_text(changelog(cl_src.read_text(encoding="utf-8")), encoding="utf-8")
    data_dir = out / "public" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "gpu_lab.json").write_text(json.dumps(finite(gpu_lab_json(r)), separators=(",", ":"), allow_nan=False,
                                                      default=str), encoding="utf-8")
    if r.has("pipeline") and r.get("pipeline", "timeline_svg"):
        svg = r.run_of("pipeline").path / r.get("pipeline", "timeline_svg")
        if svg.exists():
            shutil.copy(svg, data_dir / "pipeline_timeline.svg")
    prov_path = (out / "results" / "provenance.json") if args.out_dir else (REPO / "results" / "provenance.json")
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps({"generated_unix": time.time(), "tokens": r.prov}, indent=1, sort_keys=True, default=str),
                         encoding="utf-8")
    size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
    print(f"rendered {', '.join(docs)} + README.md, CHANGELOG.md; refused synthetic T4 runs: {len(r.refused)}; "
          f"public/data {size / 1e6:.2f} MB")
    if size > DATA_LIMIT_BYTES:
        print("public/data exceeds 5 MB", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
