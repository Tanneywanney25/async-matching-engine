"""Render the Nsight Systems trace of the three-stream pipeline as an SVG timeline.

    python -m bench.timeline_svg --sqlite results/<run>/raw/pipeline.sqlite \
        --out results/<run>/raw/pipeline_timeline.svg [--nvtx-range overlapped_pass] [--title "..."]

Input is the SQLite export of a report (``nsys export --type sqlite``):

``CUPTI_ACTIVITY_KIND_MEMCPY``  ``start``, ``end`` (ns), ``copyKind`` (1 = host to
                                device, 2 = device to host), ``streamId``, ``bytes``
``CUPTI_ACTIVITY_KIND_KERNEL``  ``start``, ``end`` (ns), ``streamId``, ``shortName``
                                -> ``StringIds.value``
``NVTX_EVENTS`` (optional)      ``start``, ``end``, ``text`` or ``textId``: when a
                                range name is given the drawing is restricted to
                                the last range with that name (the child process
                                wraps its one measured pass in ``overlapped_pass``).

The SVG has three lanes (copy in, compute, copy out), one bar per operation, a
time axis in milliseconds, lane labels, and a title with the measured overlap
fraction: the share of the drawn window (first GPU operation start to last end)
during which at least two lanes are busy. It uses a ``viewBox`` and
``width="100%"`` so it scales to phone width, and colours checked with the
data-viz palette validator against both a white page and the dashboard's
``#00112b`` background (blue ``#3987e5``, orange ``#d95926``, aqua ``#199e70``;
text ``#6f7a8d`` has at least 4.3:1 contrast on both).

Pure Python (``sqlite3`` plus string building, no third-party imports) so a unit
test can build a fake database with those two tables and check the output.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

Interval = Tuple[int, int]

# key, short label drawn in the gutter, long label, colour
LANES: Tuple[Tuple[str, str, str, str], ...] = (
    ("h2d", "H2D", "copy in (host to device)", "#3987e5"),
    ("kernel", "GPU", "compute (kernels)", "#d95926"),
    ("d2h", "D2H", "copy out (device to host)", "#199e70"),
)
LANE_KEYS = tuple(lane[0] for lane in LANES)
COPY_KIND_H2D = 1
COPY_KIND_D2H = 2
TEXT = "#6f7a8d"       # reads on white (4.3:1) and on #00112b (4.3:1)
BAND = "#8892a6"       # lane background at low opacity, neutral on both surfaces
WIDTH = 800            # viewBox width; everything scales from here
NVTX_RANGE_DEFAULT = "overlapped_pass"


# --------------------------------------------------------------------------- #
# SQLite reading
# --------------------------------------------------------------------------- #


def _tables(conn: sqlite3.Connection) -> set:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def nvtx_ranges(conn: sqlite3.Connection, name: str) -> List[Interval]:
    """All closed NVTX ranges whose text (inline or registered) equals ``name``."""
    tables = _tables(conn)
    if "NVTX_EVENTS" not in tables:
        return []
    cols = _columns(conn, "NVTX_EVENTS")
    if "start" not in cols or "end" not in cols:
        return []
    text_expr = "n.text" if "text" in cols else "NULL"
    if "textId" in cols and "StringIds" in tables:
        sql = (f'SELECT n.start, n."end", {text_expr}, s.value FROM NVTX_EVENTS n '
               'LEFT JOIN StringIds s ON s.id = n.textId WHERE n."end" IS NOT NULL')
    else:
        sql = f'SELECT n.start, n."end", {text_expr}, NULL FROM NVTX_EVENTS n WHERE n."end" IS NOT NULL'
    out: List[Interval] = []
    for start, end, text, registered in conn.execute(sql):
        label = text if text is not None else registered
        if label == name:
            out.append((int(start), int(end)))
    out.sort()
    return out


def _read_memcpy(conn: sqlite3.Connection, lanes: Dict[str, list]) -> int:
    """Fill the h2d / d2h lanes; return the number of copies of other kinds."""
    cols = _columns(conn, "CUPTI_ACTIVITY_KIND_MEMCPY")
    stream = "streamId" if "streamId" in cols else "NULL"
    nbytes = "bytes" if "bytes" in cols else "NULL"
    other = 0
    sql = f'SELECT start, "end", copyKind, {stream}, {nbytes} FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start'
    for start, end, kind, stream_id, size in conn.execute(sql):
        op = {"start": int(start), "end": int(end), "stream": stream_id, "bytes": size}
        if kind == COPY_KIND_H2D:
            op["name"] = "memcpy host to device"
            lanes["h2d"].append(op)
        elif kind == COPY_KIND_D2H:
            op["name"] = "memcpy device to host"
            lanes["d2h"].append(op)
        else:
            other += 1
    return other


def _read_kernels(conn: sqlite3.Connection, lanes: Dict[str, list]) -> None:
    cols = _columns(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    tables = _tables(conn)
    stream = "k.streamId" if "streamId" in cols else "NULL"
    name_col = "shortName" if "shortName" in cols else ("demangledName" if "demangledName" in cols else None)
    if name_col and "StringIds" in tables:
        sql = (f'SELECT k.start, k."end", {stream}, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
               f'LEFT JOIN StringIds s ON s.id = k.{name_col} ORDER BY k.start')
    else:
        sql = f'SELECT k.start, k."end", {stream}, NULL FROM CUPTI_ACTIVITY_KIND_KERNEL k ORDER BY k.start'
    for start, end, stream_id, name in conn.execute(sql):
        lanes["kernel"].append({"start": int(start), "end": int(end), "stream": stream_id,
                                "bytes": None, "name": name or "kernel"})


def read_trace(sqlite_path, nvtx_range: Optional[str] = None,
               window: Optional[Interval] = None) -> dict:
    """Read the copy and kernel operations of a trace into three lanes.

    ``window`` (ns) or the last NVTX range called ``nvtx_range`` restricts the
    operations to those overlapping it (clipped to it). The returned ``t0``/``t1``
    are the extent of the kept operations (the window when there are none).
    """
    sqlite_path = Path(sqlite_path)
    lanes: Dict[str, list] = {key: [] for key in LANE_KEYS}
    other = 0
    found: Optional[bool] = None
    conn = sqlite3.connect(str(sqlite_path))
    try:
        tables = _tables(conn)
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            other = _read_memcpy(conn, lanes)
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            _read_kernels(conn, lanes)
        if nvtx_range:
            ranges = nvtx_ranges(conn, nvtx_range)
            found = bool(ranges)
            if ranges and window is None:
                window = ranges[-1]
    finally:
        conn.close()
    if window is not None:
        w0, w1 = int(window[0]), int(window[1])
        for key in LANE_KEYS:
            kept = []
            for op in lanes[key]:
                if op["end"] > w0 and op["start"] < w1:
                    op = dict(op)
                    op["start"] = max(op["start"], w0)
                    op["end"] = min(op["end"], w1)
                    kept.append(op)
            lanes[key] = kept
    starts = [op["start"] for ops in lanes.values() for op in ops]
    ends = [op["end"] for ops in lanes.values() for op in ops]
    if starts:
        t0, t1 = min(starts), max(ends)
    elif window is not None:
        t0, t1 = int(window[0]), int(window[1])
    else:
        t0, t1 = 0, 0
    return {"lanes": lanes, "t0": t0, "t1": t1, "window": window, "nvtx_range": nvtx_range,
            "nvtx_range_found": found, "other_memcpy": other, "source": str(sqlite_path)}


# --------------------------------------------------------------------------- #
# Interval arithmetic
# --------------------------------------------------------------------------- #


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    """Union of intervals as a sorted list of disjoint intervals."""
    out: List[Interval] = []
    for start, end in sorted((int(s), int(e)) for s, e in intervals if e > s):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def busy_ns(intervals: Sequence[Interval]) -> int:
    return sum(e - s for s, e in merge_intervals(intervals))


def overlap_fraction(lanes: Dict[str, Sequence[Interval]], span: Optional[Interval] = None) -> float:
    """Share of the span during which at least two lanes are busy.

    Each lane is first merged (two kernels back to back on one stream are one
    lane, not two), then a sweep over the endpoints accumulates the time with
    two or more lanes active. ``span`` defaults to the extent of all intervals.
    """
    events: List[Tuple[int, int]] = []
    all_ivs: List[Interval] = []
    for ivs in lanes.values():
        merged = merge_intervals(ivs)
        all_ivs.extend(merged)
        for start, end in merged:
            events.append((start, 1))
            events.append((end, -1))
    if not events:
        return 0.0
    if span is None:
        span = (min(s for s, _ in all_ivs), max(e for _, e in all_ivs))
    total = span[1] - span[0]
    if total <= 0:
        return 0.0
    events.sort(key=lambda ev: (ev[0], ev[1]))  # at equal times, ends before starts
    active = 0
    last_t: Optional[int] = None
    overlap = 0
    for t, delta in events:
        if last_t is not None and active >= 2:
            overlap += max(0, min(t, span[1]) - max(last_t, span[0]))
        active += delta
        last_t = t
    return min(1.0, overlap / total)


def trace_stats(trace: dict) -> dict:
    """Bar counts, busy time, bytes and the overlap fraction of a trace."""
    lanes = trace["lanes"]
    t0, t1 = trace["t0"], trace["t1"]
    ivs = {key: [(op["start"], op["end"]) for op in lanes[key]] for key in LANE_KEYS}
    span = (t0, t1) if t1 > t0 else None
    frac = overlap_fraction(ivs, span)
    kernels: Dict[str, int] = {}
    for op in lanes["kernel"]:
        kernels[op["name"]] = kernels.get(op["name"], 0) + 1
    return {
        "overlap_fraction": frac,
        "overlap_pct": frac * 100.0,
        "window_ms": (t1 - t0) / 1e6,
        "bars": {key: len(lanes[key]) for key in LANE_KEYS},
        "busy_ms": {key: busy_ns(ivs[key]) / 1e6 for key in LANE_KEYS},
        "bytes": {key: sum(int(op["bytes"] or 0) for op in lanes[key]) for key in ("h2d", "d2h")},
        "kernels": kernels,
        "nvtx_range": trace.get("nvtx_range"),
        "nvtx_range_found": trace.get("nvtx_range_found"),
        "other_memcpy": trace.get("other_memcpy", 0),
    }


# --------------------------------------------------------------------------- #
# SVG
# --------------------------------------------------------------------------- #


def _nice_step(span_ms: float, target_ticks: int = 8) -> float:
    if span_ms <= 0:
        return 1.0
    raw = span_ms / target_ticks
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 5.0, 10.0):
        if mult * magnitude >= raw:
            return mult * magnitude
    return 10.0 * magnitude


def _fmt_ms(value: float, step: float) -> str:
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return f"{value:.{decimals}f}"


def _fmt_bytes(nbytes) -> str:
    if not nbytes:
        return ""
    nbytes = int(nbytes)
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MiB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.1f} KiB"
    return f"{nbytes} B"


def render_svg(trace: dict, title: Optional[str] = None, width: int = WIDTH) -> str:
    """Build the SVG document for a trace returned by :func:`read_trace`."""
    lanes = trace["lanes"]
    stats = trace_stats(trace)
    t0, t1 = trace["t0"], trace["t1"]
    span = t1 - t0
    span_ms = span / 1e6
    n_ops = sum(stats["bars"].values())

    left, right = 84, 18
    title_h = 48
    lane_h, bar_h = 40, 24
    axis_h = 44
    height = title_h + lane_h * len(LANES) + axis_h
    plot_w = width - left - right

    if title is None:
        title = (f"Three-stream pipeline: {stats['overlap_pct']:.1f}% of the trace has two or more lanes active"
                 if n_ops else "Three-stream pipeline: no GPU operations in the trace window")
    legend_parts = []
    for key, short, long_label, _colour in LANES:
        unit = "kernels" if key == "kernel" else "copies"
        legend_parts.append(f"{long_label}: {stats['bars'][key]} {unit}, {stats['busy_ms'][key]:.2f} ms busy")
    legend = " · ".join(legend_parts) + f" · window {span_ms:.2f} ms"

    font = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-labelledby="pipeline-timeline-title" '
        f'style="max-width:100%;height:auto;font-family:{font}">',
        f'<title id="pipeline-timeline-title">{escape(title)}</title>',
        f'<desc>{escape(legend)}</desc>',
        f'<text class="title" x="{left}" y="20" font-size="15" font-weight="600" fill="{TEXT}">{escape(title)}</text>',
        f'<text class="legend" x="{left}" y="38" font-size="10.5" fill="{TEXT}">{escape(legend)}</text>',
    ]

    def x_of(t: int) -> float:
        return left + ((t - t0) / span * plot_w if span > 0 else 0.0)

    y = title_h
    for idx, (key, short, long_label, colour) in enumerate(LANES):
        ops = lanes.get(key, [])
        parts.append(f'<g class="lane" data-lane="{key}">')
        parts.append(f'<rect class="band" x="{left}" y="{y}" width="{plot_w}" height="{lane_h}" '
                     f'fill="{BAND}" fill-opacity="{0.10 if idx % 2 == 0 else 0.05}"/>')
        parts.append(f'<rect class="swatch" x="{left - 16}" y="{y + lane_h / 2 - 5:.1f}" width="10" height="10" rx="2" fill="{colour}"/>')
        parts.append(f'<text class="label" x="{left - 22}" y="{y + lane_h / 2 + 5:.1f}" text-anchor="end" '
                     f'font-size="15" font-weight="600" fill="{TEXT}">{escape(short)}<title>{escape(long_label)}</title></text>')
        bar_y = y + (lane_h - bar_h) / 2
        for op in ops:
            start, end = op["start"], op["end"]
            x = x_of(start)
            w = max((end - start) / span * plot_w, 0.75) if span > 0 else plot_w
            if w > 2.0:
                w -= 0.6  # hairline gap between touching bars
            dur_ms = (end - start) / 1e6
            at_ms = (start - t0) / 1e6
            size = _fmt_bytes(op.get("bytes"))
            tip = f"{long_label}: {op.get('name') or key}, {dur_ms:.3f} ms at {at_ms:.3f} ms"
            if size:
                tip += f", {size}"
            if op.get("stream") is not None:
                tip += f", stream {op['stream']}"
            parts.append(f'<rect class="bar" data-lane="{key}" x="{x:.2f}" y="{bar_y:.1f}" width="{w:.2f}" '
                         f'height="{bar_h}" rx="2" fill="{colour}"><title>{escape(tip)}</title></rect>')
        parts.append("</g>")
        y += lane_h

    axis_y = y + 8
    parts.append(f'<g class="axis" fill="{TEXT}" font-size="11">')
    parts.append(f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" stroke="{TEXT}" '
                 f'stroke-opacity="0.6" stroke-width="1"/>')
    if span_ms > 0:
        step = _nice_step(span_ms)
        tick = 0.0
        while tick <= span_ms + step * 1e-6:
            x = left + tick / span_ms * plot_w
            parts.append(f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" y2="{axis_y + 5}" stroke="{TEXT}" stroke-opacity="0.6"/>')
            parts.append(f'<text x="{x:.2f}" y="{axis_y + 18}" text-anchor="middle">{_fmt_ms(tick, step)}</text>')
            tick += step
    parts.append(f'<text x="{left + plot_w}" y="{axis_y + 33}" text-anchor="end">time (ms)</text>')
    parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render(sqlite_path, svg_path, nvtx_range: Optional[str] = NVTX_RANGE_DEFAULT,
           title: Optional[str] = None, window: Optional[Interval] = None) -> dict:
    """Read a trace, write the SVG and return its statistics."""
    trace = read_trace(sqlite_path, nvtx_range=nvtx_range, window=window)
    svg = render_svg(trace, title=title)
    svg_path = Path(svg_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")
    stats = trace_stats(trace)
    stats["svg"] = str(svg_path)
    stats["sqlite"] = str(sqlite_path)
    return stats


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render an Nsight Systems SQLite export as a three-lane SVG timeline")
    parser.add_argument("--sqlite", required=True, help="output of `nsys export --type sqlite`")
    parser.add_argument("--out", required=True, help="SVG path to write")
    parser.add_argument("--nvtx-range", default=NVTX_RANGE_DEFAULT,
                        help="restrict the drawing to the last NVTX range with this name ('' = whole trace)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    stats = render(args.sqlite, args.out, nvtx_range=args.nvtx_range or None, title=args.title)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
