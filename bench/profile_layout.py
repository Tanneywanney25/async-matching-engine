"""Profile the AoS and SoA layout kernels under Nsight Systems and parse bullet 2's figure.

Runs, from the repository root (paths under ``<out>/raw/profiles``)::

    nsys profile --trace=cuda,nvtx --stats=true -o <out>/raw/profiles/aos python bench/bench_layout.py --layout aos --archive <dir>
    nsys profile --trace=cuda,nvtx --stats=true -o <out>/raw/profiles/soa python bench/bench_layout.py --layout soa --archive <dir>
    nsys stats --report cuda_gpu_kern_sum --format csv --output <out>/raw/profiles/aos <out>/raw/profiles/aos.nsys-rep
    nsys stats --report cuda_gpu_kern_sum --format csv --output <out>/raw/profiles/soa <out>/raw/profiles/soa.nsys-rep

then sums the ``layout_*`` kernels' total time in each CSV and reports
``reduction_pct = (aos - soa) / aos * 100``. Nsight Compute is tried once and its outcome recorded.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from bench import ref_scans as ref
from bench.common import REPO_ROOT


def _run(cmd, log_path: Path, timeout: int = 1200) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(map(str, cmd))}\n[exit {proc.returncode}]\n{proc.stdout[-20000:]}\n{proc.stderr[-5000:]}\n")
    return proc


def _child_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("LAYOUT_RESULT "):
            return json.loads(line[len("LAYOUT_RESULT "):])
    return {}


def profile(out_dir: Path, archive_dir: Path, iters: int = 30, n_limit: int = 0) -> dict:
    out_dir = Path(out_dir)
    prof = out_dir / "raw" / "profiles"
    prof.mkdir(parents=True, exist_ok=True)
    log_path = prof / "nsys_layout.log"
    nsys = os.environ.get("NSYS") or shutil.which("nsys")
    section: dict = {"iters": iters, "nsys_version": None}
    if not nsys:
        section["error"] = "nsys not found on PATH"
        return section
    section["nsys_version"] = subprocess.run([nsys, "--version"], capture_output=True, text=True).stdout.strip()
    children = {}
    rows = {}
    for layout in ("aos", "soa"):
        child = [sys.executable, "bench/bench_layout.py", "--layout", layout, "--archive", str(archive_dir),
                 "--iters", str(iters)] + (["--n", str(n_limit)] if n_limit else [])
        cmd = [nsys, "profile", "--trace=cuda,nvtx", "--stats=true", "--force-overwrite", "true",
               "-o", str(prof / layout)] + child
        proc = _run(cmd, log_path)
        children[layout] = _child_result(proc.stdout)
        rep = prof / f"{layout}.nsys-rep"
        if proc.returncode != 0 or not rep.exists():
            section["error"] = f"nsys profile {layout} exited {proc.returncode}: {(proc.stderr or proc.stdout)[-800:]}"
            return section
        stats = _run([nsys, "stats", "--force-export=true", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                      "--output", str(prof / layout), str(rep)], log_path)
        csvs = sorted(prof.glob(f"{layout}*cuda_gpu_kern_sum*.csv"))
        if stats.returncode != 0 or not csvs:
            section["error"] = f"nsys stats {layout} exited {stats.returncode}: {(stats.stderr or stats.stdout)[-800:]}"
            return section
        csv_path = csvs[0]
        final = prof / f"{layout}_kern_sum.csv"
        if csv_path != final:
            shutil.copy(csv_path, final)
        rows[layout] = ref.parse_kern_sum_csv(final.read_text(encoding="utf-8"))
    aos_ns = ref.total_kernel_ns(rows["aos"])
    soa_ns = ref.total_kernel_ns(rows["soa"])
    section.update({
        "aos_total_kernel_ms": aos_ns / 1e6, "soa_total_kernel_ms": soa_ns / 1e6,
        "reduction_pct": ref.reduction_pct(aos_ns, soa_ns),
        "kernels": ref.kernel_pairs(rows["aos"], rows["soa"]),
        "children": children, "verified": bool(children.get("aos", {}).get("verified") and
                                               children.get("soa", {}).get("verified")),
        "n_snapshots": children.get("soa", {}).get("n"),
        "reports": {"aos": "raw/profiles/aos.nsys-rep", "soa": "raw/profiles/soa.nsys-rep",
                    "aos_csv": "raw/profiles/aos_kern_sum.csv", "soa_csv": "raw/profiles/soa_kern_sum.csv"},
        "measured_by": "nsys stats cuda_gpu_kern_sum, total time of the layout_* kernels over all iterations",
    })
    # Nsight Compute, once
    ncu = shutil.which("ncu")
    if ncu:
        child = [sys.executable, "bench/bench_layout.py", "--layout", "soa", "--archive", str(archive_dir), "--iters", "1",
                 "--n", str(n_limit or 100_000)]
        proc = _run([ncu, "--metrics", "gpu__time_duration.sum"] + child, log_path, timeout=900)
        text = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0 and "ERR_NVGPUCTRPERM" not in text and "==ERROR==" not in text
        msg = next((ln for ln in text.splitlines() if "ERR_" in ln or "==ERROR==" in ln), "")
        section["ncu"] = {"attempted": True, "ok": ok, "message": msg[:400] or ("ok" if ok else text[-400:])}
    else:
        section["ncu"] = {"attempted": False, "ok": False, "message": "ncu not on PATH"}
    return section
