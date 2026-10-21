"""Kaggle backend for the remote GPU runner.

    python -m remote.kaggle probe                     # environment probe on a T4
    python -m remote.kaggle run --job bench_all --dataset tanushbhatia/exchange-engine-archive-synth --source synthetic
    python -m remote.kaggle wait --job bench_all --run-id <id>   # poll, pull, ledger
    python -m remote.kaggle dataset --archive data/synth --slug exchange-engine-archive-synth
    python -m remote.kaggle quota

Every job's first remote action is the T4 check in the bootstrap (``remote/base.py``);
a wrong card exits non-zero and ``run`` re-pushes up to three times. Results land in
``results/<run_id>/`` and every run is appended to ``remote/quota.json``.

The Kaggle access token is read by the ``kaggle`` executable from
``~/.kaggle/access_token``; this module never reads, prints or stores it, and
scrubs anything token-shaped from command output.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .base import (BOOTSTRAP, GIT_URL, REPO_ROOT, RESULTS_DIR, TOKEN_PREFIX, JobSpec, git_sha, sanitize,
                   write_json)

USER = "tanushbhatia"
LEDGER = REPO_ROOT / "remote" / "quota.json"
PUSH_DIR = REPO_ROOT / "remote" / ".push"
DATASET_DIR = REPO_ROOT / "remote" / ".datasets"
JOBS_DIR = REPO_ROOT / "remote" / "jobs"
GPU_HOURS_BUDGET = 20.0
TERMINAL = {"complete", "error", "cancelacknowledged", "cancelled", "failed"}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def kaggle(*args: str, timeout: int = 1800, check: bool = True) -> str:
    """Run the kaggle CLI and return its scrubbed combined output."""
    proc = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    out = sanitize((proc.stdout or "") + (proc.stderr or ""))
    if check and proc.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed ({proc.returncode}):\n{out}")
    return out


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def load_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"budget_gpu_hours": GPU_HOURS_BUDGET, "runs": [], "gpu_hours_used_estimate": 0.0,
            "kaggle_reported": None}


def save_ledger(ledger: dict) -> None:
    ledger["gpu_hours_used_estimate"] = round(
        sum(r.get("gpu_minutes") or 0.0 for r in ledger["runs"]) / 60.0, 3
    )
    write_json(LEDGER, ledger)


def ledger_update(run_id: str, **fields) -> dict:
    ledger = load_ledger()
    for run in ledger["runs"]:
        if run["run_id"] == run_id:
            run.update(fields)
            break
    else:
        ledger["runs"].append({"run_id": run_id, **fields})
    save_ledger(ledger)
    return ledger


def quota_report() -> dict:
    """Parse ``kaggle quota`` (used/remaining/total GPU hours) and store it in the ledger."""
    out = kaggle("quota", check=False)
    report = {"raw": out.strip(), "checked_at": now_iso()}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] in ("GPU", "TPU") and len(parts) >= 4:
            try:
                report[parts[0].lower()] = {
                    "used_hours": float(parts[1].rstrip("h")),
                    "remaining_hours": float(parts[2].rstrip("h")),
                    "total_hours": float(parts[3].rstrip("h")),
                }
            except ValueError:
                pass
    ledger = load_ledger()
    ledger["kaggle_reported"] = report
    save_ledger(ledger)
    return report


def budget_guard(expected_minutes: int) -> None:
    ledger = load_ledger()
    used = ledger.get("gpu_hours_used_estimate", 0.0)
    if used + expected_minutes / 60.0 > ledger.get("budget_gpu_hours", GPU_HOURS_BUDGET):
        raise SystemExit(f"GPU budget exhausted: {used:.2f} h used, job needs {expected_minutes} min")


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #


def kernel_slug(job_name: str) -> str:
    return f"exchange-engine-gpu-{job_name.replace('_', '-')}"


def kernel_id(job_name: str) -> str:
    return f"{USER}/{kernel_slug(job_name)}"


def compose(spec: JobSpec, run_id: str, sha: str) -> str:
    header = BOOTSTRAP.format(
        run_id=run_id, job_name=spec.name, git_sha=sha, git_url=GIT_URL, branch=spec.branch,
        needs_repo=spec.needs_repo, source=spec.source, archive_hash=spec.archive_hash,
    )
    footer = r'''
# ---- bundle results so `kaggle kernels output` always gets a single file ----
import tarfile as _tarfile
with _tarfile.open(f"/kaggle/working/results_{RUN_ID}.tar.gz", "w:gz") as _tf:
    _tf.add(OUT, arcname=RUN_ID)
print("RESULT_BUNDLE", f"results_{RUN_ID}.tar.gz", flush=True)
'''
    return header + "\n" + spec.body + "\n" + footer


def load_job(name: str) -> str:
    path = JOBS_DIR / f"{name}.py"
    if not path.exists():
        raise SystemExit(f"no job body at {path}")
    return path.read_text(encoding="utf-8")


def push(spec: JobSpec, run_id: Optional[str] = None) -> str:
    """Push a kernel version for ``spec``; returns the run id."""
    budget_guard(spec.timeout_minutes)
    run_id = run_id or f"{spec.name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    sha = spec.git_sha or git_sha()
    folder = PUSH_DIR / run_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(compose(spec, run_id, sha), encoding="utf-8")
    meta = {
        "id": kernel_id(spec.name),
        "title": kernel_slug(spec.name),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": list(spec.datasets),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    write_json(folder / "kernel-metadata.json", meta)
    out = kaggle("kernels", "push", "-p", str(folder), "--timeout", str(spec.timeout_minutes * 60))
    print(out.strip())
    if "error" in out.lower() and "pushed" not in out.lower():
        raise RuntimeError(f"kernel push failed: {out}")
    ledger_update(run_id, kernel=kernel_id(spec.name), job=spec.name, pushed_at=now_iso(),
                  git_sha=sha, status="pushed", source=spec.source, datasets=list(spec.datasets),
                  timeout_minutes=spec.timeout_minutes)
    return run_id


def status(job_name: str) -> str:
    out = kaggle("kernels", "status", kernel_id(job_name), check=False)
    m = re.search(r'status\s+"?(?:KernelWorkerStatus\.)?([A-Za-z]+)"?', out)
    return (m.group(1) if m else out.strip()).lower()


def wait(job_name: str, run_id: str, poll_seconds: int = 30, max_minutes: int = 70) -> str:
    """Poll until the kernel finishes; records running time in the ledger."""
    started_running: Optional[float] = None
    t0 = time.time()
    last = None
    while time.time() - t0 < max_minutes * 60:
        st = status(job_name)
        if st != last:
            print(f"[{time.strftime('%H:%M:%S')}] {kernel_id(job_name)} -> {st}", flush=True)
            last = st
        if st == "running" and started_running is None:
            started_running = time.time()
            ledger_update(run_id, status="running", running_at=now_iso())
        if st in TERMINAL:
            minutes = (time.time() - (started_running or t0)) / 60.0
            ledger_update(run_id, status=st, finished_at=now_iso(), gpu_minutes=round(minutes, 2))
            return st
        time.sleep(poll_seconds)
    ledger_update(run_id, status="timeout-waiting")
    return "timeout"


def pull(job_name: str, run_id: str) -> Path:
    """Download the kernel output into ``results/<run_id>/``."""
    tmp = PUSH_DIR / run_id / "output"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    out = kaggle("kernels", "output", kernel_id(job_name), "-p", str(tmp), check=False)
    print(out.strip()[-500:])
    dest = RESULTS_DIR / run_id
    dest.mkdir(parents=True, exist_ok=True)
    bundle = next(tmp.glob("results_*.tar.gz"), None)
    if bundle is not None:
        with tarfile.open(bundle, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.name.startswith(run_id + "/") or m.name == run_id]
            tf.extractall(tmp / "bundle", members=members, filter="data")
        src = tmp / "bundle" / run_id
        if src.exists():
            shutil.copytree(src, dest, dirs_exist_ok=True)
    for log in tmp.glob("*.log"):
        shutil.copy(log, dest / "kernel.log")
    for extra in tmp.iterdir():
        if extra.is_file() and extra.suffix in (".json", ".csv", ".svg", ".txt") and not (dest / extra.name).exists():
            shutil.copy(extra, dest / extra.name)
    manifest_path = dest / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["kaggle_kernel"] = kernel_id(job_name)
        manifest["pulled_at"] = now_iso()
        write_json(manifest_path, manifest)
        ledger_update(run_id, wall_seconds=manifest.get("wall_seconds"),
                      gpu_name=manifest.get("gpu_name"), result_status=manifest.get("status"))
    scrub_dir(dest)
    print(f"results in {dest}")
    return dest


def scrub_dir(path: Path) -> None:
    """Belt and braces: make sure nothing token-shaped is in pulled text files."""
    for file in path.rglob("*"):
        if file.is_file() and file.suffix in (".log", ".json", ".txt", ".csv", ".md"):
            text = file.read_text(encoding="utf-8", errors="replace")
            if TOKEN_PREFIX in text:
                file.write_text(sanitize(text), encoding="utf-8")


def aborted_wrong_card(dest: Path) -> bool:
    manifest = dest / "manifest.json"
    if manifest.exists():
        try:
            if json.loads(manifest.read_text(encoding="utf-8")).get("aborted"):
                return True
        except ValueError:
            pass
    log = dest / "kernel.log"
    return log.exists() and "ABORT: expected an NVIDIA T4" in log.read_text(encoding="utf-8", errors="replace")


def run(spec: JobSpec, attempts: int = 3, poll_seconds: int = 30) -> Path:
    """Push, wait, pull; re-push up to ``attempts`` times if the card was not a T4."""
    for attempt in range(1, attempts + 1):
        run_id = push(spec)
        st = wait(spec.name, run_id, poll_seconds=poll_seconds, max_minutes=spec.timeout_minutes + 25)
        dest = pull(spec.name, run_id)
        quota_report()
        if aborted_wrong_card(dest):
            print(f"attempt {attempt}: wrong GPU, retrying")
            ledger_update(run_id, status="wrong-gpu")
            continue
        print(f"run {run_id} finished with status {st}")
        return dest
    raise SystemExit("gave up: no T4 after three attempts")


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def stage_archive(archive_dir: Path, slug: str, compress: bool = True) -> Path:
    """Copy an archive into a dataset folder, zstd-compressing the big arrays."""
    import zstandard as zstd

    folder = DATASET_DIR / slug
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    meta = json.loads((archive_dir / "archive.json").read_text(encoding="utf-8"))
    shutil.copy(archive_dir / "archive.json", folder / "archive.json")
    cctx = zstd.ZstdCompressor(level=6)
    for file in sorted(archive_dir.iterdir()):
        if file.name == "archive.json" or file.name.startswith("."):
            continue
        if compress:
            with open(file, "rb") as src, open(folder / (file.name + ".zst"), "wb") as dst:
                cctx.copy_stream(src, dst)
        else:
            shutil.copy(file, folder / file.name)
    write_json(folder / "dataset-metadata.json", {
        "title": slug,
        "id": f"{USER}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": (f"exchange-engine snapshot archive ({meta['source']}): {meta['n_snapshots']} snapshots, "
                        f"{meta['n_trades']} trades, content hash {meta['content_hash']}"),
    })
    return folder


def dataset_exists(slug: str) -> bool:
    out = kaggle("datasets", "list", "--mine", "--search", slug, check=False)
    return f"{USER}/{slug}" in out


def dataset_upload(archive_dir: Path, slug: str, compress: bool = True) -> str:
    folder = stage_archive(archive_dir, slug, compress=compress)
    meta = json.loads((archive_dir / "archive.json").read_text(encoding="utf-8"))
    size_mb = sum(f.stat().st_size for f in folder.iterdir()) / 1e6
    print(f"uploading {size_mb:.1f} MB as {USER}/{slug} (source={meta['source']})", flush=True)
    t0 = time.time()
    if dataset_exists(slug):
        out = kaggle("datasets", "version", "-p", str(folder), "-m",
                     f"{meta['source']} archive {meta['n_snapshots']} snapshots hash {meta['content_hash'][:12]}",
                     "--dir-mode", "skip", timeout=7200)
    else:
        out = kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "skip", timeout=7200)
    print(out.strip()[-400:])
    print(f"upload took {time.time() - t0:.0f}s", flush=True)
    ledger = load_ledger()
    ledger.setdefault("datasets", []).append({
        "id": f"{USER}/{slug}", "uploaded_at": now_iso(), "source": meta["source"],
        "n_snapshots": meta["n_snapshots"], "content_hash": meta["content_hash"], "mb": round(size_mb, 1),
    })
    save_ledger(ledger)
    return f"{USER}/{slug}"


def dataset_ready(slug: str, max_minutes: int = 30) -> bool:
    """Wait until Kaggle finishes processing the dataset."""
    t0 = time.time()
    while time.time() - t0 < max_minutes * 60:
        out = kaggle("datasets", "status", f"{USER}/{slug}", check=False)
        print(f"[{time.strftime('%H:%M:%S')}] dataset status: {out.strip()[:80]}", flush=True)
        if "ready" in out.lower():
            return True
        if "error" in out.lower() and "not found" not in out.lower():
            return False
        time.sleep(20)
    return False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Kaggle T4 runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="push the environment probe and wait for it")
    p.add_argument("--no-wait", action="store_true")

    p = sub.add_parser("run", help="push a job and wait for it")
    p.add_argument("--job", required=True)
    p.add_argument("--dataset", action="append", default=[])
    p.add_argument("--source", default="synthetic", choices=["real", "synthetic"])
    p.add_argument("--archive-hash", default=None)
    p.add_argument("--timeout-minutes", type=int, default=45)
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--sha", default=None)

    p = sub.add_parser("wait", help="poll a pushed job, then pull it")
    p.add_argument("--job", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--max-minutes", type=int, default=70)

    p = sub.add_parser("pull", help="download output for a run")
    p.add_argument("--job", required=True)
    p.add_argument("--run-id", required=True)

    p = sub.add_parser("status")
    p.add_argument("--job", required=True)

    p = sub.add_parser("dataset", help="upload an archive as a private dataset (or new version)")
    p.add_argument("--archive", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--wait-ready", action="store_true")

    sub.add_parser("quota", help="record kaggle's reported GPU quota in the ledger")

    args = parser.parse_args(argv)
    if args.cmd == "probe":
        spec = JobSpec(name="probe", body=load_job("probe"), needs_repo=False, timeout_minutes=20)
        if args.no_wait:
            print("run id:", push(spec))
        else:
            run(spec)
    elif args.cmd == "run":
        spec = JobSpec(name=args.job, body=load_job(args.job), datasets=args.dataset, source=args.source,
                       archive_hash=args.archive_hash, timeout_minutes=args.timeout_minutes, git_sha=args.sha)
        if args.no_wait:
            print("run id:", push(spec))
        else:
            run(spec)
    elif args.cmd == "wait":
        st = wait(args.job, args.run_id, max_minutes=args.max_minutes)
        pull(args.job, args.run_id)
        quota_report()
        print("status:", st)
    elif args.cmd == "pull":
        pull(args.job, args.run_id)
    elif args.cmd == "status":
        print(status(args.job))
    elif args.cmd == "dataset":
        did = dataset_upload(Path(args.archive), args.slug, compress=not args.no_compress)
        if args.wait_ready:
            print("ready:", dataset_ready(args.slug))
        print(did)
    elif args.cmd == "quota":
        print(json.dumps(quota_report(), indent=2))


if __name__ == "__main__":
    main()
