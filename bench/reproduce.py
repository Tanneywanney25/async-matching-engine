"""Rebuild everything from the raw log: archive, remote T4 run, results, documents.

    python -m bench.reproduce --source real [--dry-run] [--skip-upload] [--quick]
    python -m bench.reproduce --source synthetic [--dry-run]

real:      build data/archive from data/raw, upload it as a new version of the private Kaggle
           dataset, run the bench_all job on a T4 pinned to that archive's content hash, pull the
           results, then render every document and run the provenance and bullet checks.
synthetic: write data/synth, upload it to the synthetic dataset and run the job; synthetic
           results never reach a document, so nothing is rendered.

Each step is printed before it runs; a failing step stops the run with its exit code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def steps(source: str, skip_upload: bool, quick: bool) -> List[List[str]]:
    job = "bench_quick" if quick else "bench_all"
    if source == "real":
        archive, slug = "data/archive", "exchange-engine-archive"
        out = [[PY, "scripts/build_archive.py", "--raw", "data/raw", "--out", archive]]
    else:
        archive, slug = "data/synth", "exchange-engine-archive-synth"
        out = [[PY, "scripts/synth_archive.py", "--out", archive]]
    if not skip_upload:
        out.append([PY, "-m", "remote.kaggle", "dataset", "--archive", archive, "--slug", slug, "--wait-ready"])
    out.append([PY, "-m", "remote.kaggle", "run", "--job", job, "--dataset", f"tanushbhatia/{slug}",
                "--source", source, "--archive-hash", "{content_hash}", "--timeout-minutes", "45"])
    if source == "real":
        out += [[PY, "scripts/render_docs.py"], [PY, "scripts/check_numbers.py"], [PY, "scripts/check_bullets.py"]]
    return out


def content_hash(archive: str) -> str:
    path = REPO / archive / "archive.json"
    return json.loads(path.read_text(encoding="utf-8"))["content_hash"] if path.exists() else "<hash after build>"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the GPU analytics results")
    parser.add_argument("--source", required=True, choices=("real", "synthetic"))
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--skip-upload", action="store_true", help="reuse the dataset version already on Kaggle")
    parser.add_argument("--quick", action="store_true", help="run the short smoke job instead of the full one")
    args = parser.parse_args(argv)
    archive = "data/archive" if args.source == "real" else "data/synth"
    for cmd in steps(args.source, args.skip_upload, args.quick):
        cmd = [c.replace("{content_hash}", content_hash(archive)) for c in cmd]
        print("$ " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc != 0:
            print(f"step failed with exit code {rc}", flush=True)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
