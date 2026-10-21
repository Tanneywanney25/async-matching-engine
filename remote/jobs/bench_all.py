# Job body: run every benchmark section on the attached archive (after the bootstrap
# in remote/base.py, which provides RUN_ID, OUT, SOURCE, ARCHIVE_HASH, REPO, sh, log,
# finish and gpu_state, and has already aborted on any card that is not a T4).
import glob

def _find_archive():
    for pattern in ("/kaggle/input/*/archive.json", "/kaggle/input/*/*/archive.json",
                    "/kaggle/input/*/*/*/archive.json"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return os.path.dirname(hits[0])
    return None

archive_dir = _find_archive()
log("archive:", archive_dir)
if archive_dir is None:
    log("ABORT: no archive.json under /kaggle/input")
    finish("error", {"error": "no archive attached"})
    sys.exit(4)
archive_meta = json.load(open(os.path.join(archive_dir, "archive.json")))
log(f"archive source={archive_meta.get('source')} snapshots={archive_meta.get('n_snapshots')} "
    f"trades={archive_meta.get('n_trades')} hash={str(archive_meta.get('content_hash'))[:16]}")
if ARCHIVE_HASH and archive_meta.get("content_hash") != ARCHIVE_HASH:
    log("ABORT: archive content hash does not match the requested hash", ARCHIVE_HASH)
    finish("error", {"error": "archive hash mismatch", "found_hash": archive_meta.get("content_hash")})
    sys.exit(4)
if archive_meta.get("source") != SOURCE:
    log("ABORT: archive source", archive_meta.get("source"), "!= job source", SOURCE)
    finish("error", {"error": "archive source mismatch"})
    sys.exit(4)
manifest["archive_hash"] = archive_meta.get("content_hash")
manifest["archive_snapshots"] = archive_meta.get("n_snapshots")

try:
    import zstandard  # noqa: F401
except ImportError:
    sh("pip install -q zstandard", timeout=600)

cmd = [sys.executable, "-u", "-m", "bench.run_all", "--archive", archive_dir, "--out", OUT, "--source", SOURCE]
if os.environ.get("QUICK") == "1":
    cmd.append("--quick")
if os.environ.get("SECTIONS"):
    cmd += ["--sections", os.environ["SECTIONS"]]
if os.environ.get("SKIP_ENGINE") == "1":
    cmd.append("--skip-engine")
if os.environ.get("SKIP_NATIVE") == "1":
    cmd.append("--skip-native")
cmd += ["--budget-minutes", os.environ.get("BUDGET_MINUTES", "36")]
log("running:", " ".join(cmd))
env_vars = dict(os.environ)
env_vars["PYTHONPATH"] = f"{REPO}:{REPO}/src:" + env_vars.get("PYTHONPATH", "")
proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env_vars)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
log("run_all exit code", rc)

errors = []
metrics_path = os.path.join(OUT, "metrics.json")
if os.path.exists(metrics_path):
    metrics = json.load(open(metrics_path))
    errors = metrics.get("errors", [])
    manifest["sections"] = sorted(metrics.get("sections", {}))
for e in errors:
    log("section error:", e.get("section"), str(e.get("error"))[-600:])
finish("complete" if (rc == 0 and not errors) else "partial",
       {"errors": [{"section": e.get("section"), "error": str(e.get("error"))[-2000:]} for e in errors],
        "run_all_exit": rc})
