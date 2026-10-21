# Re-run selected sections on the real archive after a fix (full iterations, no --quick);
# the renderer merges sections across real T4 runs, latest run winning per section.
os.environ.setdefault("SECTIONS", "ladder,scans,precision")
os.environ.setdefault("SKIP_ENGINE", "1")
_body = open(f"{REPO}/remote/jobs/bench_all.py", encoding="utf-8").read()
exec(compile(_body, "bench_all.py", "exec"))
