# Quick smoke variant of bench_all: QUICK=1 so every section runs a short pass on a
# subset; SECTIONS limits which GPU sections run. Shakes out compile errors before a
# full run. Never used for documented numbers.
os.environ["QUICK"] = "1"
os.environ.setdefault("SECTIONS", "ladder,scans,precision,layout,slippage,ofi,pipeline")
_body = open(f"{REPO}/remote/jobs/bench_all.py", encoding="utf-8").read()
exec(compile(_body, "bench_all.py", "exec"))
