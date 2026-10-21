# Quick smoke variant of bench_all: same body, QUICK=1 so every section runs a
# short pass on a subset. Used to shake out compile errors before a full run.
os.environ["QUICK"] = "1"
_body = open(f"{REPO}/remote/jobs/bench_all.py", encoding="utf-8").read()
exec(compile(_body, "bench_all.py", "exec"))
