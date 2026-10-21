"""Poll the recorder sidecar once a minute until the snapshot target is reached.

    python scripts/poll_recorder.py [--target 1200000] [--margin 0.05] [--max-minutes 110]

Prints one status line per poll (total snapshots, rate over the last interval, gaps, liveness).
Exit codes: 0 target plus margin reached (the recorder is then stopped through its PID file),
2 the recorder process is not alive, 3 ``--max-minutes`` elapsed first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Wait for the recorder to reach its target")
    parser.add_argument("--data", default=str(REPO / "data"))
    parser.add_argument("--target", type=int, default=1_200_000)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--max-minutes", type=float, default=110.0)
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO / "scripts"))
    from record import status

    goal = int(args.target * (1 + args.margin))
    t0 = time.monotonic()
    last_n, last_t = None, None
    while True:
        st = status(Path(args.data))
        n = int(st.get("snapshots_total") or 0)
        now = time.monotonic()
        rate = (n - last_n) / (now - last_t) if last_n is not None and now > last_t else 0.0
        print(f"[{time.strftime('%H:%M:%S')}] snapshots {n:,}/{goal:,} rate {rate:.1f}/s gaps "
              f"{st.get('gaps_this_run', 0)} alive {st.get('process_alive')}", flush=True)
        if n >= goal:
            subprocess.run([sys.executable, str(REPO / "scripts" / "record.py"), "--data", args.data, "--stop"])
            return 0
        if not st.get("process_alive"):
            return 2
        if (now - t0) / 60.0 >= args.max_minutes:
            return 3
        last_n, last_t = n, now
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
