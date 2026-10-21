"""Run, stop, or inspect the raw Coinbase recorder.

Start (foreground; use ``Start-Process`` or ``nohup`` to detach)::

    python scripts/record.py [--data data] [--target 1200000] [--max-hours 10]

Graceful stop through the PID file (drops a stop file the process polls, waits
for exit, and only then falls back to ``taskkill``)::

    python scripts/record.py --stop

Status (reads the sidecar and checks the PID is alive)::

    python scripts/record.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pid_alive(pid: int) -> bool:
    """Return True if ``pid`` is a running process (Windows and POSIX)."""
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    import os

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_pid(data_dir: Path) -> int | None:
    try:
        return int((data_dir / "recorder.pid").read_text().strip())
    except (OSError, ValueError):
        return None


def status(data_dir: Path) -> dict:
    sidecar = data_dir / "raw" / "recorder_state.json"
    state = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    pid = read_pid(data_dir)
    state["pid_file"] = pid
    state["process_alive"] = bool(pid and pid_alive(pid))
    return state


def stop(data_dir: Path, wait_seconds: float = 60.0) -> int:
    pid = read_pid(data_dir)
    if not pid or not pid_alive(pid):
        print(f"recorder not running (pid file: {pid})")
        return 0
    stop_file = data_dir / "recorder.stop"
    stop_file.write_text("stop", encoding="utf-8")
    print(f"stop file written; waiting for pid {pid} to exit...")
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            print("recorder exited cleanly")
            return 0
        time.sleep(1.0)
    print("recorder did not exit in time; forcing termination")
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
    else:
        import os
        import signal

        os.kill(pid, signal.SIGTERM)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw Coinbase recorder")
    parser.add_argument("--data", default="data", help="data directory (default: data)")
    parser.add_argument("--products", nargs="+", default=["BTC-USD", "ETH-USD"])
    parser.add_argument("--target", type=int, default=1_200_000, help="snapshot target")
    parser.add_argument("--margin", type=float, default=0.05, help="stop at target*(1+margin)")
    parser.add_argument("--max-hours", type=float, default=10.0, help="deadline from first start")
    parser.add_argument("--stop", action="store_true", help="stop a running recorder")
    parser.add_argument("--status", action="store_true", help="print the sidecar state")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if args.status:
        print(json.dumps(status(data_dir), indent=2))
        return
    if args.stop:
        sys.exit(stop(data_dir))

    data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(data_dir / "recorder.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    import asyncio

    from exchange_engine.recorder import Recorder

    recorder = Recorder(
        data_dir,
        product_ids=args.products,
        target_snapshots=args.target,
        margin=args.margin,
        max_hours=args.max_hours,
    )
    asyncio.run(recorder.run())


if __name__ == "__main__":
    main()
