"""Check the resume bullets in BULLETS.md against the style rules.

    python scripts/check_bullets.py [--file BULLETS.md]

Each bullet follows a ``<!-- bullet:N -->`` marker and must be exactly two lines of 98 to 110
characters (200 to 220 in total, so there is no short dangling last line), contain no em dash,
spell out terms except the allowed product names, and, for the alternates (every bullet after the
three template bullets), avoid a three-item list. Bullets 1 to 5 must be present; later
alternates are optional and checked by the same rules. Prints PASS or FAIL per bullet with the
measured lengths; exits non-zero on FAIL.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent
MIN_LINE, MAX_LINE = 98, 110
MIN_TOTAL, MAX_TOTAL = 200, 220
ALLOWED_CAPS = {"CUDA", "NVIDIA", "T4"}  # Nsight, NumPy, Python, C++, FastAPI, Coinbase have no capital runs
TEMPLATE = {"1", "2", "3"}
REQUIRED = ("1", "2", "3", "4", "5")
THREE_ITEM = re.compile(r"\b[\w'-]+(?: [\w'-]+){0,3}, [\w'-]+(?: [\w'-]+){0,3},? and ")


def split_two_lines(text: str) -> Optional[List[str]]:
    """Break ``text`` at the space nearest its middle that leaves both lines 98 to 110 long."""
    if not (MIN_TOTAL <= len(text) <= MAX_TOTAL):
        return None
    best = None
    for i, ch in enumerate(text):
        if ch != " ":
            continue
        a, b = text[:i], text[i + 1:]
        if MIN_LINE <= len(a) <= MAX_LINE and MIN_LINE <= len(b) <= MAX_LINE:
            score = abs(len(a) - len(b))
            if best is None or score < best[0]:
                best = (score, [a, b])
    return best[1] if best else None


def problems(lines: List[str], alternate: bool = False) -> List[str]:
    out = []
    if len(lines) != 2:
        out.append(f"{len(lines)} lines, expected 2")
    for i, ln in enumerate(lines, 1):
        if not (MIN_LINE <= len(ln) <= MAX_LINE):
            out.append(f"line {i} is {len(ln)} characters")
    text = " ".join(lines)
    if not (MIN_TOTAL <= len(text) <= MAX_TOTAL):
        out.append(f"total {len(text)} characters")
    if "—" in text:
        out.append("contains an em dash")
    for word in re.findall(r"\b[A-Z][A-Z0-9]+\b", text):
        if word not in ALLOWED_CAPS:
            out.append(f"abbreviation {word!r}")
    if alternate and THREE_ITEM.search(text):
        out.append(f"three-item list: {THREE_ITEM.search(text).group(0)!r}")
    return out


def read_bullets(text: str) -> dict:
    bullets, current = {}, None
    for line in text.splitlines():
        m = re.match(r"<!-- bullet:(\w+) -->", line.strip())
        if m:
            current = m.group(1)
            bullets[current] = []
            continue
        if current is not None:
            if not line.strip():
                if bullets[current]:
                    current = None
                continue
            bullets[current].append(line.rstrip("\n"))
    return bullets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bullet length and style check")
    parser.add_argument("--file", default=str(REPO / "BULLETS.md"))
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"FAIL: {path} does not exist")
        return 1
    bullets = read_bullets(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in bullets]
    ok = not missing
    if missing:
        print(f"FAIL: bullets {', '.join(missing)} missing (found {len(bullets)})")
    for key, lines in bullets.items():
        probs = problems(lines, alternate=key not in TEMPLATE)
        lengths = ", ".join(str(len(ln)) for ln in lines)
        total = len(" ".join(lines))
        print(f"bullet {key}: {'PASS' if not probs else 'FAIL'} (lines {lengths}; total {total})"
              + ("" if not probs else " " + "; ".join(probs)))
        ok = ok and not probs
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
