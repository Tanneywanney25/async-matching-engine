"""Verify that every number in the rendered documents traces to a real T4 result.

    python scripts/check_numbers.py [--root .]

Scans ``BENCHMARKS.md``, ``BULLETS.md``, ``DESIGN.md``, ``INTERVIEW_NOTES.md`` and the README
section between the gpu-lab markers. Fenced code blocks, inline code spans, link targets, HTML
comments, ISO dates and times are ignored. Every remaining numeric token must either be a key of
``results/provenance.json`` (written by ``scripts/render_docs.py``) with at least one source that
was rendered into the same document and whose run manifest says ``"source": "real"`` with a GPU
name containing ``T4``, or be listed with a justification in ``scripts/doc_constants.json``. A
number that matches a value rendered into some other document does not count. Prints PASS, or FAIL
with every offending token.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NUM_RE = re.compile(r"(?<![A-Za-z_\d.])\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?")
MARK_START, MARK_END = "<!-- gpu-lab:start -->", "<!-- gpu-lab:end -->"
DOCUMENTS = ("BENCHMARKS.md", "BULLETS.md", "DESIGN.md", "INTERVIEW_NOTES.md")
README_DOC = "README.md (GPU section)"


def strip_ignored(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)        # fenced code
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)        # HTML comments
    text = re.sub(r"`[^`\n]*`", " ", text)                     # inline code
    text = re.sub(r"\]\([^)]*\)", "]", text)                   # link targets
    text = re.sub(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z?)?", " ", text)  # ISO dates and times
    return text


def tokens(text: str) -> list:
    return NUM_RE.findall(strip_ignored(text))


def documents(root: Path) -> dict:
    docs = {}
    for name in DOCUMENTS:
        path = root / name
        docs[name] = path.read_text(encoding="utf-8") if path.exists() else None
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    if MARK_START in readme and MARK_END in readme:
        docs[README_DOC] = readme[readme.index(MARK_START):readme.index(MARK_END)]
    else:
        docs[README_DOC] = None
    return docs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Provenance check for documented numbers")
    parser.add_argument("--root", default=str(REPO), help="directory holding the documents")
    parser.add_argument("--results", default=None, help="results directory (default <root>/results)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    results = Path(args.results) if args.results else root / "results"
    prov_path = results / "provenance.json"
    if not prov_path.exists():
        print(f"FAIL: {prov_path} missing (run scripts/render_docs.py)")
        return 1
    prov = json.loads(prov_path.read_text(encoding="utf-8"))["tokens"]
    constants = json.loads((REPO / "scripts" / "doc_constants.json").read_text(encoding="utf-8"))
    manifests: dict = {}

    def run_ok(run_id: str) -> bool:
        if run_id not in manifests:
            path = results / run_id / "manifest.json"
            if not path.exists():
                path = REPO / "results" / run_id / "manifest.json"
            m = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            metrics_path = path.parent / "metrics.json"
            src = json.loads(metrics_path.read_text(encoding="utf-8")).get("source") if metrics_path.exists() else None
            manifests[run_id] = m.get("source") == "real" and src == "real" and "T4" in str(m.get("gpu_name") or "")
        return manifests[run_id]

    bad = []
    counted = 0
    for name, text in documents(root).items():
        if text is None:
            bad.append((name, "<document missing>", "not rendered"))
            continue
        for tok in tokens(text):
            counted += 1
            if tok in constants:
                continue
            # entries written before documents were tagged carry no "doc" and match any document
            sources = [s for s in prov.get(tok, []) if s.get("doc") in (None, name)]
            if not sources:
                elsewhere = sorted({s.get("doc") for s in prov.get(tok, [])})
                bad.append((name, tok, "no provenance" + (f" in this document (rendered into {', '.join(elsewhere)})"
                                                         if elsewhere else "")))
            elif not any(run_ok(s["run_id"]) for s in sources):
                bad.append((name, tok, "source run is not real T4 data: " + ", ".join(sorted({s["run_id"] for s in sources}))))
    if bad:
        for name, tok, why in bad:
            print(f"FAIL {name}: {tok!r} ({why})")
        print(f"FAIL: {len(bad)} of {counted} numbers do not trace to real T4 results")
        return 1
    print(f"PASS: {counted} numbers in {', '.join(DOCUMENTS)} and the README GPU section trace to real T4 results "
          f"rendered into the same document, or to justified constants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
