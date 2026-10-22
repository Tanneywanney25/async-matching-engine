"""No rendered document may contain unrendered template syntax.

0.2.0 shipped INTERVIEW_NOTES.md question 2 with a raw ``{r.fmt('ladder', ...)}`` expression in it:
the answer was an f-string implicitly concatenated with a plain string, so the second half was
written out instead of evaluated. These tests check the committed documents, a fresh render of the
committed results, and the renderer's own source for the same class of slip.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

import render_docs

REPO = Path(__file__).resolve().parent.parent
RENDERED = ("BENCHMARKS.md", "DESIGN.md", "INTERVIEW_NOTES.md", "BULLETS.md", "README.md", "CHANGELOG.md")
SHIPPED = ("251.1 GB/s of minimum traffic and {r.fmt('ladder', 'rungs.' + str(best_index(r)) + "
           "'.pct_of_theoretical', 'pct')}% of the theoretical figure.")
CALL_IN_PLAIN_STRING = re.compile(r"\{\w+(?:\.\w+)*\(")
MISSING = re.compile(r"\bn/a\b")


def _prose(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    return re.sub(r"`[^`\n]*`", " ", text)


def test_detector_catches_the_shipped_leftover() -> None:
    assert render_docs.template_leftovers(SHIPPED)
    assert render_docs.template_leftovers("{A} times faster than single-threaded C++")
    assert render_docs.template_leftovers("costs {{ cost }} ms")
    clean = 'Inline `{"a": 1}` is code.\n\n```json\n{"b": 2}\n```\n\n78.5% of the theoretical figure.'
    assert render_docs.template_leftovers(clean) == []


@pytest.mark.parametrize("name", RENDERED)
def test_committed_documents_are_fully_rendered(name: str) -> None:
    text = (REPO / name).read_text(encoding="utf-8")
    assert render_docs.template_leftovers(text) == []
    assert not MISSING.search(_prose(text)), f"{name} shows n/a where a measured value is missing"


def test_fresh_render_of_committed_results_is_fully_rendered(tmp_path: Path) -> None:
    assert render_docs.main(["--results", str(REPO / "results"), "--out-dir", str(tmp_path)]) == 0
    for name in RENDERED:
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert render_docs.template_leftovers(text) == [], name
        assert not MISSING.search(_prose(text)), name


def test_renderer_refuses_to_write_a_leftover(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(render_docs, "design_md", lambda r: "# Design decisions\n\n" + SHIPPED + "\n")
    assert render_docs.main(["--results", str(REPO / "results"), "--out-dir", str(tmp_path)]) == 4
    assert not (tmp_path / "DESIGN.md").exists()


def test_renderer_source_has_no_template_in_a_plain_string() -> None:
    src = (REPO / "scripts" / "render_docs.py").read_text(encoding="utf-8")
    offenders = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.STRING:  # f-strings are FSTRING_* tokens on 3.12+ and f-prefixed STRINGs before
            continue
        prefix = re.match(r"[A-Za-z]*", tok.string).group(0).lower()
        if "f" not in prefix and CALL_IN_PLAIN_STRING.search(tok.string):
            offenders.append((tok.start[0], tok.string[:80]))
    assert offenders == []
