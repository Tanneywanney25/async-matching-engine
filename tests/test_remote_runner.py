"""Kaggle runner guards: pushed SHA, phase 2 budget, per-job limit, resumable waits (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from remote import base, kaggle
from remote.base import JobSpec


def _fake_git(remote_sha: str, status: str = "", branch: str = "gpu-phase2"):
    def git(*args: str) -> str:
        if args[0] == "ls-remote":
            return f"{remote_sha}\trefs/heads/{branch}" if remote_sha else ""
        if args[0] == "status":
            return status
        raise AssertionError(args)
    return git


def test_pushed_sha_problem(monkeypatch) -> None:
    sha = "a" * 40
    monkeypatch.setattr(base, "git", _fake_git(sha))
    assert base.pushed_sha_problem("gpu-phase2", sha) is None
    assert "push first" in base.pushed_sha_problem("gpu-phase2", "b" * 40)
    assert "detached" in base.pushed_sha_problem("HEAD", sha)
    monkeypatch.setattr(base, "git", _fake_git(""))
    assert "not on origin" in base.pushed_sha_problem("gpu-phase2", sha)
    monkeypatch.setattr(base, "git", _fake_git(sha, " M bench/bench_scans.py\n M results/provenance.json\n"))
    assert "bench/bench_scans.py" in base.pushed_sha_problem("gpu-phase2", sha)
    monkeypatch.setattr(base, "git", _fake_git(sha, " M remote/quota.json\n M results/provenance.json\n"))
    assert base.pushed_sha_problem("gpu-phase2", sha) is None  # files a job never reads


@pytest.fixture()
def ledger(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "quota.json"
    monkeypatch.setattr(kaggle, "LEDGER", path)
    return path


def _write(path: Path, runs: list) -> None:
    path.write_text(json.dumps({"budget_gpu_hours": 20.0, "runs": runs, "phase2": dict(kaggle.PHASE2)}), encoding="utf-8")


def test_budget_guard_counts_phase2_runs_only(ledger: Path) -> None:
    _write(ledger, [{"run_id": "old", "pushed_at": "2026-10-21T19:00:00Z", "gpu_minutes": 300.0},
                    {"run_id": "new", "pushed_at": "2026-10-22T01:00:00Z", "gpu_minutes": 330.0}])
    kaggle.budget_guard(25)  # 5.5 h + 25 min fits under 6 h even though the phase 1 run is large
    with pytest.raises(SystemExit, match="phase 2 GPU budget"):
        kaggle.budget_guard(45)
    with pytest.raises(SystemExit, match="45-minute limit"):
        kaggle.budget_guard(50)


def test_compose_sets_job_environment_before_the_body() -> None:
    spec = JobSpec(name="bench_quick", body="print(os.environ['SECTIONS'])", branch="gpu-phase2",
                   env={"SECTIONS": "resources"})
    text = kaggle.compose(spec, "bench_quick-x", "c" * 40)
    assert "BRANCH = 'gpu-phase2'" in text
    assert text.index("os.environ['SECTIONS'] = 'resources'") < text.index("print(os.environ['SECTIONS'])")


def test_wait_counts_minutes_from_the_first_running_sighting(ledger: Path, monkeypatch) -> None:
    _write(ledger, [{"run_id": "r1", "pushed_at": "2026-10-22T01:00:00Z", "running_at": "2026-10-22T01:02:00Z"}])
    monkeypatch.setattr(kaggle, "status", lambda job: "complete")
    monkeypatch.setattr(kaggle.time, "time", lambda: kaggle._unix("2026-10-22T01:32:00Z"))
    assert kaggle.wait("bench_quick", "r1", poll_seconds=0, max_minutes=1) == "complete"
    run = json.loads(ledger.read_text(encoding="utf-8"))["runs"][0]
    assert run["gpu_minutes"] == pytest.approx(30.0)
    assert json.loads(ledger.read_text(encoding="utf-8"))["phase2"]["gpu_hours_used_estimate"] == pytest.approx(0.5)


def test_push_refuses_an_unpushed_head(ledger: Path, monkeypatch) -> None:
    _write(ledger, [])
    monkeypatch.setattr(kaggle, "current_branch", lambda: "gpu-phase2")
    monkeypatch.setattr(kaggle, "pushed_sha_problem", lambda b, s: "origin/gpu-phase2 is behind: push first")
    monkeypatch.setattr(kaggle, "kaggle", lambda *a, **k: pytest.fail("must not reach kaggle"))
    with pytest.raises(SystemExit, match="push first"):
        kaggle.push(JobSpec(name="bench_quick", body="", git_sha="d" * 40))


@pytest.mark.parametrize("job", ["bench_all", "bench_quick", "bench_rerun", "probe", "nsys_probe"])
def test_every_job_composes_into_valid_python(job: str) -> None:
    import ast

    spec = JobSpec(name=job, body=kaggle.load_job(job), branch="gpu-phase2", env={"SECTIONS": "resources"})
    ast.parse(kaggle.compose(spec, f"{job}-test", "e" * 40))
