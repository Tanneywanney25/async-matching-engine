"""The /analytics routes serve the published GPU analytics JSON or a 404 (no network, no lifespan)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from starlette.testclient import TestClient

import exchange_engine.engine as engine_mod

REPO = Path(__file__).resolve().parent.parent


def _serverless():
    spec = importlib.util.spec_from_file_location("api_index_under_test", REPO / "api" / "index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_engine_analytics_present_and_absent(tmp_path, monkeypatch) -> None:
    path = tmp_path / "gpu_lab.json"
    path.write_text(json.dumps({"ladder": {"rungs": []}, "statement": "offline"}), encoding="utf-8")
    monkeypatch.setattr(engine_mod, "ANALYTICS_PATH", path)
    client = TestClient(engine_mod.app)
    resp = client.get("/analytics")
    assert resp.status_code == 200 and resp.json()["statement"] == "offline"
    monkeypatch.setattr(engine_mod, "ANALYTICS_PATH", tmp_path / "missing.json")
    resp = client.get("/analytics")
    assert resp.status_code == 404 and resp.json() == {"detail": "analytics not published"}


def test_serverless_analytics_routes(tmp_path, monkeypatch) -> None:
    mod = _serverless()
    client = TestClient(mod.app)
    monkeypatch.setattr(mod, "ANALYTICS_PATH", tmp_path / "missing.json")
    assert client.get("/api/analytics").status_code == 404
    path = tmp_path / "gpu_lab.json"
    path.write_text(json.dumps({"engine": {"p99_us": 1.0}}), encoding="utf-8")
    monkeypatch.setattr(mod, "ANALYTICS_PATH", path)
    for route in ("/analytics", "/api/analytics"):
        resp = client.get(route)
        assert resp.status_code == 200 and resp.json()["engine"]["p99_us"] == 1.0
