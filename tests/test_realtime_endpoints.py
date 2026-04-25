"""Tests for the /realtime/* HTTP surface and the background-thread agent run.

Mocks the website-side `_http` so no real network calls are made. Covers:
  * /realtime/connect — validates and stores config
  * /realtime/inject — fires /ops/break on the connected site
  * /realtime/heal — fires /ops/heal
  * /realtime/run-agent — kicks off background work, status reports completion
  * /realtime/config — reflects the connected state
  * Path-traversal sanitization on /realtime/status/<run_id>
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

import incident_commander_env.server.app as app_module
import incident_commander_env.server.backends.website as website_module
from incident_commander_env.server.app import app


@pytest.fixture(autouse=True)
def reset_realtime_state(monkeypatch):
    """Each test starts with a clean in-memory store + no connected site."""
    app_module._REALTIME_RUNS.clear()
    app_module._REALTIME_CONFIG.update({
        "site_url": None,
        "repo_url": None,
        "repo_token": None,
        "service_names": ["frontend", "api", "postgres"],
    })
    yield
    app_module._REALTIME_RUNS.clear()


@pytest.fixture
def fake_http(monkeypatch):
    calls: List[Dict[str, Any]] = []
    responses: Dict[str, Any] = {}

    def _fake(method: str, url: str, json_body=None, timeout=8.0):
        calls.append({"method": method, "url": url, "body": json_body})
        for key, val in responses.items():
            if key in url:
                return val
        return website_module.HttpResult(ok=True, status=200, body={"status": "ok"})

    # Patch in BOTH places — the website backend uses one reference, the
    # app module imports the same function for the realtime endpoints.
    monkeypatch.setattr(website_module, "_http", _fake)
    monkeypatch.setattr(app_module, "_site_http", _fake)
    _fake.calls = calls
    _fake.responses = responses
    return _fake


# ---------------------------------------------------------------------------
# /realtime/connect
# ---------------------------------------------------------------------------

class TestConnect:
    def test_requires_site_url(self, fake_http):
        c = TestClient(app)
        r = c.post("/realtime/connect", json={"site_url": ""})
        assert r.status_code == 200
        body = r.json()
        assert body["connected"] is False
        assert body["error"]

    def test_health_failure_reports_error(self, fake_http):
        fake_http.responses["/ops/health"] = website_module.HttpResult(
            ok=False, status=502, error="bad gateway"
        )
        c = TestClient(app)
        r = c.post("/realtime/connect", json={"site_url": "https://demo.example.com"})
        body = r.json()
        assert body["connected"] is False
        assert "GET /ops/health failed" in body["error"]

    def test_successful_connect_stores_config(self, fake_http):
        fake_http.responses["/ops/health"] = website_module.HttpResult(
            ok=True, status=200, body={
                "status": "ok",
                "services": [
                    {"name": "frontend", "health": "healthy"},
                    {"name": "api", "health": "healthy"},
                    {"name": "postgres", "health": "healthy"},
                ],
            },
        )
        c = TestClient(app)
        r = c.post("/realtime/connect", json={
            "site_url": "https://demo.example.com",
            "repo_url": "https://github.com/foo/bar",
        })
        body = r.json()
        assert body["connected"] is True
        assert body["site_url"] == "https://demo.example.com"
        assert body["status"] == "ok"
        assert body["services_discovered"] == ["frontend", "api", "postgres"]
        assert body["repo_linked"] is True
        # Config endpoint reflects state
        cfg = c.get("/realtime/config").json()
        assert cfg["site_url"] == "https://demo.example.com"
        assert cfg["repo_linked"] is True


# ---------------------------------------------------------------------------
# /realtime/inject + /realtime/heal
# ---------------------------------------------------------------------------

class TestInject:
    def test_inject_without_connect_errors(self, fake_http):
        c = TestClient(app)
        r = c.post("/realtime/inject", json={"scenario": "oom_crash"})
        body = r.json()
        assert body["injected"] is False
        assert "no site connected" in body["error"]

    def test_inject_unknown_scenario_errors(self, fake_http):
        # Connect first
        fake_http.responses["/ops/health"] = website_module.HttpResult(
            ok=True, status=200, body={"status": "ok", "services": []})
        c = TestClient(app)
        c.post("/realtime/connect", json={"site_url": "https://demo.example.com"})
        r = c.post("/realtime/inject", json={"scenario": "made_up"})
        body = r.json()
        assert body["injected"] is False
        assert "unknown scenario" in body["error"]

    def test_inject_calls_break_endpoint(self, fake_http):
        fake_http.responses["/ops/health"] = website_module.HttpResult(
            ok=True, status=200, body={"status": "ok", "services": []})
        c = TestClient(app)
        c.post("/realtime/connect", json={"site_url": "https://demo.example.com"})
        fake_http.calls.clear()
        r = c.post("/realtime/inject", json={"scenario": "oom_crash"})
        assert r.json()["injected"] is True
        # The fake_http records the break call
        break_calls = [x for x in fake_http.calls if "/ops/break" in x["url"]]
        assert break_calls
        assert break_calls[0]["body"]["scenario"] == "oom_crash"

    def test_heal_calls_heal_endpoint(self, fake_http):
        fake_http.responses["/ops/health"] = website_module.HttpResult(
            ok=True, status=200, body={"status": "ok", "services": []})
        c = TestClient(app)
        c.post("/realtime/connect", json={"site_url": "https://demo.example.com"})
        fake_http.calls.clear()
        r = c.post("/realtime/heal")
        assert r.json()["healed"] is True
        heal_calls = [x for x in fake_http.calls if "/ops/heal" in x["url"]]
        assert heal_calls


# ---------------------------------------------------------------------------
# /realtime/run-agent
# ---------------------------------------------------------------------------

class TestRunAgent:
    def test_run_without_connect_errors(self, fake_http):
        c = TestClient(app)
        r = c.post("/realtime/run-agent", json={"scenario": "oom_crash"})
        body = r.json()
        assert body["run_id"] is None
        assert "no site connected" in body["error"]

    def test_run_unknown_scenario_errors(self, fake_http):
        fake_http.responses["/ops/health"] = website_module.HttpResult(
            ok=True, status=200, body={"status": "ok", "services": []})
        c = TestClient(app)
        c.post("/realtime/connect", json={"site_url": "https://demo.example.com"})
        r = c.post("/realtime/run-agent", json={"scenario": "fakey"})
        body = r.json()
        assert body["run_id"] is None
        assert "unknown scenario" in body["error"]

    def test_run_creates_record_and_runs_to_completion(self, fake_http):
        # Site reports "ok" on /ops/health throughout, so tier1 should "succeed".
        fake_http.responses["/ops/health"] = website_module.HttpResult(
            ok=True, status=200, body={"status": "ok", "services": []})
        c = TestClient(app)
        c.post("/realtime/connect", json={"site_url": "https://demo.example.com"})
        r = c.post("/realtime/run-agent", json={"scenario": "oom_crash", "enable_tier2": False})
        run_id = r.json()["run_id"]
        assert run_id

        # Poll until it completes (cap iterations to avoid infinite loops in CI)
        for _ in range(60):
            status = c.get(f"/realtime/status/{run_id}").json()
            if status.get("status") and status["status"] != "running":
                break
            time.sleep(0.1)
        assert status["status"] in {"resolved", "unresolved_no_tier2"}
        assert status["events"]
        # Should have a 'start' event and at least one 'step' event
        types = {e.get("type") for e in status["events"]}
        assert "start" in types
        assert "step" in types

    def test_status_path_traversal_sanitized(self, fake_http):
        c = TestClient(app)
        r = c.get("/realtime/status/..secret")
        # The route accepts the path-style ID and returns "unknown run_id"
        assert r.status_code == 200
        body = r.json()
        assert "error" in body


# ---------------------------------------------------------------------------
# Tier 1 unhealed → tier 2 escalation triggers code investigator
# ---------------------------------------------------------------------------

class TestTier2Escalation:
    def test_when_health_stays_degraded_and_no_repo_status_is_no_tier2(self, fake_http, monkeypatch):
        # /ops/health returns "degraded" so tier1 won't claim resolution.
        fake_http.responses["/ops/health"] = website_module.HttpResult(
            ok=True, status=200, body={"status": "degraded", "services": []})
        # Speed up backend.check_resolved by patching it to immediately return False
        from incident_commander_env.server.backends.website import WebsiteBackend
        monkeypatch.setattr(WebsiteBackend, "check_resolved", lambda self, scenario: False)
        c = TestClient(app)
        c.post("/realtime/connect", json={"site_url": "https://demo.example.com"})
        r = c.post("/realtime/run-agent", json={"scenario": "oom_crash", "enable_tier2": False})
        run_id = r.json()["run_id"]
        for _ in range(80):
            status = c.get(f"/realtime/status/{run_id}").json()
            if status.get("status") and status["status"] != "running":
                break
            time.sleep(0.1)
        assert status["status"] == "unresolved_no_tier2"

    def test_tier2_escalation_runs_when_repo_provided(self, fake_http, monkeypatch):
        fake_http.responses["/ops/health"] = website_module.HttpResult(
            ok=True, status=200, body={"status": "degraded", "services": []})
        from incident_commander_env.server.backends.website import WebsiteBackend
        monkeypatch.setattr(WebsiteBackend, "check_resolved", lambda self, scenario: False)
        # Stub the investigate function so we don't hit network/git
        from training import code_investigator
        def fake_investigate(repo_url, scenario, target_service=None,
                             repo_token=None, llm_call=None, cloned_root=None):
            return code_investigator.CodeEscalationReport(
                repo_url=repo_url, scenario=scenario, target_service=target_service,
                summary="stub summary", suggested_fix="stub fix", findings=[],
            )
        monkeypatch.setattr(code_investigator, "investigate", fake_investigate)
        c = TestClient(app)
        c.post("/realtime/connect", json={
            "site_url": "https://demo.example.com",
            "repo_url": "https://github.com/foo/bar",
        })
        r = c.post("/realtime/run-agent", json={"scenario": "oom_crash", "enable_tier2": True})
        run_id = r.json()["run_id"]
        for _ in range(80):
            status = c.get(f"/realtime/status/{run_id}").json()
            if status.get("status") and status["status"] != "running":
                break
            time.sleep(0.1)
        assert status["status"] == "tier2_complete"
        assert status["tier2_report"]
        assert status["tier2_report"]["summary"] == "stub summary"
