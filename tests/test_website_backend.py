"""WebsiteBackend tests — exercise HTTP-based action translation with the
underlying `_http` shell-out monkeypatched. Verifies:

  * each action_type produces a typed IncidentObservation
  * stub mode returns clear errors when site_url is missing
  * snapshot() composes /ops/health + /ops/metrics responses correctly
  * reset() calls /ops/heal + /ops/break
  * config update + memory limit are tracked in shadow state
  * rollback-to-self is rejected
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from incident_commander_env.models import IncidentAction
from incident_commander_env.server.backends import (
    WebsiteBackend,
    get_backend,
)
from incident_commander_env.server.backends import website as website_module
from incident_commander_env.server.backends.website import (
    DEFAULT_SERVICE_NAMES,
    _parse_mem,
)
from incident_commander_env.server.scenarios.scenario_oom_crash import OOMCrashScenario


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeHTTP:
    """Records calls and returns programmable responses keyed by (method, path)."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.responses: Dict[str, website_module.HttpResult] = {}
        self.default = website_module.HttpResult(ok=True, status=200, body={})

    def __call__(self, method: str, url: str, json_body=None, timeout=8.0):
        self.calls.append({"method": method, "url": url, "body": json_body})
        # Match by suffix path (so we don't care about the host)
        for key, val in self.responses.items():
            if key in url and (key in url):
                return val
        return self.default

    def set(self, path_substring: str, result: website_module.HttpResult) -> None:
        self.responses[path_substring] = result


@pytest.fixture
def fake_http(monkeypatch):
    fh = FakeHTTP()
    monkeypatch.setattr(website_module, "_http", fh)
    return fh


# ---------------------------------------------------------------------------
# Stub mode (no site_url)
# ---------------------------------------------------------------------------

class TestStubMode:
    def test_no_site_url_falls_back_to_stub(self):
        be = WebsiteBackend(site_url=None)
        be.reset(OOMCrashScenario(seed=1, difficulty=0.5))
        assert be._stub_mode is True
        snap = be.snapshot()
        # 3 default services healthy
        assert len(snap.services) == 3
        for svc in snap.services.values():
            assert svc.health == "healthy"

    def test_execute_before_reset_returns_error(self):
        be = WebsiteBackend(site_url=None)
        obs = be.execute(
            IncidentAction(action_type="list_services"),
            OOMCrashScenario(seed=1, difficulty=0.5),
        )
        assert obs.error == "Backend not initialized"


# ---------------------------------------------------------------------------
# Reset wiring — /ops/heal + /ops/break
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_calls_heal_and_break(self, fake_http):
        be = WebsiteBackend(site_url="https://demo.example.com")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario)
        paths = [c["url"] for c in fake_http.calls]
        assert any("/ops/heal" in p for p in paths)
        assert any("/ops/break" in p for p in paths)

    def test_reset_resets_per_episode_state(self, fake_http):
        be = WebsiteBackend(site_url="https://demo.example.com")
        be._mem_limits_mb["api"] = 256
        be._image_tag = "v1.1"
        be._restart_history.append("api")
        be.reset(OOMCrashScenario(seed=1, difficulty=0.5))
        assert be._mem_limits_mb == {}
        assert be._image_tag is None
        assert be._restart_history == []

    def test_break_failure_drops_to_stub(self, fake_http):
        fake_http.set("/ops/break",
                      website_module.HttpResult(ok=False, status=500,
                                                error="server error"))
        be = WebsiteBackend(site_url="https://demo.example.com")
        be.reset(OOMCrashScenario(seed=1, difficulty=0.5))
        assert be._stub_mode is True
        assert "break call failed" in (be._stub_reason or "")


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

class TestActions:
    def test_restart_with_memory_tracks_state(self, fake_http):
        be = WebsiteBackend(site_url="https://demo.example.com")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario)
        obs = be.execute(
            IncidentAction(action_type="restart_service", target_service="api",
                           parameters={"memory_limit": "1024Mi"}),
            scenario,
        )
        assert obs.error is None
        assert "Restarted api" in obs.message
        assert be._mem_limits_mb["api"] == 1024
        assert "api" in be._restart_history
        # Must have hit /ops/restart with memory_limit_mb in body
        restart_calls = [c for c in fake_http.calls if "/ops/restart" in c["url"]]
        assert restart_calls
        assert restart_calls[-1]["body"]["memory_limit_mb"] == 1024

    def test_rollback_to_self_rejected(self, fake_http):
        be = WebsiteBackend(site_url="https://demo.example.com")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario)
        be._image_tag = "v1.0"
        obs = be.execute(
            IncidentAction(action_type="rollback_deployment",
                           parameters={"to_version": "v1.0"}),
            scenario,
        )
        assert obs.error == "rollback_to_self"

    def test_rollback_to_different_tag_updates_state(self, fake_http):
        be = WebsiteBackend(site_url="https://demo.example.com")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario)
        obs = be.execute(
            IncidentAction(action_type="rollback_deployment",
                           parameters={"to_version": "v0.9"}),
            scenario,
        )
        assert obs.error is None
        assert be._image_tag == "v0.9"

    def test_unknown_config_key_rejected(self, fake_http):
        be = WebsiteBackend(site_url="https://demo.example.com")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario)
        obs = be.execute(
            IncidentAction(action_type="update_config", target_service="api",
                           parameters={"key": "magic.fix.everything", "value": True}),
            scenario,
        )
        assert obs.error is not None
        assert "Unknown config key" in obs.error

    def test_pool_size_config_tracked(self, fake_http):
        be = WebsiteBackend(site_url="https://demo.example.com")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario)
        obs = be.execute(
            IncidentAction(action_type="update_config", target_service="postgres",
                           parameters={"key": "db.pool.max_size", "value": 100}),
            scenario,
        )
        assert obs.error is None
        assert be._pool_sizes["postgres"] == 100

    def test_scale_clamps_replicas(self, fake_http):
        be = WebsiteBackend(site_url="https://demo.example.com")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario)
        be.execute(
            IncidentAction(action_type="scale_service", target_service="api",
                           parameters={"replicas": 999}),
            scenario,
        )
        scale_calls = [c for c in fake_http.calls if "/ops/scale" in c["url"]]
        assert scale_calls[-1]["body"]["replicas"] == 10  # clamped

    def test_resolve_incident_requires_root_cause(self, fake_http):
        be = WebsiteBackend(site_url="https://demo.example.com")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario)
        obs = be.execute(
            IncidentAction(action_type="resolve_incident", parameters={}),
            scenario,
        )
        assert obs.error is not None

    def test_read_logs_returns_log_lines(self, fake_http):
        fake_http.set("/ops/logs",
                      website_module.HttpResult(ok=True, status=200,
                                                body={"logs": ["line1", "line2", "ERROR foo"]}))
        be = WebsiteBackend(site_url="https://demo.example.com")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario)
        obs = be.execute(
            IncidentAction(action_type="read_logs", target_service="api",
                           parameters={"lines": 50}),
            scenario,
        )
        assert obs.error is None
        assert obs.logs == ["line1", "line2", "ERROR foo"]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_aggregates_metrics(self, fake_http):
        # /ops/health says api is degraded; metrics for api shows high cpu+mem
        fake_http.set(
            "/ops/health",
            website_module.HttpResult(
                ok=True, status=200,
                body={
                    "status": "degraded",
                    "services": [
                        {"name": "frontend", "health": "healthy"},
                        {"name": "api", "health": "degraded"},
                        {"name": "postgres", "health": "healthy"},
                    ],
                },
            ),
        )
        # Per-service metrics
        def metrics_for(svc):
            return website_module.HttpResult(ok=True, status=200, body={
                "cpu_percent": 80 if svc == "api" else 5,
                "memory_mb": 250 if svc == "api" else 50,
                "memory_limit_mb": 256,
                "error_rate_percent": 12.0 if svc == "api" else 0.0,
                "request_latency_p99_ms": 0,
                "active_connections": 0,
                "requests_per_second": 0,
            })
        # Hack: use a resolver that picks based on the URL's service= query
        def resolve(method, url, json_body=None, timeout=8.0):
            fake_http.calls.append({"method": method, "url": url, "body": json_body})
            if "/ops/health" in url:
                return fake_http.responses["/ops/health"]
            if "/ops/metrics" in url:
                if "service=frontend" in url: return metrics_for("frontend")
                if "service=api" in url: return metrics_for("api")
                if "service=postgres" in url: return metrics_for("postgres")
            return fake_http.default

        # Replace the bound _http with our smarter resolver
        import incident_commander_env.server.backends.website as wb
        wb._http = resolve
        try:
            be = WebsiteBackend(site_url="https://demo.example.com")
            scenario = OOMCrashScenario(seed=1, difficulty=0.5)
            be.reset(scenario)
            snap = be.snapshot()
            assert snap.get_service("api").health == "degraded"
            assert snap.get_service("api").cpu_percent == pytest.approx(80, rel=0.01)
            assert snap.get_service("api").memory_mb == pytest.approx(250, rel=0.01)
            assert snap.get_service("frontend").health == "healthy"
        finally:
            wb._http = fake_http  # restore


# ---------------------------------------------------------------------------
# get_backend dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_get_backend_website(self):
        be = get_backend("website")
        assert isinstance(be, WebsiteBackend)
        assert be.name == "website"


# ---------------------------------------------------------------------------
# _parse_mem helper
# ---------------------------------------------------------------------------

class TestParseMem:
    @pytest.mark.parametrize("inp,expected", [
        ("1024Mi", 1024),
        ("1Gi", 1024),
        ("512", 512),
        ("256m", 256),
        ("2Gi", 2048),
        ("0.5Gi", 512),
        ("", None),
        ("garbage", None),
    ])
    def test_parse(self, inp, expected):
        assert _parse_mem(inp) == expected
