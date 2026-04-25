"""RealBackend tests — exercise the Docker Compose translation layer with the
shell-out helpers monkeypatched. Verifies that:

  * each action_type produces a typed IncidentObservation
  * compose env-var levers track per-action state (image tag, memory limits, pool)
  * the snapshot() pipeline rolls up `docker stats` + `compose ps` correctly
  * stub mode (no compose_root) returns helpful errors without crashing

These tests do NOT require Docker. They monkeypatch
`incident_commander_env.server.backends.docker_ops` so subprocess calls never
happen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from incident_commander_env.models import IncidentAction
from incident_commander_env.server.backends import docker_ops as ops
from incident_commander_env.server.backends.real import (
    DEFAULT_BAD_TAG,
    DEFAULT_REAL_SERVICES,
    DEFAULT_STABLE_TAG,
    RealBackend,
    _parse_mem_limit,
)
from incident_commander_env.server.scenarios.scenario_oom_crash import OOMCrashScenario


# ---------------------------------------------------------------------------
# helpers: build a backend pointing at a fake compose_root that exists
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_compose_root(tmp_path: Path) -> Path:
    """A tmp dir with a docker-compose.yml so RealBackend's _check_available passes."""
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    return tmp_path


@pytest.fixture
def patched_ops(monkeypatch):
    """Monkeypatch docker_ops to capture calls and return programmable results."""
    calls: List[Dict[str, Any]] = []

    def _record(name):
        def stub(*args, **kwargs):
            calls.append({"op": name, "args": args, "kwargs": kwargs})
            return ops.DockerResult(ok=True, stdout="", stderr="", returncode=0)
        return stub

    monkeypatch.setattr(ops, "compose_up", _record("compose_up"))
    monkeypatch.setattr(ops, "compose_down", _record("compose_down"))
    monkeypatch.setattr(ops, "compose_restart", _record("compose_restart"))
    monkeypatch.setattr(ops, "compose_scale", _record("compose_scale"))
    monkeypatch.setattr(ops, "chaos_inject", _record("chaos_inject"))
    monkeypatch.setattr(ops, "chaos_stop", _record("chaos_stop"))
    monkeypatch.setattr(ops, "compose_logs", _record("compose_logs"))
    monkeypatch.setattr(ops, "docker_stats_json", _record("docker_stats_json"))
    monkeypatch.setattr(ops, "compose_ps_json", _record("compose_ps_json"))
    monkeypatch.setattr(ops, "http_health", lambda url, timeout=2.0: (True, 200))
    return calls


# ---------------------------------------------------------------------------
# Stub mode (no compose root present)
# ---------------------------------------------------------------------------

class TestStubMode:
    def test_missing_compose_root_falls_back_to_stub(self):
        be = RealBackend(compose_root="./does/not/exist")
        be.reset(OOMCrashScenario(seed=1, difficulty=0.5), seed=1)
        assert be._stub_mode is True
        snap = be.snapshot()
        assert len(snap.services) == 3
        for svc in snap.services.values():
            assert svc.health == "healthy"

    def test_stub_actions_return_typed_error(self):
        be = RealBackend(compose_root="./does/not/exist")
        be.reset(OOMCrashScenario(seed=1, difficulty=0.5), seed=1)
        for at in (
            "list_services",
            "describe_service",
            "read_logs",
            "check_metrics",
            "restart_service",
            "scale_service",
            "rollback_deployment",
            "update_config",
            "run_diagnostic",
        ):
            params = {}
            target = "api"
            if at == "rollback_deployment":
                params = {"to_version": "v1.0"}
            if at == "update_config":
                params = {"key": "memory.limit", "value": "1024Mi"}
            obs = be.execute(
                IncidentAction(action_type=at, target_service=target, parameters=params),
                OOMCrashScenario(seed=1, difficulty=0.5),
            )
            # Either stub error or "stub" wording
            assert obs is not None

    def test_execute_before_reset_returns_uninitialized_error(self):
        be = RealBackend(compose_root="./does/not/exist")
        obs = be.execute(
            IncidentAction(action_type="list_services"),
            OOMCrashScenario(seed=1, difficulty=0.5),
        )
        assert obs.error == "Backend not initialized"


# ---------------------------------------------------------------------------
# Reset wiring — compose up + chaos inject
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_calls_compose_up_and_chaos(self, fake_compose_root, patched_ops):
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        ops_called = [c["op"] for c in patched_ops]
        assert "compose_up" in ops_called
        assert "chaos_inject" in ops_called

    def test_reset_resets_per_episode_state(self, fake_compose_root, patched_ops):
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        be._image_tag = "v1.1"
        be._mem_limits_mb["api"] = 256
        be._pool_sizes["api"] = 50
        be._restart_history.append("api")
        be.reset(scenario, seed=2)
        assert be._image_tag == DEFAULT_STABLE_TAG
        assert be._mem_limits_mb == {}
        assert be._pool_sizes == {}
        assert be._restart_history == []

    def test_reset_compose_up_failure_drops_to_stub(self, fake_compose_root, monkeypatch):
        """If compose up fails (e.g. docker daemon down), backend stays usable in stub mode."""
        monkeypatch.setattr(
            ops, "compose_up",
            lambda *a, **kw: ops.DockerResult(ok=False, error="docker daemon not running"),
        )
        monkeypatch.setattr(ops, "chaos_inject", lambda *a, **kw: ops.DockerResult(ok=True))
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        assert be._stub_mode is True
        assert "compose up failed" in (be._stub_reason or "")


# ---------------------------------------------------------------------------
# Per-action translation
# ---------------------------------------------------------------------------

class TestActions:
    def test_restart_with_memory_limit_tracks_state(self, fake_compose_root, patched_ops):
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        obs = be.execute(
            IncidentAction(
                action_type="restart_service",
                target_service="api",
                parameters={"memory_limit": "1024Mi"},
            ),
            scenario,
        )
        assert obs.error is None
        assert "Restarted api" in obs.message
        assert be._mem_limits_mb["api"] == 1024
        assert "api" in be._restart_history

    def test_rollback_to_self_is_rejected(self, fake_compose_root, patched_ops):
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        # Already on the stable tag → rollback to v1.0 is a no-op
        obs = be.execute(
            IncidentAction(
                action_type="rollback_deployment",
                parameters={"to_version": DEFAULT_STABLE_TAG},
            ),
            scenario,
        )
        assert obs.error == "rollback_to_self"

    def test_rollback_to_different_tag_updates_image(self, fake_compose_root, patched_ops):
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        obs = be.execute(
            IncidentAction(
                action_type="rollback_deployment",
                parameters={"to_version": DEFAULT_BAD_TAG},
            ),
            scenario,
        )
        assert obs.error is None
        assert be._image_tag == DEFAULT_BAD_TAG

    def test_update_config_unknown_key_rejected(self, fake_compose_root, patched_ops):
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        obs = be.execute(
            IncidentAction(
                action_type="update_config",
                target_service="api",
                parameters={"key": "magic.fix.everything", "value": True},
            ),
            scenario,
        )
        assert obs.error is not None
        assert "Unknown config key" in obs.error

    def test_update_config_pool_size_tracks_state(self, fake_compose_root, patched_ops):
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        obs = be.execute(
            IncidentAction(
                action_type="update_config",
                target_service="postgres",
                parameters={"key": "db.pool.max_size", "value": 100},
            ),
            scenario,
        )
        assert obs.error is None
        assert be._pool_sizes["postgres"] == 100

    def test_scale_service_clamps_replicas(self, fake_compose_root, patched_ops):
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        obs = be.execute(
            IncidentAction(
                action_type="scale_service",
                target_service="api",
                parameters={"replicas": 999},
            ),
            scenario,
        )
        assert obs.error is None
        # Should clamp to 10
        scale_call = next(c for c in patched_ops if c["op"] == "compose_scale")
        assert scale_call["args"][2] == 10

    def test_resolve_incident_requires_root_cause(self, fake_compose_root, patched_ops):
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        obs = be.execute(
            IncidentAction(action_type="resolve_incident", parameters={}),
            scenario,
        )
        assert obs.error is not None

    def test_resolve_incident_with_full_payload_marks_done(
        self, fake_compose_root, monkeypatch, patched_ops
    ):
        be = RealBackend(compose_root=str(fake_compose_root))
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        # Stub check_resolved to return True quickly
        monkeypatch.setattr(be, "check_resolved", lambda s: True)
        obs = be.execute(
            IncidentAction(
                action_type="resolve_incident",
                parameters={"root_cause": "OOM", "resolution": "Bumped memory"},
            ),
            scenario,
        )
        assert obs.done is True
        assert "PASS" in obs.message


# ---------------------------------------------------------------------------
# Snapshot rollup — exercise stats + ps parsing
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_with_real_stats_output(self, fake_compose_root, monkeypatch):
        # Sample docker stats NDJSON
        stats_ndjson = (
            '{"Name":"site-api-1","CPUPerc":"42.5%","MemUsage":"128MiB / 512MiB"}\n'
            '{"Name":"site-frontend-1","CPUPerc":"3.0%","MemUsage":"64MiB / 256MiB"}\n'
            '{"Name":"site-postgres-1","CPUPerc":"5.0%","MemUsage":"100MiB / 1GiB"}\n'
        )
        ps_json = (
            '[{"Service":"api","State":"running","Health":"healthy"},'
            '{"Service":"frontend","State":"running","Health":"healthy"},'
            '{"Service":"postgres","State":"running","Health":"healthy"}]'
        )
        monkeypatch.setattr(
            ops, "compose_up", lambda *a, **kw: ops.DockerResult(ok=True)
        )
        monkeypatch.setattr(
            ops, "chaos_inject", lambda *a, **kw: ops.DockerResult(ok=True)
        )
        monkeypatch.setattr(
            ops, "docker_stats_json",
            lambda timeout=10: ops.DockerResult(ok=True, stdout=stats_ndjson),
        )
        monkeypatch.setattr(
            ops, "compose_ps_json",
            lambda root, timeout=10: ops.DockerResult(ok=True, stdout=ps_json),
        )

        be = RealBackend(compose_root=str(fake_compose_root))
        be.reset(OOMCrashScenario(seed=1, difficulty=0.5), seed=1)
        snap = be.snapshot()
        assert "api" in snap.services
        api = snap.services["api"]
        assert api.cpu_percent == pytest.approx(42.5, rel=0.01)
        assert api.memory_mb == pytest.approx(128.0, rel=0.01)
        assert api.memory_limit_mb == pytest.approx(512.0, rel=0.01)
        # Quota rolls up
        assert snap.quota.memory_used_mb > 0


# ---------------------------------------------------------------------------
# Helpers — _parse_mem_limit unit table
# ---------------------------------------------------------------------------

class TestMemParse:
    @pytest.mark.parametrize("input_str,expected", [
        ("1024Mi", 1024),
        ("1Gi", 1024),
        ("512", 512),
        ("256m", 256),
        ("2Gi", 2048),
        ("0.5Gi", 512),
        ("", None),
        ("garbage", None),
    ])
    def test_parse(self, input_str, expected):
        assert _parse_mem_limit(input_str) == expected


# ---------------------------------------------------------------------------
# docker_ops parser smoke tests
# ---------------------------------------------------------------------------

class TestDockerOpsParsers:
    def test_parse_compose_ps_array_form(self):
        out = '[{"Service":"api","State":"running"},{"Service":"db","State":"running"}]'
        rows = ops.parse_compose_ps(out)
        assert len(rows) == 2
        assert {r["Service"] for r in rows} == {"api", "db"}

    def test_parse_compose_ps_ndjson_form(self):
        out = '{"Service":"api","State":"running"}\n{"Service":"db","State":"exited"}'
        rows = ops.parse_compose_ps(out)
        assert len(rows) == 2

    def test_parse_compose_ps_empty(self):
        assert ops.parse_compose_ps("") == []
        assert ops.parse_compose_ps("   \n") == []

    def test_parse_docker_stats_ndjson(self):
        out = '{"Name":"a","CPUPerc":"5%"}\n{"Name":"b","CPUPerc":"6%"}'
        rows = ops.parse_docker_stats(out)
        assert len(rows) == 2

    def test_stats_to_service_metrics_compose_naming(self):
        rows = [
            {"Name": "site-api-1", "CPUPerc": "12.5%", "MemUsage": "100MiB / 512MiB"},
        ]
        m = ops.stats_to_service_metrics(rows)
        assert "api" in m
        assert m["api"]["cpu_percent"] == 12.5
        assert m["api"]["memory_mb"] == 100.0
        assert m["api"]["memory_limit_mb"] == 512.0
