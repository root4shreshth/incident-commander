"""Backend Protocol tests.

These pin the contract every backend implementation must honor:
  - reset() prepares a fresh episode
  - execute() returns a typed IncidentObservation per action
  - snapshot() returns a typed BackendSnapshot with the expected shape
  - check_resolved() answers correctly given current state
  - tick() advances simulation
  - teardown() releases resources

The SimulatedBackend is the canonical implementation; these tests assert
its behavior. Future backends (RealBackend, CodeAwareBackend) will be
covered by their own integration tests in Phase 6 / post-hackathon.
"""

from __future__ import annotations

import pytest

from incident_commander_env.models import IncidentAction
from incident_commander_env.server.backends import (
    BackendSnapshot,
    QuotaSnapshot,
    RealBackend,
    ServiceSnapshot,
    SimulatedBackend,
    get_backend,
)
from incident_commander_env.server.environment import IncidentCommanderEnv
from incident_commander_env.server.scenarios.scenario_oom_crash import OOMCrashScenario


# ---------------------------------------------------------------------------
# Backend Protocol contract — every backend must support this surface
# ---------------------------------------------------------------------------

class TestSimulatedBackendContract:
    def test_initial_state_pre_reset(self):
        be = SimulatedBackend()
        assert be.name == "sim"
        # Snapshot before reset is empty but valid
        snap = be.snapshot()
        assert isinstance(snap, BackendSnapshot)
        assert snap.services == {}
        assert isinstance(snap.quota, QuotaSnapshot)

    def test_reset_then_snapshot_has_all_services(self):
        be = SimulatedBackend()
        scenario = OOMCrashScenario(seed=42, difficulty=0.5)
        be.reset(scenario, seed=42)
        snap = be.snapshot()
        assert isinstance(snap, BackendSnapshot)
        # 9 default services in the simulated cluster
        assert len(snap.services) == 9
        # Every service is a typed ServiceSnapshot
        for name, svc in snap.services.items():
            assert isinstance(svc, ServiceSnapshot)
            assert svc.name == name
            assert svc.health in {"healthy", "degraded", "unhealthy", "crashed", "restarting"}

    def test_target_service_is_crashed_after_reset(self):
        be = SimulatedBackend()
        scenario = OOMCrashScenario(seed=42, difficulty=0.5)
        be.reset(scenario, seed=42)
        snap = be.snapshot()
        target = scenario.target_service
        assert snap.services[target].health == "crashed"

    def test_execute_routes_through_action_handlers(self):
        be = SimulatedBackend()
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        action = IncidentAction(action_type="list_services")
        obs = be.execute(action, scenario)
        assert obs.error is None
        assert obs.services_summary is not None
        assert len(obs.services_summary) == 9

    def test_execute_with_unknown_action_returns_typed_error(self):
        be = SimulatedBackend()
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        # Bypass IncidentAction validation by constructing an invalid action manually
        # (we want to exercise the handler-not-found branch)
        class Fake:
            action_type = "nonexistent_action"
            target_service = None
            parameters = {}
        obs = be.execute(Fake(), scenario)
        assert obs.error is not None
        assert "Invalid" in obs.error

    def test_check_resolved_pre_fix(self):
        be = SimulatedBackend()
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        assert be.check_resolved(scenario) is False

    def test_check_resolved_after_correct_fix(self):
        be = SimulatedBackend()
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        # Apply the correct fix: restart with higher memory
        action = IncidentAction(
            action_type="restart_service",
            target_service=scenario.target_service,
            parameters={"memory_limit": "1024Mi"},
        )
        be.execute(action, scenario)
        assert be.check_resolved(scenario) is True

    def test_tick_advances_simulation(self):
        be = SimulatedBackend()
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        cluster = be.cluster
        before = cluster._tick_count
        be.tick()
        assert cluster._tick_count == before + 1

    def test_teardown_releases_cluster(self):
        be = SimulatedBackend()
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        assert be.cluster is not None
        be.teardown()
        assert be.cluster is None


# ---------------------------------------------------------------------------
# RealBackend stub: must boot under BACKEND=real even without Docker
# ---------------------------------------------------------------------------

class TestRealBackendStub:
    def test_boot_without_compose_root(self):
        # Tolerates a missing compose root so the env can load in environments
        # without Docker (pytest CI, HF Space booting before user vibecoded site lands)
        be = RealBackend(compose_root="./this/path/does/not/exist")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)  # must not raise
        snap = be.snapshot()
        assert isinstance(snap, BackendSnapshot)
        # Stub reports 3 default services as healthy
        assert len(snap.services) == 3
        for name, svc in snap.services.items():
            assert svc.health == "healthy"

    def test_execute_returns_typed_not_implemented(self):
        be = RealBackend(compose_root="./this/path/does/not/exist")
        scenario = OOMCrashScenario(seed=1, difficulty=0.5)
        be.reset(scenario, seed=1)
        action = IncidentAction(action_type="list_services")
        obs = be.execute(action, scenario)
        # Phase 2 stub: returns observation with error, doesn't crash
        assert obs.error is not None


# ---------------------------------------------------------------------------
# get_backend() dispatch
# ---------------------------------------------------------------------------

class TestGetBackend:
    def test_default_returns_sim(self):
        be = get_backend()
        assert isinstance(be, SimulatedBackend)
        assert be.name == "sim"

    def test_explicit_sim(self):
        be = get_backend("sim")
        assert isinstance(be, SimulatedBackend)

    def test_explicit_real(self):
        be = get_backend("real")
        assert isinstance(be, RealBackend)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_backend("nonexistent")

    def test_code_aware_raises(self):
        # Reserved for post-hackathon roadmap
        with pytest.raises(ValueError):
            get_backend("code_aware")


# ---------------------------------------------------------------------------
# IncidentCommanderEnv works with explicit backend injection
# ---------------------------------------------------------------------------

class TestEnvWithBackend:
    def test_default_uses_simulated_backend(self):
        env = IncidentCommanderEnv()
        assert isinstance(env.backend, SimulatedBackend)

    def test_inject_custom_backend(self):
        be = SimulatedBackend()
        env = IncidentCommanderEnv(backend=be)
        assert env.backend is be

    def test_full_episode_through_backend(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="oom_crash", seed=42)
        obs = env.step(IncidentAction(
            action_type="restart_service",
            target_service=env._scenario.target_service,
            parameters={"memory_limit": "512Mi"},
        ))
        # Episode resolves on the correct fix
        assert obs.done is True
        assert env.state.incident_resolved is True
        assert 0.0 <= env.state.current_score <= 1.0
