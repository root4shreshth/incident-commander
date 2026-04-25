"""Focused tests for the three new scenario families.

Each scenario is exercised end-to-end:
  1. Setup injects the right anomaly + logs.
  2. The wrong fix doesn't heal.
  3. The right fix heals (check_resolved -> True).
  4. is_correct_op() agrees with the rubric.
  5. The auto-classifier (in the Real-Time path) recognizes the log signature.
"""

from __future__ import annotations

import pytest

from incident_commander_env.models import IncidentAction, ActionRecord
from incident_commander_env.server.environment import IncidentCommanderEnv
from incident_commander_env.server.scenarios import (
    CertExpiryScenario,
    DiskFullScenario,
    SCENARIO_REGISTRY,
    SlowQueryScenario,
)


# ---------------------------------------------------------------------------
# Disk-full scenario
# ---------------------------------------------------------------------------

class TestDiskFullScenario:
    def test_setup_injects_disk_full_anomaly(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="disk_full", seed=1)
        scenario = env._scenario
        cluster = env._cluster
        target = scenario.target_service
        svc = cluster.get_service(target)
        assert "disk_full" in svc._anomalies
        assert svc.health.value == "degraded"
        # Logs mention ENOSPC / disk
        joined = " ".join(svc.log_buffer).lower()
        assert "no space left" in joined or "disk usage" in joined

    def test_restart_heals(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="disk_full", seed=1)
        scenario = env._scenario
        target = scenario.target_service
        env.step(IncidentAction(action_type="read_logs", target_service=target))
        obs = env.step(IncidentAction(action_type="restart_service", target_service=target))
        assert env._cluster.get_service(target).health.value == "healthy"
        assert scenario.check_resolved(env._cluster) is True

    def test_rollback_does_not_heal(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="disk_full", seed=1)
        scenario = env._scenario
        target = scenario.target_service
        env.step(IncidentAction(
            action_type="rollback_deployment", target_service=target,
            parameters={"to_version": "v0.9"},
        ))
        # Disk still full
        assert "disk_full" in env._cluster.get_service(target)._anomalies

    def test_is_correct_op_only_target_restart(self):
        scenario = DiskFullScenario(seed=2, difficulty=0.5)
        target = scenario.target_service
        good = ActionRecord(step=1, action_type="restart_service", target_service=target, parameters={})
        wrong_target = ActionRecord(step=1, action_type="restart_service", target_service="api-gateway", parameters={})
        wrong_action = ActionRecord(step=1, action_type="rollback_deployment", target_service=target, parameters={})
        assert scenario.is_correct_op(good, None) is True
        assert scenario.is_correct_op(wrong_target, None) is False
        assert scenario.is_correct_op(wrong_action, None) is False


# ---------------------------------------------------------------------------
# Slow-query / lock-contention scenario
# ---------------------------------------------------------------------------

class TestSlowQueryScenario:
    def test_setup_marks_bad_version(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="slow_query", seed=1)
        scenario = env._scenario
        target = scenario.target_service
        svc = env._cluster.get_service(target)
        assert svc.config.version == scenario.bad_version
        assert "lock_contention" in svc._anomalies

    def test_restart_does_not_fully_fix(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="slow_query", seed=1)
        scenario = env._scenario
        target = scenario.target_service
        env.step(IncidentAction(action_type="restart_service", target_service=target))
        # Bad version still active → check_resolved must be False
        assert scenario.check_resolved(env._cluster) is False

    def test_rollback_heals(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="slow_query", seed=1)
        scenario = env._scenario
        target = scenario.target_service
        env.step(IncidentAction(
            action_type="rollback_deployment",
            target_service=target,
            parameters={"to_version": scenario.stable_version},
        ))
        assert scenario.check_resolved(env._cluster) is True

    def test_rollback_to_self_rejected_by_handler(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="slow_query", seed=1)
        scenario = env._scenario
        target = scenario.target_service
        bad = scenario.bad_version
        obs = env.step(IncidentAction(
            action_type="rollback_deployment",
            target_service=target,
            parameters={"to_version": bad},
        ))
        # The rollback handler reports errors via the `message` text (not the
        # typed `error` field). Either way the rollback must NOT have healed.
        assert "Error" in (obs.message or "") or obs.error is not None
        assert "lock_contention" in env._cluster.get_service(target)._anomalies

    def test_is_correct_op_requires_rollback_to_different_version(self):
        scenario = SlowQueryScenario(seed=2, difficulty=0.5)
        target = scenario.target_service
        good = ActionRecord(
            step=1, action_type="rollback_deployment", target_service=target,
            parameters={"to_version": scenario.stable_version},
        )
        rollback_to_self = ActionRecord(
            step=1, action_type="rollback_deployment", target_service=target,
            parameters={"to_version": scenario.bad_version},
        )
        restart = ActionRecord(
            step=1, action_type="restart_service", target_service=target, parameters={},
        )
        assert scenario.is_correct_op(good, None) is True
        assert scenario.is_correct_op(rollback_to_self, None) is False
        assert scenario.is_correct_op(restart, None) is False


# ---------------------------------------------------------------------------
# Cert-expiry scenario
# ---------------------------------------------------------------------------

class TestCertExpiryScenario:
    def test_setup_signals_cert_problem_in_logs(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="cert_expiry", seed=1)
        scenario = env._scenario
        svc = env._cluster.get_service(scenario.target_service)
        assert "cert_expired" in svc._anomalies
        joined = " ".join(svc.log_buffer).lower()
        assert "tls" in joined or "ssl" in joined or "certificate" in joined

    def test_metrics_look_almost_normal(self):
        """The whole point of cert_expiry is that metrics LOOK fine — just no traffic."""
        env = IncidentCommanderEnv()
        env.reset(task_id="cert_expiry", seed=1)
        scenario = env._scenario
        svc = env._cluster.get_service(scenario.target_service)
        # CPU low, memory low, but error_rate high
        assert svc.metrics.cpu_percent < 30
        assert svc.metrics.error_rate_percent > 80
        assert svc.metrics.requests_per_second < 5

    def test_restart_heals(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="cert_expiry", seed=1)
        scenario = env._scenario
        target = scenario.target_service
        env.step(IncidentAction(action_type="restart_service", target_service=target))
        assert scenario.check_resolved(env._cluster) is True

    def test_rollback_does_not_heal(self):
        env = IncidentCommanderEnv()
        env.reset(task_id="cert_expiry", seed=1)
        scenario = env._scenario
        target = scenario.target_service
        # Rollback shouldn't fix a cert problem
        env.step(IncidentAction(
            action_type="rollback_deployment", target_service=target,
            parameters={"to_version": "v0.9"},
        ))
        assert "cert_expired" in env._cluster.get_service(target)._anomalies


# ---------------------------------------------------------------------------
# Auto-classify heuristics (sanity check the log-pattern matchers)
# ---------------------------------------------------------------------------

class TestAutoClassifyHeuristics:
    """The Real-Time auto-classifier looks at log strings to infer the fault.
    These tests pin the log signatures the new scenarios produce against the
    matchers in app._classify_current_fault.
    """

    @pytest.mark.parametrize("task_id,signature", [
        ("disk_full", "no space left"),
        ("slow_query", "lock wait"),
        ("cert_expiry", "certificate has expired"),
    ])
    def test_log_signatures_present(self, task_id: str, signature: str):
        env = IncidentCommanderEnv()
        env.reset(task_id=task_id, seed=1)
        target = env._scenario.target_service
        joined = " ".join(env._cluster.get_service(target).log_buffer).lower()
        assert signature in joined, f"expected {signature!r} in {task_id} logs but found:\n{joined[:500]}"


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_six_built_in_families(self):
        # 6 built-in Python scenarios; YAML scenarios may add more.
        assert len(SCENARIO_REGISTRY) >= 6
        for fam in ("oom_crash", "db_pool_exhaustion", "bad_deployment_cascade",
                    "disk_full", "slow_query", "cert_expiry"):
            assert fam in SCENARIO_REGISTRY

    def test_each_scenario_has_root_cause_keywords(self):
        # Required for the resolution-reward component
        for fam, cls in SCENARIO_REGISTRY.items():
            instance = cls(seed=1, difficulty=0.5)
            assert hasattr(instance, "root_cause_keywords")
            assert len(instance.root_cause_keywords) >= 3, f"{fam} needs more keywords"
