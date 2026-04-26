"""Regression tests for the four documented reward-hack exploits.

Each test asserts that an exploit path which previously yielded reward without
solving the underlying problem now correctly fails to score. These tests exist
because the OpenEnv hackathon judging guide explicitly calls out reward-hacking
as a top failure mode and asks for "anti-cheating checks" - this file is the
receipts.

Exploits covered:
    a) update_config string-match heal - any key containing "pool" + "size"
       used to silently clear db_pool_exhaustion. Now requires the strict
       allowlist key `db.pool.max_size` AND a value > 50, delegated to
       scenario.on_config_update.
    b) restart() unconditionally cleared all anomalies. Now restart only cures
       anomalies in `Service._RESTART_CURABLE` (oom, connection_leak), and OOM
       only if the memory limit was actually raised.
    c) Redundancy bypass via parameter tweak - exact-dict comparison meant
       `{lines: 50}` and `{lines: 51}` slipped through. Now redundancy is keyed
       on (action_type, target_service) within the last 3 actions.
    d) Rollback to currently-active version silently appended a fresh "active"
       deployment row and cleared anomalies. Now errors as no-op.
"""

from __future__ import annotations

import pytest

from incident_commander_env.models import ActionRecord, IncidentAction
from incident_commander_env.server.actions.handlers import handle_update_config
from incident_commander_env.server.grading.reward import (
    REDUNDANT_PENALTY,
    TIME_DECAY,
    compute_step_reward,
)
from incident_commander_env.server.scenarios.scenario_db_pool import DBPoolScenario
from incident_commander_env.server.scenarios.scenario_oom_crash import OOMCrashScenario
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.service import ServiceHealth


# ---------------------------------------------------------------------------
# Exploit (a): update_config string-match heal
# ---------------------------------------------------------------------------

class TestUpdateConfigHeal:
    def _setup_db_pool(self):
        cluster = Cluster()
        cluster.initialize()
        scenario = DBPoolScenario()
        scenario.setup(cluster)
        return cluster, scenario

    def test_unknown_config_key_is_rejected(self):
        cluster, scenario = self._setup_db_pool()
        action = IncidentAction(
            action_type="update_config",
            target_service="postgres-db",
            parameters={"key": "pool_size_hack", "value": 999},
        )
        obs = handle_update_config(action, cluster, scenario)
        assert obs.error == "Unknown config key"
        assert cluster.get_service("postgres-db").has_anomaly("db_pool_exhaustion")

    def test_old_string_match_payload_no_longer_heals(self):
        # Previously: any key containing both "pool" and "size" healed the DB.
        cluster, scenario = self._setup_db_pool()
        action = IncidentAction(
            action_type="update_config",
            target_service="postgres-db",
            parameters={"key": "DB_POOL_SIZE_HACK", "value": 999},
        )
        obs = handle_update_config(action, cluster, scenario)
        # Rejected because key is not in allowlist
        assert obs.error == "Unknown config key"
        # The DB anomaly is still there
        assert cluster.get_service("postgres-db").has_anomaly("db_pool_exhaustion")

    def test_correct_key_with_insufficient_value_does_not_heal(self):
        # Allowlisted key, but the scenario gates the heal on value > 50.
        cluster, scenario = self._setup_db_pool()
        action = IncidentAction(
            action_type="update_config",
            target_service="postgres-db",
            parameters={"key": "db.pool.max_size", "value": 30},
        )
        obs = handle_update_config(action, cluster, scenario)
        assert obs.error is None
        # Anomaly still present because 30 isn't enough
        assert cluster.get_service("postgres-db").has_anomaly("db_pool_exhaustion")

    def test_correct_key_with_sufficient_value_does_heal(self):
        cluster, scenario = self._setup_db_pool()
        action = IncidentAction(
            action_type="update_config",
            target_service="postgres-db",
            parameters={"key": "db.pool.max_size", "value": 100},
        )
        obs = handle_update_config(action, cluster, scenario)
        assert obs.error is None
        assert not cluster.get_service("postgres-db").has_anomaly("db_pool_exhaustion")

    def test_config_on_wrong_service_does_not_heal(self):
        # Setting db.pool.max_size on order-service must not heal postgres
        cluster, scenario = self._setup_db_pool()
        action = IncidentAction(
            action_type="update_config",
            target_service="order-service",
            parameters={"key": "db.pool.max_size", "value": 200},
        )
        obs = handle_update_config(action, cluster, scenario)
        assert obs.error is None
        # postgres anomaly untouched
        assert cluster.get_service("postgres-db").has_anomaly("db_pool_exhaustion")


# ---------------------------------------------------------------------------
# Exploit (b): restart-clears-all-anomalies
# ---------------------------------------------------------------------------

class TestRestartCurable:
    def test_restart_clears_oom_when_memory_increased(self):
        cluster = Cluster()
        cluster.initialize()
        scenario = OOMCrashScenario()
        scenario.setup(cluster)
        svc = cluster.get_service("payment-service")
        assert svc.has_anomaly("oom")
        msg = svc.restart(new_memory_limit="512Mi")
        assert not svc.has_anomaly("oom")
        assert svc.health == ServiceHealth.HEALTHY
        assert "successfully" in msg

    def test_restart_does_not_clear_oom_without_memory_bump(self):
        cluster = Cluster()
        cluster.initialize()
        scenario = OOMCrashScenario()
        scenario.setup(cluster)
        svc = cluster.get_service("payment-service")
        msg = svc.restart()  # no new_memory_limit
        # OOM survives because the underlying constraint (memory cap) wasn't addressed
        assert svc.has_anomaly("oom")
        assert svc.health == ServiceHealth.DEGRADED
        assert "Anomalies remain" in msg

    def test_restart_does_not_clear_memory_leak(self):
        # memory_leak in bad-deploy scenario survives a plain restart
        cluster = Cluster()
        cluster.initialize()
        svc = cluster.get_service("order-service")
        svc.set_anomaly("memory_leak")
        msg = svc.restart(new_memory_limit="1024Mi")
        assert svc.has_anomaly("memory_leak")
        assert svc.health == ServiceHealth.DEGRADED
        assert "Anomalies remain" in msg

    def test_restart_does_not_clear_db_pool_exhaustion(self):
        cluster = Cluster()
        cluster.initialize()
        scenario = DBPoolScenario()
        scenario.setup(cluster)
        db = cluster.get_service("postgres-db")
        msg = db.restart()
        assert db.has_anomaly("db_pool_exhaustion")
        assert db.health == ServiceHealth.DEGRADED

    def test_restart_does_clear_connection_leak(self):
        # connection_leak is in _RESTART_CURABLE - restart does clear it
        cluster = Cluster()
        cluster.initialize()
        scenario = DBPoolScenario()
        scenario.setup(cluster)
        order = cluster.get_service("order-service")
        assert order.has_anomaly("connection_leak")
        msg = order.restart()
        assert not order.has_anomaly("connection_leak")
        assert order.health == ServiceHealth.HEALTHY


# ---------------------------------------------------------------------------
# Exploit (c): redundancy bypass via parameter tweak
# ---------------------------------------------------------------------------

class TestRedundancyDetection:
    def test_param_tweak_does_not_bypass_redundancy(self):
        # Same action_type + target with different params should still trigger
        # the redundancy penalty *component*, regardless of the total reward.
        # The total combines r_diagnostic (positive) + r_format (positive) +
        # r_penalty (negative redundancy), so we verify the penalty component
        # specifically.
        from incident_commander_env.server.grading.components import (
            R_PENALTY_REDUNDANT, r_penalty,
        )
        from incident_commander_env.server.grading.episode_context import EpisodeContext

        prev = ActionRecord(
            step=1,
            action_type="read_logs",
            target_service="payment-service",
            parameters={"lines": 50, "severity": "ERROR"},
        )
        action = ActionRecord(
            step=2,
            action_type="read_logs",
            target_service="payment-service",
            parameters={"lines": 51, "severity": "ERROR"},  # tweaked param
        )
        ctx = EpisodeContext(
            scenario=None,
            previous_actions=[prev],
            relevant_services={"payment-service"},
            healthy_services=set(),
            step_count=2,
            max_steps=15,
        )
        # The redundancy penalty component fires regardless of param differences
        assert r_penalty(action, ctx) == R_PENALTY_REDUNDANT

    def test_redundancy_window_is_3_actions(self):
        # If the same action was 4+ actions ago, it's not redundant anymore
        prev_old = ActionRecord(
            step=1,
            action_type="read_logs",
            target_service="payment-service",
            parameters={},
        )
        filler = [
            ActionRecord(step=2, action_type="list_services", target_service=None, parameters={}),
            ActionRecord(step=3, action_type="check_metrics", target_service="auth-service", parameters={}),
            ActionRecord(step=4, action_type="describe_service", target_service="auth-service", parameters={}),
        ]
        action = ActionRecord(
            step=5,
            action_type="read_logs",
            target_service="payment-service",
            parameters={},
        )
        reward = compute_step_reward(
            action=action,
            step=5,
            previous_actions=[prev_old, *filler],
            relevant_services={"payment-service"},
            healthy_services=set(),
        )
        # Reading the relevant service's logs again is no longer flagged as redundant
        # (the original read fell out of the 3-action window). It should be a positive
        # diagnostic reward instead.
        assert reward > 0

    def test_different_target_is_not_redundant(self):
        prev = ActionRecord(
            step=1,
            action_type="read_logs",
            target_service="payment-service",
            parameters={"lines": 50},
        )
        action = ActionRecord(
            step=2,
            action_type="read_logs",
            target_service="postgres-db",
            parameters={"lines": 50},
        )
        reward = compute_step_reward(
            action=action,
            step=2,
            previous_actions=[prev],
            relevant_services={"payment-service"},
            healthy_services=set(),
        )
        assert reward > 0


# ---------------------------------------------------------------------------
# Exploit (d): rollback-to-self
# ---------------------------------------------------------------------------

class TestRollbackToSelf:
    def test_rollback_to_current_version_errors(self):
        cluster = Cluster()
        cluster.initialize()
        order = cluster.get_service("order-service")
        order.set_anomaly("memory_leak")
        current = order.config.version
        msg = order.rollback(current)
        assert "no-op" in msg.lower() or "error" in msg.lower()
        # Anomaly NOT cleared because rollback was a no-op
        assert order.has_anomaly("memory_leak")
        # Version unchanged
        assert order.config.version == current

    def test_rollback_to_real_prior_version_works(self):
        # Sanity: legitimate rollback path still works
        cluster = Cluster()
        cluster.initialize()
        order = cluster.get_service("order-service")
        original_version = order.config.version
        # Push order onto a "bad" version so we can roll back to original
        from incident_commander_env.server.simulation.service import Deployment
        order.config.version = "v99.0.0-bad"
        order.deployment_history.append(
            Deployment(version="v99.0.0-bad", timestamp="2026-03-29T09:00:00Z", status="active")
        )
        order.set_anomaly("memory_leak")
        msg = order.rollback(original_version)
        assert "rolled back" in msg.lower()
        assert order.config.version == original_version
        # memory_leak is in _ROLLBACK_CURABLE so it gets cleared
        assert not order.has_anomaly("memory_leak")
