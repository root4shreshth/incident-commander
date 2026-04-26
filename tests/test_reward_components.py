"""Per-component reward tests.

Each of the 6 reward components (r_diagnostic, r_correct_op, r_resolution,
r_format, r_efficiency, r_penalty) has isolated unit tests that pin its
behavior. The hackathon judging guide demands "multiple independent reward
functions, not just one"; these tests are the receipts.

The components are tested at the function level - no env, no cluster
required - by hand-building an `EpisodeContext` and asserting the
component's float output matches the documented constants.
"""

from __future__ import annotations

import pytest

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.grading.components import (
    R_CORRECT_OP,
    R_DIAG_ADJACENT,
    R_DIAG_RELEVANT,
    R_EFFICIENCY,
    R_FORMAT,
    R_PENALTY_HARMFUL,
    R_PENALTY_HANDLER_ERROR,
    R_PENALTY_REDUNDANT,
    R_RESOLUTION_ACCURATE,
    R_RESOLUTION_FALSE,
    R_RESOLUTION_VAGUE,
    RewardBreakdown,
    compute_step_breakdown,
    r_correct_op,
    r_diagnostic,
    r_efficiency,
    r_format,
    r_penalty,
    r_resolution,
)
from incident_commander_env.server.grading.episode_context import EpisodeContext
from incident_commander_env.server.scenarios.scenario_bad_deploy import BadDeployScenario
from incident_commander_env.server.scenarios.scenario_db_pool import DBPoolScenario
from incident_commander_env.server.scenarios.scenario_oom_crash import OOMCrashScenario
from incident_commander_env.server.simulation.cluster import Cluster


def _ctx(scenario, *, prev=None, relevant=None, healthy=None, step=1, max_steps=15,
         is_terminal=False, is_resolved=False, error=None):
    cluster = Cluster()
    cluster.initialize()
    return EpisodeContext(
        scenario=scenario,
        previous_actions=prev or [],
        relevant_services=relevant or set(),
        healthy_services=healthy or set(),
        step_count=step,
        max_steps=max_steps,
        is_terminal=is_terminal,
        is_resolved=is_resolved,
        cluster=cluster,
        last_observation_error=error,
    )


def _action(action_type, target=None, **params):
    return ActionRecord(step=1, action_type=action_type, target_service=target, parameters=params)


# ---------------------------------------------------------------------------
# r_diagnostic
# ---------------------------------------------------------------------------

class TestRDiagnostic:
    def test_relevant_service_yields_relevant_reward(self):
        ctx = _ctx(OOMCrashScenario(), relevant={"payment-service"})
        action = _action("read_logs", "payment-service")
        assert r_diagnostic(action, ctx) == R_DIAG_RELEVANT

    def test_irrelevant_service_yields_adjacent_reward(self):
        ctx = _ctx(OOMCrashScenario(), relevant={"payment-service"})
        action = _action("read_logs", "auth-service")
        assert r_diagnostic(action, ctx) == R_DIAG_ADJACENT

    def test_remediation_action_yields_zero(self):
        ctx = _ctx(OOMCrashScenario(), relevant={"payment-service"})
        action = _action("restart_service", "payment-service")
        assert r_diagnostic(action, ctx) == 0.0

    def test_list_services_yields_adjacent(self):
        ctx = _ctx(OOMCrashScenario(), relevant={"payment-service"})
        action = _action("list_services")
        # No target, so target_service not in relevant set => adjacent reward
        assert r_diagnostic(action, ctx) == R_DIAG_ADJACENT


# ---------------------------------------------------------------------------
# r_correct_op
# ---------------------------------------------------------------------------

class TestRCorrectOp:
    def test_oom_correct_restart_with_more_memory(self):
        sc = OOMCrashScenario()
        ctx = _ctx(sc, relevant={"payment-service"})
        action = _action("restart_service", "payment-service", memory_limit="512Mi")
        assert r_correct_op(action, ctx) == R_CORRECT_OP

    def test_oom_restart_without_memory_bump_is_not_correct(self):
        sc = OOMCrashScenario()
        ctx = _ctx(sc, relevant={"payment-service"})
        action = _action("restart_service", "payment-service")  # no memory_limit
        assert r_correct_op(action, ctx) == 0.0

    def test_oom_restart_with_lower_memory_is_not_correct(self):
        sc = OOMCrashScenario()
        ctx = _ctx(sc, relevant={"payment-service"})
        action = _action("restart_service", "payment-service", memory_limit="128Mi")
        assert r_correct_op(action, ctx) == 0.0

    def test_db_pool_correct_config_update(self):
        sc = DBPoolScenario()
        ctx = _ctx(sc, relevant={"postgres-db"})
        action = _action("update_config", "postgres-db", key="db.pool.max_size", value=100)
        assert r_correct_op(action, ctx) == R_CORRECT_OP

    def test_db_pool_correct_restart_order_service(self):
        sc = DBPoolScenario()
        ctx = _ctx(sc, relevant={"order-service"})
        action = _action("restart_service", "order-service")
        assert r_correct_op(action, ctx) == R_CORRECT_OP

    def test_bad_deploy_rollback_is_correct(self):
        sc = BadDeployScenario()
        ctx = _ctx(sc, relevant={"order-service"})
        action = _action("rollback_deployment", "order-service", to_version="v2.3.1")
        assert r_correct_op(action, ctx) == R_CORRECT_OP

    def test_bad_deploy_restart_order_is_NOT_correct(self):
        # The whole point of this scenario: restarting order-service is the WRONG move
        sc = BadDeployScenario()
        ctx = _ctx(sc, relevant={"order-service"})
        action = _action("restart_service", "order-service")
        assert r_correct_op(action, ctx) == 0.0

    def test_bad_deploy_restart_starved_dependents_is_correct(self):
        sc = BadDeployScenario()
        ctx = _ctx(sc, relevant={"inventory-service"})
        action = _action("restart_service", "inventory-service")
        assert r_correct_op(action, ctx) == R_CORRECT_OP


# ---------------------------------------------------------------------------
# r_resolution
# ---------------------------------------------------------------------------

class TestRResolution:
    def test_accurate_root_cause_with_keyword_match(self):
        ctx = _ctx(OOMCrashScenario(), is_terminal=True, is_resolved=True)
        action = _action("resolve_incident",
                         root_cause="payment-service hit OOM due to low memory limit",
                         resolution="restarted with 512Mi")
        assert r_resolution(action, ctx) == R_RESOLUTION_ACCURATE

    def test_resolved_but_vague_root_cause(self):
        ctx = _ctx(OOMCrashScenario(), is_terminal=True, is_resolved=True)
        action = _action("resolve_incident", root_cause="something broke", resolution="fixed it")
        assert r_resolution(action, ctx) == R_RESOLUTION_VAGUE

    def test_resolve_when_not_actually_resolved_is_penalized(self):
        # Anti-cheat: declaring resolved without solving costs reward
        ctx = _ctx(OOMCrashScenario(), is_terminal=True, is_resolved=False)
        action = _action("resolve_incident", root_cause="OOM in payment-service",
                         resolution="fingers crossed")
        assert r_resolution(action, ctx) == R_RESOLUTION_FALSE

    def test_non_resolve_action_yields_zero(self):
        ctx = _ctx(OOMCrashScenario(), is_terminal=False)
        action = _action("read_logs", "payment-service")
        assert r_resolution(action, ctx) == 0.0


# ---------------------------------------------------------------------------
# r_format
# ---------------------------------------------------------------------------

class TestRFormat:
    def test_well_formed_action_gets_format_credit(self):
        ctx = _ctx(OOMCrashScenario())
        action = _action("read_logs", "payment-service")
        assert r_format(action, ctx) == R_FORMAT


# ---------------------------------------------------------------------------
# r_efficiency
# ---------------------------------------------------------------------------

class TestREfficiency:
    def test_efficient_resolution_within_50pct_budget(self):
        ctx = _ctx(OOMCrashScenario(), step=4, max_steps=15, is_terminal=True, is_resolved=True)
        action = _action("resolve_incident")
        assert r_efficiency(action, ctx) == R_EFFICIENCY

    def test_slow_resolution_yields_zero(self):
        ctx = _ctx(OOMCrashScenario(), step=12, max_steps=15, is_terminal=True, is_resolved=True)
        action = _action("resolve_incident")
        assert r_efficiency(action, ctx) == 0.0

    def test_unresolved_terminal_yields_zero(self):
        ctx = _ctx(OOMCrashScenario(), step=4, max_steps=15, is_terminal=True, is_resolved=False)
        action = _action("resolve_incident")
        assert r_efficiency(action, ctx) == 0.0

    def test_non_terminal_step_yields_zero(self):
        ctx = _ctx(OOMCrashScenario(), step=4, max_steps=15, is_terminal=False)
        action = _action("read_logs", "payment-service")
        assert r_efficiency(action, ctx) == 0.0


# ---------------------------------------------------------------------------
# r_penalty
# ---------------------------------------------------------------------------

class TestRPenalty:
    def test_redundancy_penalty(self):
        prev = _action("read_logs", "payment-service")
        ctx = _ctx(OOMCrashScenario(), prev=[prev])
        action = _action("read_logs", "payment-service")
        assert r_penalty(action, ctx) == R_PENALTY_REDUNDANT

    def test_harmful_restart_penalty(self):
        ctx = _ctx(OOMCrashScenario(), healthy={"auth-service"})
        action = _action("restart_service", "auth-service")
        assert r_penalty(action, ctx) == R_PENALTY_HARMFUL

    def test_handler_error_penalty(self):
        ctx = _ctx(OOMCrashScenario(), error="Unknown config key")
        action = _action("update_config", "postgres-db", key="bogus", value=1)
        assert r_penalty(action, ctx) == R_PENALTY_HANDLER_ERROR

    def test_redundancy_plus_harmful_stack(self):
        # Redundant restart of healthy service: both penalties apply
        prev = _action("restart_service", "auth-service")
        ctx = _ctx(OOMCrashScenario(), prev=[prev], healthy={"auth-service"})
        action = _action("restart_service", "auth-service")
        assert r_penalty(action, ctx) == R_PENALTY_REDUNDANT + R_PENALTY_HARMFUL

    def test_clean_action_no_penalty(self):
        ctx = _ctx(OOMCrashScenario(), relevant={"payment-service"})
        action = _action("read_logs", "payment-service")
        assert r_penalty(action, ctx) == 0.0


# ---------------------------------------------------------------------------
# RewardBreakdown aggregate
# ---------------------------------------------------------------------------

class TestRewardBreakdown:
    def test_total_sums_components(self):
        bd = RewardBreakdown(
            diagnostic=0.05, correct_op=0.15, resolution=0.30,
            format=0.01, efficiency=0.10, penalty=-0.03,
        )
        assert bd.total() == pytest.approx(0.58, abs=1e-9)

    def test_zero_returns_blank_breakdown(self):
        bd = RewardBreakdown.zero()
        assert bd.total() == 0.0
        assert bd.to_dict() == {
            "diagnostic": 0.0, "correct_op": 0.0, "resolution": 0.0,
            "format": 0.0, "efficiency": 0.0, "penalty": 0.0,
        }

    def test_compose_breakdown_for_perfect_resolve(self):
        sc = OOMCrashScenario()
        ctx = _ctx(sc, relevant={"payment-service"}, step=4, max_steps=15,
                   is_terminal=True, is_resolved=True)
        action = _action("resolve_incident",
                         root_cause="payment-service OOM, raised memory limit")
        bd = compute_step_breakdown(action, ctx)
        # Diagnostic=0 (resolve_incident isn't diagnostic)
        # correct_op=0 (resolve_incident isn't remediative)
        # resolution=R_RESOLUTION_ACCURATE (matches "payment" + "memory")
        # format=R_FORMAT
        # efficiency=R_EFFICIENCY (4 <= 15*0.5=7.5)
        # penalty=0
        assert bd.diagnostic == 0.0
        assert bd.correct_op == 0.0
        assert bd.resolution == R_RESOLUTION_ACCURATE
        assert bd.format == R_FORMAT
        assert bd.efficiency == R_EFFICIENCY
        assert bd.penalty == 0.0
        assert bd.total() == pytest.approx(R_RESOLUTION_ACCURATE + R_FORMAT + R_EFFICIENCY)
