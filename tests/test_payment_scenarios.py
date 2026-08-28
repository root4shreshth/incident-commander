"""Tests for the 4 payments-industry scenarios (Razorpay-shaped incidents).

Covers:
  - `payment_gateway_timeout` (YAML) - upstream processor 5xx, correct fix is scale_service
  - `webhook_delivery_backlog` (YAML) - stuck delivery workers, correct fix is restart
  - `fraud_check_memory_blowup` (YAML) - preemptive memory-headroom restart
  - `refund_race_deadlock` (Python subclass) - ORDERING-sensitive rollback then restart

Structural tests (registration, env reset, is_correct_op accept/reject) live
here rather than in test_scenarios.py so a targeted `pytest -k payment` run
covers the whole payments library in one go.
"""

from __future__ import annotations

import pytest

from incident_commander_env.models import ActionRecord, IncidentAction
from incident_commander_env.server.coach import IDEAL_TRAJECTORIES, LEARNING_CONTEXT
from incident_commander_env.server.environment import IncidentCommanderEnv
from incident_commander_env.server.scenarios import SCENARIO_REGISTRY
from incident_commander_env.server.scenarios.scenario_refund_race import (
    BAD_VERSION,
    STABLE_VERSION,
    RefundRaceScenario,
)


PAYMENT_FAMILIES = [
    "payment_gateway_timeout",
    "webhook_delivery_backlog",
    "fraud_check_memory_blowup",
    "refund_race_deadlock",
]


class TestPaymentScenarioRegistration:
    def test_all_four_payment_scenarios_registered(self) -> None:
        for fam in PAYMENT_FAMILIES:
            assert fam in SCENARIO_REGISTRY, f"scenario {fam!r} not in SCENARIO_REGISTRY"

    def test_registry_size_at_least_12(self) -> None:
        """6 built-in Python + 1 new Python (refund_race) + 5 YAML (3 payment + dns + rate_limit)."""
        assert len(SCENARIO_REGISTRY) >= 12

    def test_coach_ideal_trajectory_present_for_each(self) -> None:
        for fam in PAYMENT_FAMILIES:
            assert fam in IDEAL_TRAJECTORIES, f"IDEAL_TRAJECTORIES missing {fam!r}"
            trajectory = IDEAL_TRAJECTORIES[fam]
            assert len(trajectory) >= 3, f"{fam} trajectory too short ({len(trajectory)} steps)"
            assert trajectory[0]["action"] == "list_services", (
                f"{fam} should start with list_services (matches other scenarios)"
            )
            assert trajectory[-1]["action"] == "resolve_incident", (
                f"{fam} should end with resolve_incident"
            )

    def test_coach_learning_context_present_for_each(self) -> None:
        for fam in PAYMENT_FAMILIES:
            assert fam in LEARNING_CONTEXT, f"LEARNING_CONTEXT missing {fam!r}"
            ctx = LEARNING_CONTEXT[fam]
            for key in ("skill_tag", "backstory", "learning_goals", "est_minutes"):
                assert key in ctx, f"{fam} learning context missing {key!r}"


class TestPaymentScenarioReset:
    @pytest.mark.parametrize("family", PAYMENT_FAMILIES)
    def test_env_resets_cleanly(self, family: str) -> None:
        env = IncidentCommanderEnv()
        obs = env.reset(task_id=family, seed=42, difficulty=0.5)
        assert obs.alert is not None and len(obs.alert) > 0
        assert obs.message is not None and len(obs.message) > 0

    @pytest.mark.parametrize("family", PAYMENT_FAMILIES)
    def test_first_step_produces_reward_breakdown(self, family: str) -> None:
        env = IncidentCommanderEnv()
        env.reset(task_id=family, seed=42, difficulty=0.5)
        env.step(IncidentAction(action_type="list_services"))
        breakdown = env._last_breakdown
        assert breakdown is not None
        # list_services on turn 1 gets format credit at minimum.
        assert breakdown.format > 0

    @pytest.mark.parametrize("family", PAYMENT_FAMILIES)
    def test_seeded_reset_is_reproducible(self, family: str) -> None:
        env_a = IncidentCommanderEnv()
        obs_a = env_a.reset(task_id=family, seed=123, difficulty=0.5)
        env_b = IncidentCommanderEnv()
        obs_b = env_b.reset(task_id=family, seed=123, difficulty=0.5)
        assert obs_a.alert == obs_b.alert
        assert obs_a.message == obs_b.message


class TestPaymentScenarioCorrectOp:
    """is_correct_op must accept the right remediation and reject the wrong one."""

    def test_payment_gateway_timeout_wants_scale(self) -> None:
        scenario = SCENARIO_REGISTRY["payment_gateway_timeout"](seed=1, difficulty=0.5)
        right = ActionRecord(step=1, action_type="scale_service",
                             target_service="payment-gateway", parameters={"replicas": 6})
        wrong_action = ActionRecord(step=1, action_type="restart_service",
                                    target_service="payment-gateway", parameters={})
        wrong_target = ActionRecord(step=1, action_type="scale_service",
                                    target_service="postgres-db", parameters={})
        assert scenario.is_correct_op(right, None) is True
        assert scenario.is_correct_op(wrong_action, None) is False
        assert scenario.is_correct_op(wrong_target, None) is False

    def test_webhook_delivery_backlog_wants_restart(self) -> None:
        scenario = SCENARIO_REGISTRY["webhook_delivery_backlog"](seed=1, difficulty=0.5)
        right = ActionRecord(step=1, action_type="restart_service",
                             target_service="webhook-consumer", parameters={})
        wrong_action = ActionRecord(step=1, action_type="scale_service",
                                    target_service="webhook-consumer", parameters={})
        assert scenario.is_correct_op(right, None) is True
        assert scenario.is_correct_op(wrong_action, None) is False

    def test_fraud_check_memory_blowup_wants_restart(self) -> None:
        scenario = SCENARIO_REGISTRY["fraud_check_memory_blowup"](seed=1, difficulty=0.5)
        right = ActionRecord(step=1, action_type="restart_service",
                             target_service="fraud-check", parameters={"memory_limit": "2048Mi"})
        wrong_target = ActionRecord(step=1, action_type="restart_service",
                                    target_service="payment-service", parameters={})
        assert scenario.is_correct_op(right, None) is True
        assert scenario.is_correct_op(wrong_target, None) is False


class TestRefundRaceScenario:
    """Refund-race is the ordering-sensitive one - deserves extra structural tests."""

    def test_uses_correct_versions(self) -> None:
        assert BAD_VERSION == "v3.2.1"
        assert STABLE_VERSION == "v3.2.0"

    def test_setup_puts_refund_service_on_bad_version(self) -> None:
        env = IncidentCommanderEnv()
        env.reset(task_id="refund_race_deadlock", seed=1, difficulty=0.5)
        # After setup, refund-service should be on the bad version.
        # The env exposes services via the sim backend snapshot.
        snap = env._backend.snapshot()
        refund = snap.services.get("refund-service")
        assert refund is not None
        assert refund.version == BAD_VERSION

    def test_rollback_to_stable_version_is_correct(self) -> None:
        scenario = RefundRaceScenario(seed=1, difficulty=0.5)
        right = ActionRecord(
            step=1, action_type="rollback_deployment",
            target_service="refund-service",
            parameters={"to_version": STABLE_VERSION},
        )
        assert scenario.is_correct_op(right, None) is True

    def test_rollback_to_wrong_version_is_not_correct(self) -> None:
        scenario = RefundRaceScenario(seed=1, difficulty=0.5)
        wrong = ActionRecord(
            step=1, action_type="rollback_deployment",
            target_service="refund-service",
            parameters={"to_version": "v3.2.2"},
        )
        assert scenario.is_correct_op(wrong, None) is False

    def test_restart_ledger_is_correct(self) -> None:
        scenario = RefundRaceScenario(seed=1, difficulty=0.5)
        right = ActionRecord(
            step=1, action_type="restart_service",
            target_service="ledger-service", parameters={},
        )
        assert scenario.is_correct_op(right, None) is True

    def test_restart_refund_service_is_wrong(self) -> None:
        """Bare restart of refund-service leaves the bug in place - explicit anti-cheat."""
        scenario = RefundRaceScenario(seed=1, difficulty=0.5)
        wrong = ActionRecord(
            step=1, action_type="restart_service",
            target_service="refund-service", parameters={},
        )
        assert scenario.is_correct_op(wrong, None) is False

    def test_penalty_for_bare_restart_of_refund_service(self) -> None:
        scenario = RefundRaceScenario(seed=1, difficulty=0.5)
        actions = [
            ActionRecord(step=1, action_type="restart_service",
                         target_service="refund-service", parameters={}),
        ]
        pen = scenario.compute_penalties(actions, cluster=None)
        assert pen <= -0.10, f"expected >= -0.10 penalty for bare restart, got {pen}"

    def test_correct_sequence_wins_ordering_credit(self) -> None:
        scenario = RefundRaceScenario(seed=1, difficulty=0.5)
        actions = [
            ActionRecord(step=1, action_type="list_services",
                         target_service=None, parameters={}),
            ActionRecord(step=2, action_type="read_logs",
                         target_service="refund-service", parameters={}),
            ActionRecord(step=3, action_type="rollback_deployment",
                         target_service="refund-service",
                         parameters={"to_version": STABLE_VERSION}),
            ActionRecord(step=4, action_type="restart_service",
                         target_service="ledger-service", parameters={}),
        ]
        rubric = scenario.get_rubric()
        # Find the "Correct ordering" criterion
        ordering_check = None
        for desc, check_fn, _weight in rubric:
            if "Correct ordering" in desc:
                ordering_check = check_fn
                break
        assert ordering_check is not None, "rubric missing the 'Correct ordering' criterion"
        assert ordering_check(actions, None) is True

    def test_reversed_sequence_fails_ordering_credit(self) -> None:
        scenario = RefundRaceScenario(seed=1, difficulty=0.5)
        actions = [
            # Restart ledger FIRST (wrong order), then rollback.
            ActionRecord(step=1, action_type="restart_service",
                         target_service="ledger-service", parameters={}),
            ActionRecord(step=2, action_type="rollback_deployment",
                         target_service="refund-service",
                         parameters={"to_version": STABLE_VERSION}),
        ]
        rubric = scenario.get_rubric()
        ordering_check = next(
            (fn for desc, fn, _ in rubric if "Correct ordering" in desc),
            None,
        )
        assert ordering_check is not None
        assert ordering_check(actions, None) is False
