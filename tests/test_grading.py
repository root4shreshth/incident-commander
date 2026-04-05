"""Tests for grading, reward computation, and penalty logic.

Covers:
- Grader scores are always in [0.0, 1.0]
- Perfect action sequences score 1.0 (or close to it)
- Penalty computation for harmful/redundant actions
- Per-step reward function returns values in expected ranges
- Reward constants and time decay behave correctly
"""

import pytest

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.environment import IncidentCommanderEnv
from incident_commander_env.server.grading.grader import IncidentGrader
from incident_commander_env.server.grading.reward import (
    CORRECT_FIX_REWARD,
    DIAGNOSTIC_REWARD,
    HARMFUL_PENALTY,
    IRRELEVANT_PENALTY,
    REDUNDANT_PENALTY,
    RELEVANT_DIAGNOSTIC,
    TIME_DECAY,
    compute_step_reward,
)
from incident_commander_env.server.scenarios import SCENARIO_REGISTRY
from incident_commander_env.server.scenarios.scenario_bad_deploy import BadDeployScenario
from incident_commander_env.server.scenarios.scenario_db_pool import DBPoolScenario
from incident_commander_env.server.scenarios.scenario_oom_crash import OOMCrashScenario
from incident_commander_env.server.simulation.cluster import Cluster


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grader():
    return IncidentGrader()


@pytest.fixture
def oom_cluster():
    """Cluster with OOM scenario set up."""
    cluster = Cluster()
    cluster.initialize()
    scenario = OOMCrashScenario()
    scenario.setup(cluster)
    return cluster, scenario


@pytest.fixture
def db_pool_cluster():
    """Cluster with DB pool exhaustion scenario set up."""
    cluster = Cluster()
    cluster.initialize()
    scenario = DBPoolScenario()
    scenario.setup(cluster)
    return cluster, scenario


@pytest.fixture
def bad_deploy_cluster():
    """Cluster with bad deployment scenario set up."""
    cluster = Cluster()
    cluster.initialize()
    scenario = BadDeployScenario()
    scenario.setup(cluster)
    return cluster, scenario


# ---------------------------------------------------------------------------
# Grader score range tests
# ---------------------------------------------------------------------------

class TestGraderScoreRange:
    """Grader must always return scores in [0.0, 1.0]."""

    @pytest.mark.parametrize("task_id,scenario_cls", list(SCENARIO_REGISTRY.items()))
    def test_empty_actions_score_in_range(self, grader, task_id, scenario_cls):
        cluster = Cluster()
        cluster.initialize()
        scenario = scenario_cls()
        scenario.setup(cluster)
        score = grader.grade(scenario, [], cluster)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("task_id,scenario_cls", list(SCENARIO_REGISTRY.items()))
    def test_random_actions_score_in_range(self, grader, task_id, scenario_cls):
        cluster = Cluster()
        cluster.initialize()
        scenario = scenario_cls()
        scenario.setup(cluster)

        actions = [
            ActionRecord(step=1, action_type="list_services"),
            ActionRecord(step=2, action_type="read_logs", target_service="postgres-db"),
            ActionRecord(step=3, action_type="check_metrics", target_service="api-gateway"),
            ActionRecord(step=4, action_type="restart_service", target_service="auth-service"),
            ActionRecord(step=5, action_type="restart_service", target_service="user-service"),
        ]
        score = grader.grade(scenario, actions, cluster)
        assert 0.0 <= score <= 1.0

    def test_score_never_negative(self, grader, oom_cluster):
        """Even with many penalties, score is clamped to 0.0."""
        cluster, scenario = oom_cluster
        # All harmful actions: restarting every healthy service
        actions = [
            ActionRecord(step=i, action_type="restart_service", target_service=svc)
            for i, svc in enumerate(
                ["auth-service", "postgres-db", "inventory-service",
                 "notification-service", "user-service", "order-service",
                 "api-gateway", "frontend-bff"],
                start=1,
            )
        ]
        score = grader.grade(scenario, actions, cluster)
        assert score >= 0.0

    def test_score_never_above_one(self, grader, oom_cluster):
        cluster, scenario = oom_cluster
        # Simulate perfect actions plus bonus -- score should still cap at 1.0
        svc = cluster.get_service("payment-service")
        svc.restart(new_memory_limit="512Mi")

        actions = [
            ActionRecord(step=1, action_type="list_services"),
            ActionRecord(step=2, action_type="read_logs", target_service="payment-service"),
            ActionRecord(step=3, action_type="check_metrics", target_service="payment-service"),
            ActionRecord(step=4, action_type="describe_service", target_service="payment-service"),
            ActionRecord(step=5, action_type="restart_service", target_service="payment-service",
                         parameters={"memory_limit": "512Mi"}),
        ]
        score = grader.grade(scenario, actions, cluster)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# Perfect action sequences
# ---------------------------------------------------------------------------

class TestPerfectScores:
    """Perfect (or near-perfect) action sequences should score high."""

    def test_oom_perfect_sequence_scores_1(self, grader):
        cluster = Cluster()
        cluster.initialize()
        scenario = OOMCrashScenario()
        scenario.setup(cluster)

        # Perfect sequence: identify, read logs, diagnose, fix
        actions = [
            ActionRecord(step=1, action_type="list_services"),
            ActionRecord(step=2, action_type="read_logs", target_service="payment-service"),
            ActionRecord(step=3, action_type="check_metrics", target_service="payment-service"),
            ActionRecord(step=4, action_type="restart_service", target_service="payment-service",
                         parameters={"memory_limit": "512Mi"}),
        ]

        # Apply the fix to the cluster so the rubric check sees it resolved
        svc = cluster.get_service("payment-service")
        svc.restart(new_memory_limit="512Mi")

        score = grader.grade(scenario, actions, cluster)
        assert score == 1.0

    def test_db_pool_perfect_sequence_scores_high(self, grader):
        cluster = Cluster()
        cluster.initialize()
        scenario = DBPoolScenario()
        scenario.setup(cluster)

        actions = [
            ActionRecord(step=1, action_type="list_services"),
            ActionRecord(step=2, action_type="read_logs", target_service="frontend-bff"),
            ActionRecord(step=3, action_type="read_logs", target_service="order-service"),
            ActionRecord(step=4, action_type="describe_service", target_service="postgres-db"),
            ActionRecord(step=5, action_type="read_logs", target_service="postgres-db"),
            ActionRecord(step=6, action_type="update_config", target_service="postgres-db",
                         parameters={"key": "db.pool.max_size", "value": 100}),
            ActionRecord(step=7, action_type="restart_service", target_service="order-service"),
            ActionRecord(step=8, action_type="resolve_incident",
                         parameters={"root_cause": "db pool exhaustion"}),
        ]

        # Apply fixes to cluster
        db = cluster.get_service("postgres-db")
        db.config.db_pool_size = 100
        db.clear_anomaly("db_pool_exhaustion")
        db.health = db.health.__class__("healthy")

        order = cluster.get_service("order-service")
        order.restart()

        score = grader.grade(scenario, actions, cluster)
        assert score >= 0.9, f"Expected >= 0.9, got {score}"

    def test_bad_deploy_perfect_sequence_scores_high(self, grader):
        cluster = Cluster()
        cluster.initialize()
        scenario = BadDeployScenario()
        scenario.setup(cluster)

        actions = [
            ActionRecord(step=1, action_type="list_services"),
            ActionRecord(step=2, action_type="read_logs", target_service="api-gateway"),
            ActionRecord(step=3, action_type="read_logs", target_service="order-service"),
            ActionRecord(step=4, action_type="describe_service", target_service="order-service"),
            ActionRecord(step=5, action_type="check_metrics", target_service="inventory-service"),
            ActionRecord(step=6, action_type="rollback_deployment",
                         target_service="order-service",
                         parameters={"to_version": "v2.3.1"}),
            ActionRecord(step=7, action_type="restart_service",
                         target_service="inventory-service"),
            ActionRecord(step=8, action_type="restart_service",
                         target_service="notification-service"),
            ActionRecord(step=9, action_type="resolve_incident",
                         parameters={"root_cause": "Bad deployment v2.4.0 caused memory leak"}),
        ]

        # Apply fixes to cluster
        order = cluster.get_service("order-service")
        order.rollback("v2.3.1")
        order.config.replicas = 2

        inv = cluster.get_service("inventory-service")
        inv.restart()

        notif = cluster.get_service("notification-service")
        notif.restart()

        score = grader.grade(scenario, actions, cluster)
        assert score >= 0.9, f"Expected >= 0.9, got {score}"


# ---------------------------------------------------------------------------
# Penalty computation tests
# ---------------------------------------------------------------------------

class TestPenalties:
    """Tests for scenario-specific penalty computation."""

    def test_oom_penalty_for_restarting_healthy(self, oom_cluster):
        cluster, scenario = oom_cluster
        actions = [
            ActionRecord(step=1, action_type="restart_service", target_service="auth-service"),
            ActionRecord(step=2, action_type="restart_service", target_service="postgres-db"),
        ]
        penalty = scenario.compute_penalties(actions, cluster)
        assert penalty < 0.0
        # Each healthy restart costs -0.10
        assert penalty == pytest.approx(-0.20, abs=0.01)

    def test_oom_no_penalty_for_target_restart(self, oom_cluster):
        cluster, scenario = oom_cluster
        actions = [
            ActionRecord(step=1, action_type="restart_service", target_service="payment-service",
                         parameters={"memory_limit": "512Mi"}),
        ]
        penalty = scenario.compute_penalties(actions, cluster)
        assert penalty == 0.0

    def test_db_pool_penalty_for_unrelated_restarts(self, db_pool_cluster):
        cluster, scenario = db_pool_cluster
        actions = [
            ActionRecord(step=1, action_type="restart_service", target_service="auth-service"),
            ActionRecord(step=2, action_type="restart_service", target_service="user-service"),
            ActionRecord(step=3, action_type="restart_service",
                         target_service="notification-service"),
        ]
        penalty = scenario.compute_penalties(actions, cluster)
        # auth-service and user-service are uninvolved (-0.05 each)
        # notification-service is not in the uninvolved set for db_pool
        assert penalty < 0.0
        assert penalty == pytest.approx(-0.10, abs=0.01)

    def test_bad_deploy_penalty_for_restarting_order(self, bad_deploy_cluster):
        cluster, scenario = bad_deploy_cluster
        actions = [
            ActionRecord(step=1, action_type="restart_service", target_service="order-service"),
        ]
        penalty = scenario.compute_penalties(actions, cluster)
        assert penalty == pytest.approx(-0.10, abs=0.01)

    def test_bad_deploy_penalty_for_unrelated_services(self, bad_deploy_cluster):
        cluster, scenario = bad_deploy_cluster
        actions = [
            ActionRecord(step=1, action_type="restart_service", target_service="auth-service"),
            ActionRecord(step=2, action_type="scale_service", target_service="payment-service",
                         parameters={"replicas": 5}),
        ]
        penalty = scenario.compute_penalties(actions, cluster)
        # auth-service restart: -0.05, payment-service scale: -0.05
        assert penalty == pytest.approx(-0.10, abs=0.01)

    def test_no_penalties_for_diagnostic_actions(self, oom_cluster):
        cluster, scenario = oom_cluster
        actions = [
            ActionRecord(step=1, action_type="list_services"),
            ActionRecord(step=2, action_type="read_logs", target_service="auth-service"),
            ActionRecord(step=3, action_type="check_metrics", target_service="postgres-db"),
        ]
        penalty = scenario.compute_penalties(actions, cluster)
        assert penalty == 0.0


# ---------------------------------------------------------------------------
# Grade details tests
# ---------------------------------------------------------------------------

class TestGradeDetails:
    """Tests for the grade_details breakdown."""

    def test_grade_details_structure(self, grader, oom_cluster):
        cluster, scenario = oom_cluster
        actions = [
            ActionRecord(step=1, action_type="list_services"),
        ]
        details = grader.grade_details(scenario, actions, cluster)
        assert "task_id" in details
        assert "criteria" in details
        assert "penalties" in details
        assert "final_score" in details
        assert details["task_id"] == "oom_crash"

    def test_grade_details_criteria_have_fields(self, grader, oom_cluster):
        cluster, scenario = oom_cluster
        details = grader.grade_details(scenario, [], cluster)
        for criterion in details["criteria"]:
            assert "criterion" in criterion
            assert "weight" in criterion
            assert "passed" in criterion
            assert isinstance(criterion["passed"], bool)

    def test_grade_details_final_score_matches_grade(self, grader, oom_cluster):
        cluster, scenario = oom_cluster
        actions = [
            ActionRecord(step=1, action_type="read_logs", target_service="payment-service"),
        ]
        score = grader.grade(scenario, actions, cluster)
        details = grader.grade_details(scenario, actions, cluster)
        assert details["final_score"] == pytest.approx(score)


# ---------------------------------------------------------------------------
# Per-step reward function tests
# ---------------------------------------------------------------------------

class TestComputeStepReward:
    """Tests for the compute_step_reward function."""

    def test_diagnostic_on_relevant_service(self):
        action = ActionRecord(step=1, action_type="read_logs", target_service="payment-service")
        reward = compute_step_reward(
            action=action,
            step=1,
            previous_actions=[],
            relevant_services={"payment-service"},
            healthy_services=set(),
        )
        expected = RELEVANT_DIAGNOSTIC * (TIME_DECAY ** 1)
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_diagnostic_on_irrelevant_service(self):
        action = ActionRecord(step=1, action_type="read_logs", target_service="auth-service")
        reward = compute_step_reward(
            action=action,
            step=1,
            previous_actions=[],
            relevant_services={"payment-service"},
            healthy_services=set(),
        )
        expected = DIAGNOSTIC_REWARD * (TIME_DECAY ** 1)
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_list_services_gives_diagnostic_reward(self):
        action = ActionRecord(step=1, action_type="list_services")
        reward = compute_step_reward(
            action=action,
            step=1,
            previous_actions=[],
            relevant_services={"payment-service"},
            healthy_services=set(),
        )
        expected = DIAGNOSTIC_REWARD * (TIME_DECAY ** 1)
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_correct_fix_on_relevant_service(self):
        action = ActionRecord(step=3, action_type="restart_service",
                              target_service="payment-service",
                              parameters={"memory_limit": "512Mi"})
        reward = compute_step_reward(
            action=action,
            step=3,
            previous_actions=[],
            relevant_services={"payment-service"},
            healthy_services=set(),
        )
        expected = CORRECT_FIX_REWARD * (TIME_DECAY ** 3)
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_harmful_restart_on_healthy_service(self):
        action = ActionRecord(step=2, action_type="restart_service",
                              target_service="auth-service")
        reward = compute_step_reward(
            action=action,
            step=2,
            previous_actions=[],
            relevant_services={"payment-service"},
            healthy_services={"auth-service"},
        )
        expected = HARMFUL_PENALTY * (TIME_DECAY ** 2)
        assert reward == pytest.approx(expected, abs=1e-6)
        assert reward < 0.0

    def test_redundant_action_penalty(self):
        prev = ActionRecord(step=1, action_type="list_services")
        action = ActionRecord(step=2, action_type="list_services")
        reward = compute_step_reward(
            action=action,
            step=2,
            previous_actions=[prev],
            relevant_services=set(),
            healthy_services=set(),
        )
        expected = REDUNDANT_PENALTY * (TIME_DECAY ** 2)
        assert reward == pytest.approx(expected, abs=1e-6)
        assert reward < 0.0

    def test_resolve_incident_reward(self):
        action = ActionRecord(step=5, action_type="resolve_incident",
                              parameters={"root_cause": "OOM", "resolution": "restarted"})
        reward = compute_step_reward(
            action=action,
            step=5,
            previous_actions=[],
            relevant_services=set(),
            healthy_services=set(),
        )
        expected = 0.05 * (TIME_DECAY ** 5)
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_time_decay_reduces_reward(self):
        """Later steps get slightly less reward due to time decay."""
        action_early = ActionRecord(step=1, action_type="list_services")
        action_late = ActionRecord(step=10, action_type="list_services")

        reward_early = compute_step_reward(
            action=action_early, step=1,
            previous_actions=[], relevant_services=set(), healthy_services=set(),
        )
        reward_late = compute_step_reward(
            action=action_late, step=10,
            previous_actions=[], relevant_services=set(), healthy_services=set(),
        )
        assert reward_early > reward_late
        assert reward_late > 0.0

    def test_irrelevant_fix_penalty(self):
        action = ActionRecord(step=2, action_type="update_config",
                              target_service="notification-service",
                              parameters={"key": "some.config", "value": 42})
        reward = compute_step_reward(
            action=action,
            step=2,
            previous_actions=[],
            relevant_services={"payment-service"},
            healthy_services=set(),
        )
        expected = IRRELEVANT_PENALTY * (TIME_DECAY ** 2)
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_all_rewards_are_finite(self):
        """No NaN or Inf in any reward path."""
        scenarios = [
            ("list_services", None, {}),
            ("read_logs", "payment-service", {}),
            ("check_metrics", "payment-service", {}),
            ("restart_service", "payment-service", {"memory_limit": "512Mi"}),
            ("resolve_incident", None, {"root_cause": "test"}),
            ("rollback_deployment", "order-service", {"to_version": "v2.3.1"}),
            ("scale_service", "order-service", {"replicas": 3}),
            ("update_config", "postgres-db", {"key": "db.pool.max_size", "value": 50}),
        ]
        for action_type, target, params in scenarios:
            action = ActionRecord(step=1, action_type=action_type,
                                  target_service=target, parameters=params)
            reward = compute_step_reward(
                action=action, step=1,
                previous_actions=[], relevant_services={"payment-service"},
                healthy_services={"auth-service"},
            )
            assert isinstance(reward, float)
            assert reward == reward  # NaN check: NaN != NaN


# ---------------------------------------------------------------------------
# Integration: grading through the environment
# ---------------------------------------------------------------------------

class TestGradingThroughEnvironment:
    """Test grading behaviour through full env.step() calls."""

    def test_resolved_episode_score_in_range(self):
        from incident_commander_env.models import IncidentAction

        env = IncidentCommanderEnv()
        env.reset(task_id="oom_crash")
        env.step(IncidentAction(action_type="read_logs", target_service="payment-service"))
        obs = env.step(IncidentAction(
            action_type="restart_service",
            target_service="payment-service",
            parameters={"memory_limit": "512Mi"},
        ))
        assert obs.done is True
        assert 0.0 <= env.state.current_score <= 1.0

    def test_timeout_episode_score_in_range(self):
        from incident_commander_env.models import IncidentAction

        env = IncidentCommanderEnv()
        env.reset(task_id="oom_crash")

        for _ in range(15):
            obs = env.step(IncidentAction(action_type="list_services"))

        assert obs.done is True
        assert 0.0 <= env.state.current_score <= 1.0
