"""Tests for IncidentCommanderEnv core environment lifecycle.

Covers:
- reset() returns valid IncidentObservation with alert and dependency_graph
- step() returns observation, reward, done, info fields
- All 3 tasks can be reset and initialized correctly
- Step budget enforcement (done=True when exceeded)
- Episode termination on incident resolution
- State tracking across steps
"""

import pytest

from incident_commander_env.models import IncidentAction, IncidentObservation
from incident_commander_env.server.environment import IncidentCommanderEnv
from incident_commander_env.server.scenarios import SCENARIO_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Fresh environment instance for each test."""
    return IncidentCommanderEnv()


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:
    """Tests for environment reset behaviour."""

    def test_reset_returns_observation(self, env: IncidentCommanderEnv):
        obs = env.reset(task_id="oom_crash")
        assert isinstance(obs, IncidentObservation)

    def test_reset_has_alert(self, env: IncidentCommanderEnv):
        obs = env.reset(task_id="oom_crash")
        assert obs.alert is not None
        assert len(obs.alert) > 0

    def test_reset_has_dependency_graph(self, env: IncidentCommanderEnv):
        obs = env.reset(task_id="oom_crash")
        assert obs.dependency_graph is not None
        assert isinstance(obs.dependency_graph, dict)
        assert len(obs.dependency_graph) > 0

    def test_reset_not_done(self, env: IncidentCommanderEnv):
        obs = env.reset(task_id="oom_crash")
        assert obs.done is False

    def test_reset_message_contains_incident_alert(self, env: IncidentCommanderEnv):
        obs = env.reset(task_id="oom_crash")
        assert "INCIDENT ALERT" in obs.message

    def test_reset_message_lists_available_actions(self, env: IncidentCommanderEnv):
        obs = env.reset(task_id="oom_crash")
        assert "list_services" in obs.message
        assert "resolve_incident" in obs.message

    def test_reset_state_initialised(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        state = env.state
        assert state.task_id == "oom_crash"
        assert state.step_count == 0
        assert state.episode_id != ""
        assert state.max_steps == 15
        assert state.incident_resolved is False

    def test_reset_unknown_task_returns_error(self, env: IncidentCommanderEnv):
        obs = env.reset(task_id="nonexistent_task")
        assert obs.done is True
        assert obs.error is not None
        assert "nonexistent_task" in obs.error


class TestAllTasksInitialise:
    """Verify every registered task can be reset without errors."""

    @pytest.mark.parametrize("task_id", list(SCENARIO_REGISTRY.keys()))
    def test_task_resets_cleanly(self, env: IncidentCommanderEnv, task_id: str):
        obs = env.reset(task_id=task_id)
        assert obs.alert is not None
        assert obs.dependency_graph is not None
        assert obs.done is False
        assert obs.error is None
        assert env.state.task_id == task_id

    @pytest.mark.parametrize("task_id", list(SCENARIO_REGISTRY.keys()))
    def test_task_state_max_steps_positive(self, env: IncidentCommanderEnv, task_id: str):
        env.reset(task_id=task_id)
        assert env.state.max_steps > 0

    def test_exactly_three_tasks_registered(self):
        assert len(SCENARIO_REGISTRY) == 3
        assert "oom_crash" in SCENARIO_REGISTRY
        assert "db_pool_exhaustion" in SCENARIO_REGISTRY
        assert "bad_deployment_cascade" in SCENARIO_REGISTRY


class TestDefaultTaskCycling:
    """reset() without task_id cycles through tasks."""

    def test_cycles_through_defaults(self, env: IncidentCommanderEnv):
        expected_order = ["oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"]
        for expected in expected_order:
            env.reset()
            assert env.state.task_id == expected


# ---------------------------------------------------------------------------
# step() tests
# ---------------------------------------------------------------------------

class TestStep:
    """Tests for environment step behaviour."""

    def test_step_without_reset_returns_error(self, env: IncidentCommanderEnv):
        action = IncidentAction(action_type="list_services")
        obs = env.step(action)
        assert obs.done is True
        assert obs.error is not None
        assert "not initialized" in obs.error.lower()

    def test_step_returns_observation(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        action = IncidentAction(action_type="list_services")
        obs = env.step(action)
        assert isinstance(obs, IncidentObservation)

    def test_step_increments_count(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        assert env.state.step_count == 0
        env.step(IncidentAction(action_type="list_services"))
        assert env.state.step_count == 1
        env.step(IncidentAction(action_type="list_services"))
        assert env.state.step_count == 2

    def test_step_records_action(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        env.step(IncidentAction(action_type="list_services"))
        assert len(env.state.actions_taken) == 1
        assert "list_services" in env.state.actions_taken[0]

    def test_step_has_reward(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        obs = env.step(IncidentAction(action_type="list_services"))
        assert isinstance(obs.reward, float)

    def test_step_list_services_returns_summaries(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        obs = env.step(IncidentAction(action_type="list_services"))
        assert obs.services_summary is not None
        assert len(obs.services_summary) > 0

    def test_step_read_logs_returns_logs(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        action = IncidentAction(
            action_type="read_logs",
            target_service="payment-service",
        )
        obs = env.step(action)
        assert obs.logs is not None
        assert len(obs.logs) > 0

    def test_step_check_metrics_returns_snapshot(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        action = IncidentAction(
            action_type="check_metrics",
            target_service="payment-service",
        )
        obs = env.step(action)
        assert obs.metrics is not None
        assert obs.metrics.service == "payment-service"

    def test_step_describe_service_returns_detail(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        action = IncidentAction(
            action_type="describe_service",
            target_service="payment-service",
        )
        obs = env.step(action)
        assert obs.service_detail is not None
        assert obs.service_detail.name == "payment-service"

    def test_step_unknown_action_returns_error(self, env: IncidentCommanderEnv):
        """Stepping with an action_type not in handlers returns error."""
        env.reset(task_id="oom_crash")
        # Build action manually to bypass Literal validation
        action = IncidentAction.model_construct(
            action_type="nonexistent_action",
            target_service=None,
            parameters={},
        )
        obs = env.step(action)
        assert obs.error is not None

    def test_step_missing_target_returns_error(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        action = IncidentAction(action_type="read_logs")
        obs = env.step(action)
        assert obs.error is not None
        assert "target_service" in obs.error.lower()

    def test_step_nonexistent_service_returns_error(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        action = IncidentAction(
            action_type="read_logs",
            target_service="nonexistent-svc",
        )
        obs = env.step(action)
        assert obs.error is not None
        assert "not found" in obs.error.lower()

    def test_step_tracks_restarts(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        action = IncidentAction(
            action_type="restart_service",
            target_service="payment-service",
            parameters={"memory_limit": "512Mi"},
        )
        env.step(action)
        assert "payment-service" in env.state.services_restarted


# ---------------------------------------------------------------------------
# Step budget and termination tests
# ---------------------------------------------------------------------------

class TestStepBudget:
    """Tests for episode termination on step budget exhaustion."""

    def test_episode_ends_at_max_steps(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        max_steps = env.state.max_steps
        assert max_steps == 15

        for i in range(max_steps):
            obs = env.step(IncidentAction(action_type="list_services"))
            if obs.done:
                break

        assert obs.done is True
        assert env.state.step_count == max_steps

    def test_step_budget_message_on_timeout(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        max_steps = env.state.max_steps

        obs = None
        for _ in range(max_steps):
            obs = env.step(IncidentAction(action_type="list_services"))
        assert obs is not None
        assert "STEP LIMIT REACHED" in obs.message

    def test_score_assigned_on_timeout(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        for _ in range(env.state.max_steps):
            env.step(IncidentAction(action_type="list_services"))
        assert env.state.current_score >= 0.0
        assert env.state.current_score <= 1.0


class TestEpisodeTermination:
    """Tests for episode termination on incident resolution."""

    def test_oom_crash_resolution(self, env: IncidentCommanderEnv):
        """Solving oom_crash: restart payment-service with more memory."""
        env.reset(task_id="oom_crash")

        # Restart payment-service with higher memory limit
        action = IncidentAction(
            action_type="restart_service",
            target_service="payment-service",
            parameters={"memory_limit": "512Mi"},
        )
        obs = env.step(action)
        assert obs.done is True
        assert "RESOLVED" in obs.message
        assert env.state.incident_resolved is True
        assert env.state.current_score > 0.0

    def test_resolved_score_in_valid_range(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        env.step(IncidentAction(
            action_type="restart_service",
            target_service="payment-service",
            parameters={"memory_limit": "512Mi"},
        ))
        assert 0.0 <= env.state.current_score <= 1.0

    def test_db_pool_resolution(self, env: IncidentCommanderEnv):
        """Solving db_pool_exhaustion: update pool config and restart order-service."""
        env.reset(task_id="db_pool_exhaustion")

        # Update DB pool size
        env.step(IncidentAction(
            action_type="update_config",
            target_service="postgres-db",
            parameters={"key": "db.pool.max_size", "value": 100},
        ))

        # Restart order-service to clear connection leak
        obs = env.step(IncidentAction(
            action_type="restart_service",
            target_service="order-service",
        ))

        assert obs.done is True
        assert env.state.incident_resolved is True

    def test_bad_deploy_resolution(self, env: IncidentCommanderEnv):
        """Solving bad_deployment_cascade: rollback, then restart dependents."""
        env.reset(task_id="bad_deployment_cascade")

        # Rollback order-service
        env.step(IncidentAction(
            action_type="rollback_deployment",
            target_service="order-service",
            parameters={"to_version": "v2.3.1"},
        ))

        # Restart starved services
        env.step(IncidentAction(
            action_type="restart_service",
            target_service="inventory-service",
        ))

        obs = env.step(IncidentAction(
            action_type="restart_service",
            target_service="notification-service",
        ))

        assert obs.done is True
        assert env.state.incident_resolved is True


class TestResetBetweenEpisodes:
    """Verify that state is properly cleaned between episodes."""

    def test_state_resets_between_episodes(self, env: IncidentCommanderEnv):
        # First episode
        env.reset(task_id="oom_crash")
        env.step(IncidentAction(action_type="list_services"))
        env.step(IncidentAction(action_type="list_services"))
        assert env.state.step_count == 2

        # Second episode
        env.reset(task_id="db_pool_exhaustion")
        assert env.state.step_count == 0
        assert env.state.task_id == "db_pool_exhaustion"
        assert len(env.state.actions_taken) == 0
        assert len(env.state.services_restarted) == 0
        assert env.state.incident_resolved is False

    def test_episode_id_changes_on_reset(self, env: IncidentCommanderEnv):
        env.reset(task_id="oom_crash")
        first_id = env.state.episode_id
        env.reset(task_id="oom_crash")
        second_id = env.state.episode_id
        assert first_id != second_id
