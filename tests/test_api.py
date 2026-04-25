"""Tests for FastAPI endpoints using Starlette TestClient.

Covers:
- GET /health returns {"status": "ok"}
- GET /tasks lists all 3 tasks
- POST /reset initialises a new episode
- POST /step executes actions and returns observation/reward/done/info
- GET /state returns current episode state
"""

import pytest
from starlette.testclient import TestClient

from incident_commander_env.server.app import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Synchronous test client for the FastAPI app.

    Each test gets a fresh client; however, the app-level `env` is module
    scoped in app.py, so we always call /reset before tests that depend
    on episode state.
    """
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "ok"}

    def test_health_is_always_available(self, client: TestClient):
        # Should succeed even before any reset
        resp = client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------

class TestTasksEndpoint:

    def test_tasks_returns_all_three(self, client: TestClient):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        tasks = data["tasks"]
        assert "oom_crash" in tasks
        assert "db_pool_exhaustion" in tasks
        assert "bad_deployment_cascade" in tasks

    def test_tasks_have_required_fields(self, client: TestClient):
        resp = client.get("/tasks")
        tasks = resp.json()["tasks"]
        for task_id, info in tasks.items():
            assert "difficulty" in info
            assert "description" in info
            assert "max_steps" in info
            assert info["max_steps"] > 0

    def test_tasks_difficulty_levels(self, client: TestClient):
        resp = client.get("/tasks")
        tasks = resp.json()["tasks"]
        assert tasks["oom_crash"]["difficulty"] == "easy"
        assert tasks["db_pool_exhaustion"]["difficulty"] == "medium"
        assert tasks["bad_deployment_cascade"]["difficulty"] == "hard"


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------

class TestResetEndpoint:

    def test_reset_default(self, client: TestClient):
        resp = client.post("/reset", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_reset_specific_task(self, client: TestClient):
        resp = client.post("/reset", json={"task_id": "oom_crash"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["task_id"] == "oom_crash"
        assert data["done"] is False

    def test_reset_observation_has_alert(self, client: TestClient):
        resp = client.post("/reset", json={"task_id": "oom_crash"})
        obs = resp.json()["observation"]
        assert obs["alert"] is not None
        assert len(obs["alert"]) > 0

    def test_reset_observation_has_dependency_graph(self, client: TestClient):
        resp = client.post("/reset", json={"task_id": "oom_crash"})
        obs = resp.json()["observation"]
        assert obs["dependency_graph"] is not None
        assert isinstance(obs["dependency_graph"], dict)

    def test_reset_info_has_episode_metadata(self, client: TestClient):
        resp = client.post("/reset", json={"task_id": "oom_crash"})
        info = resp.json()["info"]
        assert "episode_id" in info
        assert "max_steps" in info
        assert info["max_steps"] > 0
        assert info["task_id"] == "oom_crash"

    def test_reset_reward_is_near_zero(self, client: TestClient):
        # /reset returns 0.01 (not 0.0) because the OpenEnv validator rejects
        # exactly 0.0 — it expects rewards in the strict (0, 1) range. Anything
        # in [0, 0.05] indicates "no real reward yet".
        resp = client.post("/reset", json={"task_id": "oom_crash"})
        assert 0.0 <= resp.json()["reward"] < 0.05

    @pytest.mark.parametrize("task_id", ["oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"])
    def test_reset_all_tasks(self, client: TestClient, task_id: str):
        resp = client.post("/reset", json={"task_id": task_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["task_id"] == task_id
        assert data["done"] is False

    def test_reset_unknown_task(self, client: TestClient):
        resp = client.post("/reset", json={"task_id": "nonexistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        obs = data["observation"]
        assert obs.get("error") is not None


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------

class TestStepEndpoint:

    def test_step_list_services(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        resp = client.post("/step", json={"action_type": "list_services"})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_is_float(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        resp = client.post("/step", json={"action_type": "list_services"})
        assert isinstance(resp.json()["reward"], float)

    def test_step_info_has_step_count(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        resp = client.post("/step", json={"action_type": "list_services"})
        info = resp.json()["info"]
        assert "step_count" in info
        assert info["step_count"] == 1

    def test_step_increments(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        client.post("/step", json={"action_type": "list_services"})
        resp = client.post("/step", json={"action_type": "list_services"})
        assert resp.json()["info"]["step_count"] == 2

    def test_step_with_target_service(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        resp = client.post("/step", json={
            "action_type": "read_logs",
            "target_service": "payment-service",
        })
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs.get("logs") is not None

    def test_step_check_metrics(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        resp = client.post("/step", json={
            "action_type": "check_metrics",
            "target_service": "payment-service",
        })
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs.get("metrics") is not None

    def test_step_describe_service(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        resp = client.post("/step", json={
            "action_type": "describe_service",
            "target_service": "payment-service",
        })
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs.get("service_detail") is not None

    def test_step_resolution_returns_final_score(self, client: TestClient):
        """Resolving an incident should return final_score in info."""
        client.post("/reset", json={"task_id": "oom_crash"})
        resp = client.post("/step", json={
            "action_type": "restart_service",
            "target_service": "payment-service",
            "parameters": {"memory_limit": "512Mi"},
        })
        data = resp.json()
        assert data["done"] is True
        assert "final_score" in data["info"]
        assert 0.0 <= data["info"]["final_score"] <= 1.0

    def test_step_resolution_returns_grade_details(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        resp = client.post("/step", json={
            "action_type": "restart_service",
            "target_service": "payment-service",
            "parameters": {"memory_limit": "512Mi"},
        })
        data = resp.json()
        assert "grade_details" in data["info"]
        details = data["info"]["grade_details"]
        assert "criteria" in details
        assert "penalties" in details
        assert "final_score" in details

    def test_step_missing_target_returns_error(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        resp = client.post("/step", json={"action_type": "read_logs"})
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs.get("error") is not None


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------

class TestStateEndpoint:

    def test_state_returns_episode_fields(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "episode_id" in data
        assert "step_count" in data
        assert "task_id" in data
        assert "incident_resolved" in data
        assert "max_steps" in data

    def test_state_reflects_steps(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        client.post("/step", json={"action_type": "list_services"})
        resp = client.get("/state")
        data = resp.json()
        assert data["step_count"] == 1
        assert len(data["actions_taken"]) == 1

    def test_state_reflects_resolution(self, client: TestClient):
        client.post("/reset", json={"task_id": "oom_crash"})
        client.post("/step", json={
            "action_type": "restart_service",
            "target_service": "payment-service",
            "parameters": {"memory_limit": "512Mi"},
        })
        resp = client.get("/state")
        data = resp.json()
        assert data["incident_resolved"] is True
        assert data["current_score"] > 0.0

    def test_state_before_reset(self, client: TestClient):
        """State should be accessible even before reset (default empty state)."""
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "step_count" in data


# ---------------------------------------------------------------------------
# Full episode lifecycle via API
# ---------------------------------------------------------------------------

class TestAPILifecycle:
    """End-to-end lifecycle tests through the API layer."""

    def test_full_oom_episode(self, client: TestClient):
        # Reset
        reset_resp = client.post("/reset", json={"task_id": "oom_crash"})
        assert reset_resp.json()["done"] is False

        # Diagnose
        client.post("/step", json={"action_type": "list_services"})
        client.post("/step", json={
            "action_type": "read_logs",
            "target_service": "payment-service",
        })
        client.post("/step", json={
            "action_type": "check_metrics",
            "target_service": "payment-service",
        })

        # Fix
        fix_resp = client.post("/step", json={
            "action_type": "restart_service",
            "target_service": "payment-service",
            "parameters": {"memory_limit": "512Mi"},
        })

        data = fix_resp.json()
        assert data["done"] is True
        assert data["info"]["final_score"] > 0.0

        # Verify state
        state = client.get("/state").json()
        assert state["incident_resolved"] is True
        assert state["step_count"] == 4
