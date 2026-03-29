"""IncidentCommanderEnv — the core OpenEnv environment implementation.

Wires together the cluster simulation, scenarios, action handlers, and grading.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Set

from incident_commander_env.models import (
    ActionRecord,
    IncidentAction,
    IncidentObservation,
    IncidentState,
)
from incident_commander_env.server.actions.handlers import ACTION_HANDLERS
from incident_commander_env.server.grading.grader import IncidentGrader
from incident_commander_env.server.grading.reward import compute_step_reward
from incident_commander_env.server.scenarios import SCENARIO_REGISTRY
from incident_commander_env.server.scenarios.base_scenario import BaseScenario
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.service import ServiceHealth


# Which services are relevant to each scenario (for reward computation)
SCENARIO_RELEVANT_SERVICES: Dict[str, Set[str]] = {
    "oom_crash": {"payment-service"},
    "db_pool_exhaustion": {
        "postgres-db", "order-service", "payment-service",
        "inventory-service", "frontend-bff", "api-gateway",
    },
    "bad_deployment_cascade": {
        "order-service", "inventory-service", "notification-service",
        "api-gateway", "frontend-bff",
    },
}

# Default task rotation for when no task_id is provided
DEFAULT_TASKS = ["oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"]


class IncidentCommanderEnv:
    """OpenEnv environment for SRE incident response.

    Implements the OpenEnv spec: reset(), step(), state property.
    Manages a simulated microservices cluster, injects incident scenarios,
    and grades agent performance via deterministic rubrics.
    """

    def __init__(self) -> None:
        self._state = IncidentState()
        self._cluster: Optional[Cluster] = None
        self._scenario: Optional[BaseScenario] = None
        self._action_history: List[ActionRecord] = []
        self._grader = IncidentGrader()
        self._task_index = 0

    def reset(self, task_id: Optional[str] = None, **kwargs: Any) -> IncidentObservation:
        """Initialize a new episode with the specified scenario.

        Args:
            task_id: Scenario to run. One of: oom_crash, db_pool_exhaustion, bad_deployment_cascade.
                     If not provided, cycles through tasks in order.
        """
        # Determine task
        if task_id is None:
            task_id = kwargs.get("task_id", DEFAULT_TASKS[self._task_index % len(DEFAULT_TASKS)])
            self._task_index += 1

        if task_id not in SCENARIO_REGISTRY:
            return IncidentObservation(
                message=f"Unknown task_id: {task_id}. Available: {list(SCENARIO_REGISTRY.keys())}",
                error=f"Invalid task_id: {task_id}",
                done=True,
            )

        # Create scenario and cluster
        self._scenario = SCENARIO_REGISTRY[task_id]()
        self._cluster = Cluster()
        self._cluster.initialize()

        # Inject the fault
        self._scenario.setup(self._cluster)

        # Reset state
        self._action_history = []
        self._state = IncidentState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            max_steps=self._scenario.max_steps,
        )

        # Build initial observation
        return IncidentObservation(
            message=(
                f"INCIDENT ALERT\n"
                f"{'=' * 60}\n"
                f"{self._scenario.alert_message}\n"
                f"{'=' * 60}\n\n"
                f"You are the on-call SRE. Diagnose the issue and resolve it.\n"
                f"Task: {self._scenario.description}\n"
                f"Difficulty: {self._scenario.difficulty}\n"
                f"Step budget: {self._scenario.max_steps}\n\n"
                f"Available actions: list_services, describe_service, read_logs, "
                f"check_metrics, restart_service, scale_service, rollback_deployment, "
                f"run_diagnostic, update_config, resolve_incident"
            ),
            alert=self._scenario.alert_message,
            dependency_graph=self._cluster.dependency_graph.to_dict(),
        )

    def step(self, action: IncidentAction) -> IncidentObservation:
        """Execute an action and return the resulting observation."""
        if not self._cluster or not self._scenario:
            return IncidentObservation(
                message="Error: environment not initialized. Call reset() first.",
                error="Environment not initialized",
                done=True,
            )

        self._state.step_count += 1

        # Record action
        record = ActionRecord(
            step=self._state.step_count,
            action_type=action.action_type,
            target_service=action.target_service,
            parameters=action.parameters,
        )
        self._action_history.append(record)
        self._state.actions_taken.append(
            f"{action.action_type}({action.target_service or ''})"
        )

        # Track restarts
        if action.action_type == "restart_service" and action.target_service:
            self._state.services_restarted.append(action.target_service)

        # Dispatch to handler
        handler = ACTION_HANDLERS.get(action.action_type)
        if not handler:
            return IncidentObservation(
                message=f"Unknown action: {action.action_type}",
                error=f"Invalid action_type: {action.action_type}",
            )

        observation = handler(action, self._cluster, self._scenario)

        # Compute per-step reward
        relevant = SCENARIO_RELEVANT_SERVICES.get(self._state.task_id, set())
        healthy = {
            name for name, svc in self._cluster.services.items()
            if svc.health == ServiceHealth.HEALTHY and name not in relevant
        }
        step_reward = compute_step_reward(
            action=record,
            step=self._state.step_count,
            previous_actions=self._action_history[:-1],
            relevant_services=relevant,
            healthy_services=healthy,
        )
        observation.reward = round(step_reward, 4)

        # Check termination
        if self._scenario.check_resolved(self._cluster):
            self._state.incident_resolved = True
            observation.done = True
            # Compute final score
            final_score = self._grader.grade(
                self._scenario, self._action_history, self._cluster
            )
            self._state.current_score = round(final_score, 4)
            observation.message += (
                f"\n\nINCIDENT RESOLVED\n"
                f"Final score: {final_score:.2f}/1.00\n"
                f"Steps used: {self._state.step_count}/{self._scenario.max_steps}"
            )
        elif self._state.step_count >= self._scenario.max_steps:
            observation.done = True
            # Grade even on timeout
            final_score = self._grader.grade(
                self._scenario, self._action_history, self._cluster
            )
            self._state.current_score = round(final_score, 4)
            observation.message += (
                f"\n\nSTEP LIMIT REACHED — Incident unresolved.\n"
                f"Partial score: {final_score:.2f}/1.00\n"
                f"Steps used: {self._state.step_count}/{self._scenario.max_steps}"
            )

        # Advance simulation
        self._cluster.tick()

        return observation

    @property
    def state(self) -> IncidentState:
        """Return current episode state."""
        return self._state

    def get_grade_details(self) -> Dict[str, Any]:
        """Return detailed grading breakdown (for debugging)."""
        if not self._scenario or not self._cluster:
            return {"error": "No active episode"}
        return self._grader.grade_details(
            self._scenario, self._action_history, self._cluster
        )
