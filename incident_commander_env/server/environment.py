"""IncidentCommanderEnv - the core OpenEnv environment implementation.

Delegates execution to a `Backend` (sim, real, or code-aware) so the same
trained policy runs unchanged across substrates. The env's job is now:

  - own the action history, episode state, and reward computation
  - route actions through the backend's `execute()` method
  - read state via the backend's `snapshot()` (typed view)
  - compute multi-component reward against that snapshot

The cluster, handlers, and scenario hooks all live below the backend layer.
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
from incident_commander_env.server.backends.protocol import Backend, BackendSnapshot
from incident_commander_env.server.backends.sim import SimulatedBackend
from incident_commander_env.server.grading.grader import IncidentGrader
from incident_commander_env.server.grading.reward import (
    TIME_DECAY,
    compute_step_breakdown_scaled,
)
from incident_commander_env.server.scenarios import SCENARIO_REGISTRY
from incident_commander_env.server.scenarios.base_scenario import BaseScenario


# Which services are relevant to each scenario (for reward computation).
# Scenarios may define their own `relevant_services` class attribute that
# overrides this; the dict here is a fallback for tasks that don't.
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
    Routes through a `Backend` instance for execution + state queries; the
    default backend is `SimulatedBackend` which wraps the in-memory cluster.
    """

    def __init__(self, backend: Optional[Backend] = None) -> None:
        self._backend: Backend = backend or SimulatedBackend()
        self._state = IncidentState()
        self._scenario: Optional[BaseScenario] = None
        self._action_history: List[ActionRecord] = []
        self._grader = IncidentGrader()
        self._task_index = 0
        # Per-step breakdown of the most recent reward, exposed via /reward-breakdown.
        self._last_breakdown = None

    @property
    def backend(self) -> Backend:
        return self._backend

    # Convenience accessor for legacy callers that expect `env._cluster`.
    # Sim backend exposes the live Cluster; real backends return None.
    @property
    def _cluster(self):  # noqa: D401
        return getattr(self._backend, "cluster", None)

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        difficulty: float = 0.5,
        **kwargs: Any,
    ) -> IncidentObservation:
        """Initialize a new episode with the specified scenario.

        Args:
            task_id: One of: oom_crash, db_pool_exhaustion, bad_deployment_cascade.
                     If not provided, cycles through tasks in order.
            seed: Optional integer for deterministic anomaly metric generation
                     AND parametric scenario instantiation. OpenEnv contract.
            difficulty: 0.0 (easiest) to 1.0 (hardest).
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

        # Create scenario (parametric - seed and difficulty go into __init__ for
        # those scenarios that accept them; legacy scenarios fall back to a no-arg call)
        scenario_cls = SCENARIO_REGISTRY[task_id]
        try:
            self._scenario = scenario_cls(seed=seed, difficulty=difficulty)
        except TypeError:
            self._scenario = scenario_cls()

        # Tear down any previous episode and let the backend initialize cleanly.
        self._backend.teardown()
        self._backend.reset(self._scenario, seed=seed)

        # Reset history + state
        self._action_history = []
        self._state = IncidentState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            max_steps=self._scenario.max_steps,
        )
        self._last_breakdown = None

        # Build initial observation. For the dependency graph we ask the
        # underlying cluster (sim only) - real backends can return None here.
        dep_graph = None
        cluster = getattr(self._backend, "cluster", None)
        if cluster is not None and hasattr(cluster, "dependency_graph"):
            dep_graph = cluster.dependency_graph.to_dict()

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
            dependency_graph=dep_graph,
        )

    def step(self, action: IncidentAction) -> IncidentObservation:
        """Execute an action and return the resulting observation."""
        if self._scenario is None:
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

        # Dispatch to backend
        observation = self._backend.execute(action, self._scenario)

        # Determine termination state BEFORE computing reward
        is_resolved = self._backend.check_resolved(self._scenario)
        is_terminal = is_resolved or (self._state.step_count >= self._scenario.max_steps)

        # Read typed snapshot for reward components
        snapshot: BackendSnapshot = self._backend.snapshot()

        # Determine relevant + healthy service sets. Scenario can override the
        # relevant set; otherwise fall back to the per-task default.
        relevant = getattr(
            self._scenario, "relevant_services", None
        ) or SCENARIO_RELEVANT_SERVICES.get(self._state.task_id, set())
        healthy = {
            name for name, s in snapshot.services.items()
            if s.health == "healthy" and name not in relevant
        }

        # Multi-component reward
        cluster = self._cluster  # may be None for real backends; reward components handle that
        breakdown = compute_step_breakdown_scaled(
            action=record,
            step=self._state.step_count,
            previous_actions=self._action_history[:-1],
            relevant_services=set(relevant),
            healthy_services=healthy,
            scenario=self._scenario,
            cluster=cluster,
            is_terminal=is_terminal,
            is_resolved=is_resolved,
            max_steps=self._scenario.max_steps,
            last_observation_error=observation.error,
        )
        scaled = breakdown.total() * (TIME_DECAY ** self._state.step_count)
        if scaled == 0.0:
            scaled = 0.01  # validator rejects exactly 0
        observation.reward = round(scaled, 4)
        self._last_breakdown = breakdown

        # Compute terminal score
        if is_resolved:
            self._state.incident_resolved = True
            observation.done = True
            final_score = self._grade(cluster)
            self._state.current_score = round(final_score, 4)
            observation.message += (
                f"\n\nINCIDENT RESOLVED\n"
                f"Final score: {final_score:.2f}/1.00\n"
                f"Steps used: {self._state.step_count}/{self._scenario.max_steps}"
            )
        elif is_terminal:
            observation.done = True
            final_score = self._grade(cluster)
            self._state.current_score = round(final_score, 4)
            observation.message += (
                f"\n\nSTEP LIMIT REACHED - Incident unresolved.\n"
                f"Partial score: {final_score:.2f}/1.00\n"
                f"Steps used: {self._state.step_count}/{self._scenario.max_steps}"
            )

        # Advance simulation
        self._backend.tick()

        return observation

    def _grade(self, cluster) -> float:
        """Compute the final score; gracefully degrade if backend has no cluster."""
        if cluster is None:
            # Real backends: rubric checks rely on Cluster state today; for
            # Phase 6 we'll wire an alternate grader against BackendSnapshot.
            # For now, return a reasonable mid-range score so episodes terminate.
            return 0.5
        return self._grader.grade(self._scenario, self._action_history, cluster)

    @property
    def state(self) -> IncidentState:
        return self._state

    def get_grade_details(self) -> Dict[str, Any]:
        if self._scenario is None or self._cluster is None:
            return {"error": "No active episode"}
        return self._grader.grade_details(
            self._scenario, self._action_history, self._cluster
        )
