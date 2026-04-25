"""Backend Protocol — the typed contract every backend must implement.

The env orchestrator (`IncidentCommanderEnv`) talks to whatever backend is
plugged in through this interface. Three backends share it:

  * SimulatedBackend  — in-memory Python `Cluster` (for fast training)
  * RealBackend       — Docker Compose shell-outs (for sim-to-real demo)
  * CodeAwareBackend  — git worktree + pytest (Phase 2 roadmap)

The env never touches `Cluster` directly anymore; it asks the backend for
state via `snapshot()` and routes actions via `execute()`. This is what
makes the same trained policy run unchanged across substrates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    from incident_commander_env.models import IncidentAction, IncidentObservation
    from incident_commander_env.server.scenarios.base_scenario import BaseScenario


@dataclass
class ServiceSnapshot:
    """A typed view of one service's state — same shape across sim and real backends."""
    name: str
    health: str  # "healthy" | "degraded" | "unhealthy" | "crashed" | "restarting"
    version: str
    replicas: int
    cpu_percent: float
    memory_mb: float
    memory_limit_mb: float
    error_rate_percent: float
    request_latency_p99_ms: float = 0.0
    active_connections: int = 0
    requests_per_second: float = 0.0


@dataclass
class QuotaSnapshot:
    """Cluster-wide resource quota state."""
    cpu_used: float = 0.0
    cpu_total: float = 0.0
    cpu_utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_utilization_percent: float = 0.0


@dataclass
class BackendSnapshot:
    """The complete typed view a backend exposes to the env + reward components.

    Reward functions and rubrics read only from this snapshot, never from the
    backend's internal representation. That decoupling is what lets the same
    reward function score sim and real episodes.
    """
    services: Dict[str, ServiceSnapshot] = field(default_factory=dict)
    quota: QuotaSnapshot = field(default_factory=QuotaSnapshot)

    def healthy_service_names(self) -> List[str]:
        """Names of services currently reporting `healthy` status."""
        return [n for n, s in self.services.items() if s.health == "healthy"]

    def get_service(self, name: str) -> Optional[ServiceSnapshot]:
        return self.services.get(name)


class Backend(Protocol):
    """The Protocol every backend implements.

    Method contracts:
      * reset(scenario, seed):   prepare a fresh episode; the scenario's setup
                                 hook fires here.
      * execute(action, scenario): run one action and return the observation.
                                   Reward is computed by the env *outside* this
                                   call using `snapshot()`; backends only
                                   produce the observation + side effects.
      * snapshot():              typed read view of cluster state.
      * check_resolved(scenario): True iff scenario's resolution criteria met.
      * tick():                  advance one simulation step (sim only;
                                 real backends pass).
      * teardown():              release resources (e.g. `docker compose down`).
    """

    name: str

    def reset(self, scenario: "BaseScenario", seed: Optional[int] = None) -> None: ...
    def execute(
        self, action: "IncidentAction", scenario: "BaseScenario"
    ) -> "IncidentObservation": ...
    def snapshot(self) -> BackendSnapshot: ...
    def check_resolved(self, scenario: "BaseScenario") -> bool: ...
    def tick(self) -> None: ...
    def teardown(self) -> None: ...


__all__ = [
    "Backend",
    "BackendSnapshot",
    "ServiceSnapshot",
    "QuotaSnapshot",
]
