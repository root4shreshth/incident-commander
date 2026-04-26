"""SimulatedBackend - wraps the existing in-memory `Cluster` simulation.

This is what the env runs by default and what the Colab training pipeline
uses. Fast, fully reproducible with a seed, no external dependencies.

Implementation note: this backend keeps the live `Cluster` accessible as
`self.cluster` for backwards compatibility with action handlers and scenario
hooks that haven't been converted to consume `BackendSnapshot` yet. The
`snapshot()` method is the canonical "view" surface that reward components
and rubrics use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from incident_commander_env.models import IncidentAction, IncidentObservation
from incident_commander_env.server.actions.handlers import ACTION_HANDLERS
from incident_commander_env.server.backends.protocol import (
    BackendSnapshot,
    QuotaSnapshot,
    ServiceSnapshot,
)
from incident_commander_env.server.simulation.cluster import Cluster

if TYPE_CHECKING:
    from incident_commander_env.server.scenarios.base_scenario import BaseScenario


class SimulatedBackend:
    """In-memory Python simulation backend."""

    name = "sim"

    def __init__(self) -> None:
        self.cluster: Optional[Cluster] = None

    def reset(self, scenario: "BaseScenario", seed: Optional[int] = None) -> None:
        self.cluster = Cluster(seed=seed)
        self.cluster.initialize(seed=seed)
        scenario.setup(self.cluster)

    def execute(
        self, action: IncidentAction, scenario: "BaseScenario"
    ) -> IncidentObservation:
        if self.cluster is None:
            return IncidentObservation(
                message="Error: backend not reset. Call reset() first.",
                error="Backend not initialized",
                done=True,
            )
        handler = ACTION_HANDLERS.get(action.action_type)
        if handler is None:
            return IncidentObservation(
                message=f"Unknown action: {action.action_type}",
                error=f"Invalid action_type: {action.action_type}",
            )
        return handler(action, self.cluster, scenario)

    def snapshot(self) -> BackendSnapshot:
        if self.cluster is None:
            return BackendSnapshot()
        services = {}
        for name, svc in self.cluster.services.items():
            m = svc.metrics
            services[name] = ServiceSnapshot(
                name=name,
                health=svc.health.value,
                version=svc.config.version,
                replicas=svc.config.replicas,
                cpu_percent=round(m.cpu_percent, 1),
                memory_mb=round(m.memory_mb, 1),
                memory_limit_mb=round(m.memory_limit_mb, 1),
                error_rate_percent=round(m.error_rate_percent, 2),
                request_latency_p99_ms=round(m.request_latency_p99_ms, 1),
                active_connections=m.active_connections,
                requests_per_second=round(m.requests_per_second, 1),
            )
        rq = self.cluster.resource_quota
        quota = QuotaSnapshot(
            cpu_used=rq.cpu_used,
            cpu_total=rq.cpu_total,
            cpu_utilization_percent=rq.cpu_utilization_percent,
            memory_used_mb=rq.memory_used_mb,
            memory_total_mb=rq.memory_total_mb,
            memory_utilization_percent=rq.memory_utilization_percent,
        )
        return BackendSnapshot(services=services, quota=quota)

    def check_resolved(self, scenario: "BaseScenario") -> bool:
        if self.cluster is None:
            return False
        return scenario.check_resolved(self.cluster)

    def tick(self) -> None:
        if self.cluster is not None:
            self.cluster.tick()

    def teardown(self) -> None:
        self.cluster = None


__all__ = ["SimulatedBackend"]
