"""Cluster orchestrator — holds all services, dependency graph, and simulation state."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from incident_commander_env.server.simulation.dependency_graph import (
    DependencyGraph,
    create_default_graph,
)
from incident_commander_env.server.simulation.metrics_engine import apply_anomalies
from incident_commander_env.server.simulation.service import (
    Service,
    ServiceConfig,
    ServiceHealth,
)


DEFAULT_SERVICES = [
    ServiceConfig(name="postgres-db", version="15.4", replicas=1, memory_limit="1024Mi", cpu_limit="1000m", db_pool_size=20, port=5432),
    ServiceConfig(name="auth-service", version="v1.2.0", replicas=2, memory_limit="256Mi", cpu_limit="250m", port=8081),
    ServiceConfig(name="payment-service", version="v3.1.0", replicas=2, memory_limit="256Mi", cpu_limit="500m", port=8082),
    ServiceConfig(name="inventory-service", version="v2.0.1", replicas=2, memory_limit="512Mi", cpu_limit="500m", port=8083),
    ServiceConfig(name="notification-service", version="v1.5.0", replicas=2, memory_limit="256Mi", cpu_limit="250m", port=8084),
    ServiceConfig(name="user-service", version="v2.3.0", replicas=2, memory_limit="256Mi", cpu_limit="250m", port=8085),
    ServiceConfig(name="order-service", version="v2.3.1", replicas=2, memory_limit="512Mi", cpu_limit="500m", port=8086),
    ServiceConfig(name="api-gateway", version="v1.8.0", replicas=3, memory_limit="512Mi", cpu_limit="500m", port=8080),
    ServiceConfig(name="frontend-bff", version="v1.3.0", replicas=2, memory_limit="256Mi", cpu_limit="250m", port=3000),
]


class ResourceQuota:
    """Simulated cluster resource quota."""

    def __init__(self, cpu_total: float = 8000.0, memory_total_mb: float = 8192.0) -> None:
        self.cpu_total = cpu_total
        self.memory_total_mb = memory_total_mb
        self.cpu_used: float = 0.0
        self.memory_used_mb: float = 0.0

    @property
    def cpu_utilization_percent(self) -> float:
        return (self.cpu_used / self.cpu_total) * 100 if self.cpu_total > 0 else 0.0

    @property
    def memory_utilization_percent(self) -> float:
        return (self.memory_used_mb / self.memory_total_mb) * 100 if self.memory_total_mb > 0 else 0.0

    def update_from_services(self, services: Dict[str, Service]) -> None:
        self.cpu_used = sum(s.metrics.cpu_percent * s.config.replicas for s in services.values())
        self.memory_used_mb = sum(s.metrics.memory_mb * s.config.replicas for s in services.values())

    def to_dict(self) -> Dict[str, str]:
        return {
            "cpu": f"{self.cpu_used:.0f}m / {self.cpu_total:.0f}m ({self.cpu_utilization_percent:.0f}%)",
            "memory": f"{self.memory_used_mb:.0f}MB / {self.memory_total_mb:.0f}MB ({self.memory_utilization_percent:.0f}%)",
        }


class Cluster:
    """Top-level container for the simulated infrastructure."""

    def __init__(self) -> None:
        self.services: Dict[str, Service] = {}
        self.dependency_graph: DependencyGraph = create_default_graph()
        self.resource_quota = ResourceQuota()
        self._tick_count = 0

    def initialize(self, service_configs: List[ServiceConfig] | None = None) -> None:
        """Create all services with default healthy state."""
        configs = service_configs or DEFAULT_SERVICES
        self.services = {}
        for cfg in configs:
            svc = Service(cfg)
            self.services[svc.name] = svc

    def get_service(self, name: str) -> Optional[Service]:
        return self.services.get(name)

    def list_services(self) -> List[Dict[str, Any]]:
        return [svc.summary_dict() for svc in self.services.values()]

    def describe_service(self, name: str) -> Optional[Dict[str, Any]]:
        svc = self.get_service(name)
        if not svc:
            return None
        deps = self.dependency_graph.get_dependencies(name)
        dependents = self.dependency_graph.get_dependents(name)
        return svc.detail_dict(deps, dependents)

    def tick(self) -> None:
        """Advance simulation by one step. Recompute metrics for anomalous services."""
        self._tick_count += 1
        for svc in self.services.values():
            apply_anomalies(svc)
        self.resource_quota.update_from_services(self.services)
