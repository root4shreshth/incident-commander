"""Service dependency DAG for the simulated microservices cluster."""

from __future__ import annotations

from typing import Dict, List, Set


class DependencyGraph:
    """Directed acyclic graph of service dependencies.

    An edge from A -> B means A depends on B (B is upstream of A).
    """

    def __init__(self) -> None:
        self._depends_on: Dict[str, List[str]] = {}

    def add_service(self, name: str, depends_on: List[str] | None = None) -> None:
        self._depends_on[name] = depends_on or []

    def get_dependencies(self, service: str) -> List[str]:
        """What does this service depend on? (upstream)"""
        return list(self._depends_on.get(service, []))

    def get_dependents(self, service: str) -> List[str]:
        """What services depend on this one? (downstream)"""
        return [s for s, deps in self._depends_on.items() if service in deps]

    def get_cascade_path(self, failed_service: str) -> List[str]:
        """BFS to find all services affected by a failure (downstream cascade)."""
        affected: List[str] = []
        visited: Set[str] = {failed_service}
        queue = [failed_service]

        while queue:
            current = queue.pop(0)
            for dependent in self.get_dependents(current):
                if dependent not in visited:
                    visited.add(dependent)
                    affected.append(dependent)
                    queue.append(dependent)

        return affected

    def to_dict(self) -> Dict[str, List[str]]:
        return {name: list(deps) for name, deps in self._depends_on.items()}

    @property
    def services(self) -> List[str]:
        return list(self._depends_on.keys())


def create_default_graph() -> DependencyGraph:
    """Create the 8-service production topology.

    Topology:
        api-gateway -> order-service -> payment-service
                                     -> inventory-service
                    -> user-service  -> auth-service
                    -> notification-service
                    -> frontend-bff

        order-service, user-service, payment-service, inventory-service -> postgres-db
    """
    g = DependencyGraph()

    g.add_service("postgres-db", depends_on=[])
    g.add_service("auth-service", depends_on=[])
    g.add_service("payment-service", depends_on=["postgres-db"])
    g.add_service("inventory-service", depends_on=["postgres-db"])
    g.add_service("notification-service", depends_on=[])
    g.add_service("user-service", depends_on=["auth-service", "postgres-db"])
    g.add_service("order-service", depends_on=["payment-service", "inventory-service", "postgres-db"])
    g.add_service("api-gateway", depends_on=["order-service", "user-service", "notification-service"])
    g.add_service("frontend-bff", depends_on=["api-gateway"])

    return g
