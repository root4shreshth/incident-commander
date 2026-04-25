"""Task 2 (Medium): Database connection pool exhaustion.

order-service has a connection leak that exhausts the postgres-db connection pool.
This causes cascading 500 errors in order-service, payment-service, and inventory-service.
frontend-bff shows generic "Service Unavailable" errors.

The agent must trace from frontend symptoms to the DB root cause.
"""

from __future__ import annotations

from typing import List, Tuple

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario, RubricCheck
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.log_generator import (
    cascading_failure_logs,
    connection_leak_logs,
    db_pool_exhaustion_logs,
    frontend_generic_error_logs,
    normal_logs,
)
from incident_commander_env.server.simulation.service import ServiceHealth


class DBPoolScenario(BaseScenario):
    task_id = "db_pool_exhaustion"
    difficulty = "medium"
    description = "Database connection pool exhaustion — cascading 5xx errors across multiple services"
    alert_message = (
        "WARNING: Elevated 5xx error rates across multiple services. "
        "Customer-facing errors reported since 14:23 UTC. "
        "Affected: frontend-bff, order-service, payment-service."
    )
    root_cause = "postgres-db connection pool exhausted by order-service connection leak"
    root_cause_keywords = ["pool", "connection", "postgres", "db_pool_exhaustion", "order-service", "leak", "exhaust"]
    relevant_services = {"postgres-db", "order-service", "payment-service", "inventory-service", "frontend-bff", "api-gateway"}
    max_steps = 25

    def __init__(self, seed=None, difficulty: float = 0.5) -> None:
        # The DB Pool family currently keeps fixed service names but accepts
        # seed + difficulty so the env's parametric `reset(seed=..., difficulty=...)`
        # threads through cleanly. `difficulty` scales the step budget so a
        # harder draw forces the agent to find the root cause faster.
        import random as _random
        rng = _random.Random(seed) if seed is not None else _random.Random(0)
        self.seed = seed
        self.difficulty_factor = float(difficulty) if difficulty is not None else 0.5
        # Pool size varies on seed (this is the value the DB starts with —
        # the `value > 50` heal threshold remains the bar to clear).
        self._pool_size_initial = rng.choice([16, 20, 24]) if seed is not None else 20
        # Step budget scales: difficulty=0 -> 38 steps, 1 -> 18 steps; default 0.5 -> 28
        self.max_steps = max(18, int(38 - 20 * max(0.0, min(1.0, self.difficulty_factor))))

    def setup(self, cluster: Cluster) -> None:
        # postgres-db: pool exhausted
        db = cluster.get_service("postgres-db")
        if db:
            db.config.db_pool_size = 20
            db.set_anomaly("db_pool_exhaustion")
            db.add_logs(db_pool_exhaustion_logs("postgres-db", pool_size=20))

        # order-service: connection leak (the actual cause)
        order = cluster.get_service("order-service")
        if order:
            order.set_anomaly("connection_leak")
            order.add_logs(connection_leak_logs("order-service"))

        # Cascading effects
        for svc_name in ("payment-service", "inventory-service"):
            svc = cluster.get_service(svc_name)
            if svc:
                svc.set_anomaly("cascade_degradation")
                svc.add_logs(cascading_failure_logs(svc_name, "postgres-db"))

        # api-gateway: shows degradation
        gw = cluster.get_service("api-gateway")
        if gw:
            gw.set_anomaly("cascade_degradation")
            gw.add_logs(cascading_failure_logs("api-gateway", "order-service"))

        # frontend-bff: generic errors (misleading)
        fe = cluster.get_service("frontend-bff")
        if fe:
            fe.set_anomaly("cascade_degradation")
            fe.add_logs(frontend_generic_error_logs("frontend-bff"))

        # Healthy services get normal logs
        for name in ("auth-service", "user-service", "notification-service"):
            svc = cluster.get_service(name)
            if svc:
                svc.add_logs(normal_logs(name, count=6))

    def check_resolved(self, cluster: Cluster) -> bool:
        db = cluster.get_service("postgres-db")
        order = cluster.get_service("order-service")
        if not db or not order:
            return False
        return (
            db.health == ServiceHealth.HEALTHY
            and order.health == ServiceHealth.HEALTHY
            and not db.has_anomaly("db_pool_exhaustion")
            and not order.has_anomaly("connection_leak")
        )

    def get_rubric(self) -> List[Tuple[str, RubricCheck, float]]:
        def investigated_frontend(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.target_service == "frontend-bff"
                and a.action_type in ("read_logs", "check_metrics", "describe_service")
                for a in actions
            )

        def traced_to_order(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.target_service == "order-service"
                and a.action_type in ("read_logs", "check_metrics", "describe_service")
                for a in actions
            )

        def identified_db_root(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.target_service == "postgres-db"
                and a.action_type in ("read_logs", "check_metrics", "describe_service")
                for a in actions
            )

        def read_db_logs(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "read_logs" and a.target_service == "postgres-db"
                for a in actions
            )

        def fixed_pool(actions: List[ActionRecord], cluster: Cluster) -> bool:
            # Check if pool size was increased or DB was restarted/reconfigured
            db = cluster.get_service("postgres-db")
            if not db:
                return False
            pool_updated = any(
                a.action_type == "update_config"
                and a.target_service == "postgres-db"
                and "pool" in str(a.parameters.get("key", "")).lower()
                for a in actions
            )
            db_restarted = any(
                a.action_type == "restart_service" and a.target_service == "postgres-db"
                for a in actions
            )
            return pool_updated or db_restarted or db.health == ServiceHealth.HEALTHY

        def restarted_order(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "restart_service" and a.target_service == "order-service"
                for a in actions
            )

        def resolved_incident(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(a.action_type == "resolve_incident" for a in actions)

        return [
            ("Investigated frontend-bff symptoms", investigated_frontend, 0.10),
            ("Traced to order-service as intermediate", traced_to_order, 0.15),
            ("Identified postgres-db as root cause layer", identified_db_root, 0.15),
            ("Read DB logs showing pool exhaustion", read_db_logs, 0.10),
            ("Fixed pool size or restarted DB connections", fixed_pool, 0.20),
            ("Restarted order-service to clear leaked connections", restarted_order, 0.20),
            ("Resolved incident with root cause", resolved_incident, 0.10),
        ]

    def compute_penalties(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        penalty = 0.0
        uninvolved = {"auth-service", "user-service", "notification-service"}
        for a in actions:
            if a.action_type == "restart_service" and a.target_service in uninvolved:
                penalty -= 0.05
        return penalty

    def is_correct_op(self, action, cluster):
        """DB pool exhaustion is fixed by raising postgres pool size sufficiently
        AND restarting the leaking service (order-service) to clear stale conns.

        Either move counts as a "correct op" — both are needed for full
        resolution; reward fires once per move.
        """
        if action.action_type == "update_config":
            if action.target_service != "postgres-db":
                return False
            if action.parameters.get("key") != "db.pool.max_size":
                return False
            try:
                return int(action.parameters.get("value", 0)) > 50
            except (ValueError, TypeError):
                return False
        if action.action_type == "restart_service":
            return action.target_service == "order-service"
        return False

    def on_config_update(self, cluster: Cluster, target_service: str, key: str, value):
        """Raising the postgres pool size to a sufficient value resolves the exhaustion."""
        if target_service != "postgres-db":
            return False
        if key != "db.pool.max_size":
            return False
        if not isinstance(value, (int, float)):
            return False
        # Pool must be raised meaningfully above the leak draw rate. 20 is the original;
        # require strictly larger than 50 to count as the fix (anti-cheat: no off-by-one heal).
        return int(value) > 50
