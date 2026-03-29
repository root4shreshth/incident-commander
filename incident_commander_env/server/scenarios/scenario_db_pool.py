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
    max_steps = 25

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
