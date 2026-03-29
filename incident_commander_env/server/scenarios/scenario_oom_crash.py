"""Task 1 (Easy): Single service OOM crash.

payment-service crashes due to OutOfMemoryError. Logs clearly show the error.
Agent must identify the service, read logs, and restart with higher memory limit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario, RubricCheck
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.log_generator import (
    normal_logs,
    oom_crash_logs,
)
from incident_commander_env.server.simulation.metrics_engine import apply_oom_anomaly
from incident_commander_env.server.simulation.service import ServiceHealth

TARGET_SERVICE = "payment-service"


class OOMCrashScenario(BaseScenario):
    task_id = "oom_crash"
    difficulty = "easy"
    description = "Single service OOM crash — payment-service is down due to OutOfMemoryError"
    alert_message = (
        "CRITICAL: payment-service is DOWN. Multiple health check failures detected. "
        "PagerDuty alert triggered at 03:42 UTC. Immediate investigation required."
    )
    root_cause = "payment-service OOM killed due to insufficient memory limit (256Mi)"
    max_steps = 15

    def setup(self, cluster: Cluster) -> None:
        svc = cluster.get_service(TARGET_SERVICE)
        if not svc:
            return

        # Set memory limit low to trigger OOM
        svc.config.memory_limit = "256Mi"
        svc.metrics.memory_limit_mb = 256.0

        # Inject OOM anomaly
        svc.set_anomaly("oom")
        apply_oom_anomaly(svc)

        # Add crash logs
        svc.add_logs(oom_crash_logs(TARGET_SERVICE, memory_limit_mb=256))

        # Add normal logs to other services so they look healthy
        for name, other_svc in cluster.services.items():
            if name != TARGET_SERVICE:
                other_svc.add_logs(normal_logs(name, count=8))

    def check_resolved(self, cluster: Cluster) -> bool:
        svc = cluster.get_service(TARGET_SERVICE)
        if not svc:
            return False
        return (
            svc.health == ServiceHealth.HEALTHY
            and svc.config.memory_limit_mb() > 256.0
        )

    def get_rubric(self) -> List[Tuple[str, RubricCheck, float]]:
        def identified_service(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(a.target_service == TARGET_SERVICE for a in actions)

        def read_logs(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "read_logs" and a.target_service == TARGET_SERVICE
                for a in actions
            )

        def diagnosed_oom(actions: List[ActionRecord], cluster: Cluster) -> bool:
            # If they read logs or checked metrics on payment-service, they saw the OOM
            return any(
                a.action_type in ("read_logs", "check_metrics", "describe_service")
                and a.target_service == TARGET_SERVICE
                for a in actions
            )

        def applied_fix(actions: List[ActionRecord], cluster: Cluster) -> bool:
            svc = cluster.get_service(TARGET_SERVICE)
            if not svc:
                return False
            return (
                svc.health == ServiceHealth.HEALTHY
                and svc.config.memory_limit_mb() > 256.0
            )

        return [
            ("Identified payment-service as the failing service", identified_service, 0.20),
            ("Read logs from payment-service", read_logs, 0.20),
            ("Diagnosed OOM error", diagnosed_oom, 0.20),
            ("Restarted with increased memory limit", applied_fix, 0.40),
        ]

    def compute_penalties(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        penalty = 0.0
        healthy_services = {
            "postgres-db", "auth-service", "inventory-service",
            "notification-service", "user-service", "order-service",
            "api-gateway", "frontend-bff",
        }
        for a in actions:
            if a.action_type == "restart_service" and a.target_service in healthy_services:
                penalty -= 0.10
        return penalty
