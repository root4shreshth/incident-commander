"""Task 4 (Easy): Disk space exhausted on a service.

A randomly chosen service hits 100% disk usage on its log volume — writes
start failing with `ENOSPC`, reads still work. Service is DEGRADED, not down.

Resolution: restart the target service. In the sim model, restart cycles
the volume mount so tmp/log files clear and the service comes back. The
agent must identify the failing service from logs (`No space left on
device`) and restart it.

Why this scenario matters: disk-full incidents are extremely common in
real production — log retention misconfigured, a runaway batch job
spilling output, a metrics agent dumping to disk. Real outage post-mortems
from companies like Slack, GitHub, and Stripe all feature it.
"""

from __future__ import annotations

import random as _random
from typing import List, Optional, Tuple

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario, RubricCheck
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.log_generator import (
    disk_full_logs,
    normal_logs,
)
from incident_commander_env.server.simulation.metrics_engine import apply_disk_full_anomaly
from incident_commander_env.server.simulation.service import ServiceHealth


_DISK_FULL_CANDIDATES = [
    "notification-service",
    "order-service",
    "inventory-service",
    "auth-service",
]


class DiskFullScenario(BaseScenario):
    task_id = "disk_full"
    difficulty = "easy"
    description = "Service degraded — disk space exhausted, writes failing"
    root_cause_keywords = ["disk", "no space", "enospc", "volume", "log rotation"]

    relevant_services = {"notification-service"}
    max_steps = 18

    def __init__(self, seed: Optional[int] = None, difficulty: float = 0.5) -> None:
        rng = _random.Random(seed) if seed is not None else _random.Random(0)
        if seed is None:
            self.target_service = "notification-service"
            self.mount = "/var/log"
        else:
            self.target_service = rng.choice(_DISK_FULL_CANDIDATES)
            self.mount = rng.choice(["/var/log", "/data", "/tmp", "/var/lib/app"])
        self.relevant_services = {self.target_service}
        self.max_steps = max(12, int(18 * (1.5 - max(0.0, min(1.0, difficulty)))))
        self.alert_message = (
            f"DEGRADED: {self.target_service} write operations are failing. "
            f"Disk usage on its log volume is reportedly at 96%+. "
            f"PagerDuty alert at 11:18 UTC. Customer impact rising."
        )
        self.root_cause = (
            f"{self.target_service}'s log volume ({self.mount}) is full — "
            f"writes return ENOSPC. Restarting cycles the volume."
        )

    def setup(self, cluster: Cluster) -> None:
        svc = cluster.get_service(self.target_service)
        if not svc:
            return
        svc.set_anomaly("disk_full")
        apply_disk_full_anomaly(svc)
        svc.add_logs(disk_full_logs(self.target_service, mount=self.mount))
        for name, other in cluster.services.items():
            if name != self.target_service:
                other.add_logs(normal_logs(name, count=8))

    def check_resolved(self, cluster: Cluster) -> bool:
        svc = cluster.get_service(self.target_service)
        if not svc:
            return False
        return svc.health == ServiceHealth.HEALTHY and "disk_full" not in svc._anomalies

    def get_rubric(self) -> List[Tuple[str, RubricCheck, float]]:
        target = self.target_service

        def identified(actions, cluster):
            return any(a.target_service == target for a in actions)

        def read_logs(actions, cluster):
            return any(a.action_type == "read_logs" and a.target_service == target for a in actions)

        def diagnosed(actions, cluster):
            return any(
                a.action_type in ("read_logs", "check_metrics", "describe_service")
                and a.target_service == target
                for a in actions
            )

        def fixed(actions, cluster):
            svc = cluster.get_service(target)
            return svc is not None and svc.health == ServiceHealth.HEALTHY

        return [
            (f"Identified {target} as the failing service", identified, 0.20),
            (f"Read logs from {target}", read_logs, 0.20),
            ("Diagnosed disk-full / ENOSPC", diagnosed, 0.20),
            ("Restarted to clear the volume", fixed, 0.40),
        ]

    def is_correct_op(self, action, cluster):
        return (
            action.action_type == "restart_service"
            and action.target_service == self.target_service
        )

    def compute_penalties(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        penalty = 0.0
        for a in actions:
            if (
                a.action_type == "restart_service"
                and a.target_service
                and a.target_service != self.target_service
            ):
                penalty -= 0.10
            # Rolling back doesn't help — it doesn't free disk.
            if a.action_type == "rollback_deployment":
                penalty -= 0.05
        return penalty
