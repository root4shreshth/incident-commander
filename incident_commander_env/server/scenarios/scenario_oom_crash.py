"""Task 1 (Easy): Single service OOM crash — parametric.

A randomly chosen service crashes due to OutOfMemoryError. Logs clearly show
the error. Agent must identify the service, read logs, and restart with a
higher memory limit. Each `(seed, difficulty)` pair gives a different
instance of the family — same fault shape, different specific service /
memory ceiling / step budget.

Why parametric: with three hardcoded scenarios an RL agent overfits in ~100
episodes. With seeded families, each /reset call materializes a fresh
instance from a distribution; the agent has to learn the *shape* of the
fault rather than memorize three constants.
"""

from __future__ import annotations

import random as _random
from typing import Any, Dict, List, Optional, Tuple

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario, RubricCheck
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.log_generator import (
    normal_logs,
    oom_crash_logs,
)
from incident_commander_env.server.simulation.metrics_engine import apply_oom_anomaly
from incident_commander_env.server.simulation.service import ServiceHealth


# Pool of services that may be the "victim" — chosen seed-deterministically.
_OOM_CANDIDATES = [
    "payment-service",
    "order-service",
    "inventory-service",
    "user-service",
]


class OOMCrashScenario(BaseScenario):
    task_id = "oom_crash"
    difficulty = "easy"
    description = "Single service OOM crash — a service is down due to OutOfMemoryError"
    root_cause_keywords = ["oom", "memory", "out of memory", "outofmemory", "memory limit"]

    # `relevant_services` is computed in __init__ but typed here for the
    # static `is_correct_op` default in BaseScenario to find a fallback.
    relevant_services = {"payment-service"}
    max_steps = 15

    def __init__(self, seed: Optional[int] = None, difficulty: float = 0.5) -> None:
        # Seed-deterministic randomization. seed=None => default behaviour
        # (payment-service, 256Mi, 15 steps) for backwards compat with tests.
        rng = _random.Random(seed) if seed is not None else _random.Random(0)
        if seed is None:
            # Legacy default: payment-service @ 256Mi
            self.target_service = "payment-service"
            self.memory_limit_mb = 256
        else:
            self.target_service = rng.choice(_OOM_CANDIDATES)
            self.memory_limit_mb = int(rng.uniform(192, 320))
        self.relevant_services = {self.target_service}
        # Difficulty knob: harder => smaller step budget. Bound to [10, 20].
        self.max_steps = max(10, int(15 * (1.5 - max(0.0, min(1.0, difficulty)))))
        self.alert_message = (
            f"CRITICAL: {self.target_service} is DOWN. Multiple health check failures detected. "
            f"PagerDuty alert triggered at 03:42 UTC. Immediate investigation required."
        )
        self.root_cause = (
            f"{self.target_service} OOM killed due to insufficient memory limit "
            f"({self.memory_limit_mb}Mi)"
        )

    def setup(self, cluster: Cluster) -> None:
        svc = cluster.get_service(self.target_service)
        if not svc:
            return

        svc.config.memory_limit = f"{self.memory_limit_mb}Mi"
        svc.metrics.memory_limit_mb = float(self.memory_limit_mb)

        svc.set_anomaly("oom")
        apply_oom_anomaly(svc)

        svc.add_logs(oom_crash_logs(self.target_service, memory_limit_mb=self.memory_limit_mb))

        for name, other_svc in cluster.services.items():
            if name != self.target_service:
                other_svc.add_logs(normal_logs(name, count=8))

    def check_resolved(self, cluster: Cluster) -> bool:
        svc = cluster.get_service(self.target_service)
        if not svc:
            return False
        return (
            svc.health == ServiceHealth.HEALTHY
            and svc.config.memory_limit_mb() > self.memory_limit_mb
        )

    def get_rubric(self) -> List[Tuple[str, RubricCheck, float]]:
        target = self.target_service
        memory_floor = self.memory_limit_mb

        def identified_service(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(a.target_service == target for a in actions)

        def read_logs(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "read_logs" and a.target_service == target
                for a in actions
            )

        def diagnosed_oom(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type in ("read_logs", "check_metrics", "describe_service")
                and a.target_service == target
                for a in actions
            )

        def applied_fix(actions: List[ActionRecord], cluster: Cluster) -> bool:
            svc = cluster.get_service(target)
            if not svc:
                return False
            return (
                svc.health == ServiceHealth.HEALTHY
                and svc.config.memory_limit_mb() > memory_floor
            )

        return [
            (f"Identified {target} as the failing service", identified_service, 0.20),
            (f"Read logs from {target}", read_logs, 0.20),
            ("Diagnosed OOM error", diagnosed_oom, 0.20),
            ("Restarted with increased memory limit", applied_fix, 0.40),
        ]

    def is_correct_op(self, action, cluster):
        """OOM is fixed by restarting the target service with a higher memory limit."""
        if action.action_type != "restart_service":
            return False
        if action.target_service != self.target_service:
            return False
        new_limit = action.parameters.get("memory_limit") if action.parameters else None
        if not new_limit:
            return False
        try:
            v = str(new_limit).replace("Mi", "").replace("Gi", "")
            mb = float(v) * (1024 if "Gi" in str(new_limit) else 1)
            return mb > self.memory_limit_mb
        except (ValueError, TypeError):
            return False

    def compute_penalties(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        penalty = 0.0
        # Penalize restart of any non-target service.
        for a in actions:
            if (
                a.action_type == "restart_service"
                and a.target_service
                and a.target_service != self.target_service
            ):
                penalty -= 0.10
        return penalty
