"""Task 5 (Medium): Slow query / lock contention from a recent deploy.

A recent deploy (v2.5.0 by default) introduced a query that holds row-locks
for too long. P99 latency spikes into the 8-12s range, throughput collapses,
the connection pool fills up with txns waiting on locks. Logs show "Lock
wait timeout exceeded" and "deadlock detected".

Resolution: rollback the offending deploy. Restarting clears the active
txns but the slow query is still in the binary, so it'll lock up again.
The scenario distinguishes the *runtime* fix (rollback) from the *quick*
fix (restart) and rewards the agent that picks the right one.

Why this scenario matters: lock contention from a single bad query is a
classic incident shape. Half of the post-mortems on the SREcon talks list
have a slow-query as root cause.
"""

from __future__ import annotations

import random as _random
from typing import List, Optional, Tuple

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario, RubricCheck
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.log_generator import (
    lock_contention_logs,
    normal_logs,
)
from incident_commander_env.server.simulation.metrics_engine import apply_lock_contention_anomaly
from incident_commander_env.server.simulation.service import Deployment, ServiceHealth


_LOCK_VICTIMS = ["order-service", "payment-service", "inventory-service"]
_BAD_VERSIONS = ["v2.5.0", "v2.4.7", "v3.1.0"]
_STABLE_VERSIONS = ["v2.4.6", "v2.4.6", "v3.0.5"]


class SlowQueryScenario(BaseScenario):
    task_id = "slow_query"
    difficulty = "medium"
    description = "Lock contention — a slow query introduced in a recent deploy"
    root_cause_keywords = [
        "lock", "deadlock", "slow query", "lock wait", "for update",
        "rollback", "deploy", "version",
    ]

    relevant_services = {"order-service"}
    max_steps = 22

    def __init__(self, seed: Optional[int] = None, difficulty: float = 0.5) -> None:
        rng = _random.Random(seed) if seed is not None else _random.Random(0)
        if seed is None:
            self.target_service = "order-service"
            self.bad_version = "v2.5.0"
            self.stable_version = "v2.4.6"
        else:
            idx = rng.randrange(len(_BAD_VERSIONS))
            self.target_service = rng.choice(_LOCK_VICTIMS)
            self.bad_version = _BAD_VERSIONS[idx]
            self.stable_version = _STABLE_VERSIONS[idx]
        self.relevant_services = {self.target_service}
        self.max_steps = max(15, int(22 * (1.5 - max(0.0, min(1.0, difficulty)))))
        self.alert_message = (
            f"WARNING: latency on {self.target_service} has exploded — p99 > 8s, "
            f"throughput collapsed. Connection pool filling with locked txns. "
            f"Recent deploy ({self.bad_version}) is suspicious."
        )
        self.root_cause = (
            f"{self.bad_version} introduced a query holding row-locks too long; "
            f"rolling back to {self.stable_version} reverts the change."
        )

    def setup(self, cluster: Cluster) -> None:
        svc = cluster.get_service(self.target_service)
        if not svc:
            return
        # Mark the service as running the bad version + leave the stable version
        # in deployment_history so rollback has somewhere to go.
        svc.config.version = self.bad_version
        svc.deployment_history.append(
            Deployment(version=self.stable_version, timestamp="2026-04-25T13:00:00Z", status="rolled_back")
        )
        svc.deployment_history.append(
            Deployment(version=self.bad_version, timestamp="2026-04-25T15:00:00Z", status="active")
        )
        svc.set_anomaly("lock_contention")
        apply_lock_contention_anomaly(svc)
        svc.add_logs(lock_contention_logs(self.target_service))
        for name, other in cluster.services.items():
            if name != self.target_service:
                other.add_logs(normal_logs(name, count=8))

    def check_resolved(self, cluster: Cluster) -> bool:
        svc = cluster.get_service(self.target_service)
        if not svc:
            return False
        return (
            svc.health == ServiceHealth.HEALTHY
            and "lock_contention" not in svc._anomalies
            and svc.config.version != self.bad_version
        )

    def get_rubric(self) -> List[Tuple[str, RubricCheck, float]]:
        target = self.target_service
        bad = self.bad_version
        stable = self.stable_version

        def identified(actions, cluster):
            return any(a.target_service == target for a in actions)

        def read_logs(actions, cluster):
            return any(a.action_type == "read_logs" and a.target_service == target for a in actions)

        def diagnosed_lock(actions, cluster):
            return any(
                a.action_type in ("read_logs", "check_metrics", "describe_service")
                and a.target_service == target
                for a in actions
            )

        def saw_recent_deploy(actions, cluster):
            return any(a.action_type == "describe_service" and a.target_service == target for a in actions)

        def applied_rollback(actions, cluster):
            svc = cluster.get_service(target)
            if not svc:
                return False
            return svc.config.version != bad and "lock_contention" not in svc._anomalies

        return [
            (f"Identified {target} as the slow service", identified, 0.15),
            (f"Read logs from {target} (saw lock-wait errors)", read_logs, 0.15),
            ("Diagnosed lock contention", diagnosed_lock, 0.15),
            (f"Examined deployment history of {target}", saw_recent_deploy, 0.15),
            (f"Rolled {target} back from {bad}", applied_rollback, 0.40),
        ]

    def is_correct_op(self, action, cluster):
        if action.action_type != "rollback_deployment":
            return False
        if action.target_service != self.target_service:
            return False
        to_v = (action.parameters or {}).get("to_version")
        return bool(to_v) and str(to_v) != self.bad_version

    def compute_penalties(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        penalty = 0.0
        for a in actions:
            # Restarting the locked service is a quick fix that doesn't last —
            # it's the wrong move for this scenario.
            if a.action_type == "restart_service" and a.target_service == self.target_service:
                penalty -= 0.10
            # Restarting unrelated services is purely harmful here.
            if (
                a.action_type == "restart_service"
                and a.target_service
                and a.target_service != self.target_service
            ):
                penalty -= 0.10
        return penalty
