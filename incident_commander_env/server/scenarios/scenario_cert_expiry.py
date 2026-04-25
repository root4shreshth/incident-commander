"""Task 6 (Easy): TLS certificate just expired on an externally-facing service.

A service's TLS cert ticked over the expiry boundary. The process is alive
and serving HTTP-internal traffic fine, but inbound HTTPS handshakes fail
outright (`ssl.SSLError: certificate has expired`). Liveness probe (HTTP)
passes, readiness probe (HTTPS) fails. Confusing for the agent — metrics
look almost normal except error_rate is at 90%+.

Resolution: restart the service. In the sim model, restart triggers the
cert renewal hook (Let's Encrypt / cert-manager / similar) and the listener
reloads with the new cert. The agent must read logs to spot the SSLError
pattern — metrics alone won't tell them what's wrong.

Why this scenario matters: cert-expiry is the most embarrassing class of
real-world outage. Every major company has been bitten by it (Spotify
2021, Microsoft Teams 2020, Microsoft Azure DevOps 2018, ...). It's
instructive precisely BECAUSE the metrics look almost normal.
"""

from __future__ import annotations

import random as _random
from typing import List, Optional, Tuple

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario, RubricCheck
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.log_generator import (
    cert_expired_logs,
    normal_logs,
)
from incident_commander_env.server.simulation.metrics_engine import apply_cert_expired_anomaly
from incident_commander_env.server.simulation.service import ServiceHealth


_CERT_VICTIMS = [
    "frontend-bff",
    "api-gateway",
    "auth-service",
    "user-service",
]


class CertExpiryScenario(BaseScenario):
    task_id = "cert_expiry"
    difficulty = "easy"
    description = "TLS certificate expired — inbound HTTPS handshakes failing"
    root_cause_keywords = [
        "tls", "ssl", "certificate", "handshake", "expired", "cert",
        "let's encrypt", "renewal",
    ]

    relevant_services = {"frontend-bff"}
    max_steps = 16

    def __init__(self, seed: Optional[int] = None, difficulty: float = 0.5) -> None:
        rng = _random.Random(seed) if seed is not None else _random.Random(0)
        if seed is None:
            self.target_service = "frontend-bff"
        else:
            self.target_service = rng.choice(_CERT_VICTIMS)
        self.relevant_services = {self.target_service}
        self.max_steps = max(10, int(16 * (1.5 - max(0.0, min(1.0, difficulty)))))
        self.alert_message = (
            f"CRITICAL: {self.target_service} returning HTTP 5xx to ~100% of clients. "
            f"Process is up. Liveness probe passes. Readiness probe fails. "
            f"Engineers reporting they cannot reach it from outside the cluster."
        )
        self.root_cause = (
            f"TLS certificate on {self.target_service} expired. "
            f"Restarting triggers cert renewal and listener reload."
        )

    def setup(self, cluster: Cluster) -> None:
        svc = cluster.get_service(self.target_service)
        if not svc:
            return
        svc.set_anomaly("cert_expired")
        apply_cert_expired_anomaly(svc)
        svc.add_logs(cert_expired_logs(self.target_service))
        for name, other in cluster.services.items():
            if name != self.target_service:
                other.add_logs(normal_logs(name, count=8))

    def check_resolved(self, cluster: Cluster) -> bool:
        svc = cluster.get_service(self.target_service)
        if not svc:
            return False
        return svc.health == ServiceHealth.HEALTHY and "cert_expired" not in svc._anomalies

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
            (f"Identified {target} as the affected service", identified, 0.15),
            (f"Read logs from {target} (spotted SSL/cert errors)", read_logs, 0.25),
            ("Diagnosed expired TLS certificate", diagnosed, 0.20),
            ("Restarted to renew cert + reload listener", fixed, 0.40),
        ]

    def is_correct_op(self, action, cluster):
        return (
            action.action_type == "restart_service"
            and action.target_service == self.target_service
        )

    def compute_penalties(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        penalty = 0.0
        # Rolling back doesn't help — the cert is on the cluster, not the binary.
        for a in actions:
            if a.action_type == "rollback_deployment" and a.target_service == self.target_service:
                penalty -= 0.10
            if (
                a.action_type == "restart_service"
                and a.target_service
                and a.target_service != self.target_service
            ):
                penalty -= 0.05
        return penalty
