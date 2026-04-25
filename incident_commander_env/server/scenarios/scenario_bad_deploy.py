"""Task 3 (Hard): Bad deployment cascading failure.

A bad deployment to order-service (v2.4.0) causes a memory leak, which triggers
the autoscaler, exhausting the cluster resource quota. Secondary services
(inventory-service, notification-service) cannot maintain replicas and degrade.

Resolution ORDER matters: rollback first, then free resources, then restart dependents.
"""

from __future__ import annotations

from typing import List, Tuple

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario, RubricCheck
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.log_generator import (
    bad_deployment_logs,
    cascading_failure_logs,
    normal_logs,
    resource_quota_logs,
    service_starved_logs,
)
from incident_commander_env.server.simulation.service import Deployment, ServiceHealth


BAD_VERSION = "v2.4.0"
STABLE_VERSION = "v2.3.1"


class BadDeployScenario(BaseScenario):
    task_id = "bad_deployment_cascade"
    difficulty = "hard"
    description = "Bad deployment cascade — memory leak triggers autoscaler, exhausts resource quota, starves secondary services"
    alert_message = (
        "CRITICAL: Multiple services degraded. Cluster resource quota at 95%. "
        "Autoscaler events detected for order-service. "
        "inventory-service and notification-service reporting failures. Started at 09:15 UTC."
    )
    root_cause = (
        "Bad deployment (order-service v2.4.0) caused memory leak, triggering autoscaler "
        "which exhausted cluster resource quota, starving inventory-service and notification-service"
    )
    root_cause_keywords = [
        "deploy", "deployment", "rollback", "v2.4.0", "order-service",
        "memory leak", "autoscaler", "quota", "starved", "cascade"
    ]
    relevant_services = {
        "order-service", "inventory-service", "notification-service",
        "api-gateway", "frontend-bff",
    }
    max_steps = 35

    def __init__(self, seed=None, difficulty: float = 0.5) -> None:
        # Accepts seed + difficulty for the env's parametric reset path. The
        # version pair is fixed (v2.4.0 -> v2.3.1) for now since the rubric
        # has rich logic tied to those specific values; difficulty scales the
        # step budget so harder draws give less time to recover.
        import random as _random
        rng = _random.Random(seed) if seed is not None else _random.Random(0)
        self.seed = seed
        self.difficulty_factor = float(difficulty) if difficulty is not None else 0.5
        # Step budget: difficulty=0 -> 50 steps, 1 -> 25 steps; default 0.5 -> ~37
        self.max_steps = max(25, int(50 - 25 * max(0.0, min(1.0, self.difficulty_factor))))

    def setup(self, cluster: Cluster) -> None:
        # order-service: bad deployment with memory leak
        order = cluster.get_service("order-service")
        if order:
            order.config.version = BAD_VERSION
            order.config.replicas = 6  # Autoscaler already scaled up
            order.deployment_history.append(
                Deployment(version=BAD_VERSION, timestamp="2026-03-29T09:10:00Z", status="active")
            )
            order.set_anomaly("memory_leak")
            order.add_logs(bad_deployment_logs("order-service", BAD_VERSION))

        # Cluster resource quota nearly exhausted
        cluster.resource_quota.memory_used_mb = 7800.0  # 95% of 8192
        cluster.resource_quota.cpu_used = 6800.0  # 85% of 8000

        # inventory-service: starved of resources
        inv = cluster.get_service("inventory-service")
        if inv:
            inv.config.replicas = 1  # Dropped from 2 to 1
            inv.set_anomaly("resource_starved")
            inv.add_logs(service_starved_logs("inventory-service"))

        # notification-service: starved of resources
        notif = cluster.get_service("notification-service")
        if notif:
            notif.config.replicas = 1  # Dropped from 2 to 1
            notif.set_anomaly("resource_starved")
            notif.add_logs(service_starved_logs("notification-service"))

        # api-gateway: showing cascading degradation
        gw = cluster.get_service("api-gateway")
        if gw:
            gw.set_anomaly("cascade_degradation")
            gw.add_logs(cascading_failure_logs("api-gateway", "order-service"))

        # frontend-bff: degraded
        fe = cluster.get_service("frontend-bff")
        if fe:
            fe.set_anomaly("cascade_degradation")
            fe.add_logs(cascading_failure_logs("frontend-bff", "api-gateway"))

        # Healthy services
        for name in ("postgres-db", "auth-service", "user-service", "payment-service"):
            svc = cluster.get_service(name)
            if svc:
                svc.add_logs(normal_logs(name, count=5))

    def check_resolved(self, cluster: Cluster) -> bool:
        order = cluster.get_service("order-service")
        inv = cluster.get_service("inventory-service")
        notif = cluster.get_service("notification-service")

        if not order or not inv or not notif:
            return False

        return (
            order.health == ServiceHealth.HEALTHY
            and order.config.version != BAD_VERSION
            and inv.health == ServiceHealth.HEALTHY
            and notif.health == ServiceHealth.HEALTHY
        )

    def get_rubric(self) -> List[Tuple[str, RubricCheck, float]]:
        def mapped_blast_radius(actions: List[ActionRecord], cluster: Cluster) -> bool:
            investigated = {a.target_service for a in actions if a.target_service and a.action_type in ("read_logs", "check_metrics", "describe_service")}
            return len(investigated) >= 3

        def identified_order_origin(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.target_service == "order-service"
                and a.action_type in ("read_logs", "check_metrics", "describe_service")
                for a in actions
            )

        def found_bad_version(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.target_service == "order-service"
                and a.action_type == "describe_service"
                for a in actions
            )

        def rolled_back(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "rollback_deployment"
                and a.target_service == "order-service"
                and a.parameters.get("to_version") == STABLE_VERSION
                for a in actions
            )

        def addressed_resources(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type in ("scale_service", "update_config")
                and (a.target_service == "order-service" or "quota" in str(a.parameters).lower())
                for a in actions
            ) or rolled_back(actions, cluster)  # Rollback inherently frees resources

        def restarted_inventory(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "restart_service" and a.target_service == "inventory-service"
                for a in actions
            )

        def restarted_notification(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "restart_service" and a.target_service == "notification-service"
                for a in actions
            )

        def correct_order(actions: List[ActionRecord], cluster: Cluster) -> bool:
            rollback_step = None
            restart_steps = []
            for a in actions:
                if a.action_type == "rollback_deployment" and a.target_service == "order-service":
                    rollback_step = a.step
                if a.action_type == "restart_service" and a.target_service in ("inventory-service", "notification-service"):
                    restart_steps.append(a.step)
            if rollback_step is None or not restart_steps:
                return False
            return all(s > rollback_step for s in restart_steps)

        def resolved_with_cause(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "resolve_incident"
                and ("deploy" in str(a.parameters.get("root_cause", "")).lower()
                     or "v2.4.0" in str(a.parameters.get("root_cause", "")).lower())
                for a in actions
            )

        def efficient_resolution(actions: List[ActionRecord], cluster: Cluster) -> bool:
            harmful = sum(
                1 for a in actions
                if a.action_type == "restart_service" and a.target_service == "order-service"
            )
            return harmful == 0 and len(actions) <= 25

        return [
            ("Mapped blast radius (investigated 3+ services)", mapped_blast_radius, 0.10),
            ("Identified order-service as the origin", identified_order_origin, 0.10),
            ("Found bad deployment version via describe_service", found_bad_version, 0.10),
            ("Rolled back order-service to v2.3.1", rolled_back, 0.15),
            ("Addressed resource quota", addressed_resources, 0.10),
            ("Restarted inventory-service", restarted_inventory, 0.10),
            ("Restarted notification-service", restarted_notification, 0.10),
            ("Correct ordering (rollback before restarts)", correct_order, 0.05),
            ("Resolved with accurate root cause", resolved_with_cause, 0.10),
            ("Efficient resolution (no harmful actions)", efficient_resolution, 0.10),
        ]

    def is_correct_op(self, action, cluster):
        """Bad-deploy is fixed by rolling back order-service then restarting starved deps.

        Critically, restart of order-service is NOT correct — must be a rollback.
        Restarting starved services (inventory, notification) IS correct because
        they need to come back up after the bad deploy is reverted.
        """
        if action.action_type == "rollback_deployment":
            if action.target_service != "order-service":
                return False
            target_v = action.parameters.get("to_version") if action.parameters else None
            return target_v == STABLE_VERSION
        if action.action_type == "restart_service":
            return action.target_service in {"inventory-service", "notification-service"}
        return False

    def compute_penalties(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        penalty = 0.0
        # Major penalty: restarting order-service instead of rolling back
        for a in actions:
            if a.action_type == "restart_service" and a.target_service == "order-service":
                penalty -= 0.10
        # Minor penalty: unnecessary actions on unrelated services
        unrelated = {"auth-service", "user-service", "postgres-db", "payment-service"}
        for a in actions:
            if a.action_type in ("restart_service", "scale_service") and a.target_service in unrelated:
                penalty -= 0.05
        return penalty
