"""Refund-race deadlock scenario - fintech-shaped, ORDERING-SENSITIVE incident.

The refund-service v3.2.1 deployment introduced a lock-acquisition order
bug: refund transactions grab the refund row FIRST, then attempt to grab
the ledger row, while the ledger-service posts new entries in the opposite
order (ledger row first, then refund row). Result: mutual deadlock, both
services hang on database locks, customer refund requests time out.

Resolution ORDER matters, exactly like `bad_deployment_cascade`:

    1. FIRST: `rollback_deployment` refund-service to the stable v3.2.0
       (which uses the correct lock-acquisition order matching ledger-service).
    2. THEN: `restart_service` ledger-service to kill the wedged transactions
       that are still holding stale locks from the pre-rollback state.

Bare restart of ledger-service without the rollback is EXPLICITLY WRONG - the
v3.2.1 refund code will just re-deadlock on the next refund request. This is
penalised -0.10 (same magnitude as the "restart order-service instead of
rollback" penalty in bad_deployment_cascade).

Restart of refund-service is also wrong - the deadlock lives in the compiled
binary logic, not in wedged state. Rollback replaces the binary and thus the
bug; restart preserves it. Penalised -0.10.

Real outages of this shape: Razorpay is fintech infrastructure - the payments
industry is riddled with deadlock-on-refund war stories (Stripe 2017 subscription
proration deadlock, Adyen 2020 double-entry ordering incident, and every internal
"weekend hotfix" a payments engineer has ever had to page-out for). This is the
canonical shape.
"""

from __future__ import annotations

from typing import List, Tuple

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario, RubricCheck
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.log_generator import (
    lock_contention_logs,
    normal_logs,
)
from incident_commander_env.server.simulation.service import Deployment, ServiceHealth


BAD_VERSION = "v3.2.1"
STABLE_VERSION = "v3.2.0"


class RefundRaceScenario(BaseScenario):
    """Refund-service v3.2.1 lock-order bug -> deadlock with ledger-service.

    Fix order: rollback refund-service THEN restart ledger-service.
    """

    task_id = "refund_race_deadlock"
    difficulty = "hard"
    description = (
        "refund-service v3.2.1 has a lock-acquisition order bug that deadlocks "
        "with ledger-service - customer refunds hanging, both services wedged"
    )
    alert_message = (
        "CRITICAL: refund-service and ledger-service both showing lock-wait "
        "timeouts. Customer refund requests hanging past 60s SLO. Refund queue "
        "at 220 pending. Deploy of refund-service v3.2.1 went live 12 minutes "
        "before symptoms started."
    )
    root_cause = (
        "refund-service v3.2.1 acquires the refund row lock BEFORE the ledger "
        "row lock; ledger-service holds them in the opposite order. Concurrent "
        "refund + ledger writes hit a mutual deadlock. Rollback of refund-service "
        "to v3.2.0 restores the correct ordering."
    )
    root_cause_keywords = [
        "refund", "ledger", "deadlock", "lock", "order", "rollback",
        "v3.2.1", "v3.2.0", "deploy", "race",
    ]
    relevant_services = {"refund-service", "ledger-service"}
    max_steps = 30

    def __init__(self, seed=None, difficulty: float = 0.5) -> None:
        # Parametric reset accepts seed + difficulty. The version pair is
        # fixed (v3.2.1 bad -> v3.2.0 stable) because the rubric is tied to
        # those exact values. Difficulty scales the step budget.
        import random as _random
        rng = _random.Random(seed) if seed is not None else _random.Random(0)
        self.seed = seed
        self.difficulty_factor = float(difficulty) if difficulty is not None else 0.5
        # 0.0 -> 45 steps, 1.0 -> 20 steps, default 0.5 -> ~32
        self.max_steps = max(20, int(45 - 25 * max(0.0, min(1.0, self.difficulty_factor))))

    def setup(self, cluster: Cluster) -> None:
        # refund-service: bad deploy with lock-order bug -> lock_contention anomaly
        refund = cluster.get_service("refund-service")
        if refund:
            refund.config.version = BAD_VERSION
            refund.deployment_history.append(
                Deployment(
                    version=BAD_VERSION,
                    timestamp="2026-08-15T14:23:00Z",
                    status="active",
                )
            )
            refund.set_anomaly("lock_contention")
            refund.add_logs(lock_contention_logs("refund-service"))
            refund.add_logs([
                f"[INFO]  refund-service - Deploy {BAD_VERSION} promoted at 14:23:00Z (12 min ago)",
                "[ERROR] refund-service - Deadlock detected acquiring ledger lock while holding refund lock",
                "[WARN]  refund-service - 220 refund requests pending, oldest waiting 62s",
                f"[INFO]  refund-service - Previous stable deploy: {STABLE_VERSION}",
            ])

        # ledger-service: cascade-degraded because refund-service is holding its locks.
        # We use `connection_leak` (restart-curable) rather than `lock_contention`
        # (rollback-only) because, once refund-service is rolled back, the wedged
        # ledger transactions look like leaked connections from ledger's own
        # perspective - a restart cycles the DB pool and clears them.
        ledger = cluster.get_service("ledger-service")
        if ledger:
            ledger.set_anomaly("connection_leak")
            ledger.add_logs(lock_contention_logs("ledger-service"))
            ledger.add_logs([
                "[ERROR] ledger-service - Transaction ledger_credit_882741 waiting on refund-service lock 58s",
                "[WARN]  ledger-service - 14 write transactions wedged; new posts queuing",
                "[INFO]  ledger-service - Restart clears wedged local transactions but does not fix upstream",
            ])

        # Healthy background services - a bit of noise so the picture reads correctly
        for name in (
            "postgres-db", "payment-service", "payment-gateway", "fraud-check",
            "webhook-consumer", "auth-service", "user-service",
        ):
            svc = cluster.get_service(name)
            if svc:
                svc.add_logs(normal_logs(name, count=4))

    def check_resolved(self, cluster: Cluster) -> bool:
        refund = cluster.get_service("refund-service")
        ledger = cluster.get_service("ledger-service")
        if not refund or not ledger:
            return False
        return (
            refund.health == ServiceHealth.HEALTHY
            and refund.config.version != BAD_VERSION
            and ledger.health == ServiceHealth.HEALTHY
        )

    def get_rubric(self) -> List[Tuple[str, RubricCheck, float]]:
        def investigated_refund(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.target_service == "refund-service"
                and a.action_type in ("read_logs", "check_metrics", "describe_service")
                for a in actions
            )

        def investigated_ledger(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.target_service == "ledger-service"
                and a.action_type in ("read_logs", "check_metrics", "describe_service")
                for a in actions
            )

        def read_deployment_history(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.target_service == "refund-service" and a.action_type == "describe_service"
                for a in actions
            )

        def rolled_back_refund(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "rollback_deployment"
                and a.target_service == "refund-service"
                and a.parameters.get("to_version") == STABLE_VERSION
                for a in actions
            )

        def restarted_ledger(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "restart_service" and a.target_service == "ledger-service"
                for a in actions
            )

        def correct_order(actions: List[ActionRecord], cluster: Cluster) -> bool:
            rollback_step = None
            restart_steps = []
            for a in actions:
                if a.action_type == "rollback_deployment" and a.target_service == "refund-service":
                    rollback_step = a.step
                if a.action_type == "restart_service" and a.target_service == "ledger-service":
                    restart_steps.append(a.step)
            if rollback_step is None or not restart_steps:
                return False
            return all(s > rollback_step for s in restart_steps)

        def resolved_with_cause(actions: List[ActionRecord], cluster: Cluster) -> bool:
            return any(
                a.action_type == "resolve_incident"
                and (
                    "deadlock" in str(a.parameters.get("root_cause", "")).lower()
                    or "v3.2.1" in str(a.parameters.get("root_cause", "")).lower()
                    or "lock" in str(a.parameters.get("root_cause", "")).lower()
                )
                for a in actions
            )

        def efficient_no_harm(actions: List[ActionRecord], cluster: Cluster) -> bool:
            harmful = sum(
                1 for a in actions
                if (a.action_type == "restart_service" and a.target_service == "refund-service")
            )
            return harmful == 0 and len(actions) <= 20

        return [
            ("Investigated refund-service", investigated_refund, 0.10),
            ("Investigated ledger-service", investigated_ledger, 0.10),
            ("Read refund-service deployment history (found v3.2.1)", read_deployment_history, 0.10),
            ("Rolled back refund-service to v3.2.0", rolled_back_refund, 0.25),
            ("Restarted ledger-service to clear wedged transactions", restarted_ledger, 0.15),
            ("Correct ordering (rollback BEFORE ledger restart)", correct_order, 0.10),
            ("Resolved with accurate root cause", resolved_with_cause, 0.10),
            ("Efficient - no bare restart of refund-service", efficient_no_harm, 0.10),
        ]

    def is_correct_op(self, action, cluster):
        """Rollback refund-service to v3.2.0, then restart ledger-service.

        Restart of refund-service = wrong (bug lives in the binary; restart
        preserves it). Restart of ledger-service = correct (clears wedged
        transactions after the upstream fix).
        """
        if action.action_type == "rollback_deployment":
            if action.target_service != "refund-service":
                return False
            target_v = (action.parameters or {}).get("to_version")
            return target_v == STABLE_VERSION
        if action.action_type == "restart_service":
            return action.target_service == "ledger-service"
        return False

    def compute_penalties(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        penalty = 0.0
        # Major: restarting refund-service instead of rolling back leaves the bug in place.
        for a in actions:
            if a.action_type == "restart_service" and a.target_service == "refund-service":
                penalty -= 0.10
        # Minor: touching unrelated services under an active incident.
        unrelated = {
            "auth-service", "user-service", "postgres-db", "notification-service",
            "frontend-bff", "api-gateway", "order-service", "inventory-service",
            "payment-gateway", "payment-service", "fraud-check", "webhook-consumer",
        }
        for a in actions:
            if a.action_type in ("restart_service", "scale_service") and a.target_service in unrelated:
                penalty -= 0.03
        return penalty
