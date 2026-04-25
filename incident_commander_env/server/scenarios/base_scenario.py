"""Abstract base class for incident scenarios."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.simulation.cluster import Cluster


RubricCheck = Callable[[List[ActionRecord], Cluster], bool]


class BaseScenario(ABC):
    """Base class for all incident scenarios.

    Each scenario injects a specific fault into the cluster and defines
    a deterministic grading rubric as a list of (check_fn, weight) tuples.
    """

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    alert_message: str
    root_cause: str
    max_steps: int

    # Keywords the scenario considers strong evidence that a `resolve_incident`
    # action's `root_cause` argument is accurate. Used by `r_resolution` to
    # distinguish a vague "yeah I fixed it" from "I correctly identified
    # the OOM in payment-service caused by insufficient memory limit".
    root_cause_keywords: List[str] = []

    @abstractmethod
    def setup(self, cluster: Cluster) -> None:
        """Inject the fault into the cluster. Called after cluster.initialize()."""

    @abstractmethod
    def check_resolved(self, cluster: Cluster) -> bool:
        """Return True if the incident has been fully resolved."""

    @abstractmethod
    def get_rubric(self) -> List[Tuple[str, RubricCheck, float]]:
        """Return grading rubric: list of (description, check_fn, weight).

        Weights should sum to ~1.0 for the positive criteria.
        """

    @abstractmethod
    def compute_penalties(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        """Compute penalty score (negative) for harmful/unnecessary actions."""

    # ---- Anti-cheat hooks (overridable; default behaviour is "no scenario-specific heal") ----

    def on_config_update(
        self, cluster: Cluster, target_service: str, key: str, value: Any
    ) -> bool:
        """Called when a known config key is set on a service.

        Return True if this config change is the correct fix for *this* scenario
        (in which case the relevant anomaly will be cleared by the handler).
        Default: False — config alone does not heal.

        Scenarios override this to declare what config change resolves their fault,
        so the handler doesn't have to string-match parameter names heuristically.
        """
        return False

    def is_correct_op(self, action: ActionRecord, cluster: Optional[Cluster]) -> bool:
        """Return True if `action` is a correct remediation move for this scenario.

        Default: any remediation action targeting a service in the scenario's
        relevant set counts. Scenarios override for tighter behavior — e.g. the
        bad-deploy scenario should reject restart-of-order-service in favor of
        rollback.

        Subclasses MUST be defensive — this is invoked from the reward path on
        every step and a crash would corrupt training. Wrap risky logic in
        try/except and return False on error.
        """
        from incident_commander_env.server.grading.components import REMEDIATIVE_ACTIONS
        if action.action_type not in REMEDIATIVE_ACTIONS:
            return False
        # Subclasses can override this list; default looks at relevant_services.
        relevant = getattr(self, "relevant_services", None) or set()
        return bool(action.target_service) and action.target_service in relevant

    def grade(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        """Compute final 0.0-1.0 score from rubric + penalties."""
        rubric = self.get_rubric()
        score = sum(weight for _, check, weight in rubric if check(actions, cluster))
        penalties = self.compute_penalties(actions, cluster)
        # Clamp to strict (0, 1) — hackathon validator rejects exactly 0.0 and 1.0
        raw = score + penalties
        return max(0.01, min(0.99, raw))

    def grade_details(self, actions: List[ActionRecord], cluster: Cluster) -> Dict[str, Any]:
        """Return detailed breakdown for debugging."""
        rubric = self.get_rubric()
        details = []
        for desc, check, weight in rubric:
            passed = check(actions, cluster)
            details.append({"criterion": desc, "weight": weight, "passed": passed})
        penalties = self.compute_penalties(actions, cluster)
        return {
            "task_id": self.task_id,
            "criteria": details,
            "penalties": penalties,
            "final_score": self.grade(actions, cluster),
        }
