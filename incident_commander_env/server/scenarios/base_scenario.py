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

    def grade(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        """Compute final 0.0-1.0 score from rubric + penalties."""
        rubric = self.get_rubric()
        score = sum(weight for _, check, weight in rubric if check(actions, cluster))
        penalties = self.compute_penalties(actions, cluster)
        return max(0.0, min(1.0, score + penalties))

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
