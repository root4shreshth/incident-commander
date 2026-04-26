"""Episode-end deterministic grader - computes 0.0-1.0 score from scenario rubric."""

from __future__ import annotations

from typing import Any, Dict, List

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario
from incident_commander_env.server.simulation.cluster import Cluster


class IncidentGrader:
    """Computes final episode score using scenario-specific rubric."""

    def grade(
        self,
        scenario: BaseScenario,
        actions: List[ActionRecord],
        cluster: Cluster,
    ) -> float:
        """Return 0.0-1.0 score."""
        return scenario.grade(actions, cluster)

    def grade_details(
        self,
        scenario: BaseScenario,
        actions: List[ActionRecord],
        cluster: Cluster,
    ) -> Dict[str, Any]:
        """Return detailed grading breakdown."""
        return scenario.grade_details(actions, cluster)
