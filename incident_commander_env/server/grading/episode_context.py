"""EpisodeContext — the typed bundle every reward component reads from.

Centralizing the read surface lets us:
1. Compute reward components in pure functions (no global state, no env coupling).
2. Plug different backends (sim, real) into the same reward pipeline because
   the components only ever query this struct.
3. Test components independently by hand-building a context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario
from incident_commander_env.server.simulation.cluster import Cluster


@dataclass
class EpisodeContext:
    """All information a reward component needs to score a single action.

    Built once per env.step() in the environment orchestrator and passed to
    every component function in `components.py`. Components are pure: they
    read fields from ctx and the action, and return a float.
    """

    scenario: BaseScenario
    """The active scenario instance for this episode."""

    previous_actions: List[ActionRecord]
    """All actions taken before the current one, in step order."""

    relevant_services: Set[str]
    """Services that are part of the scenario's incident.

    Reading or remediating these earns reward; reading other services earns
    a smaller diagnostic credit; remediating other healthy services costs a
    penalty.
    """

    healthy_services: Set[str]
    """Currently-healthy services that should *not* be touched by remediation.

    Mostly used for the harmful_restart penalty.
    """

    step_count: int
    """1-indexed step number (incremented before reward is computed)."""

    max_steps: int
    """Episode budget. Used by `r_efficiency` and time-decay."""

    is_terminal: bool = False
    """True if the env has decided this is the last step (resolution or timeout)."""

    is_resolved: bool = False
    """True if the scenario was actually resolved on this step."""

    cluster: Optional[Cluster] = None
    """The live cluster snapshot. Components use this for state-dependent checks
    (e.g. did the agent's restart actually heal the service?). Optional so
    components can be tested without a full cluster."""

    last_observation_error: Optional[str] = None
    """Error string from the action handler, if any. Drives anti-cheat penalties
    (e.g. unknown config key, missing target). Empty for successful actions."""
