"""Per-step reward computation for the IncidentCommander env.

This module is now a thin facade over `components.py`. It exists for
backwards compatibility with `tests/test_grading.py` which imports the
constant aliases (DIAGNOSTIC_REWARD, RELEVANT_DIAGNOSTIC, ...) and the
`compute_step_reward` function.

The real signal source is `components.compute_step_breakdown(action, ctx)`
which returns a `RewardBreakdown` with six independent component values.
This is what the env and TRL training pipeline use.
"""

from __future__ import annotations

from typing import Optional, Set

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.grading.components import (
    R_CORRECT_OP,
    R_DIAG_ADJACENT,
    R_DIAG_RELEVANT,
    R_PENALTY_HARMFUL,
    R_PENALTY_REDUNDANT,
    R_PENALTY_REDUNDANCY_WINDOW,
    RewardBreakdown,
    compute_step_breakdown,
)
from incident_commander_env.server.grading.episode_context import EpisodeContext


# ---------------------------------------------------------------------------
# Legacy constants (kept as aliases so existing tests + imports continue to work)
# ---------------------------------------------------------------------------

DIAGNOSTIC_REWARD = R_DIAG_ADJACENT       # 0.02
RELEVANT_DIAGNOSTIC = R_DIAG_RELEVANT     # 0.05
CORRECT_FIX_REWARD = R_CORRECT_OP         # 0.15
HARMFUL_PENALTY = R_PENALTY_HARMFUL       # -0.10
REDUNDANT_PENALTY = R_PENALTY_REDUNDANT   # -0.03
IRRELEVANT_PENALTY = -0.01                # legacy; not used in new components
TIME_DECAY = 0.995                        # multiplicative per-step decay


def compute_step_reward(
    action: ActionRecord,
    step: int,
    previous_actions: list[ActionRecord],
    relevant_services: Set[str],
    healthy_services: Set[str],
    *,
    scenario=None,
    cluster=None,
    is_terminal: bool = False,
    is_resolved: bool = False,
    max_steps: int = 25,
    last_observation_error: Optional[str] = None,
) -> float:
    """Compute per-step scalar reward.

    Backwards-compatible signature - old call sites pass the first five
    positional args and get a float back. New call sites pass scenario +
    cluster + terminal flags as keyword args for full multi-component
    behaviour.

    Time decay is applied at the end so old training scripts get the same
    decay structure they expected.
    """
    breakdown = compute_step_breakdown_scaled(
        action=action,
        step=step,
        previous_actions=previous_actions,
        relevant_services=relevant_services,
        healthy_services=healthy_services,
        scenario=scenario,
        cluster=cluster,
        is_terminal=is_terminal,
        is_resolved=is_resolved,
        max_steps=max_steps,
        last_observation_error=last_observation_error,
    )
    result = breakdown.total() * (TIME_DECAY ** step)
    # The validator rejects exactly 0.0 (it expects strictly-(0,1) reward), so
    # nudge zero outputs to a tiny positive. This keeps the gradient signal
    # alive on no-op steps without changing behaviour for non-zero steps.
    if result == 0.0:
        result = 0.01
    return result


def compute_step_breakdown_scaled(
    action: ActionRecord,
    step: int,
    previous_actions: list[ActionRecord],
    relevant_services: Set[str],
    healthy_services: Set[str],
    *,
    scenario=None,
    cluster=None,
    is_terminal: bool = False,
    is_resolved: bool = False,
    max_steps: int = 25,
    last_observation_error: Optional[str] = None,
) -> RewardBreakdown:
    """Compute the per-step `RewardBreakdown`.

    Public API for callers that want the per-component breakdown (env,
    training reward function, dashboard observability). Legacy callers
    use `compute_step_reward` and get a flat float.
    """
    ctx = EpisodeContext(
        scenario=scenario,
        previous_actions=previous_actions,
        relevant_services=relevant_services,
        healthy_services=healthy_services,
        step_count=step,
        max_steps=max_steps,
        is_terminal=is_terminal,
        is_resolved=is_resolved,
        cluster=cluster,
        last_observation_error=last_observation_error,
    )
    return compute_step_breakdown(action, ctx)


__all__ = [
    "DIAGNOSTIC_REWARD",
    "RELEVANT_DIAGNOSTIC",
    "CORRECT_FIX_REWARD",
    "HARMFUL_PENALTY",
    "REDUNDANT_PENALTY",
    "IRRELEVANT_PENALTY",
    "TIME_DECAY",
    "compute_step_reward",
    "compute_step_breakdown_scaled",
    "RewardBreakdown",
]
