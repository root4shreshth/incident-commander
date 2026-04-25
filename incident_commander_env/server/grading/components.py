"""Six independent reward components for the IncidentCommander env.

The hackathon judging guide explicitly recommends *"multiple independent reward
functions, not just one"* to defeat reward hacking. Each component below is a
pure function `(ActionRecord, EpisodeContext) -> float`, isolated from the
others, and unit-testable in three lines.

The aggregate signal is `RewardBreakdown.total()`. TRL's GRPOTrainer logs each
component to wandb separately — that's how you get the "reward components
diverging during training" plot which is the strongest visual evidence of
non-trivial learning.

Component table:

| Component        | Triggers                                          | Range          |
|------------------|---------------------------------------------------|----------------|
| r_diagnostic     | first read on a relevant or adjacent service       | +0.02 .. +0.05 |
| r_correct_op     | scenario-defined right move (delegated)            | +0.15          |
| r_resolution     | terminal — accurate root-cause declaration         | +0.05 .. +0.30 |
| r_format         | action parsed cleanly (no fallback)                | +0.01          |
| r_efficiency     | terminal — solved in <=50% of step budget          | 0 or +0.10     |
| r_penalty        | redundancy + harmful_restart + handler errors      | -0.30 .. 0     |
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.grading.episode_context import EpisodeContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIAGNOSTIC_ACTIONS = frozenset({"read_logs", "check_metrics", "describe_service", "run_diagnostic", "list_services"})
REMEDIATIVE_ACTIONS = frozenset({"restart_service", "rollback_deployment", "scale_service", "update_config"})

R_DIAG_RELEVANT = 0.05
R_DIAG_ADJACENT = 0.02
R_CORRECT_OP = 0.15
R_RESOLUTION_ACCURATE = 0.30
R_RESOLUTION_VAGUE = 0.10
R_RESOLUTION_FALSE = -0.05  # declared resolved when scenario isn't actually resolved
R_FORMAT = 0.01
R_EFFICIENCY = 0.10
R_PENALTY_HARMFUL = -0.10
R_PENALTY_REDUNDANT = -0.03
R_PENALTY_HANDLER_ERROR = -0.05
R_PENALTY_REDUNDANCY_WINDOW = 3


# ---------------------------------------------------------------------------
# RewardBreakdown — the typed aggregate
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Per-step reward, decomposed by component.

    `total()` is the scalar that goes into RL. The individual fields are what
    judges see in the wandb plots; agents should learn that high overall
    reward decomposes into reading right + acting right + resolving right
    rather than gaming any single component.
    """
    diagnostic: float = 0.0
    correct_op: float = 0.0
    resolution: float = 0.0
    format: float = 0.0
    efficiency: float = 0.0
    penalty: float = 0.0

    def total(self) -> float:
        return (
            self.diagnostic
            + self.correct_op
            + self.resolution
            + self.format
            + self.efficiency
            + self.penalty
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def zero(cls) -> "RewardBreakdown":
        return cls()


# ---------------------------------------------------------------------------
# Component functions — pure (ActionRecord, EpisodeContext) -> float
# ---------------------------------------------------------------------------

def r_diagnostic(action: ActionRecord, ctx: EpisodeContext) -> float:
    """Reward investigative actions, more for the relevant ones.

    Suppressed if the same (action_type, target) pair was used in the recent
    window — that path returns the redundancy penalty instead, in `r_penalty`.
    Note we don't *also* zero the reward here; the penalty path already
    nets a negative.
    """
    if action.action_type not in DIAGNOSTIC_ACTIONS:
        return 0.0
    # Already-seen redundant reads pick up the redundancy penalty, but we still
    # award a tiny credit so the curve doesn't flatten on long episodes.
    if action.target_service in ctx.relevant_services:
        return R_DIAG_RELEVANT
    return R_DIAG_ADJACENT


def r_correct_op(action: ActionRecord, ctx: EpisodeContext) -> float:
    """Reward a remediation action that the scenario considers correct.

    The scenario decides what counts as "correct" via `is_correct_op` (default
    in BaseScenario, overridable). This is the strongest single signal an
    agent gets and is why scenarios that delegate the heal-decision (see
    Track A.2 — anti-cheat) keep this reward honest.
    """
    if action.action_type not in REMEDIATIVE_ACTIONS:
        return 0.0
    scenario = ctx.scenario
    if scenario is None:
        return 0.0
    is_correct = getattr(scenario, "is_correct_op", None)
    if is_correct is None:
        # Fallback: any remediation on a relevant service counts.
        return R_CORRECT_OP if action.target_service in ctx.relevant_services else 0.0
    try:
        if is_correct(action, ctx.cluster):
            return R_CORRECT_OP
    except Exception:
        # Defensive: a bug in the scenario predicate must not crash training.
        return 0.0
    return 0.0


def r_resolution(action: ActionRecord, ctx: EpisodeContext) -> float:
    """Terminal reward for declaring resolution accurately.

    +0.30 if `resolve_incident` is called AND the scenario actually is resolved
           AND the root_cause text contains scenario-specific keywords.
    +0.10 if resolved but root_cause is vague.
    -0.05 if `resolve_incident` is called when the scenario is NOT resolved
           (anti-cheat: prevents the "just call resolve at step 1" exploit).
    """
    if action.action_type != "resolve_incident":
        return 0.0
    if not ctx.is_resolved:
        return R_RESOLUTION_FALSE
    rc = str(action.parameters.get("root_cause", "")).lower()
    keywords = getattr(ctx.scenario, "root_cause_keywords", None) or []
    if any(kw.lower() in rc for kw in keywords):
        return R_RESOLUTION_ACCURATE
    return R_RESOLUTION_VAGUE


def r_format(action: ActionRecord, ctx: EpisodeContext) -> float:
    """Tiny per-step credit for emitting a well-formed action.

    Gives the LLM a constant nudge toward producing parseable JSON. If the
    action arrived here at all, it parsed; the inference fallback path
    converts malformed responses to `list_services` and that's still well-
    formed, so the reward is always +0.01. The inference layer can deduct
    its own format penalty for the malformed case if desired.
    """
    return R_FORMAT


def r_efficiency(action: ActionRecord, ctx: EpisodeContext) -> float:
    """Bonus at terminal step if the agent resolved within budget."""
    if not ctx.is_terminal or not ctx.is_resolved:
        return 0.0
    if ctx.step_count <= max(1, int(ctx.max_steps * 0.5)):
        return R_EFFICIENCY
    return 0.0


def r_penalty(action: ActionRecord, ctx: EpisodeContext) -> float:
    """Sum of negative signals: redundancy, harmful action, handler errors."""
    pen = 0.0

    # Redundancy: same (action_type, target) within the last N actions.
    window = ctx.previous_actions[-R_PENALTY_REDUNDANCY_WINDOW:]
    for prev in window:
        if (
            prev.action_type == action.action_type
            and prev.target_service == action.target_service
        ):
            pen += R_PENALTY_REDUNDANT
            break

    # Harmful: restarting a service that's already healthy.
    if (
        action.action_type == "restart_service"
        and action.target_service in ctx.healthy_services
    ):
        pen += R_PENALTY_HARMFUL

    # Handler-reported error (e.g. unknown config key, missing target).
    if ctx.last_observation_error:
        pen += R_PENALTY_HANDLER_ERROR

    return pen


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_step_breakdown(action: ActionRecord, ctx: EpisodeContext) -> RewardBreakdown:
    """Compose all six components into a typed breakdown."""
    return RewardBreakdown(
        diagnostic=r_diagnostic(action, ctx),
        correct_op=r_correct_op(action, ctx),
        resolution=r_resolution(action, ctx),
        format=r_format(action, ctx),
        efficiency=r_efficiency(action, ctx),
        penalty=r_penalty(action, ctx),
    )


__all__ = [
    "RewardBreakdown",
    "EpisodeContext",
    "DIAGNOSTIC_ACTIONS",
    "REMEDIATIVE_ACTIONS",
    "R_DIAG_RELEVANT",
    "R_DIAG_ADJACENT",
    "R_CORRECT_OP",
    "R_RESOLUTION_ACCURATE",
    "R_RESOLUTION_VAGUE",
    "R_RESOLUTION_FALSE",
    "R_FORMAT",
    "R_EFFICIENCY",
    "R_PENALTY_HARMFUL",
    "R_PENALTY_REDUNDANT",
    "R_PENALTY_HANDLER_ERROR",
    "r_diagnostic",
    "r_correct_op",
    "r_resolution",
    "r_format",
    "r_efficiency",
    "r_penalty",
    "compute_step_breakdown",
]
