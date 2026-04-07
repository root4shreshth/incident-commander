"""Per-step reward computation for the incident response environment."""

from __future__ import annotations

from typing import Optional, Set

from incident_commander_env.models import ActionRecord


# Reward constants
DIAGNOSTIC_REWARD = 0.02       # Reading logs/metrics of any service
RELEVANT_DIAGNOSTIC = 0.03     # Reading logs/metrics of a relevant service
CORRECT_FIX_REWARD = 0.15      # Correct remediation action
HARMFUL_PENALTY = -0.10        # Restarting a healthy service
REDUNDANT_PENALTY = -0.03      # Repeating the same action
IRRELEVANT_PENALTY = -0.01     # Action on clearly unrelated service
TIME_DECAY = 0.995             # Multiplicative decay per step


def compute_step_reward(
    action: ActionRecord,
    step: int,
    previous_actions: list[ActionRecord],
    relevant_services: Set[str],
    healthy_services: Set[str],
) -> float:
    """Compute reward for a single step.

    Args:
        action: The action just taken
        step: Current step number
        previous_actions: All previous actions
        relevant_services: Services involved in the incident
        healthy_services: Services that are healthy (penalty for touching)
    """
    reward = 0.0

    # Check for redundant action
    for prev in previous_actions:
        if (prev.action_type == action.action_type
                and prev.target_service == action.target_service
                and prev.parameters == action.parameters):
            return REDUNDANT_PENALTY * (TIME_DECAY ** step)

    # Diagnostic actions
    if action.action_type in ("read_logs", "check_metrics", "describe_service", "run_diagnostic"):
        if action.target_service in relevant_services:
            reward = RELEVANT_DIAGNOSTIC
        else:
            reward = DIAGNOSTIC_REWARD

    # Exploration
    elif action.action_type == "list_services":
        reward = DIAGNOSTIC_REWARD

    # Remediation actions
    elif action.action_type in ("restart_service", "rollback_deployment", "scale_service", "update_config"):
        if action.target_service in relevant_services:
            reward = CORRECT_FIX_REWARD
        elif action.target_service in healthy_services:
            reward = HARMFUL_PENALTY
        else:
            reward = IRRELEVANT_PENALTY

    # Resolve
    elif action.action_type == "resolve_incident":
        reward = 0.05

    # Ensure reward is never exactly 0.0 — validator requires strict (0, 1)
    result = reward * (TIME_DECAY ** step)
    if result == 0.0:
        result = 0.01
    return result
