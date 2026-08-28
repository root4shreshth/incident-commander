"""GRPO reward function - wraps the env's 6-component breakdown for TRL.

TRL's ``GRPOTrainer`` calls a reward function with
``(prompts, completions, **kwargs)`` and expects a list of floats, one per
completion. The trainer relativises those rewards within each prompt's group
of N completions to compute the policy gradient (that's what makes it "Group
Relative" Policy Optimization).

For each completion we:
  1. Parse the LLM output into an ``IncidentAction`` (via the same parser
     the eval runner uses, so training + eval share exactly one JSON path).
  2. Spin up a fresh ``IncidentCommanderEnv`` seeded to the prompt's
     ``(task_id, seed, difficulty)`` triple that the dataset carries in
     ``**kwargs``.
  3. Step the env once with the parsed action.
  4. Read ``env._last_breakdown`` - the typed ``RewardBreakdown`` the env
     stashes on every step - and return its scalar total as the reward.

Each per-completion ``RewardBreakdown`` is also appended to the sidecar
``_LAST_BREAKDOWNS`` list so the training notebook can read per-component
curves back out at plot time. That's the "reward components diverging
during training" plot (``training/plots.py:make_reward_components``) that
is the strongest single visual for the RLVR story: ``r_correct_op`` rising
while ``r_penalty`` stays flat means the agent is doing more RIGHT things,
not just more things.

Anti-reward-hacking note: because the underlying reward is the verifiable
6-component rubric (RLVR), every component is computable from action
history + cluster state. No learned reward model and no LLM-as-judge to
game.
"""

from __future__ import annotations

from typing import Any, List, Optional

from incident_commander_env.models import IncidentAction
from incident_commander_env.server.environment import IncidentCommanderEnv
from incident_commander_env.server.grading.components import RewardBreakdown
from training.eval_runner import parse_action_response


# Sidecar so the training notebook can read per-completion breakdowns for
# the per-component reward plot. Cleared by ``reset_history()`` between runs.
_LAST_BREAKDOWNS: List[RewardBreakdown] = []


def reset_history() -> None:
    """Clear the sidecar between runs / epochs / notebook re-executions."""
    _LAST_BREAKDOWNS.clear()


def get_recent_breakdowns(n: Optional[int] = None) -> List[RewardBreakdown]:
    """Read the last N breakdowns (or all if ``n=None``).

    Called by the notebook's plot cell after training completes. Order
    matches the order the trainer sampled completions, so ``[i]`` is the
    i-th scored completion across the full run.
    """
    if n is None:
        return list(_LAST_BREAKDOWNS)
    return list(_LAST_BREAKDOWNS[-n:])


def _extract_completion_text(comp: Any) -> str:
    """TRL passes ``completions`` as either strings or lists of chat-message dicts.

    Dataset format decides the shape - a string dataset gives strings,
    a conversational dataset gives lists of ``{"role": ..., "content": ...}``.
    Normalise to the assistant's text either way; fall back to ``str(comp)``.
    """
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        # Take the last assistant message; if none, join the tail contents.
        for msg in reversed(comp):
            if msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        return "".join(str(m.get("content", "")) for m in comp)
    return str(comp)


def _score_one(
    completion_text: str,
    task_id: str,
    seed: int,
    difficulty: float,
) -> float:
    """Score a single completion: parse -> reset env -> step env -> total().

    Any failure returns 0.0 rather than crashing the trainer. The sidecar
    gets a zeroed breakdown in the failure case so the plot's x-axis stays
    aligned with the trainer's step count.
    """
    parsed = parse_action_response(completion_text)
    try:
        action = IncidentAction(
            action_type=parsed.get("action_type") or "list_services",
            target_service=parsed.get("target_service"),
            parameters=parsed.get("parameters") or {},
        )
    except Exception:
        action = IncidentAction(action_type="list_services")

    env = IncidentCommanderEnv()
    try:
        env.reset(task_id=task_id, seed=seed, difficulty=difficulty)
        obs = env.step(action)
        breakdown = getattr(env, "_last_breakdown", None)
        if breakdown is None:
            breakdown = RewardBreakdown.zero()
            # Fall back to the observation's scalar reward if the env didn't
            # attach a breakdown (older env versions).
            fallback = float(getattr(obs, "reward", 0.0) or 0.0)
            _LAST_BREAKDOWNS.append(breakdown)
            return fallback
        _LAST_BREAKDOWNS.append(breakdown)
        return float(breakdown.total())
    except Exception:
        _LAST_BREAKDOWNS.append(RewardBreakdown.zero())
        return 0.0


def grpo_reward_fn(
    prompts: List[Any],
    completions: List[Any],
    **kwargs: Any,
) -> List[float]:
    """The reward function TRL's ``GRPOTrainer`` calls.

    Args:
        prompts:     list of prompt values (usually chat message lists),
                     one per completion. Not read by us - the env is fully
                     seeded by kwargs - but part of the required signature.
        completions: list of LLM outputs to score. Length is
                     ``len(prompt_batch) * num_generations``.
        kwargs:      TRL forwards every non-``prompt`` column from the
                     dataset here as parallel lists. We read ``task_id``,
                     ``seed``, ``difficulty``. Defaults kick in if a column
                     is missing so a smoke test with a bare-bones dataset
                     still runs.

    Returns:
        List of floats, one per completion. Absolute scale doesn't matter
        for GRPO (it relativises within each group); we return
        ``RewardBreakdown.total()`` directly so absolute values also line up
        with the eval runner's per-episode ``breakdown_totals`` for
        cross-checking.
    """
    task_ids: List[str] = kwargs.get("task_id", []) or []
    seeds: List[int] = kwargs.get("seed", []) or []
    difficulties: List[float] = kwargs.get("difficulty", []) or []

    rewards: List[float] = []
    for i, completion in enumerate(completions):
        task_id = task_ids[i] if i < len(task_ids) else "oom_crash"
        seed = int(seeds[i]) if i < len(seeds) else i
        difficulty = float(difficulties[i]) if i < len(difficulties) else 0.5
        text = _extract_completion_text(completion)
        rewards.append(_score_one(text, task_id, seed, difficulty))
    return rewards


__all__ = [
    "grpo_reward_fn",
    "reset_history",
    "get_recent_breakdowns",
]
