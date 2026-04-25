"""SFT dataset builder for IncidentCommanderEnv.

Converts the senior-SRE behavioral-clone trajectories in `coach.IDEAL_TRAJECTORIES`
into a HuggingFace `Dataset` of `[system, user, assistant]` chat conversations.

Each (state, action) pair from a trajectory becomes one training row:
    system    -> SRE persona + action vocabulary (matches inference.py SYSTEM_PROMPT)
    user      -> rendered observation text the env actually emits at that step
    assistant -> JSON action with a `thinking` field set to the trajectory's `why`

We replay every trajectory under multiple seeds (default 5 per scenario family)
to expose the model to varied surface forms of the same incident — different
log timestamps, slightly different metric noise, different OOM target services
(in the OOM family) — so the policy learns the *shape* of the right move
rather than memorizing one fixed render.

Total dataset size with defaults:
    3 families x 5 seeds x ~6 steps avg = ~90 rows.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from incident_commander_env.models import IncidentAction
from incident_commander_env.server.coach import IDEAL_TRAJECTORIES, LEARNING_CONTEXT
from incident_commander_env.server.environment import IncidentCommanderEnv


# The system prompt mirrors the one in inference.py so SFT training format
# matches the deployment chat shape exactly.
SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) responding to a production incident in a microservices cluster.

You must diagnose the issue and resolve it by taking actions. Always respond with a single JSON object (no markdown, no explanation outside the JSON):

{
  "thinking": "Brief reasoning about what to do next",
  "action_type": "one of: list_services, describe_service, read_logs, check_metrics, restart_service, scale_service, rollback_deployment, run_diagnostic, update_config, resolve_incident",
  "target_service": "service-name or null",
  "parameters": {}
}

Available actions:
- list_services: Get overview of all services (no target needed)
- describe_service: Get detailed info for a service (config, deployment history, dependencies)
- read_logs: Read service logs. params: {lines: 50, severity: "ERROR"}
- check_metrics: Check CPU, memory, latency, error rate. params: {metric: "all"}
- restart_service: Restart a service. params: {memory_limit: "512Mi"} (optional)
- scale_service: Change replicas. params: {replicas: 3}
- rollback_deployment: Rollback to version. params: {to_version: "v2.3.1"}
- run_diagnostic: Run diagnostic. params: {command: "check_connectivity|check_health|check_resources"}
- update_config: Update config. params: {key: "db.pool.max_size", value: 50}
- resolve_incident: Declare resolved. params: {root_cause: "...", resolution: "..."}

Strategy:
1. Start with list_services to see the cluster state
2. Investigate unhealthy services: read_logs, check_metrics, describe_service
3. Trace the dependency chain to find the root cause
4. Apply the correct fix (restart, rollback, config change)
5. Call resolve_incident when done

IMPORTANT: Don't restart services unless you've diagnosed the issue. Rollback is preferred over restart when a bad deployment is the cause."""


def _format_action_completion(step: Dict[str, Any]) -> str:
    """Render a trajectory step as the assistant's JSON-with-thinking response."""
    payload = {
        "thinking": step.get("why", ""),
        "action_type": step["action"],
        "target_service": step.get("target"),
        "parameters": step.get("params", {}) or {},
    }
    return json.dumps(payload, indent=2)


def _build_user_message(observation_message: str, alert: Optional[str], step_idx: int) -> str:
    """Construct the user-turn that an LLM sees for this step.

    For step 0 we include the alert framing; for subsequent steps we just hand
    over the observation text and ask for the next action. This matches the
    pattern inference.py uses at deployment.
    """
    if step_idx == 0 and alert:
        return (
            f"INCIDENT ALERT:\n{observation_message}\n\n"
            "Begin investigation. What is your first action?"
        )
    return f"Action result:\n{observation_message}\n\nWhat is your next action?"


def replay_trajectory(
    task_id: str,
    seed: int,
    trajectory: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Replay one IDEAL trajectory under a fresh seed and emit chat rows.

    Returns one row per step in chat-message format:
        [{"role": "system", "content": ...},
         {"role": "user",   "content": ...},
         {"role": "assistant", "content": ...}]

    Side note: the assistant content is the canonical "right answer" — the
    senior-SRE move + reasoning. SFT teaches the policy to emit this shape.
    """
    env = IncidentCommanderEnv()
    reset_obs = env.reset(task_id=task_id, seed=seed)

    rows: List[Dict[str, str]] = []
    last_message = reset_obs.message
    last_alert = reset_obs.alert

    for step_idx, traj_step in enumerate(trajectory):
        user_msg = _build_user_message(last_message, last_alert, step_idx)
        assistant_msg = _format_action_completion(traj_step)
        rows.append({
            "system": SYSTEM_PROMPT,
            "user": user_msg,
            "assistant": assistant_msg,
        })

        # Step the env forward using the trajectory's prescribed action so the
        # next user message reflects the post-action observation.
        action = IncidentAction(
            action_type=traj_step["action"],
            target_service=traj_step.get("target"),
            parameters=traj_step.get("params") or {},
        )
        try:
            obs = env.step(action)
            last_message = obs.message
            last_alert = None
            if obs.done:
                break
        except Exception:
            # Defensive: if the trajectory step fails for a particular seed
            # (e.g., parametric scenarios randomized away from the hardcoded
            # service in IDEAL_TRAJECTORIES), stop replay early. The rows
            # collected so far are still valid (state, action) pairs.
            break

    return rows


def build_sft_dataset(
    n_seeds_per_family: int = 5,
    families: Optional[List[str]] = None,
    seed_offset: int = 0,
) -> List[Dict[str, str]]:
    """Build the full SFT dataset as a list of chat rows.

    Args:
        n_seeds_per_family: number of distinct seed replays per scenario family
        families: subset of family names; default = all in IDEAL_TRAJECTORIES
        seed_offset: shift seeds so train and held-out eval don't overlap

    Returns:
        List of chat-row dicts. Convert to HuggingFace Dataset via
        `datasets.Dataset.from_list(rows)` in the Colab.
    """
    families = families or list(IDEAL_TRAJECTORIES.keys())
    rows: List[Dict[str, str]] = []
    for family in families:
        trajectory = IDEAL_TRAJECTORIES.get(family)
        if not trajectory:
            continue
        for s in range(n_seeds_per_family):
            seed = seed_offset + s
            rows.extend(replay_trajectory(family, seed, trajectory))
    return rows


def to_hf_dataset(rows: List[Dict[str, str]]):
    """Convert chat rows to a HuggingFace Dataset (lazy import — Colab-only)."""
    from datasets import Dataset
    return Dataset.from_list(rows)


def to_chat_messages(row: Dict[str, str]) -> List[Dict[str, str]]:
    """Convert a single row to the chat-message list format that
    `tokenizer.apply_chat_template` expects.
    """
    return [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
        {"role": "assistant", "content": row["assistant"]},
    ]


__all__ = [
    "SYSTEM_PROMPT",
    "build_sft_dataset",
    "replay_trajectory",
    "to_hf_dataset",
    "to_chat_messages",
]
