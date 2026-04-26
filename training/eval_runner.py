"""Eval runner for IncidentCommanderEnv.

Runs N episodes against a configurable policy (random, base model, SFT,
SFT+GRPO) and returns a structured `EvalReport` with the numbers that go
into the README's results table:

  - success_rate (per scenario family)
  - avg_score (per family)
  - avg_steps_used (per family)
  - action_distribution (per family)
  - per_episode_breakdown (the 6 reward components summed across the episode)

The runner is model-agnostic — it takes an `act` callable that maps
(observation_message, conversation_history) -> action_dict. Concrete
adapters for HF transformers pipelines + a random baseline are provided.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from incident_commander_env.models import IncidentAction
from incident_commander_env.server.environment import IncidentCommanderEnv
from training.episode_logger import EpisodeLogger


# A policy is a callable: given the conversation history (a list of {role, content}),
# return the next action as a dict {action_type, target_service, parameters}.
ActFn = Callable[[List[Dict[str, str]]], Dict[str, Any]]


@dataclass
class EpisodeRecord:
    """One episode's complete trace for downstream analysis + plotting."""
    task_id: str
    seed: int
    steps_used: int
    score: float
    resolved: bool
    rewards: List[float] = field(default_factory=list)
    breakdown_totals: Dict[str, float] = field(default_factory=dict)  # summed components
    actions: List[Tuple[str, Optional[str]]] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated metrics across N episodes per scenario family."""
    condition_name: str  # "random" | "base" | "sft" | "sft+grpo"
    n_episodes: int
    by_family: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    episodes: List[EpisodeRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_name": self.condition_name,
            "n_episodes": self.n_episodes,
            "by_family": self.by_family,
            "episodes": [
                {
                    "task_id": e.task_id,
                    "seed": e.seed,
                    "steps_used": e.steps_used,
                    "score": e.score,
                    "resolved": e.resolved,
                    "rewards": e.rewards,
                    "breakdown_totals": e.breakdown_totals,
                    "actions": [list(a) for a in e.actions],
                }
                for e in self.episodes
            ],
        }


# ---------------------------------------------------------------------------
# Action-string parsing — robust against various LLM output shapes
# ---------------------------------------------------------------------------

_FALLBACK_ACTION = {
    "action_type": "list_services",
    "target_service": None,
    "parameters": {},
}


def parse_action_response(response_text: str) -> Dict[str, Any]:
    """Extract a structured action dict from raw LLM output.

    Handles:
      - bare JSON object
      - JSON wrapped in ```json fences
      - Extra prose surrounding a JSON object
      - Malformed JSON (returns fallback list_services so the episode continues)
    """
    text = (response_text or "").strip()
    if text.startswith("```"):
        # Strip code fences
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    try:
        parsed = json.loads(text)
        return _coerce_action(parsed)
    except json.JSONDecodeError:
        pass
    # Try to extract a {...} substring
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return _coerce_action(parsed)
        except json.JSONDecodeError:
            pass
    return dict(_FALLBACK_ACTION)


def _coerce_action(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_type": parsed.get("action_type", "list_services"),
        "target_service": parsed.get("target_service"),
        "parameters": parsed.get("parameters", {}) or {},
    }


# ---------------------------------------------------------------------------
# Policies — concrete `ActFn` implementations
# ---------------------------------------------------------------------------

def random_policy(rng_seed: int = 0) -> ActFn:
    """Uniform-random policy across the action space.

    Used as the floor baseline in the eval table. If even the random policy
    occasionally resolves a scenario, that's a sign the task isn't impossibly
    hard — which is what the docs explicitly warn against ("RL only works if
    success probability is greater than zero").
    """
    import random as _random
    rng = _random.Random(rng_seed)
    actions = [
        ("list_services", None),
        ("read_logs", "payment-service"),
        ("read_logs", "order-service"),
        ("read_logs", "postgres-db"),
        ("check_metrics", "payment-service"),
        ("describe_service", "order-service"),
        ("restart_service", "payment-service"),
        ("rollback_deployment", "order-service"),
    ]
    def _act(history: List[Dict[str, str]]) -> Dict[str, Any]:
        a, t = rng.choice(actions)
        params: Dict[str, Any] = {}
        if a == "restart_service":
            params = {"memory_limit": rng.choice(["256Mi", "512Mi", "1024Mi"])}
        elif a == "rollback_deployment":
            params = {"to_version": "v2.3.1"}
        return {"action_type": a, "target_service": t, "parameters": params}
    return _act


def hf_pipeline_policy(generate_fn: Callable[[List[Dict[str, str]]], str]) -> ActFn:
    """Wrap any HF chat-completion callable into an `ActFn`.

    The `generate_fn` takes a list of chat messages and returns the assistant's
    raw text. We parse that text into an action dict. Used by the Colab to
    plug in `tokenizer.apply_chat_template + model.generate`.
    """
    def _act(history: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            response_text = generate_fn(history)
        except Exception:
            return dict(_FALLBACK_ACTION)
        return parse_action_response(response_text)
    return _act


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def run_episode(
    task_id: str,
    seed: int,
    act: ActFn,
    *,
    system_prompt: str,
    max_history_messages: int = 16,
    difficulty: float = 0.5,
    runs_root: Optional[str] = None,
    run_id: Optional[str] = None,
    model_label: Optional[str] = None,
) -> EpisodeRecord:
    """Run one episode against the given policy.

    Returns a typed `EpisodeRecord` with per-step rewards + the summed reward
    component breakdown for later plotting.

    If `runs_root` is provided, the episode is also written as a JSONL trace
    under `runs_root/<run_id>/episode.jsonl` for the dashboard's observe
    mode (`/watch/<run_id>`).
    """
    env = IncidentCommanderEnv()
    reset_obs = env.reset(task_id=task_id, seed=seed, difficulty=difficulty)

    logger: Optional[EpisodeLogger] = None
    if runs_root is not None:
        logger = EpisodeLogger.for_run(
            runs_root, task_id, seed=seed, run_id=run_id
        )
        logger.__enter__()
        logger.start({
            "task_id": task_id,
            "seed": seed,
            "difficulty": difficulty,
            "model": model_label,
            "alert": reset_obs.alert,
            "max_steps": env.state.max_steps,
            "episode_id": env.state.episode_id,
        })

    history: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"INCIDENT ALERT:\n{reset_obs.message}\n\n"
                "Begin investigation. What is your first action?"
            ),
        },
    ]

    rewards: List[float] = []
    breakdown_totals: Dict[str, float] = defaultdict(float)
    actions: List[Tuple[str, Optional[str]]] = []
    steps_used = 0
    resolved = False
    score = 0.0

    while True:
        action_dict = act(history)
        try:
            action = IncidentAction(
                action_type=action_dict["action_type"],
                target_service=action_dict.get("target_service"),
                parameters=action_dict.get("parameters") or {},
            )
        except Exception:
            action = IncidentAction(**_FALLBACK_ACTION)

        obs = env.step(action)
        steps_used = env.state.step_count
        rewards.append(float(obs.reward or 0.0))
        actions.append((action.action_type, action.target_service))

        # Pull the per-step breakdown the env stashed and accumulate
        bd = getattr(env, "_last_breakdown", None)
        if bd is not None:
            for k, v in bd.to_dict().items():
                breakdown_totals[k] += float(v)

        if logger is not None:
            logger.step(
                step_num=steps_used,
                action={
                    "action_type": action.action_type,
                    "target_service": action.target_service,
                    "parameters": action.parameters,
                },
                observation=obs,
                reward_breakdown=bd.to_dict() if bd is not None else None,
                message=obs.message,
            )

        # Update conversation history for the next call
        history.append({
            "role": "assistant",
            "content": json.dumps({
                "action_type": action.action_type,
                "target_service": action.target_service,
                "parameters": action.parameters,
            }),
        })
        history.append({
            "role": "user",
            "content": (
                f"Action result:\n{obs.message}\n\nWhat is your next action?"
            ),
        })
        # Trim to keep within model context — keep system + last N
        if len(history) > max_history_messages + 1:
            history = [history[0]] + history[-(max_history_messages):]

        if obs.done:
            score = float(env.state.current_score or 0.0)
            resolved = bool(env.state.incident_resolved)
            break

    if logger is not None:
        logger.end({
            "resolved": resolved,
            "score": score,
            "steps_used": steps_used,
            "breakdown_totals": dict(breakdown_totals),
        })
        logger.close()
        # Phase 2 — auto-write postmortem.md alongside episode.jsonl and append
        # a row to runs/RUNBOOK.md. Failures are non-fatal: trace data is the
        # source of truth, postmortem is a derived artifact.
        try:
            from training.postmortem_writer import write_postmortem
            from pathlib import Path as _Path
            jsonl_path = _Path(logger.file_path)
            runs_root_path = jsonl_path.parent.parent
            write_postmortem(jsonl_path, runbook_path=runs_root_path / "RUNBOOK.md")
        except Exception:
            pass

    return EpisodeRecord(
        task_id=task_id,
        seed=seed,
        steps_used=steps_used,
        score=score,
        resolved=resolved,
        rewards=rewards,
        breakdown_totals=dict(breakdown_totals),
        actions=actions,
    )


def evaluate(
    condition_name: str,
    act: ActFn,
    families: List[str],
    seeds: List[int],
    *,
    system_prompt: str,
    difficulty: float = 0.5,
    runs_root: Optional[str] = None,
    on_episode: Optional[Callable[[int, int, str, int, "EpisodeRecord"], None]] = None,
) -> EvalReport:
    """Run `len(families) * len(seeds)` episodes and aggregate.

    If `runs_root` is set, every episode also emits a JSONL trace at
    `runs_root/<run_id>/episode.jsonl` so the dashboard observe mode can
    replay it back. The `condition_name` is used as part of the run id.

    `on_episode(idx, total, family, seed, record)` is called after every
    completed episode. Use it to print live progress in long-running notebooks
    so the cell doesn't appear to hang. Defaults to None (silent).
    """
    episodes: List[EpisodeRecord] = []
    total = len(families) * len(seeds)
    idx = 0
    for family in families:
        for seed in seeds:
            idx += 1
            ep = run_episode(
                task_id=family,
                seed=seed,
                act=act,
                system_prompt=system_prompt,
                difficulty=difficulty,
                runs_root=runs_root,
                model_label=condition_name,
            )
            episodes.append(ep)
            if on_episode is not None:
                try:
                    on_episode(idx, total, family, seed, ep)
                except Exception:
                    # Progress callback never blocks the eval — failures are silent.
                    pass

    by_family: Dict[str, Dict[str, Any]] = {}
    for family in families:
        eps = [e for e in episodes if e.task_id == family]
        if not eps:
            continue
        n = len(eps)
        by_family[family] = {
            "n_episodes": n,
            "success_rate": sum(1 for e in eps if e.resolved) / n,
            "avg_score": sum(e.score for e in eps) / n,
            "avg_steps_used": sum(e.steps_used for e in eps) / n,
            "action_distribution": dict(Counter(a[0] for e in eps for a in e.actions)),
            "avg_breakdown_totals": {
                k: sum(e.breakdown_totals.get(k, 0.0) for e in eps) / n
                for k in (
                    "diagnostic", "correct_op", "resolution",
                    "format", "efficiency", "penalty",
                )
            },
        }

    return EvalReport(
        condition_name=condition_name,
        n_episodes=len(episodes),
        by_family=by_family,
        episodes=episodes,
    )


__all__ = [
    "ActFn",
    "EpisodeRecord",
    "EvalReport",
    "parse_action_response",
    "random_policy",
    "hf_pipeline_policy",
    "run_episode",
    "evaluate",
]
