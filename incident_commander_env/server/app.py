"""FastAPI application for IncidentCommanderEnv.

Exposes POST /reset, POST /step, GET /state endpoints as required by OpenEnv spec.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from incident_commander_env.models import IncidentAction, IncidentObservation, IncidentState
from incident_commander_env.server.backends import get_backend
from incident_commander_env.server.environment import IncidentCommanderEnv
from incident_commander_env.server.coach import (
    IDEAL_TRAJECTORIES,
    LEARNING_CONTEXT,
    build_postmortem,
    compute_hint,
    explain_observation,
)

STATIC_DIR = Path(__file__).parent / "static"

# Where eval-runner writes per-episode JSONL traces. The dashboard's
# observe mode consumes these via /watch/<run_id>.
RUNS_ROOT = Path(os.getenv("RUNS_ROOT", "runs")).resolve()


app = FastAPI(
    title="IncidentCommanderEnv",
    description=(
        "SRE/DevOps Cloud Incident Response & Diagnostics environment. "
        "An AI agent acts as an on-call SRE, diagnosing and remediating "
        "production incidents across a simulated microservices cluster."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# BACKEND env var picks the substrate: "sim" (default), "real", or "code_aware".
# Same OpenEnv API regardless; only the execution path differs.
env = IncidentCommanderEnv(backend=get_backend())

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    """Serve the interactive UI."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {
        "name": "IncidentCommanderEnv",
        "version": "0.1.0",
        "description": "SRE/DevOps Cloud Incident Response & Diagnostics",
        "endpoints": {
            "POST /reset": "Start new incident episode",
            "POST /step": "Execute an SRE action",
            "GET /state": "Get current episode state",
            "GET /health": "Liveness check",
            "GET /tasks": "List available tasks",
        },
    }


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None  # OpenEnv: deterministic episodes for reproducible training
    difficulty: float = 0.5  # 0.0 easiest, 1.0 hardest; drives parametric scenario instance


class StepRequest(BaseModel):
    action_type: str
    target_service: Optional[str] = None
    parameters: Dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: IncidentObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """Reset environment and start a new incident episode.

    Optional `seed` makes the episode deterministic — same seed plus same
    action sequence yields identical observations and rewards. This is the
    OpenEnv contract for reproducible RL training.
    """
    obs = env.reset(
        task_id=request.task_id,
        seed=request.seed,
        difficulty=request.difficulty,
    )
    return {
        "observation": obs.model_dump(),
        "reward": 0.01,
        "done": obs.done,
        "info": {
            "task_id": env.state.task_id,
            "max_steps": env.state.max_steps,
            "episode_id": env.state.episode_id,
            "seed": request.seed,
            "difficulty": request.difficulty,
        },
    }


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    """Execute an action and return observation, reward, done, info."""
    action = IncidentAction(
        action_type=request.action_type,
        target_service=request.target_service,
        parameters=request.parameters,
    )
    obs = env.step(action)

    info: Dict[str, Any] = {
        "step_count": env.state.step_count,
        "task_id": env.state.task_id,
    }
    if obs.done:
        info["final_score"] = env.state.current_score
        info["grade_details"] = env.get_grade_details()

    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    """Get current episode state."""
    return env.state.model_dump()


@app.get("/reward-breakdown")
def reward_breakdown() -> Dict[str, Any]:
    """Per-component breakdown of the most recent step's reward.

    Returns the six independent components (diagnostic, correct_op, resolution,
    format, efficiency, penalty) plus their sum. Useful for the dashboard
    observability mode and for confirming TRL's wandb logs match what the env
    actually emitted.
    """
    bd = getattr(env, "_last_breakdown", None)
    if bd is None:
        return {"breakdown": None, "total": None, "step": env.state.step_count}
    return {
        "breakdown": bd.to_dict(),
        "total": bd.total(),
        "step": env.state.step_count,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/backend")
def backend_info() -> Dict[str, Any]:
    """Which backend is the env wired to (sim/real/code_aware)."""
    return {
        "name": env.backend.name,
        "available_backends": ["sim", "real", "code_aware"],
        "default": "sim",
    }


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List available tasks."""
    from incident_commander_env.server.scenarios import SCENARIO_REGISTRY

    tasks = {}
    for task_id, scenario_cls in SCENARIO_REGISTRY.items():
        s = scenario_cls()
        ctx = LEARNING_CONTEXT.get(task_id, {})
        tasks[task_id] = {
            "difficulty": s.difficulty,
            "description": s.description,
            "max_steps": s.max_steps,
            "backstory": ctx.get("backstory", ""),
            "learning_goals": ctx.get("learning_goals", []),
            "est_minutes": ctx.get("est_minutes", 10),
            "prerequisite": ctx.get("prerequisite"),
            "skill_tag": ctx.get("skill_tag", s.difficulty.upper()),
        }
    return {"tasks": tasks}


@app.get("/coach/hint")
def coach_hint() -> Dict[str, Any]:
    """Return a contextual, rule-based hint based on the current game state."""
    if not env._scenario or not env._cluster:
        return {
            "hint": "Pick a scenario and click Start Incident. Your AI coach will appear once the incident begins.",
            "suggested_action": None,
            "tone": "neutral",
        }
    return compute_hint(
        task_id=env._state.task_id,
        cluster=env._cluster,
        action_history=env._action_history,
        step_count=env._state.step_count,
        max_steps=env._scenario.max_steps,
    )


class ExplainRequest(BaseModel):
    last_action: Optional[Dict[str, Any]] = None
    last_message: Optional[str] = None


@app.post("/coach/explain")
def coach_explain(request: ExplainRequest) -> Dict[str, Any]:
    """Explain the most recent observation in plain English."""
    if not env._scenario:
        return {"explanation": "Start an incident first, then click 'Why?' on any action result for an explanation."}
    return explain_observation(
        task_id=env._state.task_id,
        last_action=request.last_action or {},
        last_message=request.last_message or "",
    )


@app.get("/postmortem")
def postmortem() -> Dict[str, Any]:
    """Build a structured post-mortem for the review screen."""
    if not env._scenario or not env._cluster:
        return {"error": "No episode has been run yet."}
    return build_postmortem(
        scenario=env._scenario,
        action_history=env._action_history,
        cluster=env._cluster,
        state=env._state,
    )


@app.get("/ideal-trajectory/{task_id}")
def ideal_trajectory(task_id: str) -> Dict[str, Any]:
    """Return what a senior SRE would have done."""
    traj = IDEAL_TRAJECTORIES.get(task_id)
    if not traj:
        return {"error": f"No ideal trajectory available for task {task_id}"}
    return {"task_id": task_id, "trajectory": traj}


# ---------------------------------------------------------------------------
# Observe-mode: the dashboard's "watch a trained agent" surface.
#
# Trained agent runs (run by eval_runner with `runs_root=...`) leave JSONL
# traces under `runs/<run_id>/episode.jsonl`. The dashboard's observe page
# (static/observe.html) lists them via /runs and replays one via /watch.
# This is what powers the sim-to-real demo recording.
# ---------------------------------------------------------------------------


@app.get("/runs")
def list_runs() -> Dict[str, Any]:
    """List available recorded trained-agent runs."""
    try:
        from training.episode_logger import iter_runs
    except Exception as exc:  # pragma: no cover — defensive
        return {"runs": [], "error": f"training extras not installed: {exc}"}
    if not RUNS_ROOT.exists():
        return {"runs": [], "runs_root": str(RUNS_ROOT)}
    runs = list(iter_runs(RUNS_ROOT))
    return {"runs": runs, "runs_root": str(RUNS_ROOT)}


@app.get("/watch/{run_id}")
def watch_run(run_id: str) -> Dict[str, Any]:
    """Return the events of a single recorded run for replay in observe mode."""
    try:
        from training.episode_logger import read_episode
    except Exception as exc:  # pragma: no cover — defensive
        return {"error": f"training extras not installed: {exc}", "events": []}
    # Sanitize: prevent traversal outside RUNS_ROOT
    safe_id = run_id.replace("..", "").replace("/", "").replace("\\", "")
    target = RUNS_ROOT / safe_id / "episode.jsonl"
    if not target.exists():
        return {"error": f"run not found: {safe_id}", "events": []}
    events = read_episode(target)
    summary: Dict[str, Any] = {"run_id": safe_id, "n_events": len(events)}
    start = next((e for e in events if e.get("type") == "start"), None)
    end = next((e for e in reversed(events) if e.get("type") == "end"), None)
    if start:
        for k in ("task_id", "seed", "model", "alert", "max_steps"):
            if k in start:
                summary[k] = start[k]
    if end:
        summary["resolved"] = end.get("resolved")
        summary["score"] = end.get("score")
        summary["steps_used"] = end.get("steps_used")
    return {"summary": summary, "events": events}


@app.get("/observe")
def observe_page():
    """Serve the agent-observation dashboard."""
    page = STATIC_DIR / "observe.html"
    if page.exists():
        return FileResponse(page)
    return {"error": "observe.html missing — copy it under static/."}


def main() -> None:
    """Entry point for the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
