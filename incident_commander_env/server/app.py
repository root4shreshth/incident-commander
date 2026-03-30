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
from incident_commander_env.server.environment import IncidentCommanderEnv

STATIC_DIR = Path(__file__).parent / "static"


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

env = IncidentCommanderEnv()


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
    """Reset environment and start a new incident episode."""
    obs = env.reset(task_id=request.task_id)
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": obs.done,
        "info": {
            "task_id": env.state.task_id,
            "max_steps": env.state.max_steps,
            "episode_id": env.state.episode_id,
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


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List available tasks."""
    from incident_commander_env.server.scenarios import SCENARIO_REGISTRY

    tasks = {}
    for task_id, scenario_cls in SCENARIO_REGISTRY.items():
        s = scenario_cls()
        tasks[task_id] = {
            "difficulty": s.difficulty,
            "description": s.description,
            "max_steps": s.max_steps,
        }
    return {"tasks": tasks}


def main() -> None:
    """Entry point for the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
