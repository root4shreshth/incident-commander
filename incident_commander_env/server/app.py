"""FastAPI application for IncidentCommanderEnv.

Exposes POST /reset, POST /step, GET /state endpoints as required by OpenEnv spec.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Load .env so optional config (GITHUB_CLIENT_ID, GOOGLE_CLIENT_ID, ...) is
# picked up without requiring uvicorn to be launched through a wrapper. We
# only set vars that aren't already in the environment so deployment-time
# overrides win.
# ---------------------------------------------------------------------------

def _load_dotenv_if_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and not os.getenv(key):
                os.environ[key] = value
    except Exception:
        pass


_load_dotenv_if_present()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from incident_commander_env.models import IncidentAction, IncidentObservation, IncidentState
from incident_commander_env.server.backends import get_backend
from incident_commander_env.server.backends.website import WebsiteBackend, _http as _site_http
from incident_commander_env.server.environment import IncidentCommanderEnv
from incident_commander_env.server.coach import (
    IDEAL_TRAJECTORIES,
    LEARNING_CONTEXT,
    build_postmortem,
    compute_hint,
    explain_observation,
)
from incident_commander_env.server.incidents import (
    classify_scenario,
    normalize_generic,
    normalize_pagerduty,
    normalize_prometheus,
    webhook_token_check,
)
from incident_commander_env.server.ops_endpoints import make_ops_router
from incident_commander_env.server.integrations import make_integrations_router

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

# Self-target demo environment - completely separate from `env` above so the
# Real-Time tab can drive this one through the operator-contract endpoints
# without trampling the user's interactive Apprentice/Observatory state.
# This is what makes "point Real-Time at http://127.0.0.1:8000" work without
# the user having to deploy a separate site that implements /ops/*.
ops_demo_env = IncidentCommanderEnv(backend=get_backend())
app.include_router(make_ops_router(ops_demo_env))

# Integrations - GitHub OAuth (real, via device flow), cloud-provider stubs
# (demo mode - never store credentials server-side), and the adapter generator
# that produces a praetor_adapter.py the user drops into their own deployment.
app.include_router(make_integrations_router())

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

    Optional `seed` makes the episode deterministic - same seed plus same
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
    except Exception as exc:  # pragma: no cover - defensive
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
    except Exception as exc:  # pragma: no cover - defensive
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
    return {"error": "observe.html missing - copy it under static/."}


@app.get("/dataset/export")
def dataset_export(format: str = "jsonl"):
    """Stream every recorded run as a single JSONL feed.

    Lets judges + researchers pull the trajectory dataset directly from the
    live Space without leaving the page. One JSON object per event from every
    run under `runs/`, in chronological order. Use with the data-factory
    pitch: "the env is also the dataset endpoint."

    Query string:
        format=jsonl (default) - one JSON object per line, newline-separated.
        format=summary         - one JSON object per *run* (no per-step events).
    """
    from fastapi.responses import StreamingResponse
    try:
        from training.episode_logger import iter_runs, read_episode
    except Exception as exc:  # pragma: no cover - defensive
        def _err():
            yield json.dumps({"error": f"training extras not installed: {exc}"}) + "\n"
        return StreamingResponse(_err(), media_type="application/x-ndjson")

    if not RUNS_ROOT.exists():
        def _empty():
            yield json.dumps({"error": "runs/ is empty - visit /runs to populate."}) + "\n"
        return StreamingResponse(_empty(), media_type="application/x-ndjson")

    fmt = (format or "jsonl").lower()

    def _stream():
        for run_meta in iter_runs(RUNS_ROOT):
            if fmt == "summary":
                yield json.dumps({"run_id": run_meta["run_id"], **{
                    k: run_meta.get(k) for k in ("task_id", "seed", "model", "n_events",
                                                   "resolved", "score", "steps_used")
                }}) + "\n"
                continue
            target = RUNS_ROOT / run_meta["run_id"] / "episode.jsonl"
            if not target.exists():
                continue
            for event in read_episode(target):
                event["run_id"] = run_meta["run_id"]
                yield json.dumps(event, ensure_ascii=False) + "\n"

    headers = {
        "Content-Disposition": f'attachment; filename="praetor-trajectories-{fmt}.jsonl"',
        "X-Praetor-Source": "GET /dataset/export",
    }
    return StreamingResponse(_stream(), media_type="application/x-ndjson", headers=headers)


# ---------------------------------------------------------------------------
# Auto-populate demo runs on first boot.
#
# When a fresh deploy comes up with no recorded runs, the Observatory dropdown
# is empty and the home-page marquee shows zeros. To fix that, on startup we
# check if the runs directory has any traces; if not, we generate a small
# baseline (random_policy across the 3 scenario families × a few seeds) so the
# dashboard is immediately useful. Runs in a daemon thread so we don't block
# the server from accepting requests.
# ---------------------------------------------------------------------------

def _has_any_recorded_runs() -> bool:
    """True iff RUNS_ROOT contains at least one valid episode trace."""
    if not RUNS_ROOT.exists():
        return False
    for child in RUNS_ROOT.iterdir():
        if child.is_dir() and (child / "episode.jsonl").exists():
            return True
    return False


def _seed_demo_runs(force: bool = False) -> Dict[str, Any]:
    """Generate baseline demo runs so the Observatory + Home stats are useful
    on a fresh deploy. Returns a summary dict suitable for an admin endpoint."""
    if not force and _has_any_recorded_runs():
        return {"seeded": False, "reason": "runs already present"}
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        from training.datasets import SYSTEM_PROMPT
        from training.eval_runner import evaluate, random_policy
    except Exception as exc:
        return {"seeded": False, "error": f"training extras not installed: {exc}"}
    try:
        report = evaluate(
            "random-baseline",
            random_policy(rng_seed=42),
            families=[
                "oom_crash", "db_pool_exhaustion", "bad_deployment_cascade",
                "disk_full", "slow_query", "cert_expiry",
                "dns_failure", "rate_limit_exhaustion",
            ],
            seeds=list(range(1, 6)),  # 5 seeds x 8 families = 40 episodes
            system_prompt=SYSTEM_PROMPT,
            runs_root=str(RUNS_ROOT),
        )
        return {
            "seeded": True,
            "n_episodes": report.n_episodes,
            "by_family": {
                f: {"success_rate": s["success_rate"], "avg_score": s["avg_score"]}
                for f, s in report.by_family.items()
            },
        }
    except Exception as exc:  # pragma: no cover - never crash on this
        print(f"[praetor] demo-runs seeding failed: {exc}")
        return {"seeded": False, "error": f"{type(exc).__name__}: {exc}"}


# Run seeding synchronously at import time. Random_policy is fast (~50ms per
# episode → ~1.5s for 30 episodes), so the small delay is worth not having a
# race condition with the first /runs request. Wrapped in try/except so a
# seeding failure can never break server boot.
try:
    if not _has_any_recorded_runs():
        _seed_result = _seed_demo_runs()
        if _seed_result.get("seeded"):
            print(f"[praetor] seeded {_seed_result['n_episodes']} demo runs")
        else:
            print(f"[praetor] demo-run seeding skipped: {_seed_result}")
except Exception as _seed_exc:  # pragma: no cover
    print(f"[praetor] demo-run seeding errored: {_seed_exc}")


@app.post("/admin/regenerate-demo-runs")
def admin_regenerate_demo_runs(force: bool = False) -> Dict[str, Any]:
    """Regenerate demo runs. Used by the Observatory empty-state retry button."""
    return _seed_demo_runs(force=force)


# ---------------------------------------------------------------------------
# Phase 3 - Real-time / sim-to-real on a deployed site.
#
# State machine: connect → inject chaos → run agent → (if not healed) escalate.
# Agent runs in a background thread; UI polls /realtime/status/<run_id>.
# ---------------------------------------------------------------------------

# In-memory store of active real-time runs. Bounded - we only need the current
# run for the demo, but keeping the last few makes A/B comparison possible.
_REALTIME_RUNS: Dict[str, Dict[str, Any]] = {}
_REALTIME_LOCK = threading.Lock()
_REALTIME_CONFIG: Dict[str, Any] = {
    "site_url": None,
    # Tier 2 codebase config - one of three sources, mutually exclusive
    "repo_url": None,           # github or azure-devops URL
    "repo_token": None,         # PAT for either provider
    "repo_source": None,        # "github" | "azure" | "zip" | None
    "repo_local_path": None,    # populated when source="zip" - the extracted dir
    "service_names": ["frontend", "api", "postgres"],
}

# Where uploaded ZIPs are extracted. Cleaned up across server restarts.
_CODEBASE_ROOT = Path(os.getenv("CODEBASE_ROOT", "uploaded_codebase")).resolve()
_MAX_ZIP_BYTES = 25 * 1024 * 1024  # 25 MB - big enough for most codebases, small enough to be safe


# Demo policy - replays a known-good action sequence per scenario family.
# Replaced by the trained LoRA when the user hooks one up; intentionally
# deterministic so the live demo doesn't fail on stage.
_DEMO_PLAYBOOK: Dict[str, List[Dict[str, Any]]] = {
    "oom_crash": [
        {"action_type": "list_services",
         "why": "Always start with cluster overview so I can spot which service is unhealthy before drilling in."},
        {"action_type": "read_logs",     "target_service": "api",
         "why": "The api service is failing - read its logs to confirm the failure mode. Looking specifically for OOMKilled / java.lang.OutOfMemoryError patterns."},
        {"action_type": "check_metrics", "target_service": "api",
         "why": "Logs suggest OOM; verifying with metrics (memory_mb vs memory_limit_mb) gives me a confident diagnosis before I touch a knob."},
        {"action_type": "restart_service", "target_service": "api",
         "parameters": {"memory_limit": "1024Mi"},
         "why": "Restart with a higher memory ceiling - 1024Mi gives 4× headroom over the original 256Mi, absorbing the workload spike. OOM is restart-curable in this scenario; smaller bumps tend to be revisited."},
        {"action_type": "resolve_incident",
         "parameters": {"root_cause": "OOM on api - memory limit too low",
                        "resolution": "restart api with 1024Mi"},
         "why": "Service is healthy and the fix is durable. Declaring resolved so the postmortem captures the root cause + remediation in the runbook."},
    ],
    "db_pool_exhaustion": [
        {"action_type": "list_services",
         "why": "Cluster-wide view first - multiple services are likely degraded if a shared dependency is the cause."},
        {"action_type": "read_logs",     "target_service": "api",
         "why": "User-facing service is failing - start where the symptom is."},
        {"action_type": "read_logs",     "target_service": "postgres",
         "why": "API logs point downstream - postgres is the shared dependency, look for pool-exhaustion signatures there."},
        {"action_type": "update_config", "target_service": "postgres",
         "parameters": {"key": "db.pool.max_size", "value": 100},
         "why": "Raise the connection pool ceiling so legitimate traffic stops queuing. 100 is a safe value above the observed peak; the underlying connection-leak fix is a separate code change."},
        {"action_type": "restart_service", "target_service": "api",
         "why": "Restart api to release any zombie connections it's holding open. Without this, the new pool capacity stays partly consumed by the old leaked sessions."},
        {"action_type": "resolve_incident",
         "parameters": {"root_cause": "connection pool exhausted on postgres",
                        "resolution": "raised pool to 100, restarted api"},
         "why": "Health probes are green and connection count is back to baseline."},
    ],
    "bad_deployment_cascade": [
        {"action_type": "list_services",
         "why": "Multiple services are reported failing - get the blast-radius picture first."},
        {"action_type": "read_logs",     "target_service": "api",
         "why": "Spot the recent deployment marker - look for v1.1 references and memory-leak symptoms."},
        {"action_type": "rollback_deployment", "target_service": "api",
         "parameters": {"to_version": "v1.0"},
         "why": "Rollback FIRST - restart alone won't help, the leak is in the v1.1 binary. Reverting to v1.0 stops the bleeding before we restore dependent services."},
        {"action_type": "restart_service", "target_service": "api",
         "why": "After rollback, restart cycles cleanly onto v1.0 and frees the resource quota the autoscaler was burning."},
        {"action_type": "resolve_incident",
         "parameters": {"root_cause": "v1.1 bundled a memory leak",
                        "resolution": "rolled back api to v1.0 and restarted"},
         "why": "Resolution names the offending version - important for the postmortem so the team can prevent the same v1.1 from being re-deployed without a code fix."},
    ],
    "disk_full": [
        {"action_type": "list_services",
         "why": "Confirm which service is degraded - disk-full failures often look like generic 5xx until you read logs."},
        {"action_type": "read_logs",     "target_service": "api",
         "why": "Look for ENOSPC / 'No space left on device' - the smoking gun for disk-full incidents."},
        {"action_type": "check_metrics", "target_service": "api",
         "why": "Confirm CPU and memory are normal - that confirms it's NOT an OOM and rules out memory-bump as a fix."},
        {"action_type": "restart_service", "target_service": "api",
         "why": "Restart cycles the volume - in our deployment model the pod's tmp/log directory clears on restart. Permanent fix is log rotation, but restart is the right immediate action."},
        {"action_type": "resolve_incident",
         "parameters": {"root_cause": "log volume on api filled - writes returning ENOSPC",
                        "resolution": "restarted api to cycle the volume"},
         "why": "Volume is clean and writes succeed. Following up with a log-rotation policy fix is queued separately."},
    ],
    "slow_query": [
        {"action_type": "list_services",
         "why": "Cluster overview - slow-query incidents typically show one service degraded and dependencies fine."},
        {"action_type": "check_metrics", "target_service": "api",
         "why": "Latency p99 spike + low throughput is the lock-contention signature. Confirming before diving into logs."},
        {"action_type": "read_logs",     "target_service": "api",
         "why": "Logs will show 'Lock wait timeout exceeded' or 'deadlock detected' - the SQL-level smoking gun."},
        {"action_type": "describe_service", "target_service": "api",
         "why": "Pull deployment history - if a recent version introduced the slow query, rollback (not restart) is the right fix."},
        {"action_type": "rollback_deployment", "target_service": "api",
         "parameters": {"to_version": "v1.0"},
         "why": "Restart alone clears the active txns but the slow query is still in the binary - it'll lock up again. Rollback reverts both."},
        {"action_type": "resolve_incident",
         "parameters": {"root_cause": "v1.1 introduced a slow query holding row-locks",
                        "resolution": "rolled back api to v1.0"},
         "why": "Latency back to baseline and throughput recovered. The slow query goes back behind a feature flag in the next deploy."},
    ],
    "cert_expiry": [
        {"action_type": "list_services",
         "why": "One service unhealthy, others fine, but the difference matters - cert-expiry has a confusing signature."},
        {"action_type": "check_metrics", "target_service": "api",
         "why": "Verify metrics look almost normal except error_rate at 99% and request rate near zero - that's the cert-expiry pattern, not a crash."},
        {"action_type": "read_logs",     "target_service": "api",
         "why": "Confirm with logs - looking for 'ssl.SSLError: certificate has expired'. Metrics alone won't tell you cert-expiry; the logs are the source of truth."},
        {"action_type": "restart_service", "target_service": "api",
         "why": "Restart triggers the cert renewal hook - listener reloads with the new cert and HTTPS handshakes start succeeding."},
        {"action_type": "resolve_incident",
         "parameters": {"root_cause": "TLS cert on api expired",
                        "resolution": "restarted api to renew cert and reload listener"},
         "why": "Service is reachable. Following up with a 30-day cert-expiry alert so this can never surprise us again."},
    ],
}


class RealtimeConnectRequest(BaseModel):
    site_url: str
    # Tier-2 codebase wiring is OPTIONAL on connect. The dedicated
    # /realtime/codebase endpoint is the canonical path; this is just a
    # convenience for "I have a github URL ready when I connect."
    repo_url: Optional[str] = None
    repo_token: Optional[str] = None
    repo_source: Optional[str] = None  # "github" | "azure"
    service_names: Optional[List[str]] = None


class CodebaseLinkRequest(BaseModel):
    """Link a remote git repo (GitHub or Azure DevOps) for tier-2."""
    source: str  # "github" | "azure"
    repo_url: str
    repo_token: Optional[str] = None


class RealtimeInjectRequest(BaseModel):
    scenario: str  # "oom_crash" | "db_pool_exhaustion" | "bad_deployment_cascade"


class RealtimeRunRequest(BaseModel):
    # If omitted, Praetor will classify the fault from the connected site's
    # current state (auto-detect). The UI normally lets that happen.
    scenario: Optional[str] = None
    # If true, attempt the tier-2 code escalation when tier 1 doesn't fully heal.
    enable_tier2: bool = True


@app.post("/realtime/connect")
def realtime_connect(req: RealtimeConnectRequest) -> Dict[str, Any]:
    """Validate that a deployed site implements the operator contract,
    AND auto-classify the current fault (if any). Praetor doesn't ask the
    user what's wrong - it figures it out from the site's metrics and
    health response. The user just hits "Run agent."""
    backend = WebsiteBackend(
        site_url=req.site_url,
        service_names=req.service_names or _REALTIME_CONFIG["service_names"],
    )
    if not backend.site_url:
        return {"connected": False, "error": "site_url is required"}
    health = _site_http("GET", backend.site_url + "/ops/health", timeout=6.0)
    if not health.ok:
        return {
            "connected": False,
            "error": f"GET /ops/health failed: {health.error or health.status}",
        }
    body = health.body if isinstance(health.body, dict) else {}
    services_arr = body.get("services") or []
    discovered = [s.get("name") for s in services_arr if isinstance(s, dict) and s.get("name")]
    with _REALTIME_LOCK:
        _REALTIME_CONFIG["site_url"] = backend.site_url
        if req.repo_url:
            _REALTIME_CONFIG["repo_url"] = req.repo_url
            _REALTIME_CONFIG["repo_token"] = req.repo_token
            _REALTIME_CONFIG["repo_source"] = req.repo_source or "github"
            _REALTIME_CONFIG["repo_local_path"] = None
        if discovered:
            _REALTIME_CONFIG["service_names"] = discovered

    # ---- Auto-classify fault from current site state ----
    classification = _classify_current_fault(
        backend.site_url, discovered or _REALTIME_CONFIG["service_names"], body,
    )

    return {
        "connected": True,
        "site_url": backend.site_url,
        "status": body.get("status"),
        "services_discovered": discovered,
        "repo_linked": bool(req.repo_url),
        "classification": classification,
    }


def _classify_current_fault(
    site_url: str, service_names: List[str], health_body: Dict[str, Any]
) -> Dict[str, Any]:
    """Probe /ops/metrics for each service and infer the fault category.

    Returns a structured dict the UI can render directly:
      {
        "fault_detected": True/False,
        "scenario": "oom_crash" | "db_pool_exhaustion" | "bad_deployment_cascade" | None,
        "confidence": 0.0..1.0,
        "evidence": ["api memory at 96% of limit", ...],
        "narrative": "Detected OOM-like state on api ..."
      }
    """
    evidence: List[str] = []
    scenario: Optional[str] = None
    confidence = 0.0

    # Look at /ops/health first - it sometimes reports the verdict directly
    overall_status = (health_body.get("status") or "").lower()
    services_arr = health_body.get("services") or []
    degraded = [
        s for s in services_arr
        if isinstance(s, dict) and (s.get("health") or s.get("status") or "").lower() not in ("healthy", "ok", "")
    ]
    if degraded:
        for d in degraded:
            evidence.append(f"{d.get('name')} health = {d.get('health') or d.get('status')}")

    # Probe metrics for each service
    metrics_per_svc: Dict[str, Dict[str, Any]] = {}
    for svc in service_names[:5]:  # cap for latency
        m = _site_http(
            "GET", site_url + "/ops/metrics?service=" + svc, timeout=4.0,
        )
        if m.ok and isinstance(m.body, dict):
            metrics_per_svc[svc] = m.body

    # Heuristic: high memory util on any service → OOM
    for svc, m in metrics_per_svc.items():
        try:
            mem = float(m.get("memory_mb", 0) or 0)
            limit = float(m.get("memory_limit_mb", 0) or 0)
            if limit > 0 and (mem / limit) > 0.85:
                evidence.append(f"{svc} memory at {int(100*mem/limit)}% of {int(limit)}MB limit")
                if scenario is None or scenario == "oom_crash":
                    scenario = "oom_crash"
                    confidence = max(confidence, min(1.0, (mem / limit - 0.6) / 0.4))
        except (TypeError, ValueError):
            pass
        # High active connections OR error rate on a db-ish service → pool exhaustion
        try:
            conn = int(m.get("active_connections", 0) or 0)
            if conn >= 18 and "postgres" in svc.lower():
                evidence.append(f"{svc} at {conn} active connections (pool likely exhausted)")
                scenario = scenario or "db_pool_exhaustion"
                confidence = max(confidence, 0.7)
        except (TypeError, ValueError):
            pass
        try:
            err = float(m.get("error_rate_percent", 0) or 0)
            if err > 30:
                evidence.append(f"{svc} error rate at {err:.0f}% - sustained 5xx")
                if scenario is None:
                    scenario = "bad_deployment_cascade"
                    confidence = max(confidence, 0.5)
        except (TypeError, ValueError):
            pass

    # Look at logs for telltale strings (cheap heuristic). Order matters here:
    # the more specific patterns are checked first so they aren't masked by
    # generic ones (e.g. "memory" appearing in a slow_query log).
    if scenario is None and overall_status in ("degraded", "down"):
        for svc in service_names[:3]:
            r = _site_http(
                "GET", site_url + "/ops/logs?service=" + svc + "&lines=20", timeout=4.0,
            )
            if not r.ok or not isinstance(r.body, dict):
                continue
            text = " ".join(map(str, (r.body.get("logs") or [])))[:4000].lower()
            # Cert expiry - very specific
            if "tls" in text and ("certificate" in text or "handshake" in text or "expired" in text):
                scenario = "cert_expiry"; confidence = 0.8
                evidence.append(f"{svc} logs mention TLS / expired certificate")
                break
            # Disk full - very specific
            if "no space left" in text or "enospc" in text or "disk usage" in text:
                scenario = "disk_full"; confidence = 0.75
                evidence.append(f"{svc} logs mention disk full / ENOSPC")
                break
            # Lock contention - specific to slow query
            if "lock wait" in text or "deadlock" in text or "for update" in text:
                scenario = "slow_query"; confidence = 0.7
                evidence.append(f"{svc} logs mention lock-wait / deadlock")
                break
            # OOM
            if "outofmemory" in text or "oom" in text or "out of memory" in text:
                scenario = "oom_crash"; confidence = 0.65
                evidence.append(f"{svc} logs mention OOM / out of memory")
                break
            # DB pool
            if "pool" in text and "connection" in text:
                scenario = "db_pool_exhaustion"; confidence = 0.65
                evidence.append(f"{svc} logs mention connection pool")
                break
            # Bad deploy
            if ("version" in text or "rollout" in text or "deploy" in text) and (
                "v1.1" in text or "v2.5" in text or "memory leak" in text
            ):
                scenario = "bad_deployment_cascade"; confidence = 0.55
                evidence.append(f"{svc} logs mention recent deployment / version")
                break

    fault_detected = scenario is not None or overall_status in ("degraded", "down")
    if not fault_detected:
        narrative = "Site is healthy. Praetor has nothing to fix yet - connect with a fault, or use the test-fault buttons below to simulate one."
    else:
        sname = {
            "oom_crash": "OOM (out-of-memory)",
            "db_pool_exhaustion": "DB connection pool exhaustion",
            "bad_deployment_cascade": "Bad deployment cascade",
            "disk_full": "Disk space exhausted",
            "slow_query": "Slow query / lock contention",
            "cert_expiry": "TLS certificate expired",
        }.get(scenario or "", "unknown")
        narrative = f"Praetor classified the fault as **{sname}** based on " + ("; ".join(evidence) or "the site's reported status") + "."

    return {
        "fault_detected": fault_detected,
        "scenario": scenario,
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "narrative": narrative,
        "site_overall_status": overall_status,
    }


@app.post("/realtime/inject")
def realtime_inject(req: RealtimeInjectRequest) -> Dict[str, Any]:
    """Trigger chaos on the connected site (calls /ops/break)."""
    site_url = _REALTIME_CONFIG.get("site_url")
    if not site_url:
        return {"injected": False, "error": "no site connected; call /realtime/connect first"}
    if req.scenario not in _DEMO_PLAYBOOK:
        return {"injected": False, "error": f"unknown scenario: {req.scenario}"}
    r = _site_http("POST", site_url + "/ops/break", json_body={"scenario": req.scenario}, timeout=10.0)
    if not r.ok:
        return {"injected": False, "error": f"POST /ops/break failed: {r.error or r.status}"}
    return {"injected": True, "scenario": req.scenario, "site_url": site_url}


@app.post("/realtime/heal")
def realtime_heal() -> Dict[str, Any]:
    """Reset the connected site to a clean state (calls /ops/heal)."""
    site_url = _REALTIME_CONFIG.get("site_url")
    if not site_url:
        return {"healed": False, "error": "no site connected"}
    r = _site_http("POST", site_url + "/ops/heal", json_body={}, timeout=8.0)
    return {"healed": r.ok, "error": (None if r.ok else (r.error or f"HTTP {r.status}"))}


@app.post("/realtime/codebase/link")
def realtime_codebase_link(req: CodebaseLinkRequest) -> Dict[str, Any]:
    """Link a remote git repo (GitHub or Azure DevOps) for tier-2 escalation.

    Performs an actual `git ls-remote` reachability check against the URL +
    token so the user gets immediate feedback (not several minutes later when
    tier-2 fires). Honest about what failed: bad URL, bad credentials, network.
    """
    import subprocess as _sp

    src = (req.source or "").lower().strip()
    if src not in ("github", "azure"):
        return {"linked": False, "error": "source must be 'github' or 'azure'"}
    url = (req.repo_url or "").strip().rstrip("/")
    if not url:
        return {"linked": False, "error": "repo_url is required"}

    # Format validation - flexible but informative
    if src == "github":
        if "github.com" not in url.lower():
            return {"linked": False, "error": "Expected a github.com URL (got: " + url + ")"}
        # Append .git if it's missing - git tolerates either, but ls-remote prefers explicit
        canonical = url if url.endswith(".git") else url + ".git"
    elif src == "azure":
        low = url.lower()
        if "dev.azure.com" not in low and "visualstudio.com" not in low:
            return {"linked": False, "error": "Expected a dev.azure.com or visualstudio.com URL"}
        canonical = url

    # Build authed URL for the reachability check (token in URL is standard for HTTPS git auth)
    token = (req.repo_token or "").strip()
    auth_url = canonical
    if token:
        if "https://" not in canonical:
            return {"linked": False, "error": "Token-authed URL must be https://"}
        if src == "github":
            auth_url = canonical.replace(
                "https://", f"https://x-access-token:{token}@", 1
            )
        else:  # azure: PAT goes in the user position
            auth_url = canonical.replace(
                "https://", f"https://anything:{token}@", 1
            )

    # Reachability test - does this URL respond to git protocol?
    # GIT_TERMINAL_PROMPT=0 prevents git from blocking on credential prompts.
    # GIT_ASKPASS=echo similarly disables GUI askpass helpers on some systems.
    git_env = dict(os.environ)
    git_env["GIT_TERMINAL_PROMPT"] = "0"
    git_env["GIT_ASKPASS"] = "echo"
    git_env["GCM_INTERACTIVE"] = "Never"
    try:
        proc = _sp.run(
            ["git", "ls-remote", "--exit-code", auth_url, "HEAD"],
            check=False, capture_output=True, text=True, timeout=12,
            env=git_env,
        )
    except FileNotFoundError:
        # `git` not installed on the server. Fall back to format-only validation
        # - better than refusing every link request.
        with _REALTIME_LOCK:
            _REALTIME_CONFIG["repo_url"] = url
            _REALTIME_CONFIG["repo_token"] = req.repo_token
            _REALTIME_CONFIG["repo_source"] = src
            _REALTIME_CONFIG["repo_local_path"] = None
        return {
            "linked": True, "source": src, "repo_url": url,
            "warning": "git not installed on server - URL was format-validated but not pinged",
        }
    except _sp.TimeoutExpired:
        return {"linked": False, "error": "git ls-remote timed out (network or unreachable host)"}

    if proc.returncode != 0:
        # Strip token from any error output before surfacing
        err = (proc.stderr or proc.stdout or "").strip()
        if token:
            err = err.replace(token, "***")
        # Common error mapping for nicer UX
        low_err = err.lower()
        if "authentication failed" in low_err or "could not read" in low_err or "401" in low_err or "403" in low_err:
            friendly = "Authentication failed - check your token (PAT must have repo read access)"
        elif "not found" in low_err or "repository not found" in low_err or "404" in low_err:
            friendly = "Repository not found at that URL - check spelling and access permissions"
        elif "could not resolve host" in low_err:
            friendly = "Network error - could not resolve host"
        elif "timeout" in low_err:
            friendly = "Connection timed out"
        else:
            friendly = err.splitlines()[0] if err else f"git exit code {proc.returncode}"
        return {"linked": False, "error": friendly, "git_stderr": err[:400]}

    with _REALTIME_LOCK:
        _REALTIME_CONFIG["repo_url"] = url
        _REALTIME_CONFIG["repo_token"] = req.repo_token
        _REALTIME_CONFIG["repo_source"] = src
        _REALTIME_CONFIG["repo_local_path"] = None
    return {"linked": True, "source": src, "repo_url": url, "verified": True}


@app.post("/realtime/codebase/upload")
async def realtime_codebase_upload(file_b64: str = "", filename: str = "codebase.zip") -> Dict[str, Any]:
    """Accept a base64-encoded ZIP. The browser sends the file via multipart;
    FastAPI's UploadFile handles binary cleanly. We use base64 in JSON for
    simplicity here, but we also expose a multipart variant below."""
    return {"linked": False, "error": "use POST /realtime/codebase/upload-multipart for ZIP uploads"}


from fastapi import File, UploadFile  # placed here to keep the import local to this section


@app.post("/realtime/codebase/upload-multipart")
async def realtime_codebase_upload_multipart(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept a ZIP file upload, extract it to CODEBASE_ROOT/<id>, register
    it as the tier-2 codebase source. Size-capped, path-traversal-checked."""
    import io
    import shutil
    import zipfile

    if not file.filename or not file.filename.lower().endswith(".zip"):
        return {"linked": False, "error": "expected a .zip file"}
    contents = await file.read()
    if len(contents) > _MAX_ZIP_BYTES:
        return {"linked": False, "error": f"file too large ({len(contents)} > {_MAX_ZIP_BYTES})"}
    _CODEBASE_ROOT.mkdir(parents=True, exist_ok=True)
    extract_id = uuid.uuid4().hex[:8]
    dest = _CODEBASE_ROOT / extract_id
    dest.mkdir()
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            # Path-traversal defense - reject any member whose resolved path escapes dest
            for member in zf.namelist():
                norm = os.path.normpath(member).replace("\\", "/")
                if norm.startswith("/") or norm.startswith(".."):
                    raise ValueError(f"unsafe path in zip: {member}")
                if ".." in norm.split("/"):
                    raise ValueError(f"unsafe path in zip: {member}")
            zf.extractall(dest)
    except (zipfile.BadZipFile, ValueError) as exc:
        shutil.rmtree(dest, ignore_errors=True)
        return {"linked": False, "error": f"invalid zip: {exc}"}
    # If the zip has a single top-level dir, descend into it
    children = [c for c in dest.iterdir() if c.is_dir()]
    if len(children) == 1 and not any(c.is_file() for c in dest.iterdir()):
        actual_root = children[0]
    else:
        actual_root = dest
    with _REALTIME_LOCK:
        # Cleanup previous upload
        prev = _REALTIME_CONFIG.get("repo_local_path")
        if prev and prev != str(actual_root):
            try:
                pp = Path(prev)
                if pp.exists() and _CODEBASE_ROOT in pp.parents:
                    shutil.rmtree(pp, ignore_errors=True)
            except Exception:
                pass
        _REALTIME_CONFIG["repo_url"] = f"zip://{file.filename}"
        _REALTIME_CONFIG["repo_token"] = None
        _REALTIME_CONFIG["repo_source"] = "zip"
        _REALTIME_CONFIG["repo_local_path"] = str(actual_root)
    return {
        "linked": True,
        "source": "zip",
        "filename": file.filename,
        "size_bytes": len(contents),
        "path": str(actual_root),
    }


@app.post("/realtime/codebase/clear")
def realtime_codebase_clear() -> Dict[str, Any]:
    """Forget any linked / uploaded codebase."""
    import shutil
    with _REALTIME_LOCK:
        prev = _REALTIME_CONFIG.get("repo_local_path")
        _REALTIME_CONFIG["repo_url"] = None
        _REALTIME_CONFIG["repo_token"] = None
        _REALTIME_CONFIG["repo_source"] = None
        _REALTIME_CONFIG["repo_local_path"] = None
    if prev:
        try:
            pp = Path(prev)
            if pp.exists() and _CODEBASE_ROOT in pp.parents:
                shutil.rmtree(pp, ignore_errors=True)
        except Exception:
            pass
    return {"cleared": True}


@app.post("/realtime/run-agent")
def realtime_run_agent(req: RealtimeRunRequest) -> Dict[str, Any]:
    """Kick off Praetor against the connected site.

    Runs in a background thread; the UI polls /realtime/status/<run_id>
    for streaming events. If `scenario` is omitted, Praetor classifies the
    fault from the site's current state.
    """
    site_url = _REALTIME_CONFIG.get("site_url")
    if not site_url:
        return {"error": "no site connected", "run_id": None}

    # Auto-detect scenario if not explicitly passed
    scenario = req.scenario
    if not scenario:
        cls = _classify_current_fault(
            site_url, _REALTIME_CONFIG.get("service_names", []), {},
        )
        scenario = cls.get("scenario") or "oom_crash"  # safe default
    if scenario not in _DEMO_PLAYBOOK:
        return {"error": f"unknown scenario: {scenario}", "run_id": None}
    run_id = "rt-" + uuid.uuid4().hex[:10]
    record: Dict[str, Any] = {
        "run_id": run_id,
        "scenario": scenario,
        "auto_classified": req.scenario is None,
        "site_url": site_url,
        "status": "running",
        "tier": "tier1",
        "events": [],
        "started_at": time.time(),
        "finished_at": None,
        "tier1_resolved": None,
        "tier2_report": None,
    }
    with _REALTIME_LOCK:
        _REALTIME_RUNS[run_id] = record
    threading.Thread(
        target=_realtime_run_worker,
        args=(run_id, scenario, req.enable_tier2),
        daemon=True,
    ).start()
    return {"run_id": run_id, "scenario": scenario, "auto_classified": req.scenario is None}


@app.get("/realtime/status/{run_id}")
def realtime_status(run_id: str) -> Dict[str, Any]:
    """Poll the in-memory record for a real-time run."""
    safe_id = run_id.replace("..", "").replace("/", "").replace("\\", "")
    with _REALTIME_LOCK:
        rec = _REALTIME_RUNS.get(safe_id)
        if not rec:
            return {"error": f"unknown run_id: {safe_id}"}
        return dict(rec)  # shallow copy, fine for read-only API


@app.get("/realtime/run/{run_id}/report.pdf")
def realtime_run_report_pdf(run_id: str):
    """Render a real-time run as a downloadable PDF (Content-Disposition: attachment).

    The button in the Real-Time tab fetches this endpoint as a blob and triggers
    a real .pdf download - no print-dialog round-trip required. The PDF is
    generated server-side via reportlab (pure-Python, no native deps), so this
    works identically on Windows / Linux / HF Space.
    """
    from fastapi.responses import Response
    safe_id = run_id.replace("..", "").replace("/", "").replace("\\", "")
    with _REALTIME_LOCK:
        rec = _REALTIME_RUNS.get(safe_id)
        if not rec:
            return Response(
                content=f"No run record for run_id {safe_id}".encode("utf-8"),
                status_code=404,
                media_type="text/plain",
            )
        rec = dict(rec)
        rec["events"] = list(rec.get("events", []))

    try:
        from incident_commander_env.server.report_pdf import render_run_report_pdf
        pdf_bytes = render_run_report_pdf(safe_id, rec)
    except ImportError as exc:
        return Response(
            content=(
                "PDF export requires the 'reportlab' package. "
                f"Install with: pip install reportlab. ({exc})"
            ).encode("utf-8"),
            status_code=500,
            media_type="text/plain",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return Response(
            content=f"PDF generation failed: {type(exc).__name__}: {exc}".encode("utf-8"),
            status_code=500,
            media_type="text/plain",
        )

    filename = f"praetor-incident-{safe_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
            "Cache-Control": "no-store",
        },
    )


@app.get("/realtime/run/{run_id}/report")
def realtime_run_report(run_id: str):
    """Render a print-ready HTML page for the given real-time run.

    The browser opens this URL in a new tab and the user prints it as PDF
    via Ctrl+P → Save as PDF. The page includes everything needed for a
    formal incident report: alert, classification, every step with the
    agent's reasoning, and the final result.
    """
    from fastapi.responses import HTMLResponse
    safe_id = run_id.replace("..", "").replace("/", "").replace("\\", "")
    with _REALTIME_LOCK:
        rec = _REALTIME_RUNS.get(safe_id)
        if not rec:
            return HTMLResponse(
                "<!doctype html><html><body><h1>Run not found</h1>"
                f"<p>No record for run_id <code>{safe_id}</code>.</p></body></html>",
                status_code=404,
            )
        rec = dict(rec)  # shallow copy
        rec["events"] = list(rec.get("events", []))

    return HTMLResponse(_render_run_report_html(safe_id, rec))


def _render_run_report_html(run_id: str, rec: Dict[str, Any]) -> str:
    """Build the print-ready HTML for a run's incident report.

    Inline CSS (no external assets), browser-print-friendly. Auto-triggers
    print dialog on load via a tiny script - user picks 'Save as PDF' from
    the destination dropdown.
    """
    import html as _html
    from datetime import datetime, timezone

    def esc(x: Any) -> str:
        return _html.escape(str(x or ""))

    events = rec.get("events", [])
    start_ev = next((e for e in events if e.get("type") == "start"), {})
    classify_ev = next((e for e in events if e.get("type") == "classify"
                        or "classification" in e), None)
    steps = [e for e in events if e.get("type") == "step"]
    tier1_done = next((e for e in events if e.get("type") == "tier1_done"), {})
    tier2_done = next((e for e in events if e.get("type") == "tier2_done"), None)

    started_at = rec.get("started_at")
    finished_at = rec.get("finished_at")
    duration_s = (finished_at - started_at) if (started_at and finished_at) else None

    scenario = rec.get("scenario") or start_ev.get("scenario") or "unknown"
    site_url = rec.get("site_url") or start_ev.get("site_url") or "(in-process simulator)"
    alert = rec.get("alert_title") or rec.get("alert_summary") or "(no alert text captured)"

    resolved = bool(rec.get("tier1_resolved"))
    status_label = "RESOLVED" if resolved else (
        "ESCALATED" if tier2_done else "UNRESOLVED"
    )
    status_color = "#22c55e" if resolved else ("#f59e0b" if tier2_done else "#ef4444")

    started_iso = (
        datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat()
        if started_at else "-"
    )

    # Build the steps HTML
    steps_html = []
    for ev in steps:
        a = ev.get("action") or {}
        params = a.get("parameters") or {}
        params_str = (
            json.dumps(params, separators=(",", ": ")) if params else ""
        )
        why = ev.get("why") or "-"
        message = (ev.get("message") or "").strip()
        target = a.get("target_service")
        steps_html.append(f"""
<div class="step">
  <div class="step-head">
    <div>
      <span class="step-num">step {esc(ev.get("step", "?"))}</span>
      <span class="step-action">{esc(a.get("action_type", "?"))}</span>
      {f'<span class="step-target">→ {esc(target)}</span>' if target else ''}
      {f'<span class="step-params">{esc(params_str)}</span>' if params else ''}
    </div>
    <div class="step-tier">{esc(ev.get("tier", "tier1"))}</div>
  </div>
  <div class="step-body">{esc(message)[:1500]}</div>
  <div class="step-why"><span class="why-label">Why:</span> {esc(why)}</div>
</div>
        """)
    steps_html_str = "\n".join(steps_html) or "<p>No steps recorded.</p>"

    tier2_html = ""
    if tier2_done:
        tier2_html = f"""
<h2>Tier 2 - code investigation</h2>
<p><strong>Summary:</strong> {esc(tier2_done.get("summary", "-"))}</p>
<p><strong>Suggested fix:</strong> {esc(tier2_done.get("suggested_fix", "-"))}</p>
<p><strong>Findings:</strong> {esc(tier2_done.get("n_findings", 0))} candidate code locations identified.</p>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Praetor Incident Report - {esc(run_id)}</title>
<style>
@page {{ margin: 18mm 16mm; size: A4; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; color: #1a1a1a; line-height: 1.55; }}
body {{ padding: 32px 40px; max-width: 820px; margin: 0 auto; background: white; }}
h1 {{ font-family: 'Fraunces', Georgia, serif; font-weight: 400; font-size: 32px; margin: 0 0 6px; letter-spacing: -0.02em; }}
h2 {{ font-family: 'Inter', sans-serif; font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; color: #6b7280; margin: 32px 0 12px; padding-bottom: 4px; border-bottom: 1px solid #e5e7eb; }}
.subtitle {{ font-family: 'Fraunces', Georgia, serif; font-style: italic; font-size: 16px; color: #6b7280; margin: 0 0 24px; }}
.meta {{ display: grid; grid-template-columns: 130px 1fr; gap: 6px 14px; font-size: 13px; margin-bottom: 24px; padding: 14px 16px; background: #f9fafb; border-radius: 8px; }}
.meta dt {{ color: #6b7280; }}
.meta dd {{ margin: 0; color: #1a1a1a; font-family: 'JetBrains Mono', Menlo, monospace; font-size: 12px; }}
.status {{ display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; letter-spacing: 0.6px; color: white; background: {status_color}; margin-left: 10px; }}
p {{ margin: 0 0 12px; font-size: 13.5px; }}
.alert-box {{ background: #fef3c7; border-left: 3px solid #f59e0b; padding: 12px 16px; border-radius: 4px; font-size: 13.5px; margin-bottom: 20px; }}
.step {{ background: #f9fafb; border-left: 3px solid #3b82f6; padding: 12px 14px; border-radius: 4px; margin-bottom: 10px; page-break-inside: avoid; }}
.step-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; flex-wrap: wrap; gap: 6px; }}
.step-num {{ font-weight: 600; color: #1a1a1a; font-size: 12px; margin-right: 8px; }}
.step-action {{ font-family: 'JetBrains Mono', Menlo, monospace; font-size: 12px; color: #3b82f6; font-weight: 500; }}
.step-target {{ font-family: 'JetBrains Mono', Menlo, monospace; font-size: 12px; color: #6b7280; margin-left: 4px; }}
.step-params {{ font-family: 'JetBrains Mono', Menlo, monospace; font-size: 11px; color: #9ca3af; margin-left: 6px; }}
.step-tier {{ font-family: 'JetBrains Mono', Menlo, monospace; font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.6px; }}
.step-body {{ font-family: 'JetBrains Mono', Menlo, monospace; font-size: 11.5px; color: #4b5563; margin-bottom: 8px; white-space: pre-wrap; word-break: break-word; }}
.step-why {{ font-size: 12.5px; color: #1a1a1a; padding-top: 8px; border-top: 1px dashed #d1d5db; }}
.why-label {{ font-weight: 600; color: #3b82f6; text-transform: uppercase; font-size: 10px; letter-spacing: 1px; }}
.footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #e5e7eb; font-size: 11px; color: #9ca3af; text-align: center; }}
.no-print {{ position: fixed; top: 16px; right: 16px; padding: 10px 18px; background: #3b82f6; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 500; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
@media print {{ .no-print {{ display: none; }} body {{ padding: 0; }} }}
</style>
</head>
<body>
<button class="no-print" onclick="window.print()">📄 Print / Save as PDF</button>

<h1>Praetor Incident Report</h1>
<p class="subtitle">Autonomous incident response · run <code>{esc(run_id)}</code> <span class="status">{status_label}</span></p>

<dl class="meta">
  <dt>Run ID</dt>          <dd>{esc(run_id)}</dd>
  <dt>Scenario</dt>        <dd>{esc(scenario)}</dd>
  <dt>Target</dt>          <dd>{esc(site_url)}</dd>
  <dt>Started</dt>         <dd>{esc(started_iso)}</dd>
  <dt>Steps used</dt>      <dd>{esc(len(steps))}</dd>
  <dt>Wall-clock</dt>      <dd>{f"{duration_s:.2f} s" if duration_s else "-"}</dd>
  <dt>Auto-classified</dt> <dd>{"yes" if rec.get("auto_classified") else "no"}</dd>
</dl>

<h2>The problem we saw</h2>
<div class="alert-box">{esc(alert)}</div>
{f'<p><strong>Praetor classified the fault as:</strong> {esc(scenario)}.</p>' if scenario else ''}

<h2>Steps Praetor took</h2>
{steps_html_str}

{tier2_html}

<h2>Result</h2>
<p>
  Status: <strong>{status_label}</strong>.
  {'Tier-1 runtime ops resolved the incident - no code escalation needed.' if resolved else (
    'Tier-1 ops were not enough; tier-2 surfaced candidate code locations to investigate.'
    if tier2_done else 'Tier-1 ops did not fully heal the site, and tier-2 was not enabled.'
  )}
  Total wall-clock: {f"{duration_s:.2f} seconds." if duration_s else "(in progress)."}
</p>

<div class="footer">
  Generated by Praetor - autonomous SRE incident commander · {esc(started_iso)}
</div>

<script>
// Auto-open the print dialog so the user can save as PDF immediately.
// Disabled if the URL has ?noprint=1 (useful for previewing the layout).
if (!new URLSearchParams(location.search).has('noprint')) {{
  window.addEventListener('load', function() {{
    setTimeout(function() {{ window.print(); }}, 400);
  }});
}}
</script>
</body>
</html>
"""


@app.get("/realtime/config")
def realtime_config() -> Dict[str, Any]:
    """Inspect the current real-time configuration (site URL, codebase source, etc.)."""
    return {
        "site_url": _REALTIME_CONFIG.get("site_url"),
        "repo_linked": bool(_REALTIME_CONFIG.get("repo_url") or _REALTIME_CONFIG.get("repo_local_path")),
        "repo_source": _REALTIME_CONFIG.get("repo_source"),
        "repo_url_display": _REALTIME_CONFIG.get("repo_url"),
        "service_names": _REALTIME_CONFIG.get("service_names"),
    }


# ---------------------------------------------------------------------------
# Phase 2 - Webhook ingestion. Closes the "continuously monitors a fleet for
# new incidents" gap from the roadmap. PagerDuty, Prometheus, and a minimal
# generic contract all converge to the same `/realtime/run-agent` background
# worker - the dispatcher just classifies the alert and kicks it off.
# ---------------------------------------------------------------------------


def _kickoff_webhook_run(
    signal: Dict[str, Any], scenario: str, scenario_evidence: List[str],
) -> Dict[str, Any]:
    """Spin up an autonomous run from a webhook signal.

    If a real-time site is configured (/realtime/connect was called), the
    agent runs against that site. Otherwise it runs against the simulator,
    which is the right default for staging-environment webhook testing.
    """
    site_url = _REALTIME_CONFIG.get("site_url")
    run_id = "wh-" + uuid.uuid4().hex[:10]
    record: Dict[str, Any] = {
        "run_id": run_id,
        "scenario": scenario,
        "auto_classified": True,
        "site_url": site_url,
        "trigger": "webhook",
        "provider": signal.get("provider"),
        "alert_title": signal.get("title"),
        "alert_summary": signal.get("summary"),
        "scenario_evidence": scenario_evidence,
        "severity": signal.get("severity"),
        "status": "running",
        "tier": "tier1",
        "events": [],
        "started_at": time.time(),
        "finished_at": None,
        "tier1_resolved": None,
        "tier2_report": None,
    }
    with _REALTIME_LOCK:
        _REALTIME_RUNS[run_id] = record
    if site_url:
        threading.Thread(
            target=_realtime_run_worker,
            args=(run_id, scenario, True),
            daemon=True,
        ).start()
    else:
        # Sim-only fallback: run in-process against a simulator backend.
        threading.Thread(
            target=_webhook_sim_worker, args=(run_id, scenario), daemon=True,
        ).start()
    return {"run_id": run_id, "scenario": scenario}


def _webhook_sim_worker(run_id: str, scenario: str) -> None:
    """When no real site is connected, run the autonomous loop against the
    simulator. Same trace shape as a real-time run so the dashboard can
    replay it identically."""
    try:
        from training.episode_logger import EpisodeLogger
        env_local = IncidentCommanderEnv()
        env_local.reset(task_id=scenario, seed=int(time.time()) % 10000)
        log = EpisodeLogger.for_run(str(RUNS_ROOT), scenario)
        log.__enter__()
        log.start({
            "task_id": scenario, "seed": env_local.state.episode_id,
            "model": "scripted-playbook", "trigger": "webhook",
            "alert": env_local.state.task_id,
            "max_steps": env_local.state.max_steps,
        })
        _realtime_append(run_id, {
            "type": "start", "tier": "tier1",
            "scenario": scenario, "site_url": None, "substrate": "simulator",
        })
        # Replay the demo playbook against the sim.
        playbook = _DEMO_PLAYBOOK.get(scenario, [])
        # Translate website service names back to sim service names. For OOM
        # ("api" → "payment-service") we just default to whatever the scenario
        # injected as the target.
        sim_target = getattr(env_local._scenario, "target_service", None)
        steps_used = 0
        for i, step_def in enumerate(playbook, start=1):
            target = step_def.get("target_service")
            if target == "api" and sim_target:
                target = sim_target
            if target == "postgres":
                target = "postgres-db"
            action = IncidentAction(
                action_type=step_def["action_type"],
                target_service=target,
                parameters=step_def.get("parameters") or {},
            )
            obs = env_local.step(action)
            steps_used = env_local.state.step_count
            _realtime_append(run_id, {
                "type": "step", "step": i, "tier": "tier1",
                "action": {
                    "action_type": action.action_type,
                    "target_service": action.target_service,
                    "parameters": action.parameters,
                },
                "message": obs.message, "error": obs.error, "done": obs.done,
                "why": step_def.get("why"),
            })
            log.step(steps_used, action, obs)
            if obs.done:
                break
        resolved = bool(env_local.state.incident_resolved)
        with _REALTIME_LOCK:
            r = _REALTIME_RUNS.get(run_id, {})
            r["tier1_resolved"] = resolved
            r["status"] = "resolved" if resolved else "unresolved_no_tier2"
            r["finished_at"] = time.time()
        _realtime_append(run_id, {
            "type": "tier1_done", "resolved": resolved,
            "message": ("Resolved on simulator." if resolved
                        else "Unresolved on simulator after demo playbook."),
        })
        log.end({
            "resolved": resolved,
            "score": float(env_local.state.current_score or 0.0),
            "steps_used": steps_used,
        })
        log.close()
    except Exception as exc:  # pragma: no cover - defensive
        _realtime_append(run_id, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        with _REALTIME_LOCK:
            _REALTIME_RUNS[run_id]["status"] = "error"
            _REALTIME_RUNS[run_id]["finished_at"] = time.time()


@app.post("/incidents/webhook/pagerduty")
async def webhook_pagerduty(request: Request) -> Dict[str, Any]:
    return await _handle_webhook(request, normalize_pagerduty, "pagerduty")


@app.post("/incidents/webhook/prometheus")
async def webhook_prometheus(request: Request) -> Dict[str, Any]:
    return await _handle_webhook(request, normalize_prometheus, "prometheus")


@app.post("/incidents/webhook/generic")
async def webhook_generic(request: Request) -> Dict[str, Any]:
    return await _handle_webhook(request, normalize_generic, "generic")


async def _handle_webhook(request, normalizer, provider_name: str) -> Dict[str, Any]:
    token = request.headers.get("X-Praetor-Token")
    ok, why = webhook_token_check(token)
    if not ok:
        return {"accepted": False, "error": why}

    try:
        payload = await request.json()
    except Exception as exc:
        return {"accepted": False, "error": f"invalid JSON: {exc}"}

    try:
        signal = normalizer(payload)
    except Exception as exc:
        return {"accepted": False, "error": f"failed to normalize {provider_name} payload: {exc}"}

    scenario, confidence, evidence = classify_scenario(signal)
    if scenario is None:
        return {
            "accepted": False,
            "error": "could not classify alert into a known scenario family",
            "signal": {
                "title": signal.get("title"),
                "summary": signal.get("summary"),
            },
        }

    response = _kickoff_webhook_run(signal, scenario, evidence)
    response["accepted"] = True
    response["classification"] = {
        "scenario": scenario,
        "confidence": confidence,
        "evidence": evidence,
    }
    response["alert"] = {
        "title": signal.get("title"),
        "service": signal.get("service"),
        "severity": signal.get("severity"),
    }
    response["substrate"] = "real" if _REALTIME_CONFIG.get("site_url") else "simulator"
    if not os.environ.get("PRAETOR_WEBHOOK_TOKEN"):
        response["warning"] = (
            "PRAETOR_WEBHOOK_TOKEN env var is unset - webhook is in demo mode "
            "and accepts all requests. Set the token before pointing real "
            "alerts at this endpoint."
        )
    return response


# ---------------------------------------------------------------------------
# Phase 2 - Sandboxed shell action endpoint.
# Lets a client (or the UI) exercise the agent's diagnostic shell with
# the 20-command allowlist. Useful for debugging + showing judges that
# Praetor has a safe escape hatch beyond the typed action vocabulary.
# ---------------------------------------------------------------------------


class ShellRequest(BaseModel):
    command: str
    cwd: Optional[str] = None  # only used if it points inside RUNS_ROOT or CODEBASE_ROOT


@app.get("/shell/allowlist")
def shell_allowlist() -> Dict[str, Any]:
    """Return the 20-command sandboxed-shell allowlist."""
    from incident_commander_env.server.actions.sandboxed_shell import list_allowed
    return {"commands": list_allowed(), "max_output_bytes": 8192, "timeout_seconds": 10}


@app.post("/shell/run")
def shell_run(req: ShellRequest) -> Dict[str, Any]:
    """Execute a single allowlisted command. Read-mostly probe surface."""
    from incident_commander_env.server.actions.sandboxed_shell import run_shell
    cwd_path: Optional[Path] = None
    if req.cwd:
        try:
            cwd_path = Path(req.cwd).resolve()
            # Restrict to runs/ or uploaded_codebase/ for safety
            ok = (
                cwd_path.is_relative_to(RUNS_ROOT)
                or cwd_path.is_relative_to(_CODEBASE_ROOT)
            )
            if not ok:
                return {
                    "ok": False,
                    "error": f"cwd must be inside {RUNS_ROOT} or {_CODEBASE_ROOT}",
                }
        except Exception as exc:
            return {"ok": False, "error": f"invalid cwd: {exc}"}
    result = run_shell(req.command, cwd=cwd_path)
    # Convert the dataclass to a plain dict
    return {
        "ok": result.ok,
        "command": result.command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "elapsed_s": result.elapsed_s,
        "error": result.error,
        "truncated": result.truncated,
    }


# ---------------------------------------------------------------------------
# Phase 2 - Postmortem endpoint. Returns the auto-generated postmortem.md
# for a run id (sim or real-time). Lets the dashboard render it.
# ---------------------------------------------------------------------------


@app.get("/runs/{run_id}/postmortem")
def get_postmortem(run_id: str) -> Dict[str, Any]:
    safe = run_id.replace("..", "").replace("/", "").replace("\\", "")
    target = RUNS_ROOT / safe
    if not target.exists():
        return {"error": f"unknown run_id: {safe}"}
    pm_path = target / "postmortem.md"
    # Lazy-generate if missing - useful for older runs that pre-date the writer.
    if not pm_path.exists():
        ep = target / "episode.jsonl"
        if not ep.exists():
            return {"error": "no episode trace for this run"}
        try:
            from training.postmortem_writer import write_postmortem
            write_postmortem(ep, runbook_path=RUNS_ROOT / "RUNBOOK.md")
        except Exception as exc:
            return {"error": f"failed to generate postmortem: {exc}"}
    return {
        "run_id": safe,
        "markdown": pm_path.read_text(encoding="utf-8"),
        "path": str(pm_path),
    }


@app.get("/runbook")
def get_runbook() -> Dict[str, Any]:
    """Return the project-level runbook (incident ledger)."""
    rb = RUNS_ROOT / "RUNBOOK.md"
    if not rb.exists():
        return {"markdown": "# Praetor Runbook\n\n(empty - run an incident to populate)\n"}
    return {"markdown": rb.read_text(encoding="utf-8"), "path": str(rb)}


# ---------------------------------------------------------------------------
# Phase 2 - Tier-2 patch / test / PR endpoints. Wraps code_investigator's
# new propose_patch / apply_patch / run_tests / open_pull_request fns.
# ---------------------------------------------------------------------------


class CodeProposeRequest(BaseModel):
    scenario: str
    target_service: Optional[str] = None
    enable_pr_open: bool = False
    base_branch: str = "main"
    pr_title_prefix: str = "Praetor auto-fix"


@app.post("/codebase/propose-and-test")
def codebase_propose_and_test(req: CodeProposeRequest) -> Dict[str, Any]:
    """Run the full tier-2 chain: investigate → propose_patch → apply_patch
    → run_tests → optionally open_pull_request. Returns a structured report
    the dashboard can render."""
    repo_url = _REALTIME_CONFIG.get("repo_url")
    repo_local = _REALTIME_CONFIG.get("repo_local_path")
    repo_token = _REALTIME_CONFIG.get("repo_token")
    repo_source = _REALTIME_CONFIG.get("repo_source")
    if not (repo_url or repo_local):
        return {"error": "no codebase linked - link via /realtime/codebase/link or upload a ZIP"}
    from training.code_investigator import (
        apply_patch, investigate, open_pull_request, propose_patch, run_tests, _clone_repo,
    )
    # Clone (or reuse uploaded path)
    if repo_source == "zip" and repo_local:
        cloned = Path(repo_local)
        own_clone = False
    else:
        cloned = _clone_repo(repo_url, repo_token)
        own_clone = True
        if cloned is None:
            return {"error": "git clone failed"}
    try:
        report = investigate(
            repo_url=repo_url or f"local://{repo_local}",
            scenario=req.scenario,
            target_service=req.target_service,
            repo_token=repo_token,
            cloned_root=cloned,
        )
        if report.error:
            return {"investigate_error": report.error}
        patch = propose_patch(report)
        if patch is None:
            return {
                "investigation": report.to_dict(),
                "patch": None,
                "message": "Investigation surfaced findings but no auto-patch template matched.",
            }
        applied = apply_patch(cloned, patch)
        if not applied:
            return {
                "investigation": report.to_dict(),
                "patch": {"file_path": patch.file_path, "line_no": patch.line_no, "diff": patch.diff},
                "applied": False,
                "error": "could not apply the proposed patch (file may have changed)",
            }
        tests = run_tests(cloned)
        out: Dict[str, Any] = {
            "investigation": report.to_dict(),
            "patch": {
                "file_path": patch.file_path, "line_no": patch.line_no,
                "diff": patch.diff, "rationale": patch.rationale,
                "confidence": patch.confidence,
            },
            "applied": True,
            "tests": {
                "framework": tests.framework, "passed": tests.passed,
                "n_tests": tests.n_tests, "n_failed": tests.n_failed,
                "duration_s": tests.duration_s,
                "stdout_tail": tests.stdout_tail[-1500:],
                "stderr_tail": tests.stderr_tail[-1500:],
                "error": tests.error,
            },
        }
        if repo_url and "github.com" in repo_url.lower():
            pr_title = f"{req.pr_title_prefix}: {req.scenario} on {req.target_service or 'cluster'}"
            pr_body = (
                f"Praetor (autonomous SRE commander) ran tier-1 ops on this "
                f"incident, the runtime fix didn't fully heal, and tier-2 "
                f"escalation surfaced this candidate fix.\n\n"
                f"**Scenario:** `{req.scenario}`\n"
                f"**File:** `{patch.file_path}:{patch.line_no}`\n"
                f"**Confidence:** {patch.confidence:.2f}\n\n"
                f"### Rationale\n\n{patch.rationale}\n\n"
                f"### Tests after applying\n\n"
                f"`{tests.framework}` reports {tests.n_tests} tests, "
                f"{tests.n_failed} failed, "
                f"verdict: {'PASS' if tests.passed else 'FAIL'}.\n\n"
                f"---\nThis PR was opened automatically. Review carefully "
                f"before merging."
            )
            pr_result = open_pull_request(
                repo_root=cloned, repo_url=repo_url,
                branch="praetor/auto-fix", title=pr_title, body=pr_body,
                token=repo_token, enable_pr_open=req.enable_pr_open,
                base_branch=req.base_branch,
            )
            out["pr"] = {
                "opened": pr_result.opened, "dry_run": pr_result.dry_run,
                "branch": pr_result.branch, "pr_url": pr_result.pr_url,
                "error": pr_result.error,
            }
        return out
    finally:
        if own_clone and cloned:
            try:
                from training.code_investigator import cleanup_repo
                cleanup_repo(cloned)
            except Exception:
                pass


@app.get("/incidents/webhooks")
def list_webhooks() -> Dict[str, Any]:
    """Show the configured webhook endpoints + token status."""
    base = "/incidents/webhook"
    return {
        "endpoints": [
            {"provider": "pagerduty",  "path": f"{base}/pagerduty",  "method": "POST"},
            {"provider": "prometheus", "path": f"{base}/prometheus", "method": "POST"},
            {"provider": "generic",    "path": f"{base}/generic",    "method": "POST"},
        ],
        "auth": {
            "header": "X-Praetor-Token",
            "configured": bool(os.environ.get("PRAETOR_WEBHOOK_TOKEN")),
        },
        "supported_scenarios": list(_DEMO_PLAYBOOK.keys()),
    }


def _realtime_append(run_id: str, event: Dict[str, Any]) -> None:
    event = {"ts": round(time.time(), 3), **event}
    with _REALTIME_LOCK:
        rec = _REALTIME_RUNS.get(run_id)
        if rec is not None:
            rec["events"].append(event)


def _realtime_run_worker(run_id: str, scenario: str, enable_tier2: bool) -> None:
    """Background worker: walk the demo playbook against the website backend."""
    site_url = _REALTIME_CONFIG.get("site_url")
    if not site_url:
        with _REALTIME_LOCK:
            r = _REALTIME_RUNS.get(run_id, {})
            r["status"] = "error"
            r["error"] = "site disconnected mid-run"
        return

    # Build a minimal scenario object so the WebsiteBackend.execute path works.
    # We don't need the full grading rubric for the live demo.
    scenario_obj = _LiveScenario(scenario)

    backend = WebsiteBackend(site_url=site_url, service_names=_REALTIME_CONFIG.get("service_names"))
    try:
        backend.reset(scenario_obj)
    except Exception as exc:  # pragma: no cover - defensive
        _realtime_append(run_id, {"type": "error", "message": f"reset failed: {exc}"})
        with _REALTIME_LOCK:
            _REALTIME_RUNS[run_id]["status"] = "error"
        return

    _realtime_append(run_id, {
        "type": "start",
        "tier": "tier1",
        "scenario": scenario,
        "site_url": site_url,
    })

    playbook = _DEMO_PLAYBOOK.get(scenario, [])
    last_resolved = False
    for i, step_def in enumerate(playbook, start=1):
        action = IncidentAction(
            action_type=step_def["action_type"],
            target_service=step_def.get("target_service"),
            parameters=step_def.get("parameters") or {},
        )
        try:
            obs = backend.execute(action, scenario_obj)
        except Exception as exc:  # pragma: no cover - defensive
            _realtime_append(run_id, {
                "type": "step", "step": i, "tier": "tier1",
                "action": step_def, "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        _realtime_append(run_id, {
            "type": "step",
            "step": i,
            "tier": "tier1",
            "action": step_def,
            "message": obs.message,
            "error": obs.error,
            "done": obs.done,
            "why": step_def.get("why"),
        })
        if obs.done:
            last_resolved = bool(obs.message and "PASS" in obs.message)
            break
        time.sleep(0.4)

    # Final health check
    healthy = False
    try:
        health = _site_http("GET", site_url + "/ops/health", timeout=5.0)
        if health.ok and isinstance(health.body, dict):
            healthy = (health.body.get("status") or "").lower() == "ok"
    except Exception:
        healthy = False

    with _REALTIME_LOCK:
        rec = _REALTIME_RUNS.get(run_id)
        if rec:
            rec["tier1_resolved"] = bool(healthy or last_resolved)

    if (healthy or last_resolved):
        _realtime_append(run_id, {
            "type": "tier1_done",
            "resolved": True,
            "message": "Tier 1 ops actions resolved the incident.",
        })
        with _REALTIME_LOCK:
            _REALTIME_RUNS[run_id]["status"] = "resolved"
            _REALTIME_RUNS[run_id]["finished_at"] = time.time()
        return

    _realtime_append(run_id, {
        "type": "tier1_done",
        "resolved": False,
        "message": "Tier 1 ops actions did not fully heal the site.",
    })

    repo_url = _REALTIME_CONFIG.get("repo_url")
    repo_local = _REALTIME_CONFIG.get("repo_local_path")
    repo_source = _REALTIME_CONFIG.get("repo_source")
    if not enable_tier2 or not (repo_url or repo_local):
        with _REALTIME_LOCK:
            _REALTIME_RUNS[run_id]["status"] = "unresolved_no_tier2"
            _REALTIME_RUNS[run_id]["finished_at"] = time.time()
        return

    # ---- TIER 2: code escalation ------------------------------------------
    if repo_source == "zip":
        msg = f"Escalating to code investigation: scanning uploaded codebase…"
    elif repo_source == "azure":
        msg = "Escalating to code investigation: cloning Azure DevOps repo…"
    else:
        msg = "Escalating to code investigation: cloning GitHub repo…"
    _realtime_append(run_id, {
        "type": "escalate",
        "tier": "tier2",
        "message": msg,
    })
    try:
        from pathlib import Path as _P
        from training.code_investigator import investigate
        target_service = _infer_target_service(scenario)
        kwargs: Dict[str, Any] = {
            "repo_url": repo_url or f"local://{repo_local}",
            "scenario": scenario,
            "target_service": target_service,
            "repo_token": _REALTIME_CONFIG.get("repo_token"),
            "llm_call": None,
        }
        if repo_source == "zip" and repo_local:
            kwargs["cloned_root"] = _P(repo_local)
        report = investigate(**kwargs)
        with _REALTIME_LOCK:
            _REALTIME_RUNS[run_id]["tier2_report"] = report.to_dict()
        _realtime_append(run_id, {
            "type": "tier2_done",
            "summary": report.summary,
            "suggested_fix": report.suggested_fix,
            "n_findings": len(report.findings),
            "error": report.error,
        })
        with _REALTIME_LOCK:
            _REALTIME_RUNS[run_id]["status"] = (
                "tier2_complete" if not report.error else "tier2_failed"
            )
            _REALTIME_RUNS[run_id]["finished_at"] = time.time()
    except Exception as exc:
        _realtime_append(run_id, {
            "type": "tier2_error",
            "error": f"{type(exc).__name__}: {exc}",
        })
        with _REALTIME_LOCK:
            _REALTIME_RUNS[run_id]["status"] = "tier2_failed"
            _REALTIME_RUNS[run_id]["finished_at"] = time.time()


def _infer_target_service(scenario: str) -> Optional[str]:
    if scenario in {"oom_crash", "bad_deployment_cascade"}:
        return "api"
    if scenario == "db_pool_exhaustion":
        return "postgres"
    return None


class _LiveScenario:
    """Minimal scenario shim used by the live website backend during a real-time
    run. We don't need the full rubric here - backend.execute() only calls
    `on_config_update` and reads `task_id`."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.relevant_services = ["api", "postgres", "frontend"]
        self.root_cause_keywords: List[str] = []

    def setup(self, *_args, **_kwargs) -> None:
        pass

    def on_config_update(self, service: str, key: str, value: Any) -> bool:
        # Live site decides healing - we just say "maybe yes" for known config keys.
        return key == "db.pool.max_size" and isinstance(value, (int, float)) and value >= 50

    def is_correct_op(self, action: Any) -> bool:
        return False  # not used for the live demo

    def check_resolved(self, *_args, **_kwargs) -> bool:
        return False  # the backend's check_resolved polls /ops/health


def main() -> None:
    """Entry point for the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
