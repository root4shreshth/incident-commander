"""FastAPI application for IncidentCommanderEnv.

Exposes POST /reset, POST /step, GET /state endpoints as required by OpenEnv spec.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
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

def _seed_demo_runs_if_empty() -> None:
    try:
        if RUNS_ROOT.exists():
            for child in RUNS_ROOT.iterdir():
                if child.is_dir():
                    return  # already populated
        else:
            RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        from training.datasets import SYSTEM_PROMPT
        from training.eval_runner import evaluate, random_policy
        evaluate(
            "demo-baseline",
            random_policy(rng_seed=42),
            families=["oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"],
            seeds=list(range(1, 11)),
            system_prompt=SYSTEM_PROMPT,
            runs_root=str(RUNS_ROOT),
        )
    except Exception as exc:  # pragma: no cover — never crash on this
        print(f"[praetor] demo-runs seeding skipped: {exc}")


@app.on_event("startup")
def _startup_seed_demo_runs() -> None:
    threading.Thread(target=_seed_demo_runs_if_empty, daemon=True).start()


# ---------------------------------------------------------------------------
# Phase 3 — Real-time / sim-to-real on a deployed site.
#
# State machine: connect → inject chaos → run agent → (if not healed) escalate.
# Agent runs in a background thread; UI polls /realtime/status/<run_id>.
# ---------------------------------------------------------------------------

# In-memory store of active real-time runs. Bounded — we only need the current
# run for the demo, but keeping the last few makes A/B comparison possible.
_REALTIME_RUNS: Dict[str, Dict[str, Any]] = {}
_REALTIME_LOCK = threading.Lock()
_REALTIME_CONFIG: Dict[str, Any] = {
    "site_url": None,
    # Tier 2 codebase config — one of three sources, mutually exclusive
    "repo_url": None,           # github or azure-devops URL
    "repo_token": None,         # PAT for either provider
    "repo_source": None,        # "github" | "azure" | "zip" | None
    "repo_local_path": None,    # populated when source="zip" — the extracted dir
    "service_names": ["frontend", "api", "postgres"],
}

# Where uploaded ZIPs are extracted. Cleaned up across server restarts.
_CODEBASE_ROOT = Path(os.getenv("CODEBASE_ROOT", "uploaded_codebase")).resolve()
_MAX_ZIP_BYTES = 25 * 1024 * 1024  # 25 MB — big enough for most codebases, small enough to be safe


# Demo policy — replays a known-good action sequence per scenario family.
# Replaced by the trained LoRA when the user hooks one up; intentionally
# deterministic so the live demo doesn't fail on stage.
_DEMO_PLAYBOOK: Dict[str, List[Dict[str, Any]]] = {
    "oom_crash": [
        {"action_type": "list_services"},
        {"action_type": "read_logs",     "target_service": "api"},
        {"action_type": "check_metrics", "target_service": "api"},
        {"action_type": "restart_service", "target_service": "api",
         "parameters": {"memory_limit": "1024Mi"}},
        {"action_type": "resolve_incident",
         "parameters": {"root_cause": "OOM on api — memory limit too low",
                        "resolution": "restart api with 1024Mi"}},
    ],
    "db_pool_exhaustion": [
        {"action_type": "list_services"},
        {"action_type": "read_logs",     "target_service": "api"},
        {"action_type": "read_logs",     "target_service": "postgres"},
        {"action_type": "update_config", "target_service": "postgres",
         "parameters": {"key": "db.pool.max_size", "value": 100}},
        {"action_type": "restart_service", "target_service": "api"},
        {"action_type": "resolve_incident",
         "parameters": {"root_cause": "connection pool exhausted on postgres",
                        "resolution": "raised pool to 100, restarted api"}},
    ],
    "bad_deployment_cascade": [
        {"action_type": "list_services"},
        {"action_type": "read_logs",     "target_service": "api"},
        {"action_type": "rollback_deployment", "target_service": "api",
         "parameters": {"to_version": "v1.0"}},
        {"action_type": "restart_service", "target_service": "api"},
        {"action_type": "resolve_incident",
         "parameters": {"root_cause": "v1.1 bundled a memory leak",
                        "resolution": "rolled back api to v1.0 and restarted"}},
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
    user what's wrong — it figures it out from the site's metrics and
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

    # Look at /ops/health first — it sometimes reports the verdict directly
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
                evidence.append(f"{svc} error rate at {err:.0f}% — sustained 5xx")
                if scenario is None:
                    scenario = "bad_deployment_cascade"
                    confidence = max(confidence, 0.5)
        except (TypeError, ValueError):
            pass

    # Look at logs for telltale strings (cheap heuristic)
    if scenario is None and overall_status in ("degraded", "down"):
        for svc in service_names[:3]:
            r = _site_http(
                "GET", site_url + "/ops/logs?service=" + svc + "&lines=20", timeout=4.0,
            )
            if not r.ok or not isinstance(r.body, dict):
                continue
            text = " ".join(map(str, (r.body.get("logs") or [])))[:4000].lower()
            if "outofmemory" in text or "oom" in text or "memory" in text:
                scenario = "oom_crash"; confidence = 0.65
                evidence.append(f"{svc} logs mention OOM / memory")
                break
            if "pool" in text and "connection" in text:
                scenario = "db_pool_exhaustion"; confidence = 0.65
                evidence.append(f"{svc} logs mention connection pool")
                break
            if "version" in text and ("rollout" in text or "v1.1" in text or "deploy" in text):
                scenario = "bad_deployment_cascade"; confidence = 0.55
                evidence.append(f"{svc} logs mention deployment / version")
                break

    fault_detected = scenario is not None or overall_status in ("degraded", "down")
    if not fault_detected:
        narrative = "Site is healthy. Praetor has nothing to fix yet — connect with a fault, or use the test-fault buttons below to simulate one."
    else:
        sname = {
            "oom_crash": "OOM (out-of-memory)",
            "db_pool_exhaustion": "DB connection pool exhaustion",
            "bad_deployment_cascade": "Bad deployment cascade",
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
    """Link a remote git repo (GitHub or Azure DevOps) for tier-2 escalation."""
    src = (req.source or "").lower().strip()
    if src not in ("github", "azure"):
        return {"linked": False, "error": "source must be 'github' or 'azure'"}
    url = (req.repo_url or "").strip()
    if not url:
        return {"linked": False, "error": "repo_url is required"}
    # Light validation — we don't actually clone here. Tier-2 clones lazily.
    if src == "github" and "github.com" not in url.lower():
        return {"linked": False, "error": "GitHub source expects a github.com URL"}
    if src == "azure" and "dev.azure.com" not in url.lower() and "visualstudio.com" not in url.lower():
        return {"linked": False, "error": "Azure source expects a dev.azure.com URL"}
    with _REALTIME_LOCK:
        _REALTIME_CONFIG["repo_url"] = url
        _REALTIME_CONFIG["repo_token"] = req.repo_token
        _REALTIME_CONFIG["repo_source"] = src
        _REALTIME_CONFIG["repo_local_path"] = None
    return {"linked": True, "source": src, "repo_url": url}


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
            # Path-traversal defense — reject any member whose resolved path escapes dest
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
    except Exception as exc:  # pragma: no cover — defensive
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
        except Exception as exc:  # pragma: no cover — defensive
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
    run. We don't need the full rubric here — backend.execute() only calls
    `on_config_update` and reads `task_id`."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.relevant_services = ["api", "postgres", "frontend"]
        self.root_cause_keywords: List[str] = []

    def setup(self, *_args, **_kwargs) -> None:
        pass

    def on_config_update(self, service: str, key: str, value: Any) -> bool:
        # Live site decides healing — we just say "maybe yes" for known config keys.
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
