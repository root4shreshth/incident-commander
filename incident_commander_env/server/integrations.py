"""Integrations - let users connect their real platform to Praetor.

Honest scoping for the hackathon submission:

* **GitHub** - real OAuth via the device-flow grant. If the user sets
  `GITHUB_CLIENT_ID` (instructions in the UI), the flow is genuinely real:
  Praetor talks to https://github.com/login/device/code, the user enters
  the code on github.com, we poll for the access token, then we list
  their repos via the GitHub REST API. Without a Client ID, we fall back
  to a clearly-labelled demo with curated example repos.

* **Cloud providers (AWS / GCP / Render / Fly)** - DEMO MODE ONLY. The
  forms exist so users can see the architecture, but credentials are
  never stored server-side and no real cloud actions are executed. The
  code that would translate a typed action into `aws ecs update-service`
  is intentionally left as a stub so we don't ship something that could
  damage someone's production. Real cloud-provider integration is
  Phase 2 - see the Integrations roadmap card on the home page.

* **Adapter generator** - REAL. The user picks their platform shape
  (FastAPI app, Render service, Fly machine, generic Express service)
  and Praetor produces a `praetor_adapter.py` (or `.js`) file that
  implements the operator contract on top of their existing app. They
  drop the file into their project, deploy once, and Praetor's Real-Time
  tab can connect to their site for real. No credentials live in
  Praetor - they live in the user's own deployment.

The whole point of this design: Praetor isn't a magic black box that
reaches into someone else's infra. It's a trained agent that works
through a contract. Integration means *helping the user expose that
contract*, not *bypassing it*.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from urllib import parse as urllib_parse
from urllib import request as urllib_request


# ---------------------------------------------------------------------------
# Module-level state - keep it small. Token storage is in-memory, per server
# process. A real product needs encrypted server-side credential storage; we
# explicitly don't go there for this submission (see honest scoping above).
# ---------------------------------------------------------------------------

_INTEGRATIONS_LOCK = threading.RLock()
_GITHUB_STATE: Dict[str, Any] = {
    "device_code": None,        # current pending device code (None when idle)
    "user_code": None,          # the code the user types on github.com
    "verification_uri": None,   # the URL the user opens
    "interval": 5,              # poll interval in seconds (server-recommended)
    "expires_at": 0,            # epoch seconds when the device code expires
    "access_token": None,       # set after successful auth
    "username": None,           # GitHub login of the connected user
    "avatar_url": None,         # for the UI
    "selected_repo": None,      # full_name of the repo the user picked
    "demo_mode": False,         # True if no Client ID was configured
    "next_poll_after": 0,       # epoch seconds - earliest the server should poll GitHub again
}


# ---------------------------------------------------------------------------
# HTTP helper - small, dependency-free, only what we need for GitHub API
# ---------------------------------------------------------------------------


def _http_request(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Tuple[int, Dict[str, Any]]:
    """Tiny HTTP wrapper. Returns (status, parsed_json_or_text_dict)."""
    data_bytes: Optional[bytes] = None
    final_headers = {"Accept": "application/json", "User-Agent": "Praetor-IncidentCommander"}
    if headers:
        final_headers.update(headers)
    if body is not None:
        data_bytes = json.dumps(body).encode("utf-8")
        final_headers.setdefault("Content-Type", "application/json")
    req = urllib_request.Request(url, data=data_bytes, headers=final_headers, method=method)
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {"_raw": raw}
            return resp.status, (parsed if isinstance(parsed, dict) else {"_array": parsed})
    except Exception as exc:
        return 0, {"error": str(exc)}


# ---------------------------------------------------------------------------
# Adapter templates - a tiny built-in template registry. Each template
# materializes a ready-to-deploy file the user drops into their codebase.
# ---------------------------------------------------------------------------


_FASTAPI_ADAPTER = '''"""Praetor adapter - implements the operator contract for your FastAPI app.

Generated by Praetor for: {project_name}
Services exposed: {services_list}

Drop this file into your FastAPI app, mount the router, and point Praetor's
Real-Time tab at your deployed URL. Praetor will then drive your app through
this same /ops/* contract that the simulator uses - same trained policy,
real infrastructure.

Customize the TODOs to wire each endpoint to your actual deployment system
(Kubernetes, ECS, systemd, Docker, whatever you run).
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/ops", tags=["praetor-ops"])


# Map operator-contract service names to the names your platform uses.
SERVICE_MAP: Dict[str, str] = {{
{service_map_entries}
}}


def _resolve(name: str) -> str:
    """Translate Praetor's service name to your platform's name."""
    return SERVICE_MAP.get(name, name)


# ---- Read endpoints ---------------------------------------------------

@router.get("/health")
def ops_health() -> Dict[str, Any]:
    """Return overall + per-service health."""
    services = []
    for praetor_name, real_name in SERVICE_MAP.items():
        # TODO: replace this with a real health probe of `real_name`.
        # Examples:
        #   - call your /healthz on each container
        #   - kubectl get pod {{real_name}} -o json | check status
        #   - aws ecs describe-services
        is_healthy = True
        services.append({{
            "name": praetor_name,
            "platform_name": real_name,
            "health": "healthy" if is_healthy else "down",
        }})
    overall = "ok" if all(s["health"] == "healthy" for s in services) else "degraded"
    return {{"status": overall, "services": services}}


@router.get("/metrics")
def ops_metrics(service: str) -> Dict[str, Any]:
    """Return current metrics for one service."""
    real = _resolve(service)
    # TODO: pull real numbers from your observability stack.
    # Examples: Prometheus, CloudWatch, Datadog, GCP Monitoring.
    return {{
        "service": service,
        "platform_service": real,
        "cpu_percent": 0.0,
        "memory_mb": 0.0,
        "memory_limit_mb": 0.0,
        "memory_utilization_percent": 0.0,
        "request_latency_p50_ms": 0.0,
        "request_latency_p99_ms": 0.0,
        "error_rate_percent": 0.0,
        "active_connections": 0,
        "requests_per_second": 0.0,
    }}


@router.get("/logs")
def ops_logs(service: str, lines: int = 50) -> Dict[str, Any]:
    """Return the last N log lines for the named service."""
    real = _resolve(service)
    # TODO: stream from your log backend.
    # Examples: Loki LogQL, CloudWatch Logs, Datadog logs API.
    return {{
        "service": service,
        "platform_service": real,
        "logs": [f"# TODO: implement log fetching for {{real}}"],
    }}


# ---- Mutation endpoints ----------------------------------------------

class _RestartBody(BaseModel):
    service: str
    memory_limit_mb: Optional[float] = None


class _ScaleBody(BaseModel):
    service: str
    replicas: int


class _ConfigBody(BaseModel):
    service: str
    key: str
    value: Any


class _RollbackBody(BaseModel):
    service: str
    to_version: str


class _BreakBody(BaseModel):
    scenario: str


@router.post("/restart")
def ops_restart(body: _RestartBody) -> Dict[str, Any]:
    """Restart a service, optionally with a new memory limit."""
    real = _resolve(body.service)
    # TODO: implement the real restart. Common patterns:
    #   k8s:    kubectl rollout restart deployment/{{real}}
    #   ECS:    aws ecs update-service --force-new-deployment
    #   Render: POST /v1/services/<id>/deploys (Render API)
    #   Fly:    fly machine restart <machine-id>
    return {{"ok": True, "message": f"TODO: restart {{real}} (mem={{body.memory_limit_mb}}Mi)"}}


@router.post("/scale")
def ops_scale(body: _ScaleBody) -> Dict[str, Any]:
    """Change replica count."""
    real = _resolve(body.service)
    # TODO: scale via your orchestrator.
    return {{"ok": True, "message": f"TODO: scale {{real}} to {{body.replicas}}"}}


@router.post("/config")
def ops_config(body: _ConfigBody) -> Dict[str, Any]:
    """Update a runtime config key."""
    real = _resolve(body.service)
    # TODO: persist the config (Consul, etcd, env-var rotation, etc).
    return {{"ok": True, "message": f"TODO: set {{body.key}}={{body.value}} on {{real}}"}}


@router.post("/rollback")
def ops_rollback(body: _RollbackBody) -> Dict[str, Any]:
    """Roll a service back to a previous version."""
    real = _resolve(body.service)
    # TODO: rollback via your release system.
    return {{"ok": True, "message": f"TODO: rollback {{real}} to {{body.to_version}}"}}


@router.post("/break")
def ops_break(body: _BreakBody) -> Dict[str, Any]:
    """Inject a fault for testing. Optional - you can omit this in prod."""
    return {{"ok": True, "scenario": body.scenario, "message": "TODO: chaos injection (optional)"}}


@router.post("/heal")
def ops_heal() -> Dict[str, Any]:
    """Reset chaos state. Optional - you can omit this in prod."""
    return {{"ok": True, "message": "TODO: clear injected faults (optional)"}}


# Mount the router in your app:
#
#   from fastapi import FastAPI
#   from praetor_adapter import router as praetor_ops
#   app = FastAPI()
#   app.include_router(praetor_ops)
#
# Then in Praetor's Real-Time tab, paste your deployed URL and click Connect.
'''


_RENDER_NOTES = '''# Praetor adapter for Render

Drop `praetor_adapter.py` into your Render service repo, then:

1. Make sure your service exposes a public HTTPS URL (Render does this by default).
2. Set the `RENDER_API_TOKEN` env var on the service so the adapter can call Render's API:
       Settings -> Environment -> Add Environment Variable
       Key:   RENDER_API_TOKEN
       Value: <your token from https://dashboard.render.com/u/settings#api-keys>
3. Implement the TODOs in praetor_adapter.py - most importantly:
       /ops/restart  -> POST https://api.render.com/v1/services/<id>/deploys
       /ops/scale    -> PATCH https://api.render.com/v1/services/<id> body {{"numInstances": N}}
       /ops/health   -> read GET https://api.render.com/v1/services/<id>
4. Deploy. Render gives you a URL like https://your-service.onrender.com/.
5. In Praetor's Real-Time tab, paste that URL and click Connect.

Render API docs: https://api-docs.render.com/reference/introduction
'''


_FLY_NOTES = '''# Praetor adapter for Fly.io

Drop `praetor_adapter.py` into your Fly app, then:

1. Set the `FLY_API_TOKEN` secret:
       fly secrets set FLY_API_TOKEN=$(fly auth token)
2. Implement the TODOs in praetor_adapter.py:
       /ops/restart  -> fly machine restart <machine-id>
                       (or call the Machines API directly)
       /ops/scale    -> fly scale count N
       /ops/health   -> fly status --json | jq
3. fly deploy
4. Praetor's Real-Time tab points at https://<your-app>.fly.dev/

Fly Machines API docs: https://fly.io/docs/machines/api/
'''


_FASTAPI_NOTES = '''# Praetor adapter for a generic FastAPI app

Drop `praetor_adapter.py` into your FastAPI project root.

1. In your main app:
       from fastapi import FastAPI
       from praetor_adapter import router as praetor_ops
       app = FastAPI()
       app.include_router(praetor_ops)
2. Implement the TODOs in praetor_adapter.py to call into your deployment
   system. The adapter is platform-agnostic - fill in the stubs for whatever
   you run (k8s, Docker Compose, systemd, ECS, etc).
3. Deploy your FastAPI app behind a public URL.
4. In Praetor's Real-Time tab, paste that URL and click Connect.

Praetor will drive the same /ops/* contract that the simulator uses, so the
trained policy works against your real infrastructure unchanged.
'''


def _render_adapter(project_name: str, services: List[str], platform: str) -> Tuple[str, str]:
    """Materialize the adapter file + README for the chosen platform.

    Returns (filename_to_README_text, filename_to_adapter_text) - actually a
    list of (filename, content) tuples. Callers either return them as a JSON
    or zip them.
    """
    services = services or ["frontend", "api", "postgres"]
    service_map_entries = ",\n".join(f'    "{s}": "{s}"' for s in services)
    services_list = ", ".join(services) or "(none)"
    adapter_py = _FASTAPI_ADAPTER.format(
        project_name=project_name or "your-project",
        services_list=services_list,
        service_map_entries=service_map_entries,
    )
    notes = {
        "fastapi": _FASTAPI_NOTES,
        "render": _RENDER_NOTES,
        "fly": _FLY_NOTES,
    }.get(platform, _FASTAPI_NOTES)
    return adapter_py, notes


# ---------------------------------------------------------------------------
# GitHub device flow - works without needing a public callback URL.
# Reference: https://docs.github.com/en/apps/creating-github-apps/writing-code-for-a-github-app/building-a-cli-with-a-github-app
# ---------------------------------------------------------------------------


def _github_client_id() -> Optional[str]:
    cid = os.getenv("GITHUB_CLIENT_ID", "").strip()
    return cid or None


def _github_start_device_flow() -> Dict[str, Any]:
    """Kick off the device-code grant. Returns the user_code + verification_uri."""
    client_id = _github_client_id()
    if not client_id:
        # Demo mode - surface a fake but obvious flow so the UI stays consistent.
        with _INTEGRATIONS_LOCK:
            _GITHUB_STATE.update({
                "device_code": "DEMO_DEVICE_CODE",
                "user_code": "DEMO-CODE",
                "verification_uri": "https://github.com/settings/applications/new",
                "interval": 0,
                "expires_at": time.time() + 900,
                "demo_mode": True,
            })
        return {
            "demo_mode": True,
            "user_code": "DEMO-CODE",
            "verification_uri": "https://github.com/settings/applications/new",
            "instructions": (
                "GitHub OAuth is in demo mode because GITHUB_CLIENT_ID is not set. "
                "To enable real OAuth: register an OAuth App at "
                "https://github.com/settings/developers (any callback URL is fine - "
                "we use device flow), then set GITHUB_CLIENT_ID in your .env and "
                "restart the server. For the demo, you can pretend the connection "
                "succeeded and use the curated example repo list."
            ),
        }

    status, body = _http_request(
        "POST",
        "https://github.com/login/device/code",
        body={"client_id": client_id, "scope": "repo read:user"},
    )
    if status != 200 or "device_code" not in body:
        return {"error": f"GitHub device-code request failed: status={status}, body={body}"}

    with _INTEGRATIONS_LOCK:
        _GITHUB_STATE.update({
            "device_code": body["device_code"],
            "user_code": body["user_code"],
            "verification_uri": body.get("verification_uri", "https://github.com/login/device"),
            "interval": int(body.get("interval", 5)),
            "expires_at": time.time() + int(body.get("expires_in", 900)),
            "demo_mode": False,
            "next_poll_after": 0,  # fresh flow - clear any throttle from a prior code
            "access_token": None,  # don't carry over a token from a previous flow
            "username": None,
            "avatar_url": None,
        })
    return {
        "demo_mode": False,
        "user_code": body["user_code"],
        "verification_uri": body.get("verification_uri", "https://github.com/login/device"),
        "interval": body.get("interval", 5),
        "expires_in_seconds": body.get("expires_in", 900),
    }


def _github_poll_for_token() -> Dict[str, Any]:
    """Poll GitHub once for an access token. Frontend polls this every interval."""
    with _INTEGRATIONS_LOCK:
        state = dict(_GITHUB_STATE)

    if state.get("access_token"):
        return {
            "status": "authorized",
            "username": state.get("username"),
            "avatar_url": state.get("avatar_url"),
            "demo_mode": state.get("demo_mode", False),
        }

    if state.get("demo_mode"):
        # Auto-complete the demo flow after the user has had a chance to read the modal.
        time.sleep(0.5)
        with _INTEGRATIONS_LOCK:
            _GITHUB_STATE.update({
                "access_token": "demo-token",
                "username": "praetor-demo",
                "avatar_url": "https://avatars.githubusercontent.com/u/9919?v=4",
            })
        return {"status": "authorized", "username": "praetor-demo", "demo_mode": True}

    if not state.get("device_code"):
        return {"status": "idle", "error": "no device flow in progress"}

    if time.time() > state.get("expires_at", 0):
        with _INTEGRATIONS_LOCK:
            _GITHUB_STATE.update({"device_code": None, "user_code": None})
        return {"status": "expired", "error": "device code expired - restart the connection"}

    # Respect GitHub's slow_down throttling. If the server told us to wait,
    # don't hit GitHub again until that window passes - frontend polls every
    # 2-5s but we only forward to GitHub at the safe interval.
    now = time.time()
    if now < state.get("next_poll_after", 0):
        wait = max(0.0, state["next_poll_after"] - now)
        return {"status": "pending", "throttled": True, "retry_in_seconds": round(wait, 1)}

    client_id = _github_client_id()
    status, body = _http_request(
        "POST",
        "https://github.com/login/oauth/access_token",
        body={
            "client_id": client_id,
            "device_code": state["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        },
    )
    if status != 200:
        return {"status": "pending", "error": f"poll failed: {body.get('error', status)}"}

    if body.get("error") == "authorization_pending":
        # Mark the next safe poll time at server interval + 1s safety margin
        with _INTEGRATIONS_LOCK:
            _GITHUB_STATE["next_poll_after"] = time.time() + state.get("interval", 5) + 1
        return {"status": "pending"}
    if body.get("error") == "slow_down":
        # GitHub asked us to back off - increase interval by 5s permanently
        with _INTEGRATIONS_LOCK:
            _GITHUB_STATE["interval"] = state.get("interval", 5) + 5
            _GITHUB_STATE["next_poll_after"] = time.time() + _GITHUB_STATE["interval"] + 1
        return {"status": "pending", "slow_down": True,
                "new_interval": _GITHUB_STATE["interval"]}
    if body.get("error"):
        return {"status": "error", "error": body.get("error")}

    access_token = body.get("access_token")
    if not access_token:
        return {"status": "pending"}

    # We got a token. Persist it immediately - the UI considers us connected
    # the moment a token exists. We attempt to fetch /user for the avatar +
    # username, but a GitHub App with limited permissions might fail this
    # call; that's OK - we can list repos with the token without /user.
    with _INTEGRATIONS_LOCK:
        _GITHUB_STATE.update({
            "access_token": access_token,
            "username": "(connected)",
            "avatar_url": None,
        })

    user_status, user = _http_request(
        "GET",
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=6.0,
    )
    if user_status == 200 and isinstance(user, dict) and user.get("login"):
        with _INTEGRATIONS_LOCK:
            _GITHUB_STATE.update({
                "username": user.get("login"),
                "avatar_url": user.get("avatar_url"),
            })

    with _INTEGRATIONS_LOCK:
        final = dict(_GITHUB_STATE)

    return {
        "status": "authorized",
        "username": final.get("username"),
        "avatar_url": final.get("avatar_url"),
        "demo_mode": False,
        "user_endpoint_ok": user_status == 200,
    }


def _github_list_repos(per_page: int = 30) -> Dict[str, Any]:
    """List the connected user's repos."""
    with _INTEGRATIONS_LOCK:
        state = dict(_GITHUB_STATE)

    if state.get("demo_mode") or state.get("access_token") == "demo-token":
        # Curated example - the user picks one to scope Tier-2 against.
        return {
            "demo_mode": True,
            "repos": [
                {"full_name": "root4shreshth/HealioX",
                 "description": "User's actual repo - connected for tier-2 code escalation",
                 "default_branch": "main",
                 "language": "Python", "stargazers_count": 0},
                {"full_name": "metamorphs/praetor-example-app",
                 "description": "Example app showing how the adapter integrates",
                 "default_branch": "main",
                 "language": "TypeScript", "stargazers_count": 12},
                {"full_name": "metamorphs/incident-commander",
                 "description": "Praetor itself",
                 "default_branch": "main",
                 "language": "Python", "stargazers_count": 1},
            ],
        }

    if not state.get("access_token"):
        return {"error": "not connected - call /integrations/github/start first"}

    status, body = _http_request(
        "GET",
        f"https://api.github.com/user/repos?per_page={per_page}&sort=pushed",
        headers={"Authorization": f"Bearer {state['access_token']}"},
    )
    if status != 200:
        return {"error": f"GitHub /user/repos failed: status={status}"}

    if isinstance(body, dict) and "_array" in body:
        repos_list = body["_array"]
    elif isinstance(body, list):
        repos_list = body
    else:
        repos_list = []
    repos = [
        {
            "full_name": r.get("full_name"),
            "description": r.get("description") or "",
            "default_branch": r.get("default_branch"),
            "language": r.get("language"),
            "stargazers_count": r.get("stargazers_count", 0),
            "private": r.get("private", False),
        }
        for r in repos_list
    ]
    return {"demo_mode": False, "repos": repos}


def _github_disconnect() -> Dict[str, Any]:
    with _INTEGRATIONS_LOCK:
        _GITHUB_STATE.update({
            "device_code": None, "user_code": None, "verification_uri": None,
            "access_token": None, "username": None, "avatar_url": None,
            "selected_repo": None, "demo_mode": False,
        })
    return {"ok": True, "status": "disconnected"}


def _github_select_repo(full_name: str) -> Dict[str, Any]:
    if not full_name:
        return {"error": "full_name required"}
    with _INTEGRATIONS_LOCK:
        _GITHUB_STATE["selected_repo"] = full_name
    return {"ok": True, "selected_repo": full_name}


def _github_status() -> Dict[str, Any]:
    with _INTEGRATIONS_LOCK:
        s = dict(_GITHUB_STATE)
    return {
        "connected": bool(s.get("access_token")),
        "username": s.get("username"),
        "avatar_url": s.get("avatar_url"),
        "selected_repo": s.get("selected_repo"),
        "demo_mode": s.get("demo_mode", False),
        "client_id_configured": bool(_github_client_id()),
    }


# ---------------------------------------------------------------------------
# Cloud provider stubs - credentials NEVER hit the server. Frontend stores
# them in localStorage and the user knows it. This endpoint just reports
# whether a provider is "connected" so the UI can render the right state.
# ---------------------------------------------------------------------------


_CLOUD_PROVIDERS = {
    "aws":    {"label": "AWS",          "fields": ["access_key_id", "secret_access_key", "region"]},
    "gcp":    {"label": "Google Cloud", "fields": ["project_id", "service_account_json"]},
    "azure":  {"label": "Azure",        "fields": ["subscription_id", "tenant_id", "client_secret"]},
    "render": {"label": "Render",       "fields": ["api_token"]},
    "fly":    {"label": "Fly.io",       "fields": ["api_token"]},
    "vercel": {"label": "Vercel",       "fields": ["api_token", "team_id"]},
}


def _list_cloud_providers() -> Dict[str, Any]:
    return {
        "providers": [
            {"id": pid, **meta}
            for pid, meta in _CLOUD_PROVIDERS.items()
        ],
        "demo_disclaimer": (
            "Cloud-provider integration is in demo mode for the hackathon submission. "
            "Credentials entered in the UI are stored in your browser's localStorage and "
            "are NEVER sent to Praetor's server. No real cloud actions are executed. "
            "To make this real, deploy the generated adapter into your cloud project - "
            "Praetor connects to that adapter, the adapter holds your credentials, "
            "and your credentials never leave your own infrastructure."
        ),
    }


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------


class _SelectRepoBody(BaseModel):
    full_name: str


class _AdapterRequestBody(BaseModel):
    project_name: str = "praetor-target"
    services: List[str] = ["frontend", "api", "postgres"]
    platform: str = "fastapi"


def make_integrations_router() -> APIRouter:
    router = APIRouter(prefix="/integrations", tags=["integrations"])

    # -- GitHub -----------------------------------------------------------

    @router.post("/github/start")
    def gh_start() -> Dict[str, Any]:
        return _github_start_device_flow()

    @router.get("/github/poll")
    def gh_poll() -> Dict[str, Any]:
        return _github_poll_for_token()

    @router.get("/github/repos")
    def gh_repos() -> Dict[str, Any]:
        return _github_list_repos()

    @router.post("/github/select-repo")
    def gh_select(body: _SelectRepoBody) -> Dict[str, Any]:
        return _github_select_repo(body.full_name)

    @router.post("/github/disconnect")
    def gh_disconnect() -> Dict[str, Any]:
        return _github_disconnect()

    @router.get("/github/status")
    def gh_status() -> Dict[str, Any]:
        return _github_status()

    # -- Cloud providers (demo) ------------------------------------------

    @router.get("/cloud/providers")
    def cloud_providers() -> Dict[str, Any]:
        return _list_cloud_providers()

    # -- Adapter generator ------------------------------------------------

    @router.post("/adapter/preview")
    def adapter_preview(body: _AdapterRequestBody) -> Dict[str, Any]:
        adapter_py, notes = _render_adapter(body.project_name, body.services, body.platform)
        return {
            "platform": body.platform,
            "project_name": body.project_name,
            "services": body.services,
            "adapter_py": adapter_py,
            "notes": notes,
            "filename": "praetor_adapter.py",
        }

    @router.post("/adapter/download")
    def adapter_download(body: _AdapterRequestBody):
        adapter_py, notes = _render_adapter(body.project_name, body.services, body.platform)
        # Stream a single combined .py file with the notes as a top-of-file
        # docstring. Keeps the download dependency-free (no zipfile import).
        combined = (
            f'"""Praetor adapter - {body.platform} platform, project={body.project_name}\n\n'
            f'{notes}\n"""\n\n{adapter_py}'
        )
        headers = {"Content-Disposition": 'attachment; filename="praetor_adapter.py"'}
        return PlainTextResponse(combined, headers=headers)

    return router
