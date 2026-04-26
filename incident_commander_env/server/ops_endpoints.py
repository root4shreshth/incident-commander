"""Self-served `/ops/*` operator-contract endpoints.

This module makes Praetor self-targetable for the Real-Time tab demo. The
WebsiteBackend expects any deployed site to expose an `/ops/*` HTTP contract
(health, metrics, logs, restart, rollback, scale, config, break, heal). Most
deployed sites — Netlify static apps, plain FastAPI projects — don't have
these endpoints out of the box. Without a sample target, the demo bricks on
"GET /ops/health failed: HTTP 404."

The fix: implement the operator contract here, on top of a separate
`IncidentCommanderEnv` instance scoped to this module. A user can then point
Real-Time at `http://127.0.0.1:8000` (or the deployed Space URL itself) and
get a fully working sim-to-real demo without standing up a separate site.

Endpoints (all under `/ops`):

    GET  /ops/health                   - {"status", "services": [{"name", "health"}, ...]}
    GET  /ops/metrics?service=<name>   - CPU, memory, latency, error rate, ...
    GET  /ops/logs?service=<name>&lines=N
    GET  /ops/services                 - list of service names this target exposes
    POST /ops/restart                  body: {"service", "memory_limit_mb"?}
    POST /ops/scale                    body: {"service", "replicas"}
    POST /ops/config                   body: {"service", "key", "value"}
    POST /ops/rollback                 body: {"service", "to_version"}
    POST /ops/break                    body: {"scenario"}
    POST /ops/heal                     resets the chaos state
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from incident_commander_env.models import IncidentAction
from incident_commander_env.server.environment import IncidentCommanderEnv
from incident_commander_env.server.simulation.service import ServiceHealth


# Map between the WebsiteBackend's short operator-contract names and the
# simulator's full service names. The Real-Time UI uses ("frontend","api","postgres")
# by default; we map those to the closest sim services so the demo just works.
DEFAULT_OPS_NAME_MAP: Dict[str, str] = {
    "frontend": "frontend-bff",
    "api": "api-gateway",
    "postgres": "postgres-db",
}

# Reverse map for /ops/health output: when the sim has services like
# "frontend-bff", report them under the operator-contract name "frontend".
SIM_TO_OPS_NAME: Dict[str, str] = {v: k for k, v in DEFAULT_OPS_NAME_MAP.items()}


def _ops_to_sim(name: str) -> str:
    """Resolve an operator-contract service name to the sim's internal name."""
    return DEFAULT_OPS_NAME_MAP.get(name, name)


def _sim_to_ops(name: str) -> str:
    """Translate sim service names back to short operator-contract names if mapped."""
    return SIM_TO_OPS_NAME.get(name, name)


def _ensure_initialized(env: IncidentCommanderEnv) -> None:
    """Lazy-init the env so the very first /ops call works without an explicit /reset."""
    if env._cluster is None:
        env.reset(task_id="oom_crash", seed=0)


def _service_health_str(service_obj: Any) -> str:
    """Coerce ServiceHealth enum to a human string for the operator contract."""
    h = getattr(service_obj, "health", None)
    if isinstance(h, ServiceHealth):
        return h.value if isinstance(h.value, str) else h.name.lower()
    if isinstance(h, str):
        return h.lower()
    return "unknown"


# ---------------------------------------------------------------------------
# Pydantic request bodies (loose — the WebsiteBackend isn't strict about them)
# ---------------------------------------------------------------------------


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
    scenario: str  # oom_crash | db_pool_exhaustion | bad_deployment_cascade


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_ops_router(env: IncidentCommanderEnv) -> APIRouter:
    """Build a FastAPI router exposing the operator contract on top of `env`.

    The router shares a single `IncidentCommanderEnv` instance with the rest
    of the app (or a dedicated demo instance — see app.py wiring).
    """
    router = APIRouter(prefix="/ops", tags=["ops-contract"])

    # ---- Read endpoints ---------------------------------------------------

    @router.get("/health")
    def ops_health() -> Dict[str, Any]:
        _ensure_initialized(env)
        cluster = env._cluster
        if cluster is None:
            return {"status": "down", "services": []}

        services_out: List[Dict[str, Any]] = []
        any_unhealthy = False
        for sim_name, svc in cluster.services.items():
            health_str = _service_health_str(svc)
            ops_name = _sim_to_ops(sim_name)
            services_out.append({
                "name": ops_name,
                "sim_name": sim_name,
                "health": health_str,
            })
            if health_str not in ("healthy", "ok"):
                any_unhealthy = True

        overall_status = "degraded" if any_unhealthy else "ok"
        # If ANY service is crashed, mark overall as "down"
        if any(s["health"] in ("crashed",) for s in services_out):
            overall_status = "down"

        return {"status": overall_status, "services": services_out}

    @router.get("/services")
    def ops_services() -> Dict[str, Any]:
        _ensure_initialized(env)
        cluster = env._cluster
        if cluster is None:
            return {"services": []}
        return {
            "services": [
                {"name": _sim_to_ops(name), "sim_name": name}
                for name in cluster.services.keys()
            ],
        }

    @router.get("/metrics")
    def ops_metrics(service: str) -> Dict[str, Any]:
        _ensure_initialized(env)
        sim_name = _ops_to_sim(service)
        obs = env.step(IncidentAction(
            action_type="check_metrics",
            target_service=sim_name,
            parameters={},
        ))
        m = getattr(obs, "metrics", None)
        if m is None:
            return {"service": service, "error": obs.error or "metrics unavailable"}
        m_dict = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        # Translate to operator-contract field names the WebsiteBackend expects.
        return {
            "service": service,
            "sim_service": sim_name,
            "cpu_percent": m_dict.get("cpu_percent"),
            "memory_mb": m_dict.get("memory_mb"),
            "memory_limit_mb": m_dict.get("memory_limit_mb"),
            "memory_utilization_percent": m_dict.get("memory_utilization_percent"),
            "request_latency_p50_ms": m_dict.get("request_latency_p50_ms"),
            "request_latency_p99_ms": m_dict.get("request_latency_p99_ms"),
            "error_rate_percent": m_dict.get("error_rate_percent"),
            "active_connections": m_dict.get("active_connections"),
            "requests_per_second": m_dict.get("requests_per_second"),
        }

    @router.get("/logs")
    def ops_logs(service: str, lines: int = 50) -> Dict[str, Any]:
        _ensure_initialized(env)
        sim_name = _ops_to_sim(service)
        obs = env.step(IncidentAction(
            action_type="read_logs",
            target_service=sim_name,
            parameters={"lines": int(max(1, min(lines, 500)))},
        ))
        return {
            "service": service,
            "sim_service": sim_name,
            "logs": obs.logs or [],
            "error": obs.error,
        }

    # ---- Mutation endpoints ----------------------------------------------

    @router.post("/restart")
    def ops_restart(body: _RestartBody) -> Dict[str, Any]:
        _ensure_initialized(env)
        sim_name = _ops_to_sim(body.service)
        params: Dict[str, Any] = {}
        if body.memory_limit_mb is not None:
            params["memory_limit"] = f"{int(body.memory_limit_mb)}Mi"
        obs = env.step(IncidentAction(
            action_type="restart_service",
            target_service=sim_name,
            parameters=params,
        ))
        return {"ok": obs.error is None, "message": obs.message, "error": obs.error}

    @router.post("/scale")
    def ops_scale(body: _ScaleBody) -> Dict[str, Any]:
        _ensure_initialized(env)
        sim_name = _ops_to_sim(body.service)
        obs = env.step(IncidentAction(
            action_type="scale_service",
            target_service=sim_name,
            parameters={"replicas": int(body.replicas)},
        ))
        return {"ok": obs.error is None, "message": obs.message, "error": obs.error}

    @router.post("/config")
    def ops_config(body: _ConfigBody) -> Dict[str, Any]:
        _ensure_initialized(env)
        sim_name = _ops_to_sim(body.service)
        obs = env.step(IncidentAction(
            action_type="update_config",
            target_service=sim_name,
            parameters={"key": body.key, "value": body.value},
        ))
        return {"ok": obs.error is None, "message": obs.message, "error": obs.error}

    @router.post("/rollback")
    def ops_rollback(body: _RollbackBody) -> Dict[str, Any]:
        _ensure_initialized(env)
        sim_name = _ops_to_sim(body.service)
        obs = env.step(IncidentAction(
            action_type="rollback_deployment",
            target_service=sim_name,
            parameters={"to_version": body.to_version},
        ))
        return {"ok": obs.error is None, "message": obs.message, "error": obs.error}

    @router.post("/break")
    def ops_break(body: _BreakBody) -> Dict[str, Any]:
        valid = {"oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"}
        if body.scenario not in valid:
            return {"ok": False, "error": f"unknown scenario: {body.scenario}",
                    "valid_scenarios": sorted(valid)}
        # Re-init the env with the chosen scenario — that injects the fault.
        obs = env.reset(task_id=body.scenario, seed=0)
        return {
            "ok": True,
            "scenario": body.scenario,
            "alert": obs.alert,
            "message": "fault injected; site is now degraded",
        }

    @router.post("/heal")
    def ops_heal() -> Dict[str, Any]:
        # Re-init at oom_crash with a healthy seed and immediately resolve via
        # restart of the failing service. Simplest: just re-create a clean cluster
        # by re-resetting on a different family at low difficulty and clearing
        # anomalies.
        _ensure_initialized(env)
        cluster = env._cluster
        if cluster is None:
            return {"ok": False, "error": "no cluster"}
        for svc in cluster.services.values():
            try:
                svc.clear_all_anomalies()
                svc.restart()
            except Exception:
                pass
        return {"ok": True, "message": "chaos cleared; all services restarted"}

    return router
