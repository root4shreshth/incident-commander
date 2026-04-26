"""WebsiteBackend - talks to a deployed site's `/ops/*` operator API over HTTP.

Replaces the Docker-Compose-shaped `RealBackend` for the actual sim-to-real
demo. Far simpler contract: any deployable HTTP service that exposes the
operator endpoints documented in the README can be the substrate for the
trained policy. No Docker daemon, no compose file, no shell-outs.

Operator contract the target site must implement:
  GET  /ops/health                    -> {"status": "ok"|"degraded"|"down", "services": [...]}
  GET  /ops/metrics?service=<name>    -> {"cpu_percent","memory_mb","memory_limit_mb",...}
  GET  /ops/logs?service=<name>&lines=N -> {"logs": [...]}
  POST /ops/restart        body: {"service":"...", "memory_limit_mb": 1024}
  POST /ops/scale          body: {"service":"...", "replicas": N}
  POST /ops/config         body: {"service":"...", "key":"...", "value": ...}
  POST /ops/rollback       body: {"service":"...", "to_version":"..."}
  POST /ops/break          body: {"scenario": "oom_crash"|"db_pool_exhaustion"|"bad_deployment_cascade"}
  POST /ops/heal           (resets chaos)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from incident_commander_env.models import (
    IncidentAction,
    IncidentObservation,
    MetricsSnapshot,
    ServiceDetail,
    ServiceSummary,
)
from incident_commander_env.server.backends.protocol import (
    BackendSnapshot,
    QuotaSnapshot,
    ServiceSnapshot,
)

if TYPE_CHECKING:
    from incident_commander_env.server.scenarios.base_scenario import BaseScenario


DEFAULT_SERVICE_NAMES = ("frontend", "api", "postgres")


@dataclass
class HttpResult:
    ok: bool
    status: int = 0
    body: Any = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Thin HTTP client - single seam tests can monkeypatch
# ---------------------------------------------------------------------------

def _http(method: str, url: str, json_body: Optional[Dict[str, Any]] = None,
          timeout: float = 8.0) -> HttpResult:
    """Single shell-out we mock in tests. Returns a typed HttpResult."""
    data: Optional[bytes] = None
    headers = {"Accept": "application/json", "User-Agent": "incident-commander/1.0"}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib_request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            body: Any = raw
            try:
                body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                pass
            return HttpResult(ok=True, status=resp.status, body=body)
    except urllib_error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = None
        return HttpResult(ok=False, status=exc.code, body=err_body, error=f"HTTP {exc.code}")
    except (urllib_error.URLError, OSError, TimeoutError) as exc:
        return HttpResult(ok=False, status=0, error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# WebsiteBackend
# ---------------------------------------------------------------------------

class WebsiteBackend:
    """Backend that drives a deployed site exposing the /ops/* operator API."""

    name = "website"

    def __init__(
        self,
        site_url: Optional[str] = None,
        service_names: Optional[List[str]] = None,
    ) -> None:
        self.site_url = self._normalize(site_url) if site_url else None
        self.service_names = list(service_names or DEFAULT_SERVICE_NAMES)
        self._reset_done = False
        self._stub_mode = False
        self._stub_reason: Optional[str] = None
        # We keep a shadow snapshot of mutations the agent has applied - rendered
        # in /backend snapshot when the live site doesn't echo them all back.
        self._mem_limits_mb: Dict[str, int] = {}
        self._pool_sizes: Dict[str, int] = {}
        self._image_tag: Optional[str] = None
        self._restart_history: List[str] = []
        # snapshot cache to amortize cost when the env asks repeatedly
        self._cached_snapshot: Optional[BackendSnapshot] = None
        self._cached_at: float = 0.0

    @staticmethod
    def _normalize(url: str) -> str:
        url = (url or "").strip()
        if not url:
            return url
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        return url.rstrip("/")

    def configure(self, site_url: str) -> None:
        """Update target URL between episodes - used by /realtime/connect."""
        self.site_url = self._normalize(site_url)

    # ---- Lifecycle ----------------------------------------------------------

    def reset(self, scenario: "BaseScenario", seed: Optional[int] = None) -> None:
        self._reset_done = True
        self._mem_limits_mb = {}
        self._pool_sizes = {}
        self._image_tag = None
        self._restart_history = []
        self._cached_snapshot = None
        self._cached_at = 0.0
        if not self.site_url:
            self._stub_mode = True
            self._stub_reason = "no site_url configured"
            return
        # Heal first so any prior chaos from a previous episode is cleared
        heal = _http("POST", self.site_url + "/ops/heal", json_body={}, timeout=8.0)
        # If heal is missing we tolerate - site might not implement it (older contracts)
        if not heal.ok and heal.status != 404:
            self._stub_mode = True
            self._stub_reason = f"heal call failed: {heal.error}"
            return
        # Inject the scenario's chaos so the trained policy sees the fault
        inj = _http(
            "POST", self.site_url + "/ops/break",
            json_body={"scenario": scenario.task_id},
            timeout=10.0,
        )
        if not inj.ok:
            self._stub_mode = True
            self._stub_reason = f"break call failed: {inj.error}"
            return
        self._stub_mode = False
        self._stub_reason = None

    def teardown(self) -> None:
        if self._stub_mode or not self.site_url:
            return
        # Best-effort heal so we don't leave the deployed site in a broken state
        _http("POST", self.site_url + "/ops/heal", json_body={}, timeout=5.0)

    def tick(self) -> None:
        # The deployed site advances on its own clock - nothing for us to do.
        return

    # ---- Snapshot -----------------------------------------------------------

    def snapshot(self) -> BackendSnapshot:
        if self._stub_mode or not self.site_url:
            return self._stub_snapshot()
        # Cache for 0.4s so the env's reward computation + rubric checks don't
        # hammer the deployed site each time they ask for state.
        if self._cached_snapshot and (time.monotonic() - self._cached_at) < 0.4:
            return self._cached_snapshot

        snap = self._build_snapshot()
        self._cached_snapshot = snap
        self._cached_at = time.monotonic()
        return snap

    def _build_snapshot(self) -> BackendSnapshot:
        services: Dict[str, ServiceSnapshot] = {}
        total_cpu = 0.0
        total_mem = 0.0
        total_mem_limit = 0.0

        # Pull the cluster-wide /ops/health once for the source-of-truth health labels
        health = _http("GET", self.site_url + "/ops/health", timeout=5.0)
        health_per_svc: Dict[str, str] = {}
        if health.ok and isinstance(health.body, dict):
            services_arr = health.body.get("services") or []
            for s in services_arr:
                if isinstance(s, dict) and s.get("name"):
                    health_per_svc[s["name"]] = (s.get("health") or s.get("status") or "healthy").lower()

        for svc in self.service_names:
            metrics = _http(
                "GET",
                self.site_url + "/ops/metrics?" + urllib_parse.urlencode({"service": svc}),
                timeout=4.0,
            )
            cpu = 0.0
            mem_mb = 0.0
            mem_limit_mb = float(self._mem_limits_mb.get(svc, 512))
            err_rate = 0.0
            latency_p99 = 0.0
            connections = 0
            rps = 0.0
            if metrics.ok and isinstance(metrics.body, dict):
                m = metrics.body
                cpu = float(m.get("cpu_percent", 0) or 0)
                mem_mb = float(m.get("memory_mb", 0) or 0)
                mem_limit_mb = float(m.get("memory_limit_mb", mem_limit_mb) or mem_limit_mb)
                err_rate = float(m.get("error_rate_percent", 0) or 0)
                latency_p99 = float(m.get("request_latency_p99_ms", 0) or 0)
                connections = int(m.get("active_connections", 0) or 0)
                rps = float(m.get("requests_per_second", 0) or 0)
            services[svc] = ServiceSnapshot(
                name=svc,
                health=health_per_svc.get(svc, "healthy" if metrics.ok else "unhealthy"),
                version=self._image_tag or "v1.0",
                replicas=1,
                cpu_percent=round(cpu, 1),
                memory_mb=round(mem_mb, 1),
                memory_limit_mb=round(mem_limit_mb, 1),
                error_rate_percent=round(err_rate, 2),
                request_latency_p99_ms=round(latency_p99, 1),
                active_connections=connections,
                requests_per_second=round(rps, 1),
            )
            total_cpu += cpu
            total_mem += mem_mb
            total_mem_limit += mem_limit_mb

        quota = QuotaSnapshot(
            cpu_used=round(total_cpu, 2),
            cpu_total=400.0,
            cpu_utilization_percent=round(total_cpu / 4.0, 1),
            memory_used_mb=round(total_mem, 1),
            memory_total_mb=round(total_mem_limit, 1),
            memory_utilization_percent=(
                round(100.0 * total_mem / total_mem_limit, 1) if total_mem_limit > 0 else 0.0
            ),
        )
        return BackendSnapshot(services=services, quota=quota)

    def _stub_snapshot(self) -> BackendSnapshot:
        services = {
            n: ServiceSnapshot(
                name=n, health="healthy", version="v1.0", replicas=1,
                cpu_percent=0.0, memory_mb=0.0,
                memory_limit_mb=float(self._mem_limits_mb.get(n, 512)),
                error_rate_percent=0.0,
            )
            for n in self.service_names
        }
        return BackendSnapshot(services=services, quota=QuotaSnapshot())

    # ---- Resolution ---------------------------------------------------------

    def check_resolved(self, scenario: "BaseScenario") -> bool:
        if self._stub_mode or not self.site_url:
            return False
        # Poll /ops/health a few times - need a stable green window so a flaky
        # restart doesn't false-positive.
        green = 0
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            r = _http("GET", self.site_url + "/ops/health", timeout=4.0)
            if r.ok and isinstance(r.body, dict):
                status = (r.body.get("status") or "").lower()
                if status == "ok":
                    green += 1
                    if green >= 2:
                        return True
                else:
                    green = 0
            else:
                green = 0
            time.sleep(2.0)
        return False

    # ---- Action dispatch ----------------------------------------------------

    def execute(
        self, action: IncidentAction, scenario: "BaseScenario"
    ) -> IncidentObservation:
        if not self._reset_done:
            return IncidentObservation(
                message="Error: backend not reset. Call reset() first.",
                error="Backend not initialized",
                done=True,
            )
        # Invalidate cache on any mutating action so the next snapshot reflects truth.
        if action.action_type not in {"list_services", "describe_service", "read_logs", "check_metrics", "run_diagnostic"}:
            self._cached_snapshot = None
        handler = _HANDLERS.get(action.action_type)
        if handler is None:
            return IncidentObservation(
                message=f"Unknown action: {action.action_type}",
                error=f"Invalid action_type: {action.action_type}",
            )
        try:
            return handler(self, action, scenario)
        except Exception as exc:  # pragma: no cover - defensive
            return IncidentObservation(
                message=f"WebsiteBackend error executing {action.action_type}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )

    # ---- Helpers ------------------------------------------------------------

    def _stub_observation(self, action_type: str) -> IncidentObservation:
        why = self._stub_reason or "site_url not configured"
        return IncidentObservation(
            message=f"WebsiteBackend in stub mode ({why}). Action '{action_type}' would call the site.",
            error="website_backend_stub",
        )


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _list_services(b: WebsiteBackend, a: IncidentAction, s) -> IncidentObservation:
    if b._stub_mode:
        return b._stub_observation("list_services")
    snap = b.snapshot()
    summaries: List[ServiceSummary] = []
    healthy = sum(1 for sv in snap.services.values() if sv.health == "healthy")
    lines = [f"Cluster overview: {healthy}/{len(snap.services)} services healthy\n"]
    for name, sv in snap.services.items():
        summaries.append(ServiceSummary(
            name=sv.name, health=sv.health, version=sv.version, replicas=sv.replicas,
            cpu_percent=sv.cpu_percent, memory_mb=sv.memory_mb,
            error_rate_percent=sv.error_rate_percent,
        ))
        icon = "OK" if sv.health == "healthy" else sv.health.upper()
        lines.append(
            f"  [{icon:>10}] {sv.name:25s} v{sv.version:8s} "
            f"replicas={sv.replicas}  cpu={sv.cpu_percent:.0f}%  "
            f"mem={sv.memory_mb:.0f}MB  err={sv.error_rate_percent:.1f}%"
        )
    return IncidentObservation(message="\n".join(lines), services_summary=summaries)


def _describe_service(b: WebsiteBackend, a: IncidentAction, s) -> IncidentObservation:
    if not a.target_service:
        return IncidentObservation(message="Error: this action requires a target_service.", error="Missing target_service")
    if b._stub_mode:
        return b._stub_observation("describe_service")
    snap = b.snapshot()
    sv = snap.get_service(a.target_service)
    if not sv:
        return IncidentObservation(message=f"Service '{a.target_service}' not found.", error="Service not found")
    detail = ServiceDetail(
        name=sv.name, health=sv.health, version=sv.version, replicas=sv.replicas,
        memory_limit=f"{int(sv.memory_limit_mb)}Mi", cpu_limit="1000m", port=8000,
        db_pool_size=b._pool_sizes.get(sv.name),
    )
    return IncidentObservation(
        message=f"{sv.name}: health={sv.health} v{sv.version} cpu={sv.cpu_percent:.0f}% mem={sv.memory_mb:.0f}/{sv.memory_limit_mb:.0f}MB",
        service_detail=detail,
    )


def _read_logs(b: WebsiteBackend, a: IncidentAction, s) -> IncidentObservation:
    if not a.target_service:
        return IncidentObservation(message="Error: this action requires a target_service.", error="Missing target_service")
    if b._stub_mode:
        return b._stub_observation("read_logs")
    lines = int(a.parameters.get("lines", 50))
    r = _http("GET", b.site_url + "/ops/logs?" + urllib_parse.urlencode({"service": a.target_service, "lines": lines}), timeout=6.0)
    if not r.ok:
        return IncidentObservation(message=f"Failed to read logs: {r.error}", error=r.error or "log read failed")
    logs = []
    if isinstance(r.body, dict):
        logs = r.body.get("logs") or []
    elif isinstance(r.body, list):
        logs = r.body
    return IncidentObservation(
        message=f"Last {len(logs)} lines from {a.target_service}:\n" + "\n".join(map(str, logs[-lines:])),
        logs=[str(x) for x in logs[-lines:]],
    )


def _check_metrics(b: WebsiteBackend, a: IncidentAction, s) -> IncidentObservation:
    if not a.target_service:
        return IncidentObservation(message="Error: this action requires a target_service.", error="Missing target_service")
    if b._stub_mode:
        return b._stub_observation("check_metrics")
    snap = b.snapshot()
    sv = snap.get_service(a.target_service)
    if not sv:
        return IncidentObservation(message=f"Service '{a.target_service}' not found.", error="Service not found")
    util = (100.0 * sv.memory_mb / sv.memory_limit_mb) if sv.memory_limit_mb > 0 else 0.0
    metrics = MetricsSnapshot(
        service=sv.name, cpu_percent=sv.cpu_percent, memory_mb=sv.memory_mb,
        memory_limit_mb=sv.memory_limit_mb, memory_utilization_percent=round(util, 1),
        request_latency_p50_ms=0.0, request_latency_p99_ms=sv.request_latency_p99_ms,
        error_rate_percent=sv.error_rate_percent,
        active_connections=sv.active_connections, requests_per_second=sv.requests_per_second,
    )
    return IncidentObservation(
        message=f"{sv.name}: cpu={sv.cpu_percent:.0f}% mem={sv.memory_mb:.0f}MB ({util:.0f}% util) errors={sv.error_rate_percent:.1f}%",
        metrics=metrics,
    )


def _restart_service(b: WebsiteBackend, a: IncidentAction, s) -> IncidentObservation:
    if not a.target_service:
        return IncidentObservation(message="Error: this action requires a target_service.", error="Missing target_service")
    raw_lim = a.parameters.get("memory_limit")
    new_mb: Optional[int] = None
    if raw_lim is not None:
        new_mb = _parse_mem(str(raw_lim))
        if new_mb is not None:
            b._mem_limits_mb[a.target_service] = new_mb
    b._restart_history.append(a.target_service)
    if b._stub_mode:
        return IncidentObservation(message=f"[stub] would restart {a.target_service}" + (f" with mem={new_mb}Mi" if new_mb else ""), error="website_backend_stub")
    payload: Dict[str, Any] = {"service": a.target_service}
    if new_mb is not None:
        payload["memory_limit_mb"] = new_mb
    r = _http("POST", b.site_url + "/ops/restart", json_body=payload, timeout=15.0)
    if not r.ok:
        return IncidentObservation(message=f"Restart failed: {r.error}", error=r.error or "restart failed")
    suffix = f" with memory_limit={new_mb}Mi" if new_mb else ""
    return IncidentObservation(message=f"Restarted {a.target_service}{suffix}.")


def _scale_service(b: WebsiteBackend, a: IncidentAction, s) -> IncidentObservation:
    if not a.target_service:
        return IncidentObservation(message="Error: this action requires a target_service.", error="Missing target_service")
    replicas = max(1, min(int(a.parameters.get("replicas", 2)), 10))
    if b._stub_mode:
        return IncidentObservation(message=f"[stub] would scale {a.target_service} to {replicas}", error="website_backend_stub")
    r = _http("POST", b.site_url + "/ops/scale", json_body={"service": a.target_service, "replicas": replicas}, timeout=10.0)
    if not r.ok:
        return IncidentObservation(message=f"Scale failed: {r.error}", error=r.error or "scale failed")
    return IncidentObservation(message=f"Scaled {a.target_service} to {replicas} replicas.")


def _rollback_deployment(b: WebsiteBackend, a: IncidentAction, s) -> IncidentObservation:
    target_version = str(a.parameters.get("to_version") or "v1.0")
    if b._image_tag and target_version == b._image_tag:
        return IncidentObservation(message=f"Already on {target_version}; rollback would be a no-op.", error="rollback_to_self")
    b._image_tag = target_version
    if b._stub_mode:
        return IncidentObservation(message=f"[stub] would roll back to {target_version}")
    payload: Dict[str, Any] = {"to_version": target_version}
    if a.target_service:
        payload["service"] = a.target_service
    r = _http("POST", b.site_url + "/ops/rollback", json_body=payload, timeout=15.0)
    if not r.ok:
        return IncidentObservation(message=f"Rollback failed: {r.error}", error=r.error or "rollback failed")
    return IncidentObservation(message=f"Rolled deployment to {target_version}.")


_KNOWN_CONFIG_KEYS = {
    "db.pool.max_size", "db.pool.min_size",
    "memory.limit", "cpu.limit", "cluster.resource.quota.memory_mb",
}


def _update_config(b: WebsiteBackend, a: IncidentAction, s) -> IncidentObservation:
    if not a.target_service:
        return IncidentObservation(message="Error: this action requires a target_service.", error="Missing target_service")
    key = str(a.parameters.get("key") or "")
    val = a.parameters.get("value")
    if key not in _KNOWN_CONFIG_KEYS:
        return IncidentObservation(message=f"Unknown config key: {key}", error=f"Unknown config key: {key}")
    if key == "db.pool.max_size":
        try: b._pool_sizes[a.target_service] = int(val)
        except Exception: return IncidentObservation(message=f"Invalid value for {key}: {val!r}", error="invalid value")
    elif key == "memory.limit":
        mb = _parse_mem(str(val)) if val is not None else None
        if mb is None: return IncidentObservation(message=f"Invalid value for {key}: {val!r}", error="invalid value")
        b._mem_limits_mb[a.target_service] = mb
    if b._stub_mode:
        return IncidentObservation(message=f"[stub] would set {key}={val} on {a.target_service}")
    r = _http("POST", b.site_url + "/ops/config", json_body={"service": a.target_service, "key": key, "value": val}, timeout=10.0)
    if not r.ok:
        return IncidentObservation(message=f"Config update failed: {r.error}", error=r.error or "config failed")
    healed = False
    try:
        healed = bool(s.on_config_update(a.target_service, key, val))
    except Exception:
        healed = False
    msg = f"Set {key}={val} on {a.target_service}."
    if healed:
        msg += " Cluster appears to be recovering."
    return IncidentObservation(message=msg)


def _run_diagnostic(b: WebsiteBackend, a: IncidentAction, s) -> IncidentObservation:
    cmd = str(a.parameters.get("command") or "check_connectivity")
    if b._stub_mode:
        return IncidentObservation(message=f"[stub] diagnostic '{cmd}'", diagnostic_result="stub")
    if cmd == "check_connectivity":
        r = _http("GET", b.site_url + "/ops/health", timeout=4.0)
        if not r.ok:
            return IncidentObservation(message=f"Health probe failed: {r.error}", diagnostic_result=f"FAIL: {r.error}")
        body = r.body if isinstance(r.body, dict) else {}
        status = body.get("status") or "unknown"
        return IncidentObservation(message=f"Site /ops/health = {status}", diagnostic_result=str(status))
    return IncidentObservation(message=f"Diagnostic '{cmd}' not implemented for WebsiteBackend.", diagnostic_result=f"unsupported: {cmd}")


def _resolve_incident(b: WebsiteBackend, a: IncidentAction, s) -> IncidentObservation:
    root = str(a.parameters.get("root_cause") or "").strip()
    fix = str(a.parameters.get("resolution") or "").strip()
    if not root or not fix:
        return IncidentObservation(message="Error: resolve_incident needs root_cause and resolution.", error="Missing root_cause/resolution")
    resolved = b.check_resolved(s)
    return IncidentObservation(
        message=(
            f"Declared resolved.\n  root_cause: {root}\n  resolution: {fix}\n"
            f"  health-check verdict: {'PASS' if resolved else 'FAIL'}"
        ),
        done=True,
    )


_HANDLERS = {
    "list_services": _list_services,
    "describe_service": _describe_service,
    "read_logs": _read_logs,
    "check_metrics": _check_metrics,
    "restart_service": _restart_service,
    "scale_service": _scale_service,
    "rollback_deployment": _rollback_deployment,
    "update_config": _update_config,
    "run_diagnostic": _run_diagnostic,
    "resolve_incident": _resolve_incident,
}


def _parse_mem(s: str) -> Optional[int]:
    if not s:
        return None
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        pass
    lower = s.lower()
    for suffix, factor in (("gi", 1024), ("g", 1000), ("mi", 1), ("m", 1)):
        if lower.endswith(suffix):
            try:
                return int(float(lower[:-len(suffix)]) * factor)
            except ValueError:
                return None
    return None


__all__ = ["WebsiteBackend", "DEFAULT_SERVICE_NAMES"]
