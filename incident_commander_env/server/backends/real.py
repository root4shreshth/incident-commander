"""RealBackend — wraps a Docker Compose stack for the sim-to-real demo.

The user vibecodes a small site under `targets/site/` exposing the contract
documented in the README:

  * `docker-compose.yml` with services named `frontend`, `api`, `postgres`
  * env-var hooks: `IMAGE_TAG`, `API_MEM_LIMIT`, `POOL_SIZE`
  * `chaos.py --scenario={oom,conn-leak,bad-deploy} | --stop`
  * `/health` on each service

RealBackend implements the same `Backend` Protocol as `SimulatedBackend` so
the trained policy and reward components run unchanged across substrates.

Robustness — if `compose_root` does not exist or `docker` is missing, this
backend degrades to a clearly-labelled stub mode that returns helpful error
observations without crashing the env. That keeps tests green on a machine
with no Docker installed and lets the user discover misconfiguration via
the `/backend` endpoint.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from incident_commander_env.models import (
    IncidentAction,
    IncidentObservation,
    MetricsSnapshot,
    ServiceDetail,
    ServiceSummary,
)
from incident_commander_env.server.backends import docker_ops as _ops
from incident_commander_env.server.backends.protocol import (
    BackendSnapshot,
    QuotaSnapshot,
    ServiceSnapshot,
)

if TYPE_CHECKING:
    from incident_commander_env.server.scenarios.base_scenario import BaseScenario


# Default service names the env expects to find in the user's vibecoded compose.
DEFAULT_REAL_SERVICES = ("frontend", "api", "postgres")

# Mapping from logical service -> http://<host>:<port>/health
DEFAULT_HEALTH_URLS = {
    "frontend": "http://localhost:8080/health",
    "api": "http://localhost:8000/health",
    "postgres": "http://localhost:5432/health",  # often a noop, but kept symmetric
}

# Default image tags the rollback action toggles between for each scenario
# family. The user's compose substitutes `${IMAGE_TAG:-v1.0}`.
DEFAULT_STABLE_TAG = "v1.0"
DEFAULT_BAD_TAG = "v1.1"


class RealBackend:
    """Docker-Compose-backed real environment for the sim-to-real demo."""

    name = "real"

    def __init__(
        self,
        compose_root: str = "./targets/site",
        service_names: Optional[List[str]] = None,
        health_urls: Optional[Dict[str, str]] = None,
    ) -> None:
        self.compose_root = Path(compose_root).resolve()
        self.service_names = list(service_names or DEFAULT_REAL_SERVICES)
        self.health_urls = dict(health_urls or DEFAULT_HEALTH_URLS)

        # Mutable per-episode state we track between actions
        self._reset_done = False
        self._stub_mode = False  # True when compose_root / docker isn't available
        self._stub_reason: Optional[str] = None
        self._image_tag = DEFAULT_STABLE_TAG
        self._mem_limits_mb: Dict[str, int] = {}
        self._pool_sizes: Dict[str, int] = {}
        self._restart_history: List[str] = []  # services we've restarted this episode

    # ---- Lifecycle ----------------------------------------------------------

    def _check_available(self) -> Optional[str]:
        """Return a reason string if we should run in stub mode, else None."""
        if not self.compose_root.exists():
            return f"compose_root does not exist: {self.compose_root}"
        compose_file = self.compose_root / "docker-compose.yml"
        if not compose_file.exists():
            compose_file = self.compose_root / "compose.yml"
        if not compose_file.exists():
            return f"no docker-compose.yml under {self.compose_root}"
        return None

    def reset(
        self, scenario: "BaseScenario", seed: Optional[int] = None
    ) -> None:
        self._reset_done = True
        self._image_tag = DEFAULT_STABLE_TAG
        self._mem_limits_mb = {}
        self._pool_sizes = {}
        self._restart_history = []

        reason = self._check_available()
        if reason:
            self._stub_mode = True
            self._stub_reason = reason
            return

        self._stub_mode = False
        self._stub_reason = None

        # Bring up the stack with default env (stable image tag).
        env_vars = self._compose_env()
        up = _ops.compose_up(self.compose_root, env_vars=env_vars, timeout=120)
        if not up.ok:
            # Still leave _reset_done=True so teardown() will attempt cleanup,
            # but mark the failure for action-level error messages.
            self._stub_mode = True
            self._stub_reason = f"compose up failed: {up.error}"
            return

        # Inject the scenario's chaos profile so the env reproduces the fault.
        chaos = _ops.chaos_inject(self.compose_root, scenario.task_id, timeout=30)
        if not chaos.ok:
            # Tolerate — the user might run chaos differently. Leave a breadcrumb.
            self._stub_reason = f"chaos.py inject failed: {chaos.error}"

    def teardown(self) -> None:
        if not self._reset_done:
            return
        if self._stub_mode and self._check_available() is not None:
            return
        try:
            _ops.chaos_stop(self.compose_root, timeout=15)
        except Exception:
            pass
        _ops.compose_down(self.compose_root, timeout=60)

    def tick(self) -> None:
        # Real Docker advances on its own clock.
        return

    # ---- Snapshot -----------------------------------------------------------

    def snapshot(self) -> BackendSnapshot:
        """Read live container metrics and synthesize a typed BackendSnapshot."""
        if self._stub_mode:
            return self._stub_snapshot()
        # Pull container metrics + compose state
        stats = _ops.docker_stats_json(timeout=10)
        ps = _ops.compose_ps_json(self.compose_root, timeout=10)
        if not stats.ok and not ps.ok:
            return self._stub_snapshot()
        per_svc_metrics = (
            _ops.stats_to_service_metrics(_ops.parse_docker_stats(stats.stdout))
            if stats.ok
            else {}
        )
        ps_rows = _ops.parse_compose_ps(ps.stdout) if ps.ok else []
        ps_state: Dict[str, Dict[str, Any]] = {}
        for row in ps_rows:
            svc = row.get("Service") or row.get("service") or ""
            if not svc:
                continue
            ps_state.setdefault(svc, []).append(row)

        services: Dict[str, ServiceSnapshot] = {}
        total_cpu = 0.0
        total_mem = 0.0
        total_mem_limit = 0.0
        for svc in self.service_names:
            metrics = per_svc_metrics.get(svc, {})
            rows = ps_state.get(svc, [])
            replicas = max(len(rows), 1)
            health = _classify_health(rows, metrics)
            cpu = float(metrics.get("cpu_percent", 0.0))
            mem_mb = float(metrics.get("memory_mb", 0.0))
            mem_limit_mb = float(
                self._mem_limits_mb.get(svc) or metrics.get("memory_limit_mb", 512.0)
            )
            total_cpu += cpu
            total_mem += mem_mb
            total_mem_limit += mem_limit_mb
            services[svc] = ServiceSnapshot(
                name=svc,
                health=health,
                version=self._image_tag,
                replicas=replicas,
                cpu_percent=round(cpu, 1),
                memory_mb=round(mem_mb, 1),
                memory_limit_mb=round(mem_limit_mb, 1),
                error_rate_percent=0.0,
                request_latency_p99_ms=0.0,
                active_connections=0,
                requests_per_second=0.0,
            )
        quota = QuotaSnapshot(
            cpu_used=round(total_cpu, 2),
            cpu_total=400.0,
            cpu_utilization_percent=round(total_cpu / 4.0, 1),
            memory_used_mb=round(total_mem, 1),
            memory_total_mb=round(total_mem_limit, 1),
            memory_utilization_percent=(
                round(100.0 * total_mem / total_mem_limit, 1)
                if total_mem_limit > 0
                else 0.0
            ),
        )
        return BackendSnapshot(services=services, quota=quota)

    def _stub_snapshot(self) -> BackendSnapshot:
        services = {
            n: ServiceSnapshot(
                name=n,
                health="healthy",
                version=self._image_tag,
                replicas=1,
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_limit_mb=float(self._mem_limits_mb.get(n, 512)),
                error_rate_percent=0.0,
            )
            for n in self.service_names
        }
        return BackendSnapshot(services=services, quota=QuotaSnapshot())

    # ---- Resolution check ---------------------------------------------------

    def check_resolved(self, scenario: "BaseScenario") -> bool:
        """Real-world resolved = relevant services answer 200 on /health."""
        if self._stub_mode:
            return False
        targets = self._relevant_services(scenario)
        if not targets:
            targets = list(self.service_names)
        # Need 30s of green: poll every 3s for up to 30s, abort early on a fail.
        deadline = time.monotonic() + 30.0
        green_streak = 0
        while time.monotonic() < deadline:
            all_ok = True
            for svc in targets:
                url = self.health_urls.get(svc)
                if not url:
                    continue
                ok, _ = _ops.http_health(url, timeout=2.0)
                if not ok:
                    all_ok = False
                    break
            if all_ok:
                green_streak += 1
                if green_streak >= 3:  # ~9s of clean polls
                    return True
            else:
                green_streak = 0
            time.sleep(3.0)
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
        handler = _REAL_HANDLERS.get(action.action_type)
        if handler is None:
            return IncidentObservation(
                message=f"Unknown action: {action.action_type}",
                error=f"Invalid action_type: {action.action_type}",
            )
        try:
            return handler(self, action, scenario)
        except Exception as exc:  # pragma: no cover — defensive
            return IncidentObservation(
                message=f"RealBackend error executing {action.action_type}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )

    # ---- Helpers ------------------------------------------------------------

    def _compose_env(self) -> Dict[str, str]:
        """Build the env-var dict passed to docker compose."""
        env: Dict[str, str] = {"IMAGE_TAG": self._image_tag}
        for svc, mb in self._mem_limits_mb.items():
            key = f"{svc.upper().replace('-', '_')}_MEM_LIMIT"
            env[key] = f"{int(mb)}m"
            if svc == "api":
                env["API_MEM_LIMIT"] = f"{int(mb)}m"
        for svc, sz in self._pool_sizes.items():
            env["POOL_SIZE"] = str(int(sz))
            env[f"{svc.upper().replace('-', '_')}_POOL_SIZE"] = str(int(sz))
        return env

    def _relevant_services(self, scenario: "BaseScenario") -> List[str]:
        rel = getattr(scenario, "relevant_services", None) or []
        # Filter to services the real stack actually has.
        return [s for s in rel if s in self.service_names]

    def _stub_observation(self, action_type: str) -> IncidentObservation:
        why = self._stub_reason or "Docker / compose root unavailable"
        return IncidentObservation(
            message=(
                f"RealBackend in stub mode ({why}). "
                f"Action '{action_type}' would have run against Docker Compose."
            ),
            error="real_backend_stub",
        )


# ---------------------------------------------------------------------------
# Action handlers — translate IncidentAction → docker shell-outs
# ---------------------------------------------------------------------------

def _classify_health(
    ps_rows: List[Dict[str, Any]], metrics: Dict[str, float]
) -> str:
    """Heuristic mapping from `docker compose ps` State → our health enum."""
    if not ps_rows:
        return "unhealthy"
    states = {str(r.get("State") or r.get("state") or "").lower() for r in ps_rows}
    healthchecks = {str(r.get("Health") or r.get("health") or "").lower() for r in ps_rows}
    if "exited" in states or "dead" in states:
        return "crashed"
    if "restarting" in states:
        return "restarting"
    if "starting" in healthchecks:
        return "restarting"
    if any(s and s != "running" for s in states):
        return "degraded"
    if "unhealthy" in healthchecks:
        return "unhealthy"
    cpu = float(metrics.get("cpu_percent", 0.0))
    mem_mb = float(metrics.get("memory_mb", 0.0))
    mem_limit = float(metrics.get("memory_limit_mb", 0.0)) or 1.0
    if cpu > 90 or (mem_limit > 0 and mem_mb / mem_limit > 0.95):
        return "degraded"
    return "healthy"


def _list_services(
    backend: "RealBackend", action: IncidentAction, scenario: "BaseScenario"
) -> IncidentObservation:
    if backend._stub_mode:
        return backend._stub_observation("list_services")
    snap = backend.snapshot()
    summaries: List[ServiceSummary] = []
    lines = [f"Cluster overview: {len(snap.healthy_service_names())}/{len(snap.services)} services healthy\n"]
    for name, svc in snap.services.items():
        summaries.append(
            ServiceSummary(
                name=svc.name,
                health=svc.health,
                version=svc.version,
                replicas=svc.replicas,
                cpu_percent=svc.cpu_percent,
                memory_mb=svc.memory_mb,
                error_rate_percent=svc.error_rate_percent,
            )
        )
        icon = "OK" if svc.health == "healthy" else svc.health.upper()
        lines.append(
            f"  [{icon:>10}] {svc.name:25s} v{svc.version:8s} "
            f"replicas={svc.replicas}  cpu={svc.cpu_percent:.0f}%  "
            f"mem={svc.memory_mb:.0f}MB"
        )
    return IncidentObservation(message="\n".join(lines), services_summary=summaries)


def _describe_service(
    backend: "RealBackend", action: IncidentAction, scenario: "BaseScenario"
) -> IncidentObservation:
    if not action.target_service:
        return IncidentObservation(
            message="Error: this action requires a target_service parameter.",
            error="Missing target_service",
        )
    if backend._stub_mode:
        return backend._stub_observation("describe_service")
    name = action.target_service
    snap = backend.snapshot()
    svc = snap.get_service(name)
    if not svc:
        return IncidentObservation(
            message=f"Error: service '{name}' not found.",
            error=f"Service not found: {name}",
        )
    detail = ServiceDetail(
        name=svc.name,
        health=svc.health,
        version=svc.version,
        replicas=svc.replicas,
        memory_limit=f"{int(svc.memory_limit_mb)}Mi",
        cpu_limit="1000m",
        port=8000,
        db_pool_size=backend._pool_sizes.get(name),
    )
    return IncidentObservation(
        message=(
            f"{svc.name}: health={svc.health} version={svc.version} "
            f"replicas={svc.replicas} cpu={svc.cpu_percent:.0f}% "
            f"mem={svc.memory_mb:.0f}/{svc.memory_limit_mb:.0f}MB"
        ),
        service_detail=detail,
    )


def _read_logs(
    backend: "RealBackend", action: IncidentAction, scenario: "BaseScenario"
) -> IncidentObservation:
    if not action.target_service:
        return IncidentObservation(
            message="Error: this action requires a target_service parameter.",
            error="Missing target_service",
        )
    if backend._stub_mode:
        return backend._stub_observation("read_logs")
    name = action.target_service
    tail = int(action.parameters.get("lines", 50))
    res = _ops.compose_logs(backend.compose_root, name, tail=tail)
    if not res.ok:
        return IncidentObservation(
            message=f"Failed to read logs for {name}: {res.error}",
            error=res.error or "log read failed",
        )
    raw = res.stdout.splitlines()[-tail:]
    return IncidentObservation(
        message=f"Last {len(raw)} lines from {name}:\n" + "\n".join(raw),
        logs=raw,
    )


def _check_metrics(
    backend: "RealBackend", action: IncidentAction, scenario: "BaseScenario"
) -> IncidentObservation:
    if not action.target_service:
        return IncidentObservation(
            message="Error: this action requires a target_service parameter.",
            error="Missing target_service",
        )
    if backend._stub_mode:
        return backend._stub_observation("check_metrics")
    name = action.target_service
    snap = backend.snapshot()
    svc = snap.get_service(name)
    if not svc:
        return IncidentObservation(
            message=f"Error: service '{name}' not found.",
            error=f"Service not found: {name}",
        )
    util = (
        100.0 * svc.memory_mb / svc.memory_limit_mb if svc.memory_limit_mb > 0 else 0.0
    )
    metrics = MetricsSnapshot(
        service=svc.name,
        cpu_percent=svc.cpu_percent,
        memory_mb=svc.memory_mb,
        memory_limit_mb=svc.memory_limit_mb,
        memory_utilization_percent=round(util, 1),
        request_latency_p50_ms=0.0,
        request_latency_p99_ms=svc.request_latency_p99_ms,
        error_rate_percent=svc.error_rate_percent,
        active_connections=svc.active_connections,
        requests_per_second=svc.requests_per_second,
    )
    return IncidentObservation(
        message=(
            f"{svc.name}: cpu={svc.cpu_percent:.0f}% mem={svc.memory_mb:.0f}MB "
            f"({util:.0f}% util) errors={svc.error_rate_percent:.1f}%"
        ),
        metrics=metrics,
    )


def _restart_service(
    backend: "RealBackend", action: IncidentAction, scenario: "BaseScenario"
) -> IncidentObservation:
    if not action.target_service:
        return IncidentObservation(
            message="Error: this action requires a target_service parameter.",
            error="Missing target_service",
        )
    name = action.target_service
    raw_limit = action.parameters.get("memory_limit")
    new_mem_mb: Optional[int] = None
    if raw_limit:
        new_mem_mb = _parse_mem_limit(str(raw_limit))
        if new_mem_mb is not None:
            backend._mem_limits_mb[name] = new_mem_mb
    backend._restart_history.append(name)
    if backend._stub_mode:
        return IncidentObservation(
            message=(
                f"[stub] Restart of {name}"
                + (f" with memory_limit={new_mem_mb}Mi" if new_mem_mb else "")
                + " queued. Real Docker not available."
            ),
            error="real_backend_stub",
        )
    env_vars = backend._compose_env() if new_mem_mb is not None else None
    res = _ops.compose_restart(backend.compose_root, name, env_vars=env_vars)
    if not res.ok:
        return IncidentObservation(
            message=f"Failed to restart {name}: {res.error}",
            error=res.error or "restart failed",
        )
    suffix = f" (memory_limit={new_mem_mb}Mi)" if new_mem_mb else ""
    return IncidentObservation(
        message=f"Restarted {name}{suffix}. Container is coming back up...",
    )


def _scale_service(
    backend: "RealBackend", action: IncidentAction, scenario: "BaseScenario"
) -> IncidentObservation:
    if not action.target_service:
        return IncidentObservation(
            message="Error: this action requires a target_service parameter.",
            error="Missing target_service",
        )
    name = action.target_service
    replicas = int(action.parameters.get("replicas", 2))
    replicas = max(1, min(replicas, 10))
    if backend._stub_mode:
        return IncidentObservation(
            message=f"[stub] Would scale {name} to {replicas} replicas.",
            error="real_backend_stub",
        )
    res = _ops.compose_scale(backend.compose_root, name, replicas)
    if not res.ok:
        return IncidentObservation(
            message=f"Failed to scale {name}: {res.error}",
            error=res.error or "scale failed",
        )
    return IncidentObservation(message=f"Scaled {name} to {replicas} replicas.")


def _rollback_deployment(
    backend: "RealBackend", action: IncidentAction, scenario: "BaseScenario"
) -> IncidentObservation:
    target_version = str(action.parameters.get("to_version") or DEFAULT_STABLE_TAG)
    if target_version == backend._image_tag:
        return IncidentObservation(
            message=(
                f"Already running image tag {target_version}; "
                "rollback would be a no-op."
            ),
            error="rollback_to_self",
        )
    backend._image_tag = target_version
    if backend._stub_mode:
        return IncidentObservation(
            message=f"[stub] Would roll cluster back to image {target_version}."
        )
    env = backend._compose_env()
    res = _ops.compose_up(backend.compose_root, env_vars=env, timeout=120)
    if not res.ok:
        return IncidentObservation(
            message=f"Rollback to {target_version} failed: {res.error}",
            error=res.error or "rollback failed",
        )
    return IncidentObservation(
        message=f"Rolled deployment back to image tag {target_version}."
    )


# Allowlist mirroring the sim handler — defined in actions.handlers but
# duplicated here so we don't reach across the package.
_KNOWN_CONFIG_KEYS = {
    "db.pool.max_size",
    "db.pool.min_size",
    "memory.limit",
    "cpu.limit",
    "cluster.resource.quota.memory_mb",
}


def _update_config(
    backend: "RealBackend", action: IncidentAction, scenario: "BaseScenario"
) -> IncidentObservation:
    if not action.target_service:
        return IncidentObservation(
            message="Error: this action requires a target_service parameter.",
            error="Missing target_service",
        )
    name = action.target_service
    key = str(action.parameters.get("key") or "")
    value = action.parameters.get("value")
    if key not in _KNOWN_CONFIG_KEYS:
        return IncidentObservation(
            message=f"Unknown config key: {key}",
            error=f"Unknown config key: {key}",
        )
    # Translate select keys into compose env levers; let the scenario judge.
    if key == "db.pool.max_size":
        try:
            backend._pool_sizes[name] = int(value)
        except Exception:
            return IncidentObservation(
                message=f"Invalid value for {key}: {value!r}",
                error="invalid value",
            )
    elif key == "memory.limit":
        mb = _parse_mem_limit(str(value)) if value is not None else None
        if mb is None:
            return IncidentObservation(
                message=f"Invalid value for {key}: {value!r}",
                error="invalid value",
            )
        backend._mem_limits_mb[name] = mb
    # Apply to compose
    if not backend._stub_mode:
        res = _ops.compose_up(backend.compose_root, env_vars=backend._compose_env())
        if not res.ok:
            return IncidentObservation(
                message=f"Failed to apply {key}: {res.error}",
                error=res.error or "apply failed",
            )
    healed = False
    try:
        healed = bool(scenario.on_config_update(name, key, value))
    except Exception:
        healed = False
    msg = f"Set {key}={value} on {name}."
    if healed:
        msg += " Cluster appears to be recovering."
    return IncidentObservation(message=msg)


def _run_diagnostic(
    backend: "RealBackend", action: IncidentAction, scenario: "BaseScenario"
) -> IncidentObservation:
    cmd = str(action.parameters.get("command") or "check_connectivity")
    if backend._stub_mode:
        return IncidentObservation(
            message=f"[stub] diagnostic '{cmd}' — Docker unavailable.",
            diagnostic_result="stub",
        )
    if cmd == "check_connectivity":
        results: List[str] = []
        for svc, url in backend.health_urls.items():
            if svc not in backend.service_names:
                continue
            ok, code = _ops.http_health(url, timeout=2.0)
            results.append(f"  {svc}: {'OK' if ok else 'FAIL'} (HTTP {code})")
        return IncidentObservation(
            message="Connectivity check:\n" + "\n".join(results),
            diagnostic_result="\n".join(results),
        )
    return IncidentObservation(
        message=f"Diagnostic '{cmd}' not implemented for RealBackend.",
        diagnostic_result=f"unsupported: {cmd}",
    )


def _resolve_incident(
    backend: "RealBackend", action: IncidentAction, scenario: "BaseScenario"
) -> IncidentObservation:
    root = str(action.parameters.get("root_cause") or "").strip()
    fix = str(action.parameters.get("resolution") or "").strip()
    if not root or not fix:
        return IncidentObservation(
            message="Error: resolve_incident needs both root_cause and resolution parameters.",
            error="Missing root_cause/resolution",
        )
    resolved = backend.check_resolved(scenario)
    return IncidentObservation(
        message=(
            f"Declared resolved.\n  root_cause: {root}\n  resolution: {fix}\n"
            f"  health-check verdict: {'PASS' if resolved else 'FAIL'}"
        ),
        done=True,
    )


_REAL_HANDLERS = {
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


def _parse_mem_limit(s: str) -> Optional[int]:
    """Parse strings like "1024Mi", "1Gi", "512", "256m" into MB."""
    if not s:
        return None
    s = s.strip()
    try:
        # raw integer = MB
        return int(s)
    except ValueError:
        pass
    lower = s.lower()
    for suffix, factor in (("gi", 1024), ("g", 1000), ("mi", 1), ("m", 1)):
        if lower.endswith(suffix):
            try:
                return int(float(lower[: -len(suffix)]) * factor)
            except ValueError:
                return None
    return None


__all__ = [
    "RealBackend",
    "DEFAULT_REAL_SERVICES",
    "DEFAULT_HEALTH_URLS",
    "DEFAULT_STABLE_TAG",
    "DEFAULT_BAD_TAG",
]
