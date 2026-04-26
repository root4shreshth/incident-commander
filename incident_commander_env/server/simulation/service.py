"""Simulated microservice with health state, metrics, logs, and deployment history."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ServiceHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRASHED = "crashed"
    RESTARTING = "restarting"


@dataclass
class ServiceConfig:
    name: str
    version: str
    replicas: int = 2
    memory_limit: str = "512Mi"
    cpu_limit: str = "500m"
    db_pool_size: Optional[int] = None
    port: int = 8080

    def memory_limit_mb(self) -> float:
        val = self.memory_limit.replace("Mi", "").replace("Gi", "")
        multiplier = 1024.0 if "Gi" in self.memory_limit else 1.0
        return float(val) * multiplier


@dataclass
class Deployment:
    version: str
    timestamp: str
    status: str = "active"  # active, rolled_back, failed

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "timestamp": self.timestamp, "status": self.status}


@dataclass
class ServiceMetrics:
    cpu_percent: float = 15.0
    memory_mb: float = 128.0
    memory_limit_mb: float = 512.0
    request_latency_p50_ms: float = 12.0
    request_latency_p99_ms: float = 45.0
    error_rate_percent: float = 0.1
    active_connections: int = 25
    requests_per_second: float = 150.0

    def to_dict(self) -> Dict[str, str]:
        return {
            "cpu_percent": f"{self.cpu_percent:.1f}%",
            "memory_mb": f"{self.memory_mb:.0f}MB / {self.memory_limit_mb:.0f}MB",
            "memory_utilization": f"{(self.memory_mb / self.memory_limit_mb * 100):.1f}%",
            "latency_p50": f"{self.request_latency_p50_ms:.0f}ms",
            "latency_p99": f"{self.request_latency_p99_ms:.0f}ms",
            "error_rate": f"{self.error_rate_percent:.2f}%",
            "active_connections": str(self.active_connections),
            "rps": f"{self.requests_per_second:.0f}",
        }


class Service:
    """A simulated microservice in the cluster."""

    # Anomalies that a plain restart genuinely fixes in the real world.
    # The rest (memory_leak, db_pool_exhaustion, cascade_degradation) require
    # their actual root-cause fix - restarting only masks them. This set drives
    # the anti-cheat behaviour in restart().
    #
    # `resource_starved` is included because in the bad-deploy cascade scenario,
    # once the upstream leak is rolled back and quota is freed, restarting the
    # starved dependents does bring them back. The "ordering matters" lesson
    # is enforced by the scenario's rubric (correct_order check), not by
    # restart() refusing to heal.
    # disk_full + cert_expired heal on restart in the sim model:
    #   - disk_full: the restart cycle clears tmp/ and rotates logs
    #   - cert_expired: the restart triggers a cert renewal hook
    _RESTART_CURABLE = frozenset({
        "oom", "connection_leak", "resource_starved",
        "disk_full", "cert_expired",
    })

    # Anomalies that a rollback genuinely fixes (because they were introduced
    # by a bad deploy). Anything else survives a rollback.
    # lock_contention is rollback-curable because it represents a slow query
    # added in a recent deploy - reverting that deploy reverts the query.
    _ROLLBACK_CURABLE = frozenset({"memory_leak", "oom", "lock_contention"})

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.health = ServiceHealth.HEALTHY
        self.metrics = ServiceMetrics(memory_limit_mb=config.memory_limit_mb())
        self.log_buffer: List[str] = []
        self.deployment_history: List[Deployment] = [
            Deployment(version=config.version, timestamp="2026-03-28T08:00:00Z", status="active")
        ]
        self._anomalies: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return self.config.version

    def set_anomaly(self, anomaly_type: str, severity: float = 1.0) -> None:
        self._anomalies[anomaly_type] = severity

    def clear_anomaly(self, anomaly_type: str) -> None:
        self._anomalies.pop(anomaly_type, None)

    def clear_all_anomalies(self) -> None:
        self._anomalies.clear()

    def has_anomaly(self, anomaly_type: str) -> bool:
        return anomaly_type in self._anomalies

    def restart(self, new_memory_limit: Optional[str] = None) -> str:
        # Memory-limit bump is the operational fix for OOM; record it before clearing anomalies
        # so the OOM-curable check can take new_memory_limit > old_memory_limit into account.
        old_mem_limit_mb = self.metrics.memory_limit_mb
        new_mem_limit_mb = old_mem_limit_mb
        if new_memory_limit:
            self.config.memory_limit = new_memory_limit
            self.metrics.memory_limit_mb = self.config.memory_limit_mb()
            new_mem_limit_mb = self.metrics.memory_limit_mb

        # Restart only cures the anomalies that a real bounce actually fixes.
        # Memory leaks, pool exhaustion, resource starvation, and cascade
        # degradation all return after the restart unless their root cause is
        # addressed - this prevents the "just restart everything" reward hack.
        # Special case: OOM is only cured if the memory limit actually went up.
        cured: List[str] = []
        for anomaly in list(self._anomalies):
            if anomaly == "oom":
                if new_mem_limit_mb > old_mem_limit_mb:
                    cured.append(anomaly)
            elif anomaly in self._RESTART_CURABLE:
                cured.append(anomaly)
        for a in cured:
            self.clear_anomaly(a)

        # If anomalies remain after restart, the service is at most degraded, not healthy.
        if self._anomalies:
            self.health = ServiceHealth.DEGRADED
        else:
            self.health = ServiceHealth.HEALTHY
            self.metrics.cpu_percent = 15.0
            self.metrics.memory_mb = min(128.0, self.metrics.memory_limit_mb * 0.25)
            self.metrics.error_rate_percent = 0.1
            self.metrics.request_latency_p50_ms = 12.0
            self.metrics.request_latency_p99_ms = 45.0

        mem_info = f" with memory_limit={new_memory_limit}" if new_memory_limit else ""
        if self._anomalies:
            remaining = ", ".join(sorted(self._anomalies))
            return f"Service {self.name} restarted{mem_info}. Anomalies remain: {remaining}. Health: DEGRADED."
        return f"Service {self.name} restarted successfully{mem_info}. Health: HEALTHY."

    def rollback(self, to_version: str) -> str:
        valid_versions = [d.version for d in self.deployment_history]
        if to_version not in valid_versions:
            return f"Error: version {to_version} not found in deployment history. Available: {valid_versions}"

        # Anti-cheat: rollback to the currently-active version is a no-op.
        # The previous behaviour silently appended a new "active" deployment
        # row and cleared anomalies, which was an exploit path.
        if to_version == self.config.version:
            return f"Error: already running {to_version}; rollback to current version is a no-op."

        for dep in self.deployment_history:
            if dep.version == self.config.version:
                dep.status = "rolled_back"

        self.config.version = to_version
        self.deployment_history.append(
            Deployment(version=to_version, timestamp="2026-03-29T10:00:00Z", status="active")
        )

        # Rollback only cures anomalies that a deploy actually introduced.
        cured = [a for a in list(self._anomalies) if a in self._ROLLBACK_CURABLE]
        for a in cured:
            self.clear_anomaly(a)

        if self._anomalies:
            self.health = ServiceHealth.DEGRADED
            remaining = ", ".join(sorted(self._anomalies))
            return f"Service {self.name} rolled back to {to_version}. Anomalies remain: {remaining}. Health: DEGRADED."

        self.health = ServiceHealth.HEALTHY
        self.metrics.cpu_percent = 15.0
        self.metrics.memory_mb = min(128.0, self.metrics.memory_limit_mb * 0.25)
        self.metrics.error_rate_percent = 0.1
        return f"Service {self.name} rolled back to {to_version}. Health: HEALTHY."

    def scale(self, replicas: int) -> str:
        old = self.config.replicas
        self.config.replicas = replicas
        return f"Service {self.name} scaled from {old} to {replicas} replicas."

    def add_logs(self, logs: List[str]) -> None:
        self.log_buffer.extend(logs)
        if len(self.log_buffer) > 500:
            self.log_buffer = self.log_buffer[-500:]

    def get_logs(self, lines: int = 50, severity: Optional[str] = None) -> List[str]:
        logs = self.log_buffer
        if severity:
            severity_upper = severity.upper()
            logs = [l for l in logs if f"[{severity_upper}]" in l]
        return logs[-lines:]

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "health": self.health.value,
            "version": self.version,
            "replicas": self.config.replicas,
            "cpu_percent": round(self.metrics.cpu_percent, 1),
            "memory_mb": round(self.metrics.memory_mb, 0),
            "error_rate_percent": round(self.metrics.error_rate_percent, 2),
        }

    def detail_dict(self, dependencies: List[str], dependents: List[str]) -> Dict[str, Any]:
        return {
            "name": self.name,
            "health": self.health.value,
            "version": self.version,
            "replicas": self.config.replicas,
            "memory_limit": self.config.memory_limit,
            "cpu_limit": self.config.cpu_limit,
            "port": self.config.port,
            "db_pool_size": self.config.db_pool_size,
            "deployment_history": [d.to_dict() for d in self.deployment_history],
            "dependencies": dependencies,
            "dependents": dependents,
        }
