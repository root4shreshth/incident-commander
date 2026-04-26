"""Action handlers - one pure function per action_type, dispatched via dict."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from incident_commander_env.models import (
    IncidentAction,
    IncidentObservation,
    MetricsSnapshot,
    ServiceDetail,
    ServiceSummary,
)
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.scenarios.base_scenario import BaseScenario


def _require_target(action: IncidentAction) -> Optional[IncidentObservation]:
    """Return an error observation if target_service is missing."""
    if not action.target_service:
        return IncidentObservation(
            message="Error: this action requires a target_service parameter.",
            error="Missing target_service",
        )
    return None


def _service_not_found(name: str) -> IncidentObservation:
    return IncidentObservation(
        message=f"Error: service '{name}' not found in the cluster.",
        error=f"Service not found: {name}",
    )


def handle_list_services(
    action: IncidentAction, cluster: Cluster, scenario: BaseScenario
) -> IncidentObservation:
    summaries = cluster.list_services()
    svc_list = [ServiceSummary(**s) for s in summaries]
    healthy = sum(1 for s in summaries if s["health"] == "healthy")
    total = len(summaries)

    lines = [f"Cluster overview: {healthy}/{total} services healthy\n"]
    for s in summaries:
        status_icon = "OK" if s["health"] == "healthy" else s["health"].upper()
        lines.append(
            f"  [{status_icon:>10}] {s['name']:25s} v{s['version']:8s} "
            f"replicas={s['replicas']}  cpu={s['cpu_percent']:.0f}%  "
            f"mem={s['memory_mb']:.0f}MB  err={s['error_rate_percent']:.1f}%"
        )

    return IncidentObservation(
        message="\n".join(lines),
        services_summary=svc_list,
    )


def handle_describe_service(
    action: IncidentAction, cluster: Cluster, scenario: BaseScenario
) -> IncidentObservation:
    err = _require_target(action)
    if err:
        return err

    detail = cluster.describe_service(action.target_service)
    if not detail:
        return _service_not_found(action.target_service)

    svc = cluster.get_service(action.target_service)
    lines = [
        f"Service: {detail['name']}",
        f"  Health:      {detail['health']}",
        f"  Version:     {detail['version']}",
        f"  Replicas:    {detail['replicas']}",
        f"  Memory:      {detail['memory_limit']}",
        f"  CPU:         {detail['cpu_limit']}",
        f"  Port:        {detail['port']}",
    ]
    if detail.get("db_pool_size"):
        lines.append(f"  DB Pool:     {detail['db_pool_size']}")
    lines.append(f"  Dependencies: {', '.join(detail['dependencies']) or 'none'}")
    lines.append(f"  Dependents:   {', '.join(detail['dependents']) or 'none'}")
    lines.append(f"  Deployment History:")
    for dep in detail["deployment_history"]:
        lines.append(f"    - {dep['version']} ({dep['status']}) deployed at {dep['timestamp']}")

    return IncidentObservation(
        message="\n".join(lines),
        service_detail=ServiceDetail(**detail),
    )


def handle_read_logs(
    action: IncidentAction, cluster: Cluster, scenario: BaseScenario
) -> IncidentObservation:
    err = _require_target(action)
    if err:
        return err

    svc = cluster.get_service(action.target_service)
    if not svc:
        return _service_not_found(action.target_service)

    lines = action.parameters.get("lines", 50)
    severity = action.parameters.get("severity")
    logs = svc.get_logs(lines=min(lines, 200), severity=severity)

    if not logs:
        msg = f"No logs found for {action.target_service}"
        if severity:
            msg += f" with severity={severity}"
    else:
        msg = f"Logs for {action.target_service} ({len(logs)} lines):\n" + "\n".join(logs)

    return IncidentObservation(
        message=msg,
        logs=logs,
    )


def handle_check_metrics(
    action: IncidentAction, cluster: Cluster, scenario: BaseScenario
) -> IncidentObservation:
    err = _require_target(action)
    if err:
        return err

    svc = cluster.get_service(action.target_service)
    if not svc:
        return _service_not_found(action.target_service)

    m = svc.metrics
    mem_util = (m.memory_mb / m.memory_limit_mb * 100) if m.memory_limit_mb > 0 else 0

    snapshot = MetricsSnapshot(
        service=action.target_service,
        cpu_percent=round(m.cpu_percent, 1),
        memory_mb=round(m.memory_mb, 1),
        memory_limit_mb=round(m.memory_limit_mb, 1),
        memory_utilization_percent=round(mem_util, 1),
        request_latency_p50_ms=round(m.request_latency_p50_ms, 1),
        request_latency_p99_ms=round(m.request_latency_p99_ms, 1),
        error_rate_percent=round(m.error_rate_percent, 2),
        active_connections=m.active_connections,
        requests_per_second=round(m.requests_per_second, 1),
    )

    lines = [
        f"Metrics for {action.target_service}:",
        f"  CPU:             {snapshot.cpu_percent}%",
        f"  Memory:          {snapshot.memory_mb}MB / {snapshot.memory_limit_mb}MB ({snapshot.memory_utilization_percent}%)",
        f"  Latency (p50):   {snapshot.request_latency_p50_ms}ms",
        f"  Latency (p99):   {snapshot.request_latency_p99_ms}ms",
        f"  Error Rate:      {snapshot.error_rate_percent}%",
        f"  Connections:     {snapshot.active_connections}",
        f"  RPS:             {snapshot.requests_per_second}",
    ]

    return IncidentObservation(
        message="\n".join(lines),
        metrics=snapshot,
    )


def handle_restart_service(
    action: IncidentAction, cluster: Cluster, scenario: BaseScenario
) -> IncidentObservation:
    err = _require_target(action)
    if err:
        return err

    svc = cluster.get_service(action.target_service)
    if not svc:
        return _service_not_found(action.target_service)

    new_mem = action.parameters.get("memory_limit")
    result = svc.restart(new_memory_limit=new_mem)

    return IncidentObservation(message=result)


def handle_scale_service(
    action: IncidentAction, cluster: Cluster, scenario: BaseScenario
) -> IncidentObservation:
    err = _require_target(action)
    if err:
        return err

    svc = cluster.get_service(action.target_service)
    if not svc:
        return _service_not_found(action.target_service)

    replicas = action.parameters.get("replicas")
    if replicas is None or not isinstance(replicas, int) or replicas < 0:
        return IncidentObservation(
            message="Error: 'replicas' parameter must be a positive integer.",
            error="Invalid replicas parameter",
        )

    result = svc.scale(replicas)

    # Free up resources if scaling down
    cluster.resource_quota.update_from_services(cluster.services)

    return IncidentObservation(message=result)


def handle_rollback_deployment(
    action: IncidentAction, cluster: Cluster, scenario: BaseScenario
) -> IncidentObservation:
    err = _require_target(action)
    if err:
        return err

    svc = cluster.get_service(action.target_service)
    if not svc:
        return _service_not_found(action.target_service)

    to_version = action.parameters.get("to_version")
    if not to_version:
        return IncidentObservation(
            message="Error: 'to_version' parameter is required for rollback.",
            error="Missing to_version parameter",
        )

    result = svc.rollback(to_version)

    # Rollback also resets replicas to a sane default
    if svc.config.replicas > 3:
        svc.config.replicas = 2
        result += f" Replicas reset to 2."
        cluster.resource_quota.update_from_services(cluster.services)

    return IncidentObservation(message=result)


def handle_run_diagnostic(
    action: IncidentAction, cluster: Cluster, scenario: BaseScenario
) -> IncidentObservation:
    err = _require_target(action)
    if err:
        return err

    svc = cluster.get_service(action.target_service)
    if not svc:
        return _service_not_found(action.target_service)

    command = action.parameters.get("command", "check_health")
    name = action.target_service

    if command == "check_connectivity":
        deps = cluster.dependency_graph.get_dependencies(name)
        results = []
        for dep_name in deps:
            dep = cluster.get_service(dep_name)
            if dep and dep.health in (ServiceHealth.HEALTHY,):
                results.append(f"  {name} -> {dep_name}: OK (latency: 2ms)")
            elif dep:
                results.append(f"  {name} -> {dep_name}: FAILED (service {dep.health.value})")
            else:
                results.append(f"  {name} -> {dep_name}: UNKNOWN")
        msg = f"Connectivity check for {name}:\n" + "\n".join(results) if results else f"No dependencies for {name}"

    elif command == "check_health":
        msg = (
            f"Health check for {name}:\n"
            f"  Status: {svc.health.value}\n"
            f"  Uptime: {'N/A (crashed)' if svc.health == ServiceHealth.CRASHED else '2h 34m'}\n"
            f"  Last restart: 2026-03-28T08:00:00Z"
        )

    elif command == "check_resources":
        quota = cluster.resource_quota
        msg = (
            f"Resource check for {name}:\n"
            f"  Service: CPU={svc.metrics.cpu_percent:.0f}%, Memory={svc.metrics.memory_mb:.0f}MB/{svc.metrics.memory_limit_mb:.0f}MB\n"
            f"  Cluster quota: {quota.to_dict()['cpu']}\n"
            f"  Cluster quota: {quota.to_dict()['memory']}"
        )

    elif command == "check_dns":
        msg = f"DNS resolution for {name}:\n  {name}.default.svc.cluster.local -> 10.0.{hash(name) % 255}.{hash(name + 'x') % 255} (OK)"

    else:
        msg = f"Unknown diagnostic command: {command}. Available: check_connectivity, check_health, check_resources, check_dns"

    return IncidentObservation(message=msg, diagnostic_result=msg)


# Import ServiceHealth for diagnostic use
from incident_commander_env.server.simulation.service import ServiceHealth


# Strict allowlist of config keys an SRE agent may set. Anything else is rejected,
# closing the "any string containing 'pool' and 'size' heals the DB" exploit path.
_KNOWN_CONFIG_KEYS = frozenset({
    "db.pool.max_size",
    "db.pool.min_size",
    "memory.limit",
    "cpu.limit",
    "cluster.resource.quota.memory_mb",
})


def handle_update_config(
    action: IncidentAction, cluster: Cluster, scenario: BaseScenario
) -> IncidentObservation:
    err = _require_target(action)
    if err:
        return err

    svc = cluster.get_service(action.target_service)
    if not svc:
        return _service_not_found(action.target_service)

    key = action.parameters.get("key", "")
    value = action.parameters.get("value")

    if not key:
        return IncidentObservation(
            message="Error: 'key' parameter is required.",
            error="Missing key parameter",
        )
    if key not in _KNOWN_CONFIG_KEYS:
        return IncidentObservation(
            message=(
                f"Error: unknown config key '{key}'. Known keys: "
                f"{sorted(_KNOWN_CONFIG_KEYS)}"
            ),
            error="Unknown config key",
        )

    # Apply the config change to cluster/service state. The mutation is mechanical;
    # whether it constitutes the *fix* for the active scenario is a question for
    # `scenario.on_config_update`, which is the only place that may clear anomalies.
    msg_tail = ""
    if key == "db.pool.max_size":
        if not isinstance(value, (int, float)) or value <= 0:
            return IncidentObservation(
                message=f"Error: invalid value for {key}: {value}",
                error="Invalid config value",
            )
        old_val = svc.config.db_pool_size
        svc.config.db_pool_size = int(value)
        msg_tail = f" (was {old_val})"
    elif key == "db.pool.min_size":
        if not isinstance(value, (int, float)) or value < 0:
            return IncidentObservation(
                message=f"Error: invalid value for {key}: {value}",
                error="Invalid config value",
            )
        # No dedicated field; recorded but no state mutation.
    elif key == "memory.limit":
        svc.config.memory_limit = str(value)
        svc.metrics.memory_limit_mb = svc.config.memory_limit_mb()
    elif key == "cpu.limit":
        svc.config.cpu_limit = str(value)
    elif key == "cluster.resource.quota.memory_mb":
        if not isinstance(value, (int, float)):
            return IncidentObservation(
                message=f"Error: invalid value for {key}: {value}",
                error="Invalid config value",
            )
        cluster.resource_quota.memory_total_mb = float(value)

    # Delegate the heal-decision to the scenario. Only the scenario knows whether
    # this exact config change is the right fix for the active fault.
    if scenario is not None and scenario.on_config_update(
        cluster, action.target_service, key, value
    ):
        # Scenario confirms this is the correct fix - clear the relevant anomalies
        # it knows about. We clear all anomalies on the service because the scenario
        # has full knowledge of which ones it injected.
        if svc.has_anomaly("db_pool_exhaustion"):
            svc.clear_anomaly("db_pool_exhaustion")
            svc.metrics.active_connections = max(1, int(value) // 2) if isinstance(value, (int, float)) else 5
            svc.metrics.error_rate_percent = 0.5
            svc.metrics.request_latency_p50_ms = 15.0
            svc.metrics.request_latency_p99_ms = 50.0
        if not svc._anomalies:
            svc.health = ServiceHealth.HEALTHY

    msg = f"Config updated: {action.target_service}.{key} = {value}{msg_tail}"
    return IncidentObservation(message=msg)


def handle_resolve_incident(
    action: IncidentAction, cluster: Cluster, scenario: BaseScenario
) -> IncidentObservation:
    root_cause = action.parameters.get("root_cause", "not specified")
    resolution = action.parameters.get("resolution", "not specified")

    msg = (
        f"Incident resolution submitted.\n"
        f"  Root Cause: {root_cause}\n"
        f"  Resolution: {resolution}\n"
        f"  Status: RESOLVED"
    )

    return IncidentObservation(message=msg)


ActionHandler = Callable[
    [IncidentAction, Cluster, BaseScenario],
    IncidentObservation,
]

ACTION_HANDLERS: Dict[str, ActionHandler] = {
    "list_services": handle_list_services,
    "describe_service": handle_describe_service,
    "read_logs": handle_read_logs,
    "check_metrics": handle_check_metrics,
    "restart_service": handle_restart_service,
    "scale_service": handle_scale_service,
    "rollback_deployment": handle_rollback_deployment,
    "run_diagnostic": handle_run_diagnostic,
    "update_config": handle_update_config,
    "resolve_incident": handle_resolve_incident,
}
