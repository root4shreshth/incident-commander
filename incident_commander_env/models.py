"""Typed Pydantic models for IncidentCommanderEnv action, observation, and state."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IncidentAction(BaseModel):
    """Agent's action in the incident response environment."""

    action_type: Literal[
        "list_services",
        "describe_service",
        "read_logs",
        "check_metrics",
        "restart_service",
        "scale_service",
        "rollback_deployment",
        "run_diagnostic",
        "update_config",
        "resolve_incident",
    ] = Field(..., description="The type of SRE action to perform")

    target_service: Optional[str] = Field(
        None, description="Target service name (e.g. 'payment-service')"
    )

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific parameters. Examples: "
            "restart_service: {memory_limit: '512Mi'}, "
            "scale_service: {replicas: 3}, "
            "rollback_deployment: {to_version: 'v2.3.1'}, "
            "read_logs: {lines: 50, severity: 'ERROR'}, "
            "check_metrics: {metric: 'cpu', window: '5m'}, "
            "update_config: {key: 'db.pool.max_size', value: 50}, "
            "run_diagnostic: {command: 'check_connectivity'}, "
            "resolve_incident: {root_cause: '...', resolution: '...'}"
        ),
    )


class ServiceSummary(BaseModel):
    """Brief overview of a service shown in list_services."""

    name: str
    health: str
    version: str
    replicas: int
    cpu_percent: float
    memory_mb: float
    error_rate_percent: float


class ServiceDetail(BaseModel):
    """Detailed service information returned by describe_service."""

    name: str
    health: str
    version: str
    replicas: int
    memory_limit: str
    cpu_limit: str
    port: int
    db_pool_size: Optional[int] = None
    deployment_history: List[Dict[str, Any]] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    dependents: List[str] = Field(default_factory=list)


class MetricsSnapshot(BaseModel):
    """Metrics data returned by check_metrics."""

    service: str
    cpu_percent: float
    memory_mb: float
    memory_limit_mb: float
    memory_utilization_percent: float
    request_latency_p50_ms: float
    request_latency_p99_ms: float
    error_rate_percent: float
    active_connections: int
    requests_per_second: float


class IncidentObservation(BaseModel):
    """What the agent sees after each action or on reset."""

    message: str = Field("", description="Human-readable response text")
    done: bool = Field(False, description="Whether the episode has ended")
    reward: float = Field(0.0, description="Reward from this step")

    alert: Optional[str] = Field(None, description="Active incident alert (set on reset)")
    services_summary: Optional[List[ServiceSummary]] = Field(
        None, description="Result of list_services"
    )
    service_detail: Optional[ServiceDetail] = Field(
        None, description="Result of describe_service"
    )
    logs: Optional[List[str]] = Field(None, description="Result of read_logs")
    metrics: Optional[MetricsSnapshot] = Field(None, description="Result of check_metrics")
    diagnostic_result: Optional[str] = Field(
        None, description="Result of run_diagnostic"
    )
    dependency_graph: Optional[Dict[str, List[str]]] = Field(
        None, description="Service dependency graph (shown on reset)"
    )
    error: Optional[str] = Field(None, description="Error message if action was invalid")


class IncidentState(BaseModel):
    """Internal environment state exposed via GET /state."""

    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    incident_resolved: bool = False
    root_cause_identified: bool = False
    services_restarted: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)
    current_score: float = 0.0
    max_steps: int = 0


class ActionRecord(BaseModel):
    """Record of an action taken during an episode, used for grading."""

    step: int
    action_type: str
    target_service: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
