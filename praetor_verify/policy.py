"""Policy DSL - YAML-authored auto-remediation rules that the verifier runs.

A policy is a small typed document. Example (`policies/oom_auto_restart.yaml`):

    name: oom_auto_restart
    version: 0.1.0
    owner: sre-team@example.com
    description: |
      Auto-restart any payment-* service that reports OutOfMemoryError,
      with a doubled memory ceiling.
    scenarios_claimed: [oom_crash]

    trigger:
      event: alert
      match:
        message_contains: [OutOfMemoryError, java.lang.OutOfMemory]
        service_pattern: "payment-*"

    actions:
      - action_type: restart_service
        target_service: "{trigger.service}"
        parameters:
          memory_limit: 2048Mi

    safeguards:
      max_actions_per_hour: 3
      require_confirmation_if:
        service_matches: ["postgres-*", "*-db"]
        action_types: [rollback_deployment]

The verifier reads this, then for each of the 12 canonical scenario families:
  1. Resets the env, gets the initial alert
  2. Checks whether the policy's trigger.match rules match the alert
  3. If matched, executes the policy.actions sequence (with template expansion)
  4. Reports (scenario, triggered?, resolved?, reward_breakdown, failure_mode)

Everything here is Pydantic-validated so a malformed policy fails fast at load
time with a clear error, rather than 30s into a CI run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# Kept in sync with incident_commander_env.models.IncidentAction. We duplicate
# the Literal here rather than import it because the DSL should be readable /
# validatable without dragging in the env's simulation deps.
POLICY_ACTION_TYPES = Literal[
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
]

# Template variables the DSL supports in `target_service` and `parameters` values.
# Keep this set small - each variable adds surface area to the runner.
_KNOWN_TEMPLATE_VARS = frozenset({
    "trigger.service",    # the service that matched trigger.match.service_pattern
    "trigger.severity",   # e.g., "CRITICAL", "WARNING"
    "scenario.target",    # the scenario's target_service, for testing convenience
})
_TEMPLATE_RE = re.compile(r"\{([a-zA-Z0-9_.]+)\}")


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

class PolicyTriggerMatch(BaseModel):
    """Match rules for a trigger. All present rules AND together; empty is `always`."""

    message_contains: List[str] = Field(
        default_factory=list,
        description="Trigger fires if the alert message contains ANY of these substrings (case-insensitive).",
    )
    service_pattern: Optional[str] = Field(
        default=None,
        description="Glob pattern for the alerting service name. * matches any chars. E.g. 'payment-*'.",
    )
    alert_severity: Optional[Literal["INFO", "WARNING", "CRITICAL"]] = Field(
        default=None,
        description="If set, alert must be at this exact severity for the trigger to fire.",
    )

    def is_empty(self) -> bool:
        return (
            not self.message_contains
            and self.service_pattern is None
            and self.alert_severity is None
        )


class PolicyTrigger(BaseModel):
    event: Literal["alert"] = Field(
        default="alert",
        description="Kind of event that triggers this policy. Only `alert` for now.",
    )
    match: PolicyTriggerMatch = Field(default_factory=PolicyTriggerMatch)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class PolicyAction(BaseModel):
    """One remediation step. Mirrors IncidentAction's shape."""

    action_type: POLICY_ACTION_TYPES
    target_service: Optional[str] = Field(
        default=None,
        description=(
            "Service name or template. `{trigger.service}` expands to the "
            "service that matched the trigger."
        ),
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_service", "parameters", mode="after")
    @classmethod
    def _validate_templates(cls, value: Any) -> Any:
        """Fail-fast if a template variable name isn't in the known set."""
        _validate_templates_recursive(value)
        return value


def _validate_templates_recursive(value: Any) -> None:
    if isinstance(value, str):
        for match in _TEMPLATE_RE.finditer(value):
            var = match.group(1)
            if var not in _KNOWN_TEMPLATE_VARS:
                raise ValueError(
                    f"unknown template variable {{{var}}}; "
                    f"known: {sorted(_KNOWN_TEMPLATE_VARS)}"
                )
    elif isinstance(value, dict):
        for v in value.values():
            _validate_templates_recursive(v)
    elif isinstance(value, list):
        for v in value:
            _validate_templates_recursive(v)


# ---------------------------------------------------------------------------
# Safeguards
# ---------------------------------------------------------------------------

class RequireConfirmation(BaseModel):
    """Conditions where an action requires a human in the loop."""

    service_matches: List[str] = Field(
        default_factory=list,
        description="Glob patterns; if the action's target_service matches, block on confirmation.",
    )
    action_types: List[POLICY_ACTION_TYPES] = Field(
        default_factory=list,
        description="Action types that always require confirmation, regardless of target.",
    )


class PolicySafeguards(BaseModel):
    max_actions_per_hour: Optional[int] = Field(
        default=None, ge=1,
        description="Rate limit; the verifier records it but does not enforce (production runtime does).",
    )
    require_confirmation_if: RequireConfirmation = Field(default_factory=RequireConfirmation)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class Policy(BaseModel):
    """A YAML-authored auto-remediation policy.

    Loaded from disk via `load_policy(path)`, run via `verify_policy(policy)`.
    """

    name: str = Field(min_length=1, description="Unique identifier, kebab-case recommended.")
    version: str = Field(default="0.1.0", description="Policy semver, for changelog + CI diff.")
    owner: Optional[str] = Field(default=None, description="Contact email / team slug.")
    description: str = Field(default="", description="Human-readable summary of what this policy does.")
    scenarios_claimed: List[str] = Field(
        default_factory=list,
        description=(
            "task_ids of scenarios this policy CLAIMS to fix. The verifier "
            "expects these to resolve under the policy; anything else it "
            "expects the policy to NOT trigger on (else it's a false positive)."
        ),
    )

    trigger: PolicyTrigger
    actions: List[PolicyAction] = Field(min_length=1)
    safeguards: PolicySafeguards = Field(default_factory=PolicySafeguards)

    @model_validator(mode="after")
    def _validate_at_least_one_action(self) -> "Policy":
        if not self.actions:
            raise ValueError("policy must declare at least one action")
        return self

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        # Allow leading underscore for internal / not-for-production policies
        # (convention: `_bad_example_*.yaml` for regression-test policies).
        if not re.match(r"^[a-z_][a-z0-9_-]*$", v):
            raise ValueError(
                "name must be kebab_case or snake_case (letters, digits, - and _; "
                "must start with a letter or underscore)"
            )
        return v

    def expand_templates(self, context: Dict[str, str]) -> List[PolicyAction]:
        """Return a copy of `self.actions` with `{trigger.service}` etc expanded.

        The verifier calls this once per triggered scenario, with `context`
        filled from the match result.
        """
        expanded: List[PolicyAction] = []
        for act in self.actions:
            expanded.append(PolicyAction(
                action_type=act.action_type,
                target_service=_expand(act.target_service, context) if act.target_service else None,
                parameters={k: _expand(v, context) for k, v in act.parameters.items()},
            ))
        return expanded


def _expand(value: Any, context: Dict[str, str]) -> Any:
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            key = m.group(1)
            return str(context.get(key, m.group(0)))  # leave unknown vars alone
        return _TEMPLATE_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, context) for v in value]
    return value


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class PolicyLoadError(ValueError):
    """Raised when a policy YAML file fails schema validation."""


def load_policy(path: Union[str, Path]) -> Policy:
    """Parse + validate a policy YAML file.

    Raises PolicyLoadError with the underlying validation message on failure.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyLoadError(f"could not read {path}: {exc}") from exc

    try:
        import yaml  # noqa: PLC0415 - deferred import so tests can run without PyYAML
    except ImportError as exc:  # pragma: no cover - PyYAML is a hard dep for the DSL
        raise PolicyLoadError(
            "PyYAML is required to load policy YAML files. `uv add pyyaml` and retry."
        ) from exc

    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise PolicyLoadError(f"YAML parse error in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise PolicyLoadError(f"{path} must contain a YAML mapping at top level")

    try:
        return Policy(**raw)
    except Exception as exc:  # ValidationError etc
        raise PolicyLoadError(f"policy validation failed for {path}: {exc}") from exc


__all__ = [
    "Policy",
    "PolicyAction",
    "PolicyTrigger",
    "PolicyTriggerMatch",
    "PolicySafeguards",
    "RequireConfirmation",
    "PolicyLoadError",
    "load_policy",
    "POLICY_ACTION_TYPES",
]
