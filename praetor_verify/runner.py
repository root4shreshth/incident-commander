"""Policy verifier - runs a Policy against the full 12-scenario library.

For each scenario in `SCENARIO_REGISTRY`:

  1. Reset a fresh IncidentCommanderEnv with the scenario.
  2. Read the alert text + infer which service the trigger would match on.
  3. Check whether the policy's `trigger.match` rules fire against that alert.
  4. If triggered: expand action templates, execute actions in sequence via
     env.step(), and accumulate the 6-component RewardBreakdown.
  5. Ask the env whether the scenario is resolved.
  6. Compare (triggered, resolved) against `policy.scenarios_claimed` to
     produce a per-scenario verdict.

The verdict matrix:

    scenario in claimed?  triggered?  resolved?  ->  verdict
    yes                   yes         yes         ->  PASS
    yes                   yes         no          ->  FAIL (policy fires but doesn't fix)
    yes                   no          -           ->  FAIL (policy claims to handle but didn't trigger)
    no                    yes         yes         ->  WARN (policy incidentally fixes a scenario it doesn't claim)
    no                    yes         no          ->  FAIL (false positive AND made things worse)
    no                    no          -           ->  PASS (correct non-trigger)

The verifier is deterministic - same policy + same seed always gives the same
report - because the underlying env is deterministic. That's what makes this
useful in CI: a policy change that regresses one scenario's verdict is
diffable.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from incident_commander_env.models import IncidentAction
from incident_commander_env.server.environment import IncidentCommanderEnv
from incident_commander_env.server.grading.components import RewardBreakdown
from incident_commander_env.server.scenarios import SCENARIO_REGISTRY

from praetor_verify.policy import Policy, PolicyAction


VerdictLiteral = Literal["PASS", "FAIL", "WARN"]


class PolicyRunResult(BaseModel):
    """Per-scenario result of running a policy through the verifier."""

    scenario: str = Field(description="task_id of the scenario this run tested against.")
    claimed: bool = Field(description="Was this scenario in policy.scenarios_claimed?")
    triggered: bool = Field(description="Did the policy's trigger.match rules fire?")
    resolved: bool = Field(description="Did env.check_resolved() return true after the policy's actions?")
    steps_taken: int = Field(description="Number of policy actions actually executed.")
    total_reward: float = Field(description="Sum of RewardBreakdown.total() across all executed steps.")
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Sum of each RewardBreakdown component across all executed steps.",
    )
    verdict: VerdictLiteral = Field(description="PASS | FAIL | WARN per the verdict matrix.")
    failure_mode: Optional[str] = Field(
        default=None,
        description="If verdict != PASS, a short machine-readable code for why.",
    )
    confirmation_required: List[str] = Field(
        default_factory=list,
        description="Actions that would have required human confirmation per policy.safeguards.",
    )
    matched_service: Optional[str] = Field(
        default=None,
        description="The service name resolved to {trigger.service} at match time.",
    )
    error: Optional[str] = Field(
        default=None,
        description="If an exception was raised during scenario execution, its message.",
    )


class PolicyReport(BaseModel):
    """Aggregate report across all scenarios the verifier ran."""

    policy_name: str
    policy_version: str
    scenarios_claimed: List[str]
    results: List[PolicyRunResult]

    @property
    def total_scenarios(self) -> int:
        return len(self.results)

    @property
    def passing(self) -> List[PolicyRunResult]:
        return [r for r in self.results if r.verdict == "PASS"]

    @property
    def failing(self) -> List[PolicyRunResult]:
        return [r for r in self.results if r.verdict == "FAIL"]

    @property
    def warning(self) -> List[PolicyRunResult]:
        return [r for r in self.results if r.verdict == "WARN"]

    @property
    def overall_verdict(self) -> VerdictLiteral:
        if self.failing:
            return "FAIL"
        if self.warning:
            return "WARN"
        return "PASS"

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return len(self.passing) / len(self.results)


# ---------------------------------------------------------------------------
# Trigger matching
# ---------------------------------------------------------------------------

def _trigger_matches(
    policy: Policy,
    alert_text: str,
    candidate_services: List[str],
) -> Tuple[bool, Optional[str]]:
    """Return (matched, matched_service_name).

    `matched_service_name` is the first candidate service whose name matches
    the policy's `service_pattern` glob, or None if no pattern is set.
    """
    match = policy.trigger.match
    if match.is_empty():
        # Empty match rules mean the trigger always fires. Unusual but valid -
        # signals "this policy is universal" (e.g., a debug logger).
        return True, candidate_services[0] if candidate_services else None

    text_lower = (alert_text or "").lower()

    # message_contains: ANY substring match
    if match.message_contains:
        if not any(needle.lower() in text_lower for needle in match.message_contains):
            return False, None

    # service_pattern: at least one candidate matches
    matched_service: Optional[str] = None
    if match.service_pattern:
        for svc in candidate_services:
            if fnmatch.fnmatch(svc, match.service_pattern):
                matched_service = svc
                break
        if matched_service is None:
            return False, None
    elif candidate_services:
        matched_service = candidate_services[0]

    # alert_severity is inferred from the alert text (best-effort)
    if match.alert_severity is not None:
        severity_hits = {
            "CRITICAL": ("CRITICAL", "critical"),
            "WARNING": ("WARNING", "WARN ", "warn "),
            "INFO": ("INFO", "info"),
        }[match.alert_severity]
        if not any(h in (alert_text or "") for h in severity_hits):
            return False, matched_service

    return True, matched_service


def _confirmation_required_for(policy: Policy, action: PolicyAction) -> bool:
    """Would this action require human confirmation per policy.safeguards?"""
    rc = policy.safeguards.require_confirmation_if
    if action.action_type in rc.action_types:
        return True
    if action.target_service:
        for pattern in rc.service_matches:
            if fnmatch.fnmatch(action.target_service, pattern):
                return True
    return False


# ---------------------------------------------------------------------------
# Verifier core
# ---------------------------------------------------------------------------

def _run_one_scenario(
    policy: Policy,
    task_id: str,
    seed: int,
    difficulty: float,
) -> PolicyRunResult:
    """Run the policy against a single scenario. Never raises."""
    claimed = task_id in policy.scenarios_claimed
    breakdown_totals: Dict[str, float] = {
        k: 0.0 for k in ("diagnostic", "correct_op", "resolution",
                         "format", "efficiency", "penalty")
    }
    total_reward = 0.0

    try:
        env = IncidentCommanderEnv()
        obs = env.reset(task_id=task_id, seed=seed, difficulty=difficulty)
        alert_text = (obs.alert or obs.message or "")

        # Candidate services for {trigger.service}: start from the scenario's
        # target if we can find it, otherwise fall back to the full cluster list
        # so glob patterns like `payment-*` still resolve.
        scenario_target = getattr(env._scenario, "target_service", None) \
            if hasattr(env, "_scenario") else None
        candidate_services = []
        if scenario_target:
            candidate_services.append(scenario_target)
        # Grab everything in the cluster as fallback
        try:
            snap = env._backend.snapshot()
            for name in snap.services.keys():
                if name not in candidate_services:
                    candidate_services.append(name)
        except Exception:
            pass

        matched, matched_service = _trigger_matches(policy, alert_text, candidate_services)

        if not matched:
            # Correct non-trigger IF the scenario isn't claimed; miss otherwise
            verdict, failure_mode = (
                ("FAIL", "claimed_but_not_triggered") if claimed
                else ("PASS", None)
            )
            return PolicyRunResult(
                scenario=task_id,
                claimed=claimed,
                triggered=False,
                resolved=False,
                steps_taken=0,
                total_reward=0.0,
                reward_breakdown=breakdown_totals,
                verdict=verdict,
                failure_mode=failure_mode,
            )

        # Triggered - execute action sequence.
        context = {
            "trigger.service": matched_service or "",
            "scenario.target": scenario_target or "",
        }
        actions = policy.expand_templates(context)
        confirmation_needed: List[str] = []
        steps = 0

        for pact in actions:
            if _confirmation_required_for(policy, pact):
                confirmation_needed.append(
                    f"{pact.action_type} on {pact.target_service or '<no target>'}"
                )
                # In verifier mode we still execute so we can report end-state.
                # Production runtime would pause here for a human.
            try:
                action = IncidentAction(
                    action_type=pact.action_type,
                    target_service=pact.target_service,
                    parameters=pact.parameters or {},
                )
            except Exception as exc:
                # Malformed action - record + skip
                confirmation_needed.append(f"SKIPPED (invalid): {exc}")
                continue

            step_obs = env.step(action)
            steps += 1
            bd = getattr(env, "_last_breakdown", None) or RewardBreakdown.zero()
            for k in breakdown_totals:
                breakdown_totals[k] += getattr(bd, k, 0.0)
            total_reward += bd.total()
            if step_obs.done:
                break

        # Was the scenario resolved after all actions ran?
        try:
            resolved = env._backend.check_resolved(env._scenario)
        except Exception:
            resolved = False

        # Verdict logic
        if claimed:
            if resolved:
                verdict, failure_mode = "PASS", None
            else:
                verdict, failure_mode = "FAIL", "triggered_but_no_resolve"
        else:
            # Policy triggered on something it doesn't claim.
            if resolved and total_reward >= 0:
                verdict, failure_mode = "WARN", "incidental_fix_outside_claim"
            elif total_reward < 0:
                verdict, failure_mode = "FAIL", "false_positive_negative_reward"
            else:
                verdict, failure_mode = "FAIL", "false_positive"

        return PolicyRunResult(
            scenario=task_id,
            claimed=claimed,
            triggered=True,
            resolved=resolved,
            steps_taken=steps,
            total_reward=total_reward,
            reward_breakdown=breakdown_totals,
            verdict=verdict,
            failure_mode=failure_mode,
            confirmation_required=confirmation_needed,
            matched_service=matched_service,
        )
    except Exception as exc:
        return PolicyRunResult(
            scenario=task_id,
            claimed=claimed,
            triggered=False,
            resolved=False,
            steps_taken=0,
            total_reward=0.0,
            reward_breakdown=breakdown_totals,
            verdict="FAIL",
            failure_mode="verifier_exception",
            error=f"{type(exc).__name__}: {exc}",
        )


def verify_policy(
    policy: Policy,
    scenarios: Optional[List[str]] = None,
    *,
    seed: int = 9000,
    difficulty: float = 0.5,
) -> PolicyReport:
    """Run `policy` against all scenarios in SCENARIO_REGISTRY (or a subset).

    Args:
        policy: the loaded Policy object
        scenarios: optional subset of task_ids to test; None = all 12
        seed: fixed seed so runs are byte-reproducible (default 9000, outside
              training/eval seed ranges)
        difficulty: float in [0,1]; default 0.5 (average scenario intensity)

    Returns:
        PolicyReport with per-scenario PolicyRunResult and aggregate verdict.
    """
    task_ids = scenarios if scenarios is not None else sorted(SCENARIO_REGISTRY.keys())
    results = [
        _run_one_scenario(policy, tid, seed=seed, difficulty=difficulty)
        for tid in task_ids
    ]
    return PolicyReport(
        policy_name=policy.name,
        policy_version=policy.version,
        scenarios_claimed=list(policy.scenarios_claimed),
        results=results,
    )


__all__ = [
    "PolicyRunResult",
    "PolicyReport",
    "verify_policy",
    "VerdictLiteral",
]
