"""Praetor Playbook Verifier.

A pre-production QA layer for ops automations. You write an auto-remediation
policy in YAML, run ``praetor verify <policy.yaml>``, and get back a per-scenario
pass/fail report against the 12 canonical incident families before the policy
ever reaches production.

Public API:

    Policy               - Pydantic model for a YAML-authored automation policy
    load_policy(path)    - Parse + validate a policy YAML into a Policy
    verify_policy(policy) -> PolicyReport
                         - Run a policy against the full scenario library

Design philosophy: the verifier answers three questions a real ops team asks
about a new remediation rule before shipping it:

  1. Does the policy actually fix the scenarios it claims to fix?
  2. Does the policy misfire (trigger + take actions) on scenarios it was
     never designed for?
  3. Do the policy's actions produce a healthy end-state, or do they
     make things worse (measured via the 6-component RewardBreakdown)?
"""

from __future__ import annotations

from praetor_verify.policy import (
    Policy,
    PolicyAction,
    PolicyTrigger,
    PolicyTriggerMatch,
    PolicySafeguards,
    RequireConfirmation,
    PolicyLoadError,
    load_policy,
)
from praetor_verify.runner import (
    PolicyRunResult,
    PolicyReport,
    verify_policy,
)


__all__ = [
    "Policy",
    "PolicyAction",
    "PolicyTrigger",
    "PolicyTriggerMatch",
    "PolicySafeguards",
    "RequireConfirmation",
    "PolicyLoadError",
    "load_policy",
    "PolicyRunResult",
    "PolicyReport",
    "verify_policy",
]
