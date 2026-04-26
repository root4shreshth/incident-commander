"""AI Coach - deterministic, rule-based guidance for humans learning SRE incident response.

Produces contextual hints, plain-English explanations, and structured post-mortems
by inspecting the live cluster state and the user's action history. No LLM calls,
so latency is zero and behavior is reproducible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from incident_commander_env.models import ActionRecord, IncidentState
from incident_commander_env.server.scenarios.base_scenario import BaseScenario
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.service import ServiceHealth


# ---------------------------------------------------------------------------
# Per-scenario learning context (shown on the scenario picker)
# ---------------------------------------------------------------------------

LEARNING_CONTEXT: Dict[str, Dict[str, Any]] = {
    "oom_crash": {
        "skill_tag": "Your first page",
        "backstory": (
            "It's 3:42 AM. Your phone buzzes - PagerDuty. "
            "The payment-service is throwing health check failures and customers "
            "can't check out. You're the on-call SRE. Let's go."
        ),
        "learning_goals": [
            "Reading service logs to spot error patterns",
            "Interpreting memory metrics (MB used vs limit)",
            "When to restart vs when to dig deeper",
        ],
        "est_minutes": 5,
        "prerequisite": None,
    },
    "db_pool_exhaustion": {
        "skill_tag": "Trace the cascade",
        "backstory": (
            "It's 2:23 PM. Support is blowing up - customers see 'Service Unavailable' "
            "on checkout. Multiple services are throwing 5xx errors. The frontend "
            "looks broken, but the real culprit is somewhere deeper."
        ),
        "learning_goals": [
            "Tracing symptoms through a dependency chain",
            "Spotting 'cascade failure' vs 'origin failure'",
            "Understanding DB connection pools",
        ],
        "est_minutes": 12,
        "prerequisite": "oom_crash",
    },
    "bad_deployment_cascade": {
        "skill_tag": "Under fire",
        "backstory": (
            "9:15 AM, Monday. A deploy went out 10 minutes ago and now three services "
            "are failing simultaneously. Cluster memory is at 95%. Your team is "
            "staring at you. What do you touch first?"
        ),
        "learning_goals": [
            "Reading deployment history for recent changes",
            "Rollback vs restart - when each is correct",
            "Sequencing fixes under cascading failure",
        ],
        "est_minutes": 20,
        "prerequisite": "db_pool_exhaustion",
    },
    "disk_full": {
        "skill_tag": "Out of space",
        "backstory": (
            "11:18 AM. Notification-service is suddenly throwing 5xx on every push. "
            "Process is alive, latency is normal, error rate at 28%. Logs say "
            "'No space left on device'. The disk is full."
        ),
        "learning_goals": [
            "Recognizing the ENOSPC error pattern in logs",
            "Knowing when a restart cycles a volume vs preserves it",
            "Distinguishing degraded-mode (read-only) from fully-down",
        ],
        "est_minutes": 7,
        "prerequisite": "oom_crash",
    },
    "slow_query": {
        "skill_tag": "Quick fix vs real fix",
        "backstory": (
            "3:09 PM. p99 latency on order-service has gone from 80ms to 8 seconds. "
            "Throughput collapsed. Connection pool full of txns waiting on row-locks. "
            "A deploy went out two hours ago. What do you do - restart, or roll back?"
        ),
        "learning_goals": [
            "Spotting lock-contention vs other latency causes",
            "Why restart is a quick fix that doesn't last",
            "Reading deployment history to tie latency to a recent change",
        ],
        "est_minutes": 14,
        "prerequisite": "db_pool_exhaustion",
    },
    "cert_expiry": {
        "skill_tag": "The embarrassing one",
        "backstory": (
            "8:00 AM. Frontend is returning 5xx to every external user. Liveness "
            "probe passes. Internal services say it's healthy. Logs say "
            "'TLS handshake failed: certificate has expired.' "
            "It's the most embarrassing class of outage. It's still your problem."
        ),
        "learning_goals": [
            "Why metrics can look normal while everything is broken",
            "TLS / cert errors and how restart-renewal hooks work",
            "Reading-logs-not-metrics as the first move on weird outages",
        ],
        "est_minutes": 8,
        "prerequisite": "oom_crash",
    },
}


# ---------------------------------------------------------------------------
# Ideal trajectories - what a senior SRE would have done
# ---------------------------------------------------------------------------

IDEAL_TRAJECTORIES: Dict[str, List[Dict[str, Any]]] = {
    "oom_crash": [
        {
            "action": "list_services",
            "target": None,
            "why": "Get the big picture first. You'll spot payment-service marked CRASHED.",
        },
        {
            "action": "read_logs",
            "target": "payment-service",
            "why": "The crashed service. Logs show 'OutOfMemoryError' - the process ran out of memory.",
        },
        {
            "action": "check_metrics",
            "target": "payment-service",
            "why": "Confirm: memory used 294 MB against a 256 MB limit. Textbook OOM.",
        },
        {
            "action": "restart_service",
            "target": "payment-service",
            "params": {"memory_limit": "512Mi"},
            "why": "Restart with a higher memory ceiling. 512Mi gives headroom while we plan a permanent fix.",
        },
    ],
    "db_pool_exhaustion": [
        {
            "action": "list_services",
            "target": None,
            "why": "Get the overall cluster health. Multiple services are degraded.",
        },
        {
            "action": "read_logs",
            "target": "frontend-bff",
            "why": "Start where the user-visible symptom is. Logs show generic 5xx - not specific.",
        },
        {
            "action": "read_logs",
            "target": "order-service",
            "why": "One layer deeper. Logs show 'connection leak - unable to acquire DB connection'.",
        },
        {
            "action": "read_logs",
            "target": "postgres-db",
            "why": "The root. Logs show 'connection pool exhausted: 20/20 in use'.",
        },
        {
            "action": "update_config",
            "target": "postgres-db",
            "params": {"key": "db.pool.max_size", "value": 100},
            "why": "Raise the pool ceiling so it stops rejecting new connections.",
        },
        {
            "action": "restart_service",
            "target": "order-service",
            "why": "Restart the service that was leaking to force it to release stale connections.",
        },
        {
            "action": "resolve_incident",
            "target": None,
            "params": {
                "root_cause": "order-service connection leak exhausted postgres-db pool",
                "resolution": "raised pool size to 100 and restarted order-service",
            },
            "why": "Declare resolved with an accurate root-cause summary.",
        },
    ],
    "bad_deployment_cascade": [
        {
            "action": "list_services",
            "target": None,
            "why": "Map the blast radius. Three services failing, cluster memory at 95%.",
        },
        {
            "action": "read_logs",
            "target": "order-service",
            "why": "The noisiest service. Logs mention recent deploy and memory leak.",
        },
        {
            "action": "describe_service",
            "target": "order-service",
            "why": "Deployment history shows v2.4.0 went live 10 minutes ago. That's the suspect.",
        },
        {
            "action": "rollback_deployment",
            "target": "order-service",
            "params": {"to_version": "v2.3.1"},
            "why": "Rollback first - this frees memory and removes the root cause in one move.",
        },
        {
            "action": "restart_service",
            "target": "inventory-service",
            "why": "Restart starved dependents AFTER rolling back. Order matters.",
        },
        {
            "action": "restart_service",
            "target": "notification-service",
            "why": "Same reason - it was starved by quota pressure.",
        },
        {
            "action": "resolve_incident",
            "target": None,
            "params": {
                "root_cause": "Bad deployment v2.4.0 introduced memory leak; autoscaler exhausted quota",
                "resolution": "Rolled back to v2.3.1 then restarted starved services",
            },
            "why": "Resolution reflects both the origin (bad deploy) and the cascade.",
        },
    ],
    "disk_full": [
        {"action": "list_services", "target": None,
         "why": "Get the overview - notification-service is marked DEGRADED."},
        {"action": "read_logs", "target": "notification-service",
         "why": "The degraded service. Logs scream 'No space left on device' on every write."},
        {"action": "check_metrics", "target": "notification-service",
         "why": "CPU and memory normal - confirming this isn't OOM. The fault is at the volume layer."},
        {"action": "restart_service", "target": "notification-service",
         "why": "Restart cycles the pod, the volume gets cleaned + log-rotated, the service comes back."},
        {"action": "resolve_incident", "target": None,
         "params": {
             "root_cause": "notification-service log volume hit 100% - writes returned ENOSPC",
             "resolution": "Restarted to cycle the volume; will follow up with log retention config",
         },
         "why": "Honest postmortem: restart bought time; permanent fix is rotation policy."},
    ],
    "slow_query": [
        {"action": "list_services", "target": None,
         "why": "Confirm scope - order-service is degraded, others normal."},
        {"action": "check_metrics", "target": "order-service",
         "why": "Latency p99 is 8s, throughput at 4 RPS. Lock contention pattern."},
        {"action": "read_logs", "target": "order-service",
         "why": "Logs spell it out: 'Lock wait timeout exceeded' from a SELECT … FOR UPDATE."},
        {"action": "describe_service", "target": "order-service",
         "why": "Deployment history. v2.5.0 went out two hours ago - that's our suspect."},
        {"action": "rollback_deployment", "target": "order-service",
         "params": {"to_version": "v2.4.6"},
         "why": "Restart would clear the active txns but the slow query is in the binary. Rollback reverts both."},
        {"action": "resolve_incident", "target": None,
         "params": {
             "root_cause": "Slow query in v2.5.0 holding row-locks on `orders` table",
             "resolution": "Rolled back order-service to v2.4.6; query will be re-introduced behind a feature flag",
         },
         "why": "Resolution names the offending version and the right next step."},
    ],
    "cert_expiry": [
        {"action": "list_services", "target": None,
         "why": "Frontend is unhealthy. Everything else looks fine. That's a clue."},
        {"action": "check_metrics", "target": "frontend-bff",
         "why": "CPU and memory tiny. Active connections near zero. Error rate at 99%. Doesn't match crash patterns."},
        {"action": "read_logs", "target": "frontend-bff",
         "why": "Right there: 'TLS handshake failed: certificate has expired'. Cert problem, not code."},
        {"action": "restart_service", "target": "frontend-bff",
         "why": "The cert renewal hook fires on restart - listener reloads with the renewed cert."},
        {"action": "resolve_incident", "target": None,
         "params": {
             "root_cause": "Frontend TLS certificate expired at 08:00 UTC",
             "resolution": "Restarted frontend-bff to trigger cert renewal; will add 30-day expiry alert",
         },
         "why": "Honest postmortem: the bigger lesson is the missing alert."},
    ],
}


# ---------------------------------------------------------------------------
# Hint engine
# ---------------------------------------------------------------------------

_INVESTIGATIVE = {"read_logs", "check_metrics", "describe_service", "run_diagnostic"}
_REMEDIATIVE = {"restart_service", "rollback_deployment", "scale_service", "update_config"}


def _action_signature(a: ActionRecord) -> str:
    return f"{a.action_type}:{a.target_service or ''}"


def compute_hint(
    task_id: str,
    cluster: Cluster,
    action_history: List[ActionRecord],
    step_count: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Produce a contextual hint for the user's current situation.

    Returns: {hint, suggested_action, tone}
    tone ∈ {neutral, encourage, warn, celebrate}
    """
    ideal = IDEAL_TRAJECTORIES.get(task_id, [])
    done_sigs = {_action_signature(a) for a in action_history}

    remaining = max_steps - step_count
    last = action_history[-1] if action_history else None

    # Celebrate a correct fix
    if last and last.action_type in _REMEDIATIVE and task_id == "oom_crash":
        if last.target_service == "payment-service":
            new_limit = str(last.parameters.get("memory_limit", "")).replace("Mi", "")
            try:
                if int(new_limit) > 256:
                    return {
                        "hint": f"Bingo. You raised memory to {new_limit}Mi - well above the old 256Mi cap. Now declare it resolved with a root cause.",
                        "suggested_action": {"action": "resolve_incident", "target": None},
                        "tone": "celebrate",
                    }
            except ValueError:
                pass

    # Warn on a restart of a healthy service
    if last and last.action_type == "restart_service":
        svc = cluster.get_service(last.target_service or "")
        if svc and svc.health == ServiceHealth.HEALTHY and task_id == "oom_crash":
            if last.target_service != "payment-service":
                return {
                    "hint": f"Ouch - {last.target_service} wasn't broken. Restarting healthy services costs you -0.10 and wastes time. The clue is the service that's *actually* crashed. Try list_services first.",
                    "suggested_action": {"action": "list_services", "target": None},
                    "tone": "warn",
                }

    # Step budget warning
    if remaining <= 3 and remaining > 0:
        return {
            "hint": f"Only {remaining} step{'s' if remaining > 1 else ''} left. Time to commit: apply the fix you've already diagnosed, or declare resolved.",
            "suggested_action": None,
            "tone": "warn",
        }

    # First action
    if not action_history:
        return {
            "hint": "Start with the big picture. Click 'See all services' (list_services) to find what's broken. Don't restart anything yet.",
            "suggested_action": {"action": "list_services", "target": None},
            "tone": "neutral",
        }

    # Walk the ideal trajectory and find the first step the user hasn't done yet
    for step in ideal:
        sig = f"{step['action']}:{step.get('target') or ''}"
        if sig not in done_sigs:
            target_phrase = f" on {step['target']}" if step.get("target") else ""
            hint_text = f"Next up: **{step['action']}{target_phrase}**. {step['why']}"
            return {
                "hint": hint_text,
                "suggested_action": {
                    "action": step["action"],
                    "target": step.get("target"),
                    "params": step.get("params", {}),
                },
                "tone": "encourage",
            }

    # User already covered the ideal path - prompt resolve
    return {
        "hint": "You've touched every key diagnostic and fix. Now declare the incident resolved with a root-cause summary.",
        "suggested_action": {"action": "resolve_incident", "target": None},
        "tone": "neutral",
    }


# ---------------------------------------------------------------------------
# Plain-English explainer
# ---------------------------------------------------------------------------

_EXPLANATIONS = {
    "OutOfMemoryError": (
        "**OutOfMemoryError** means the service tried to use more memory than its configured limit. "
        "The operating system killed it to protect the rest of the machine. "
        "The fix is usually either (a) raise the memory limit, or (b) find what's using too much memory."
    ),
    "connection pool exhausted": (
        "**Connection pool exhausted** means every database connection slot is already in use and "
        "new requests can't get one. Usually caused by a service that opens connections but forgets "
        "to release them - a 'connection leak'. Fix by raising the pool size *and* restarting the leaker."
    ),
    "connection leak": (
        "A **connection leak** is when code borrows a DB connection but never returns it. Over time "
        "every connection is stuck 'in use' by a dead request. Restarting the leaking service resets "
        "the count, but the permanent fix is in the code."
    ),
    "memory leak": (
        "A **memory leak** is when a process keeps allocating memory but never frees it. Memory usage "
        "climbs until the process crashes or the autoscaler spawns more replicas to cope, which "
        "exhausts cluster quota. A bad deploy is the usual cause - rollback is the right call."
    ),
    "autoscaler": (
        "The **autoscaler** spawns more replicas when load is high. That's helpful for real traffic "
        "spikes, but with a memory leak it just eats your cluster quota without fixing anything."
    ),
    "resource quota": (
        "**Resource quota** is the cluster-wide memory/CPU ceiling. When it's hit, new pods can't "
        "start and existing services can't grow. Free it by removing a bad deploy or scaling something down."
    ),
    "cascade": (
        "A **cascading failure** is when one broken service takes down everything that depends on it. "
        "The surface symptom is in the frontend; the root is usually 2-3 layers deeper."
    ),
    "healthy": (
        "**Healthy** means the service is passing its health checks and serving traffic normally. "
        "Don't restart healthy services - that costs you points and doesn't fix anything."
    ),
    "rollback": (
        "**Rollback** reverts a service to its previous version. Use this when a recent deploy "
        "introduced the bug. Unlike a restart, rollback actually removes the bad code."
    ),
}


def explain_observation(task_id: str, last_action: Dict[str, Any], last_message: str) -> Dict[str, Any]:
    """Explain the user's most recent observation in plain language."""
    message = (last_message or "").lower()

    matched: List[str] = []
    for keyword, explanation in _EXPLANATIONS.items():
        if keyword.lower() in message:
            matched.append(explanation)

    if matched:
        return {
            "explanation": "\n\n".join(matched[:2]),
            "matched_terms": [k for k in _EXPLANATIONS if k.lower() in message][:3],
        }

    action_type = last_action.get("action_type", "")
    target = last_action.get("target_service", "") or ""

    if action_type == "list_services":
        return {
            "explanation": (
                "**list_services** is your 'everything OK?' command. It shows all 9 services with their "
                "current health and key metrics. Use it first, and again after any major change to verify."
            ),
            "matched_terms": [],
        }
    if action_type == "read_logs":
        return {
            "explanation": (
                f"**read_logs** pulls recent log lines from **{target}**. Look for lines tagged [ERROR] or "
                "[CRITICAL] - they usually name the bug directly. Normal services mostly log [INFO]."
            ),
            "matched_terms": [],
        }
    if action_type == "check_metrics":
        return {
            "explanation": (
                f"**check_metrics** shows the live numbers for **{target}** - CPU %, memory (used vs limit), "
                "latency, error rate, request rate. Compare memory_mb to memory_limit_mb: if they're close, "
                "you're about to OOM."
            ),
            "matched_terms": [],
        }
    if action_type == "describe_service":
        return {
            "explanation": (
                f"**describe_service** shows **{target}**'s config and deployment history. The history is "
                "gold when a recent deploy broke things - look for timestamps near the incident start."
            ),
            "matched_terms": [],
        }

    return {
        "explanation": "No specific terms to explain in that output. Try another action, or ask your coach for the next move.",
        "matched_terms": [],
    }


# ---------------------------------------------------------------------------
# Post-mortem builder
# ---------------------------------------------------------------------------

def _grade_letter(score: float) -> str:
    if score >= 0.90: return "A"
    if score >= 0.75: return "B"
    if score >= 0.60: return "C"
    if score >= 0.40: return "D"
    return "F"


def build_postmortem(
    scenario: BaseScenario,
    action_history: List[ActionRecord],
    cluster: Cluster,
    state: IncidentState,
) -> Dict[str, Any]:
    """Build a rich end-of-episode review."""
    details = scenario.grade_details(action_history, cluster)
    score = details["final_score"]
    task_id = scenario.task_id

    user_trajectory = [
        {
            "step": a.step,
            "action": a.action_type,
            "target": a.target_service,
            "params": a.parameters,
        }
        for a in action_history
    ]

    ideal = IDEAL_TRAJECTORIES.get(task_id, [])
    ctx = LEARNING_CONTEXT.get(task_id, {})

    # Highlights - what went right and what was missed
    passed_criteria = [c for c in details["criteria"] if c["passed"]]
    failed_criteria = [c for c in details["criteria"] if not c["passed"]]

    # One study recommendation based on the biggest missed criterion
    study_link = None
    if failed_criteria:
        biggest_miss = max(failed_criteria, key=lambda c: c["weight"])
        study_link = {
            "title": f"Deep dive: {biggest_miss['criterion']}",
            "topic": biggest_miss["criterion"],
            "weight_missed": biggest_miss["weight"],
        }

    return {
        "task_id": task_id,
        "resolved": state.incident_resolved,
        "score": score,
        "grade_letter": _grade_letter(score),
        "steps_used": state.step_count,
        "max_steps": scenario.max_steps,
        "criteria": details["criteria"],
        "penalties": details["penalties"],
        "user_trajectory": user_trajectory,
        "ideal_trajectory": ideal,
        "highlights": {
            "passed": [c["criterion"] for c in passed_criteria],
            "missed": [c["criterion"] for c in failed_criteria],
        },
        "study_link": study_link,
        "learning_goals": ctx.get("learning_goals", []),
    }
