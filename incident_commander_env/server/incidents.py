"""Webhook ingestion — turns external alerts into autonomous Praetor episodes.

Three providers are supported out of the box:

  * PagerDuty   — receives a v3 incident webhook
  * Prometheus  — receives an Alertmanager v4 webhook
  * Generic     — minimal contract for anything else (Datadog, Sentry, custom)

Each handler maps the alert payload to one of our scenario families using
keyword heuristics, then kicks off a Praetor run in a background thread —
exactly the same path /realtime/run-agent uses, just triggered automatically.

This closes the "continuously monitors a fleet for new incidents" gap in
the Phase 2 roadmap: once paged, no humans in the loop.

Auth posture: webhook handlers accept an `X-Praetor-Token` header. If a
token is configured via the `PRAETOR_WEBHOOK_TOKEN` env var, requests
without a matching header are rejected. If no token is configured (demo
mode), requests are accepted but a warning is added to the response so
operators know to enable auth before pointing real alerts at the URL.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Provider-specific normalizers — each turns its native payload shape into
# our internal `IncidentSignal` dict so the dispatcher doesn't care about
# provider semantics.
# ---------------------------------------------------------------------------


def normalize_pagerduty(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the firing incident from a PagerDuty v3 webhook payload."""
    # PagerDuty v3 wraps everything under `event.data` (incident.triggered)
    event = payload.get("event") or {}
    data = event.get("data") or payload
    incident = data.get("incident") or data
    title = (
        incident.get("title")
        or data.get("title")
        or payload.get("summary")
        or "Unknown PagerDuty alert"
    )
    summary = (
        incident.get("description")
        or data.get("description")
        or title
    )
    service = None
    svc_obj = incident.get("service") or data.get("service")
    if isinstance(svc_obj, dict):
        service = svc_obj.get("summary") or svc_obj.get("name")
    severity = (
        incident.get("priority", {}).get("name")
        if isinstance(incident.get("priority"), dict)
        else (incident.get("urgency") or data.get("severity") or "high")
    )
    return {
        "provider": "pagerduty",
        "title": title,
        "summary": summary,
        "service": service,
        "severity": severity,
        "raw": payload,
    }


def normalize_prometheus(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the firing alert from a Prometheus Alertmanager v4 webhook."""
    # Alertmanager sends {alerts: [{status, labels, annotations, ...}]}
    alerts = payload.get("alerts") or []
    firing = next((a for a in alerts if (a.get("status") or "").lower() == "firing"), None)
    if firing is None and alerts:
        firing = alerts[0]
    if firing is None:
        return {
            "provider": "prometheus",
            "title": "Empty Alertmanager payload",
            "summary": "",
            "service": None,
            "severity": "unknown",
            "raw": payload,
        }
    labels = firing.get("labels") or {}
    annotations = firing.get("annotations") or {}
    return {
        "provider": "prometheus",
        "title": labels.get("alertname") or annotations.get("summary") or "Unknown alert",
        "summary": annotations.get("description") or annotations.get("summary") or "",
        "service": labels.get("service") or labels.get("job") or labels.get("instance"),
        "severity": labels.get("severity") or "warning",
        "raw": payload,
    }


def normalize_generic(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generic contract: {alert: str, service?: str, scenario?: str, severity?: str}."""
    return {
        "provider": "generic",
        "title": payload.get("alert") or payload.get("title") or "Generic incident",
        "summary": payload.get("description") or payload.get("summary") or payload.get("alert") or "",
        "service": payload.get("service"),
        "severity": payload.get("severity") or "warning",
        "scenario_hint": payload.get("scenario"),
        "site_url": payload.get("site_url"),
        "raw": payload,
    }


# ---------------------------------------------------------------------------
# Scenario classification from alert title + summary text
# ---------------------------------------------------------------------------

# Keyword → scenario family. Order matters when keywords overlap; we match
# against a sentinel set of well-disambiguated patterns first.
_CLASSIFIER_RULES: List[Tuple[str, List[str]]] = [
    # Most specific first
    ("cert_expiry", [
        "certificate", "tls handshake", "ssl handshake", "cert expired",
        "cert renewal", "expired certificate",
    ]),
    ("disk_full", [
        "disk full", "no space left", "enospc", "volume full", "disk usage",
        "filesystem full", "out of disk",
    ]),
    ("slow_query", [
        "lock wait", "deadlock", "slow query", "lock contention",
        "for update", "long-running query",
    ]),
    ("oom_crash", [
        "outofmemory", "oom", "out of memory", "memory limit",
        "java heap", "container oomkilled", "exit code 137",
    ]),
    ("db_pool_exhaustion", [
        "pool exhausted", "connection pool", "connection leak",
        "psqlexception", "max connections",
    ]),
    ("bad_deployment_cascade", [
        "bad deploy", "rollout", "memory leak", "autoscaler", "cascading",
        "release v", "v1.1", "deployment", "version bump",
    ]),
]


def classify_scenario(signal: Dict[str, Any]) -> Tuple[Optional[str], float, List[str]]:
    """Heuristic: pick the scenario family from the alert text.

    Returns (scenario_name | None, confidence in [0, 1], evidence list).
    """
    if signal.get("scenario_hint") and signal["scenario_hint"]:
        return signal["scenario_hint"], 1.0, ["scenario explicitly provided in payload"]
    text_parts = [
        str(signal.get("title") or ""),
        str(signal.get("summary") or ""),
        str(signal.get("service") or ""),
    ]
    text = " ".join(text_parts).lower()
    if not text.strip():
        return None, 0.0, []
    best: Optional[str] = None
    best_hits: List[str] = []
    for scenario, keywords in _CLASSIFIER_RULES:
        hits = [kw for kw in keywords if kw in text]
        if hits and (best is None or len(hits) > len(best_hits)):
            best = scenario
            best_hits = hits
    if best is None:
        return None, 0.0, []
    confidence = min(1.0, 0.4 + 0.2 * len(best_hits))
    evidence = [f"matched: {kw!r}" for kw in best_hits[:4]]
    return best, round(confidence, 2), evidence


# ---------------------------------------------------------------------------
# Auth — token check
# ---------------------------------------------------------------------------

def webhook_token_check(provided: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Validate the X-Praetor-Token header.

    Returns (ok, reason_if_not_ok).
    """
    expected = os.environ.get("PRAETOR_WEBHOOK_TOKEN")
    if not expected:
        # Demo mode — accept all but caller will surface a warning.
        return True, None
    if not provided or provided != expected:
        return False, "invalid X-Praetor-Token header"
    return True, None


__all__ = [
    "normalize_pagerduty",
    "normalize_prometheus",
    "normalize_generic",
    "classify_scenario",
    "webhook_token_check",
]
