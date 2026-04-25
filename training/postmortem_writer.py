"""Generate structured Markdown post-mortems from episode JSONL traces.

Closes the "writes incident retro, updates runbook" gap from the Phase 2
roadmap. Every completed episode (sim or real-time) gets a postmortem.md
saved alongside its episode.jsonl, and a one-line entry appended to the
project-level RUNBOOK.md so the on-call team has a running ledger.

Format follows the standard SRE post-mortem skeleton:
  - Summary (one paragraph: what broke, what fixed it, how long)
  - Timeline (timestamped action-by-action play-by-play)
  - Root cause (the agent's resolve_incident message)
  - Resolution (what specifically restored health)
  - What went well / what didn't (auto-derived from rewards)
  - Follow-ups (suggested action items + Phase-2 nods)

The writer is dependency-free — pure stdlib — so it runs anywhere the
env runs. The output is plain Markdown, so it renders nicely on GitHub,
in any IDE, or piped to a CLI viewer.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCENARIO_PRETTY = {
    "oom_crash": "Out-of-memory crash",
    "db_pool_exhaustion": "Database connection pool exhaustion",
    "bad_deployment_cascade": "Bad-deployment cascade",
    "disk_full": "Disk space exhausted",
    "slow_query": "Slow query / lock contention",
    "cert_expiry": "TLS certificate expired",
}


def _ts(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_action(action: Dict[str, Any]) -> str:
    """Render an action as e.g. `restart_service api {memory_limit: 1024Mi}`."""
    if not isinstance(action, dict):
        return str(action)
    name = action.get("action_type") or "?"
    target = action.get("target_service")
    params = action.get("parameters") or {}
    parts = [f"`{name}`"]
    if target:
        parts.append(f"`{target}`")
    if params:
        parts.append(f"`{json.dumps(params, separators=(',', ':'))}`")
    return " ".join(parts)


def _summarize_rewards(events: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {
        "diagnostic": 0.0, "correct_op": 0.0, "resolution": 0.0,
        "format": 0.0, "efficiency": 0.0, "penalty": 0.0,
    }
    for ev in events:
        if ev.get("type") != "step":
            continue
        bd = ev.get("reward_breakdown") or {}
        for k in totals:
            try:
                totals[k] += float(bd.get(k, 0) or 0)
            except (TypeError, ValueError):
                pass
    return totals


def _what_went(events: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Auto-derive 'what went well' / 'what didn't' bullets from the trace."""
    well: List[str] = []
    not_well: List[str] = []
    totals = _summarize_rewards(events)
    if totals["diagnostic"] >= 0.05:
        well.append(f"Investigation phase scored well: r_diagnostic = +{totals['diagnostic']:.2f}")
    if totals["correct_op"] >= 0.10:
        well.append(f"Picked the correct remediation: r_correct_op = +{totals['correct_op']:.2f}")
    if totals["resolution"] >= 0.20:
        well.append("Final resolution matched the rubric (root cause + correct fix).")
    if totals["efficiency"] > 0:
        well.append(f"Resolved well within step budget: r_efficiency = +{totals['efficiency']:.2f}")
    if totals["penalty"] < -0.05:
        not_well.append(f"Took penalised actions: r_penalty = {totals['penalty']:.2f} "
                        "(restarted unrelated service, or attempted rollback-to-self).")
    if totals["format"] < 0.05:
        not_well.append("Action-format penalty: at least one action failed to parse cleanly.")
    if totals["resolution"] < 0.05:
        not_well.append("Final resolve_incident did not match the scenario's root_cause keywords.")
    if not well:
        well.append("Episode produced no meaningfully positive signal.")
    if not not_well:
        not_well.append("No notable failure modes detected in this run.")
    return {"well": well, "not_well": not_well}


def _suggested_followups(scenario: str, totals: Dict[str, float]) -> List[str]:
    """Per-scenario follow-up suggestions an SRE team would file as tickets."""
    base: Dict[str, List[str]] = {
        "oom_crash": [
            "Add a bounded-cache profile to the offending service (LRU with a maxsize).",
            "Wire memory_used / memory_limit ratio into the pre-deploy SLO check.",
            "Document the new memory_limit in the service's runbook.",
        ],
        "db_pool_exhaustion": [
            "Audit all `engine.connect()` call sites for missing context-manager use.",
            "Add a connection-pool saturation alert at 80% of `db.pool.max_size`.",
            "Run a connection-lifetime histogram weekly to catch slow leaks.",
        ],
        "bad_deployment_cascade": [
            "Add a canary stage to the deploy pipeline (10% traffic for 30 min).",
            "Make autoscaler events feed the on-call channel for review.",
            "Pin `cluster.resource.quota.memory_mb` headroom alerts at 85%.",
        ],
        "disk_full": [
            "Audit log rotation policy; consider shipping logs to a remote sink.",
            "Add a disk-usage alert at 80% per service mount.",
            "Move ephemeral workdirs onto a tmpfs with explicit bounds.",
        ],
        "slow_query": [
            "Add the offending query to the pre-deploy `EXPLAIN ANALYZE` gate.",
            "Cap `lock_wait_timeout` to a low value so stuck txns fail fast.",
            "Land query timing metrics in the deploy postmortem checklist.",
        ],
        "cert_expiry": [
            "Add a 30-day cert-expiry alert for every fronting cert in the fleet.",
            "Verify cert-renewal hooks fire successfully on staging weekly.",
            "Inventory all TLS certs in a single dashboard.",
        ],
    }
    items = list(base.get(scenario, ["File a generic follow-up ticket for this incident class."]))
    if totals.get("penalty", 0) < -0.05:
        items.append("Tighten the scenario's penalty bookkeeping — agent took actions that didn't help.")
    return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_postmortem_markdown(events: List[Dict[str, Any]]) -> str:
    """Render a Markdown postmortem from a list of JSONL events."""
    start = next((e for e in events if e.get("type") == "start"), {})
    end = next((e for e in reversed(events) if e.get("type") == "end"), {})
    steps = [e for e in events if e.get("type") == "step"]

    task_id = start.get("task_id") or "unknown_scenario"
    pretty = _SCENARIO_PRETTY.get(task_id, task_id.replace("_", " ").title())
    started_at = start.get("ts")
    ended_at = end.get("ts")
    duration_s = (
        (ended_at - started_at) if (started_at and ended_at) else 0.0
    )
    score = float(end.get("score") or 0.0)
    resolved = bool(end.get("resolved"))
    steps_used = int(end.get("steps_used") or len(steps))
    model = start.get("model") or "praetor"
    seed = start.get("seed")
    alert = start.get("alert") or "(alert text not captured)"

    # Find the resolve_incident step if any — its message contains the
    # agent's stated root cause + resolution.
    resolve_msg = ""
    resolve_action: Dict[str, Any] = {}
    for ev in reversed(steps):
        action = ev.get("action") or {}
        if action.get("action_type") == "resolve_incident":
            resolve_msg = (ev.get("message") or "")[:1200]
            resolve_action = action
            break

    totals = _summarize_rewards(events)
    went = _what_went(events)
    followups = _suggested_followups(task_id, totals)

    # Build the document
    lines: List[str] = []
    lines.append(f"# Postmortem: {pretty}")
    lines.append("")
    lines.append(f"**Status:** {'✅ Resolved' if resolved else '⚠️ Unresolved'}  ·  "
                 f"**Score:** {score:.2f} / 1.00  ·  "
                 f"**Steps used:** {steps_used}  ·  "
                 f"**Wall-clock:** {duration_s:.1f}s")
    lines.append("")
    lines.append(f"- **Scenario family:** `{task_id}`")
    lines.append(f"- **Seed:** `{seed if seed is not None else '—'}`")
    lines.append(f"- **Author:** `{model}` (autonomous)")
    lines.append(f"- **Started:** {_ts(started_at)}")
    lines.append(f"- **Resolved:** {_ts(ended_at)}")
    lines.append("")
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(_one_paragraph_summary(task_id, resolved, score, steps_used, duration_s))
    lines.append("")
    # Alert
    lines.append("## Alert")
    lines.append("")
    lines.append(f"> {alert}")
    lines.append("")
    # Root cause + resolution
    if resolve_action.get("parameters"):
        params = resolve_action["parameters"]
        rc = params.get("root_cause") or "(not stated)"
        fix = params.get("resolution") or "(not stated)"
        lines.append("## Root cause")
        lines.append("")
        lines.append(rc)
        lines.append("")
        lines.append("## Resolution")
        lines.append("")
        lines.append(fix)
        lines.append("")
    # Timeline
    lines.append("## Timeline")
    lines.append("")
    lines.append("| # | Action | Reward | Outcome |")
    lines.append("|---|---|---:|---|")
    for ev in steps:
        n = ev.get("step", "?")
        a = ev.get("action") or {}
        bd = ev.get("reward_breakdown") or {}
        total = sum(float(v or 0) for v in bd.values() if isinstance(v, (int, float)))
        msg = (ev.get("message") or "").split("\n")[0][:80]
        lines.append(f"| {n} | {_format_action(a)} | {total:+.2f} | {msg} |")
    lines.append("")
    # Reward decomp
    lines.append("## Reward decomposition")
    lines.append("")
    lines.append("| Component | Total |")
    lines.append("|---|---:|")
    for k in ("diagnostic", "correct_op", "resolution", "format", "efficiency", "penalty"):
        v = totals.get(k, 0.0)
        lines.append(f"| {k} | {v:+.2f} |")
    grand = sum(totals.values())
    lines.append(f"| **Total** | **{grand:+.2f}** |")
    lines.append("")
    # Reflection
    lines.append("## What went well")
    lines.append("")
    for it in went["well"]:
        lines.append(f"- {it}")
    lines.append("")
    lines.append("## What did not")
    lines.append("")
    for it in went["not_well"]:
        lines.append(f"- {it}")
    lines.append("")
    # Follow-ups
    lines.append("## Action items")
    lines.append("")
    for it in followups:
        lines.append(f"- [ ] {it}")
    lines.append("")
    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Generated automatically by Praetor — incident commander for SREs. "
        f"Trace: episode.jsonl in this directory.*"
    )
    lines.append("")
    return "\n".join(lines)


def _one_paragraph_summary(
    task_id: str, resolved: bool, score: float, steps: int, duration_s: float,
) -> str:
    pretty = _SCENARIO_PRETTY.get(task_id, task_id)
    verdict = "fully recovered" if resolved else "partially recovered"
    band = (
        "with a strong score" if score >= 0.7 else
        "with a moderate score" if score >= 0.4 else
        "with a weak score"
    )
    return (
        f"Praetor handled an incident in the *{pretty}* family autonomously. "
        f"The site {verdict} after {steps} typed-action steps over {duration_s:.1f}s, "
        f"{band} ({score:.2f} / 1.00) on the deterministic 6-component rubric. "
        f"Action items below are surfaced from the per-scenario operational checklist."
    )


def write_postmortem(jsonl_path: Path, runbook_path: Optional[Path] = None) -> Path:
    """Read `episode.jsonl`, write `postmortem.md` next to it, append to runbook.

    Returns the path to the written postmortem.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"episode.jsonl not found at {jsonl_path}")
    events: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    md = build_postmortem_markdown(events)
    out_path = jsonl_path.parent / "postmortem.md"
    out_path.write_text(md, encoding="utf-8")
    if runbook_path is not None:
        _append_to_runbook(events, runbook_path, out_path)
    return out_path


def _append_to_runbook(
    events: List[Dict[str, Any]], runbook_path: Path, postmortem_path: Path,
) -> None:
    """Append a one-line entry to RUNBOOK.md for fleet-level traceability."""
    start = next((e for e in events if e.get("type") == "start"), {})
    end = next((e for e in reversed(events) if e.get("type") == "end"), {})
    if not start:
        return
    runbook_path = Path(runbook_path)
    if not runbook_path.exists():
        header = (
            "# Praetor Runbook\n\n"
            "Running ledger of every incident Praetor has handled. One row "
            "per resolved (or unresolved) episode. Click the postmortem "
            "link for the full play-by-play.\n\n"
            "| Time | Scenario | Resolved | Score | Steps | Postmortem |\n"
            "|---|---|---|---:|---:|---|\n"
        )
        runbook_path.parent.mkdir(parents=True, exist_ok=True)
        runbook_path.write_text(header, encoding="utf-8")
    ts = _ts(start.get("ts"))
    task_id = start.get("task_id") or "unknown"
    resolved = "✓" if end.get("resolved") else "✗"
    score = float(end.get("score") or 0.0)
    steps = int(end.get("steps_used") or 0)
    rel_pm = postmortem_path.relative_to(runbook_path.parent) if runbook_path.parent in postmortem_path.parents else postmortem_path
    line = f"| {ts} | `{task_id}` | {resolved} | {score:.2f} | {steps} | [view]({rel_pm}) |\n"
    with runbook_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


__all__ = [
    "build_postmortem_markdown",
    "write_postmortem",
]
