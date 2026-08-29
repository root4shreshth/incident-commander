"""Report formatters - terminal, markdown, JSON.

Three output modes so the same PolicyReport can drive both a developer's
terminal and a CI system's PR comment:

  - `format_terminal(report)` - ANSI-coloured single-page summary
  - `format_markdown(report)` - GitHub-flavoured Markdown (for PR comments)
  - `format_json(report)`     - stable machine-readable payload

All three are pure functions over a PolicyReport.
"""

from __future__ import annotations

import json
from typing import Optional

from praetor_verify.runner import PolicyReport, PolicyRunResult


_VERDICT_TO_TERM = {
    "PASS": ("\x1b[32m", "PASS"),   # green
    "FAIL": ("\x1b[31m", "FAIL"),   # red
    "WARN": ("\x1b[33m", "WARN"),   # yellow
}
_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"


def format_terminal(report: PolicyReport, *, colour: bool = True) -> str:
    """Human-readable single-page summary. ANSI-coloured by default."""
    lines: list[str] = []

    def c(text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if colour else text

    header = f"Praetor Playbook Verifier | {report.policy_name} v{report.policy_version}"
    lines.append(c(header, _BOLD))
    lines.append("=" * min(len(header), 80))

    if report.scenarios_claimed:
        lines.append(f"Claims to fix: {', '.join(report.scenarios_claimed)}")
    lines.append("")

    for r in report.results:
        colour_code, verdict_text = _VERDICT_TO_TERM[r.verdict]
        badge = c(f"[{verdict_text:>4s}]", colour_code)
        trigger_mark = "trig" if r.triggered else "----"
        resolve_mark = "res " if r.resolved else "----"
        claimed_mark = "C" if r.claimed else "-"
        r_str = f"R={r.total_reward:+.2f}"
        line = (
            f"  {badge}  {r.scenario:<28s}  "
            f"{claimed_mark} {trigger_mark} {resolve_mark}  "
            f"steps={r.steps_taken:<2d}  {r_str}"
        )
        if r.failure_mode:
            line += c(f"  ({r.failure_mode})", "\x1b[2m")
        lines.append(line)
        if r.error:
            lines.append(f"           error: {r.error}")
        if r.confirmation_required:
            lines.append(
                "           needs-confirmation: "
                + ", ".join(r.confirmation_required)
            )

    lines.append("")
    lines.append("-" * 60)
    verdict_colour, verdict_text = _VERDICT_TO_TERM[report.overall_verdict]
    summary = (
        f"Overall: {c(verdict_text, verdict_colour + _BOLD)}   "
        f"pass={len(report.passing)}  warn={len(report.warning)}  fail={len(report.failing)}   "
        f"({report.pass_rate * 100:.0f}% pass rate on {report.total_scenarios} scenarios)"
    )
    lines.append(summary)

    if report.failing:
        lines.append("")
        lines.append(c("Fix these before merging:", _BOLD))
        for r in report.failing:
            lines.append(f"  * {r.scenario}: {r.failure_mode}")

    return "\n".join(lines)


def format_markdown(report: PolicyReport) -> str:
    """GitHub-flavoured Markdown. Designed to render cleanly as a PR comment."""
    verdict_emoji = {"PASS": "OK", "FAIL": "FAIL", "WARN": "WARN"}
    lines: list[str] = []
    lines.append(f"## Praetor Playbook Verifier — `{report.policy_name}` v{report.policy_version}")
    lines.append("")
    lines.append(
        f"**Verdict:** `{report.overall_verdict}` — "
        f"{len(report.passing)} pass / {len(report.warning)} warn / "
        f"{len(report.failing)} fail across {report.total_scenarios} scenarios "
        f"({report.pass_rate * 100:.0f}% pass rate)."
    )
    if report.scenarios_claimed:
        claimed = ", ".join(f"`{s}`" for s in report.scenarios_claimed)
        lines.append("")
        lines.append(f"**Claims to handle:** {claimed}")
    lines.append("")

    lines.append("| Scenario | Verdict | Claimed | Triggered | Resolved | Steps | Reward | Notes |")
    lines.append("|---|:-:|:-:|:-:|:-:|---:|---:|---|")

    for r in report.results:
        badge = verdict_emoji[r.verdict]
        claimed = "yes" if r.claimed else "-"
        triggered = "yes" if r.triggered else "-"
        resolved = "yes" if r.resolved else "-"
        notes_bits = []
        if r.failure_mode:
            notes_bits.append(f"`{r.failure_mode}`")
        if r.confirmation_required:
            notes_bits.append(f"needs {len(r.confirmation_required)} human confirm(s)")
        if r.matched_service:
            notes_bits.append(f"matched `{r.matched_service}`")
        notes = "; ".join(notes_bits) or "-"
        lines.append(
            f"| `{r.scenario}` | **{badge}** | {claimed} | {triggered} | "
            f"{resolved} | {r.steps_taken} | {r.total_reward:+.2f} | {notes} |"
        )

    if report.failing:
        lines.append("")
        lines.append("### Fix before merging")
        for r in report.failing:
            reason = r.failure_mode or "unknown"
            lines.append(f"- **`{r.scenario}`** — `{reason}`")
            if r.error:
                lines.append(f"  - error: `{r.error}`")

    lines.append("")
    lines.append(
        "<sub>Generated by `praetor verify`. "
        "See [PROJECT.md](PROJECT.md) for how the 12-scenario library grades policies.</sub>"
    )
    return "\n".join(lines)


def format_json(report: PolicyReport) -> str:
    """Stable JSON payload for CI systems to parse."""
    payload = {
        "policy_name": report.policy_name,
        "policy_version": report.policy_version,
        "overall_verdict": report.overall_verdict,
        "scenarios_claimed": report.scenarios_claimed,
        "counts": {
            "total": report.total_scenarios,
            "pass": len(report.passing),
            "warn": len(report.warning),
            "fail": len(report.failing),
        },
        "pass_rate": report.pass_rate,
        "results": [
            {
                "scenario": r.scenario,
                "verdict": r.verdict,
                "claimed": r.claimed,
                "triggered": r.triggered,
                "resolved": r.resolved,
                "steps_taken": r.steps_taken,
                "total_reward": r.total_reward,
                "reward_breakdown": r.reward_breakdown,
                "failure_mode": r.failure_mode,
                "confirmation_required": r.confirmation_required,
                "matched_service": r.matched_service,
                "error": r.error,
            }
            for r in report.results
        ],
    }
    return json.dumps(payload, indent=2)


__all__ = ["format_terminal", "format_markdown", "format_json"]
