#!/usr/bin/env python
"""praetor verify — Playbook Verifier CLI.

Usage:
    python scripts/verify_policy.py <policy.yaml> [<policy2.yaml> ...] [options]

Options:
    --format {terminal,markdown,json}   Output format (default: terminal)
    --output <path>                      Write to file instead of stdout
    --scenarios <task_id,task_id,...>    Subset of scenarios to test against
    --seed <int>                         RNG seed (default: 9000)
    --no-color                           Disable ANSI colours in terminal output
    --strict                             Exit 1 even if only WARN (no FAIL)

Exit codes:
    0   All policies PASS (WARN allowed unless --strict)
    1   At least one policy FAILed
    2   Policy YAML failed to load / validate

Examples:

    python scripts/verify_policy.py policies/oom_auto_restart.yaml
    python scripts/verify_policy.py policies/*.yaml --format json > report.json
    python scripts/verify_policy.py my_policy.yaml --format markdown \\
        --output pr_comment.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from praetor_verify import PolicyLoadError, load_policy, verify_policy
from praetor_verify.report import format_json, format_markdown, format_terminal
from praetor_verify.runner import PolicyReport


EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_LOAD_ERROR = 2


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="praetor verify",
        description="Verify auto-remediation policies against the 12-scenario Praetor library.",
    )
    parser.add_argument(
        "policies", nargs="+", type=Path,
        help="Policy YAML file(s) to verify. Globs expanded by the shell.",
    )
    parser.add_argument(
        "--format", choices=("terminal", "markdown", "json"),
        default="terminal", help="Output format (default: terminal).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write to file instead of stdout.",
    )
    parser.add_argument(
        "--scenarios", type=str, default=None,
        help="Comma-separated task_ids; default is all 12 scenarios.",
    )
    parser.add_argument(
        "--seed", type=int, default=9000,
        help="RNG seed for scenario reset (default: 9000).",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colours in terminal output.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 1 if any policy has WARN verdicts (not just FAIL).",
    )
    return parser.parse_args(argv)


def _load_or_die(path: Path) -> Optional[object]:
    """Load a policy YAML; print an error and return None on failure."""
    try:
        return load_policy(path)
    except PolicyLoadError as exc:
        print(f"[praetor verify] load error in {path}:", file=sys.stderr)
        print(f"    {exc}", file=sys.stderr)
        return None


def _render(report: PolicyReport, fmt: str, use_color: bool) -> str:
    if fmt == "markdown":
        return format_markdown(report)
    if fmt == "json":
        return format_json(report)
    return format_terminal(report, colour=use_color)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    scenario_subset = None
    if args.scenarios:
        scenario_subset = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    # Load all policies up-front so a single malformed file doesn't waste
    # the runtime of the earlier ones.
    policies = []
    for path in args.policies:
        policy = _load_or_die(path)
        if policy is None:
            return EXIT_LOAD_ERROR
        policies.append((path, policy))

    # Run each policy.
    reports: List[PolicyReport] = []
    for path, policy in policies:
        report = verify_policy(policy, scenarios=scenario_subset, seed=args.seed)
        reports.append(report)

    # Render.
    chunks: List[str] = []
    for (path, _policy), report in zip(policies, reports):
        chunks.append(f"# {path.name}\n" if args.format == "markdown" and len(reports) > 1 else "")
        chunks.append(_render(report, args.format, use_color=not args.no_color))
        chunks.append("")

    output = "\n".join(c for c in chunks if c is not None)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"[praetor verify] wrote {args.output}", file=sys.stderr)
    else:
        print(output)

    # Exit code.
    any_fail = any(r.overall_verdict == "FAIL" for r in reports)
    any_warn = any(r.overall_verdict == "WARN" for r in reports)
    if any_fail or (args.strict and any_warn):
        return EXIT_FAIL
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
