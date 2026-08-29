"""Tests for the Playbook Verifier runner + reports.

Covers:
  - Runner triggers correctly on claimed scenarios and NOT on unclaimed ones
  - Verdict matrix produces the expected label for each (claimed, triggered, resolved) triple
  - Deterministic: same policy + same seed always yields the same report
  - Reports render in all three formats (terminal, markdown, json) without crashing
  - The bad-example policy is correctly rejected (FAIL) - regression guard
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from praetor_verify import (
    Policy,
    PolicyAction,
    PolicyLoadError,
    PolicyTrigger,
    PolicyTriggerMatch,
    load_policy,
    verify_policy,
)
from praetor_verify.report import format_json, format_markdown, format_terminal


POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def well_scoped_oom_policy() -> Policy:
    """Auto-restart on CRITICAL DOWN alert. Claims and fixes oom_crash."""
    return Policy(
        name="well_scoped_oom",
        version="1.0.0",
        scenarios_claimed=["oom_crash"],
        trigger=PolicyTrigger(match=PolicyTriggerMatch(
            alert_severity="CRITICAL",
            message_contains=["is DOWN", "CRASHED"],
        )),
        actions=[
            PolicyAction(action_type="restart_service",
                         target_service="{trigger.service}",
                         parameters={"memory_limit": "2048Mi"}),
        ],
    )


@pytest.fixture
def over_broad_policy() -> Policy:
    """Restarts payment-service on ANY alert - should misfire everywhere."""
    return Policy(
        name="over_broad",
        version="0.0.1",
        scenarios_claimed=["oom_crash"],
        trigger=PolicyTrigger(match=PolicyTriggerMatch()),  # empty = always fires
        actions=[PolicyAction(action_type="restart_service", target_service="payment-service")],
    )


# ---------------------------------------------------------------------------
# Runner behaviour
# ---------------------------------------------------------------------------

class TestRunnerBasics:
    def test_verify_returns_report_for_every_registered_scenario(self, well_scoped_oom_policy):
        report = verify_policy(well_scoped_oom_policy)
        # >= 12 scenarios (7 built-in Python + 5 YAML; more if extended)
        assert report.total_scenarios >= 12
        for r in report.results:
            assert r.scenario in {res.scenario for res in report.results}
            assert r.verdict in ("PASS", "FAIL", "WARN")

    def test_deterministic_reproducibility(self, well_scoped_oom_policy):
        r1 = verify_policy(well_scoped_oom_policy, seed=9000)
        r2 = verify_policy(well_scoped_oom_policy, seed=9000)
        for a, b in zip(r1.results, r2.results):
            assert a.verdict == b.verdict
            assert a.triggered == b.triggered
            assert a.resolved == b.resolved
            assert a.steps_taken == b.steps_taken
            assert abs(a.total_reward - b.total_reward) < 1e-9

    def test_subset_of_scenarios(self, well_scoped_oom_policy):
        report = verify_policy(well_scoped_oom_policy, scenarios=["oom_crash", "cert_expiry"])
        assert report.total_scenarios == 2
        assert {r.scenario for r in report.results} == {"oom_crash", "cert_expiry"}


class TestVerdictMatrix:
    def test_well_scoped_policy_passes_all(self, well_scoped_oom_policy):
        """A tight policy that fixes oom_crash and doesn't false-positive elsewhere -> all PASS."""
        report = verify_policy(well_scoped_oom_policy)
        # Every scenario should PASS: the claimed one because it resolves,
        # the unclaimed ones because the CRITICAL + DOWN filter is strict.
        for r in report.results:
            assert r.verdict == "PASS", (
                f"expected PASS on {r.scenario}, got {r.verdict} ({r.failure_mode})"
            )
        assert report.overall_verdict == "PASS"

    def test_over_broad_policy_fails(self, over_broad_policy):
        """An always-triggering policy misfires everywhere -> overall FAIL."""
        report = verify_policy(over_broad_policy)
        assert report.overall_verdict == "FAIL"
        # Every scenario should have TRIGGERED because the match is empty.
        assert all(r.triggered for r in report.results), (
            "over_broad policy should trigger on every scenario"
        )

    def test_claimed_but_not_triggered_is_fail(self):
        policy = Policy(
            name="claims_but_wont_match",
            scenarios_claimed=["oom_crash"],
            trigger=PolicyTrigger(match=PolicyTriggerMatch(
                message_contains=["this-string-appears-in-no-alert-anywhere-xxx"],
            )),
            actions=[PolicyAction(action_type="restart_service", target_service="payment-service")],
        )
        report = verify_policy(policy)
        oom_result = next(r for r in report.results if r.scenario == "oom_crash")
        assert oom_result.verdict == "FAIL"
        assert oom_result.failure_mode == "claimed_but_not_triggered"


class TestSafeguards:
    def test_confirmation_required_is_recorded(self):
        from praetor_verify.policy import PolicySafeguards, RequireConfirmation
        policy = Policy(
            name="requires_confirm_test",
            scenarios_claimed=["refund_race_deadlock"],
            trigger=PolicyTrigger(match=PolicyTriggerMatch(
                alert_severity="CRITICAL",
                message_contains=["refund-service"],
            )),
            actions=[
                PolicyAction(action_type="rollback_deployment",
                             target_service="refund-service",
                             parameters={"to_version": "v3.2.0"}),
            ],
            safeguards=PolicySafeguards(
                require_confirmation_if=RequireConfirmation(action_types=["rollback_deployment"]),
            ),
        )
        report = verify_policy(policy)
        rr = next(r for r in report.results if r.scenario == "refund_race_deadlock")
        if rr.triggered:
            # The rollback action needed confirmation - runner records but still runs it
            assert any("rollback_deployment" in c for c in rr.confirmation_required)


# ---------------------------------------------------------------------------
# Report formatters
# ---------------------------------------------------------------------------

class TestReportFormatters:
    def test_terminal_format_runs(self, well_scoped_oom_policy):
        report = verify_policy(well_scoped_oom_policy)
        out = format_terminal(report, colour=False)
        assert "Praetor Playbook Verifier" in out
        assert well_scoped_oom_policy.name in out
        assert "Overall:" in out

    def test_terminal_format_with_colour(self, well_scoped_oom_policy):
        report = verify_policy(well_scoped_oom_policy)
        out = format_terminal(report, colour=True)
        assert "\x1b[" in out  # ANSI escapes present

    def test_markdown_format_runs(self, well_scoped_oom_policy):
        report = verify_policy(well_scoped_oom_policy)
        out = format_markdown(report)
        assert out.startswith("## Praetor Playbook Verifier")
        assert "| Scenario | Verdict |" in out

    def test_json_format_is_parseable(self, well_scoped_oom_policy):
        report = verify_policy(well_scoped_oom_policy)
        out = format_json(report)
        parsed = json.loads(out)
        assert parsed["policy_name"] == well_scoped_oom_policy.name
        assert "overall_verdict" in parsed
        assert "results" in parsed
        assert len(parsed["results"]) == report.total_scenarios


# ---------------------------------------------------------------------------
# Shipped example policies
# ---------------------------------------------------------------------------

class TestShippedPolicies:
    def test_oom_auto_restart_passes(self):
        """The flagship example policy should PASS on all 12 scenarios."""
        policy = load_policy(POLICY_DIR / "oom_auto_restart.yaml")
        report = verify_policy(policy)
        assert report.overall_verdict in ("PASS", "WARN"), (
            f"oom_auto_restart went FAIL - policy or scenario regression? "
            f"{[r.failure_mode for r in report.failing]}"
        )

    def test_webhook_backlog_drain_passes(self):
        policy = load_policy(POLICY_DIR / "webhook_backlog_drain.yaml")
        report = verify_policy(policy)
        assert report.overall_verdict in ("PASS", "WARN")

    def test_bad_example_policy_fails(self):
        """Regression guard on the verifier itself: the bad example MUST fail."""
        policy = load_policy(POLICY_DIR / "_bad_example_trigger_happy.yaml")
        report = verify_policy(policy)
        assert report.overall_verdict == "FAIL", (
            "Verifier regression: bad-example policy should FAIL but produced "
            f"{report.overall_verdict}. False-positive detection is broken."
        )
        # Should have at least one false-positive
        false_pos_modes = {"false_positive", "false_positive_negative_reward"}
        assert any(r.failure_mode in false_pos_modes for r in report.failing), (
            f"expected at least one false_positive verdict; got failure modes: "
            f"{[r.failure_mode for r in report.failing]}"
        )
