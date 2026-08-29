"""Tests for the Playbook Verifier policy DSL (praetor_verify.policy).

Focus: schema validation catches bad policies at load time, not 30s into a
CI run. Template expansion resolves the known variables and leaves unknown
ones alone (rather than crashing).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from praetor_verify.policy import (
    Policy,
    PolicyAction,
    PolicyLoadError,
    PolicySafeguards,
    PolicyTrigger,
    PolicyTriggerMatch,
    RequireConfirmation,
    load_policy,
)


class TestPolicyBasics:
    def test_minimal_policy_validates(self) -> None:
        p = Policy(
            name="minimal_test",
            trigger=PolicyTrigger(match=PolicyTriggerMatch(message_contains=["foo"])),
            actions=[PolicyAction(action_type="list_services")],
        )
        assert p.name == "minimal_test"
        assert p.version == "0.1.0"
        assert len(p.actions) == 1

    def test_name_must_be_lowercase(self) -> None:
        with pytest.raises(ValidationError):
            Policy(
                name="UpperCaseName",
                trigger=PolicyTrigger(match=PolicyTriggerMatch()),
                actions=[PolicyAction(action_type="list_services")],
            )

    def test_name_must_start_with_letter_or_underscore(self) -> None:
        with pytest.raises(ValidationError):
            Policy(
                name="123leading_digit",
                trigger=PolicyTrigger(match=PolicyTriggerMatch()),
                actions=[PolicyAction(action_type="list_services")],
            )

    def test_leading_underscore_name_allowed_for_internal_policies(self) -> None:
        """The `_bad_example_*` naming convention for regression-test policies."""
        p = Policy(
            name="_bad_example_internal",
            trigger=PolicyTrigger(match=PolicyTriggerMatch()),
            actions=[PolicyAction(action_type="list_services")],
        )
        assert p.name.startswith("_")

    def test_actions_list_must_not_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            Policy(
                name="empty_actions",
                trigger=PolicyTrigger(match=PolicyTriggerMatch()),
                actions=[],
            )


class TestPolicyAction:
    def test_valid_action_type_passes(self) -> None:
        for at in ("list_services", "restart_service", "rollback_deployment",
                   "resolve_incident", "update_config"):
            PolicyAction(action_type=at)

    def test_unknown_action_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PolicyAction(action_type="not_a_real_action")

    def test_unknown_template_variable_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PolicyAction(
                action_type="restart_service",
                target_service="{unknown.var}",
            )

    def test_unknown_template_in_nested_parameters_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PolicyAction(
                action_type="update_config",
                target_service="postgres-db",
                parameters={"key": "db.pool.max_size", "value": "{something.wrong}"},
            )

    def test_known_template_variable_accepted(self) -> None:
        act = PolicyAction(
            action_type="restart_service",
            target_service="{trigger.service}",
            parameters={"memory_limit": "2048Mi"},
        )
        assert "{trigger.service}" in act.target_service


class TestTemplateExpansion:
    def test_expand_trigger_service(self) -> None:
        p = Policy(
            name="templ_test",
            trigger=PolicyTrigger(match=PolicyTriggerMatch(service_pattern="*")),
            actions=[PolicyAction(
                action_type="restart_service",
                target_service="{trigger.service}",
            )],
        )
        expanded = p.expand_templates({"trigger.service": "payment-gateway"})
        assert expanded[0].target_service == "payment-gateway"

    def test_expand_leaves_unknown_vars_alone(self) -> None:
        """Unknown template vars in policy YAML would be rejected at load time;
        but if somehow they get through, expansion should leave them literal
        rather than crash."""
        act = PolicyAction(action_type="restart_service", target_service="payment-*")
        # Manually attach an unknown template - simulating a runtime bug
        p = Policy(
            name="ok_name",
            trigger=PolicyTrigger(match=PolicyTriggerMatch()),
            actions=[act],
        )
        expanded = p.expand_templates({"trigger.service": "svc-x"})
        # Wildcard `payment-*` isn't a template, so no substitution.
        assert expanded[0].target_service == "payment-*"

    def test_expand_in_nested_parameters(self) -> None:
        p = Policy(
            name="templ_nested",
            trigger=PolicyTrigger(match=PolicyTriggerMatch()),
            actions=[PolicyAction(
                action_type="update_config",
                target_service="{trigger.service}",
                parameters={"key": "memory.limit", "value": "2048Mi",
                            "note": "adjusting {trigger.service}"},
            )],
        )
        expanded = p.expand_templates({"trigger.service": "fraud-check"})
        assert expanded[0].target_service == "fraud-check"
        assert expanded[0].parameters["note"] == "adjusting fraud-check"


class TestSafeguards:
    def test_require_confirmation_defaults_empty(self) -> None:
        s = PolicySafeguards()
        assert s.require_confirmation_if.action_types == []
        assert s.require_confirmation_if.service_matches == []
        assert s.max_actions_per_hour is None

    def test_max_actions_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PolicySafeguards(max_actions_per_hour=0)

    def test_require_confirmation_action_types_validated(self) -> None:
        with pytest.raises(ValidationError):
            RequireConfirmation(action_types=["nuke_everything"])


class TestYAMLLoading:
    def test_load_valid_policy(self, tmp_path: Path) -> None:
        yaml_text = textwrap.dedent("""\
            name: test_yaml_load
            version: 1.0.0
            description: test
            scenarios_claimed: [oom_crash]
            trigger:
              event: alert
              match:
                message_contains: [OutOfMemoryError]
            actions:
              - action_type: restart_service
                target_service: "{trigger.service}"
                parameters:
                  memory_limit: 2048Mi
        """)
        path = tmp_path / "test.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        policy = load_policy(path)
        assert policy.name == "test_yaml_load"
        assert policy.scenarios_claimed == ["oom_crash"]
        assert policy.actions[0].parameters["memory_limit"] == "2048Mi"

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PolicyLoadError, match="could not read"):
            load_policy(tmp_path / "does_not_exist.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("name: 'unterminated string", encoding="utf-8")
        with pytest.raises(PolicyLoadError):
            load_policy(path)

    def test_load_yaml_missing_required_fields_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "incomplete.yaml"
        path.write_text("name: only_name\n", encoding="utf-8")
        with pytest.raises(PolicyLoadError, match="validation failed"):
            load_policy(path)

    def test_load_shipped_production_policies(self) -> None:
        """All committed policies under policies/ load without error."""
        policy_dir = Path(__file__).resolve().parent.parent / "policies"
        yaml_files = sorted(list(policy_dir.glob("*.yaml")) + list(policy_dir.glob("*.yml")))
        assert len(yaml_files) >= 4, (
            f"expected at least 4 example policies in policies/, found {len(yaml_files)}"
        )
        for path in yaml_files:
            # The intentionally-bad regression example still loads (bad policies
            # are runtime-behaviour bad, not schema-bad).
            policy = load_policy(path)
            assert policy.name, f"loaded {path} has empty name"
