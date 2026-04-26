"""Crowdsourced scenario library - load scenarios from YAML files.

Closes the "YAML scenario authoring DSL" gap from the Phase 2 roadmap.
A community contributor writes a small YAML file (no Python required),
drops it under `scenarios/yaml/`, and Praetor picks it up at startup
as a new scenario family.

Schema (minimum viable):

    task_id: stripe_outage          # unique scenario id
    difficulty: medium              # easy | medium | hard
    description: "Stripe API timing out for /v1/charges"
    target_service: payment-service
    anomaly: connection_leak        # any anomaly type known to metrics_engine
    max_steps: 25
    alert: "PagerDuty: payment-service timing out on /charges"
    root_cause: "Stripe API is rate-limiting due to a bad retry loop"
    root_cause_keywords: [stripe, rate limit, retry]
    correct_action:
      action_type: restart_service
      target_service: payment-service
    log_lines:
      - "[ERROR] payment-service - Stripe API call timed out after 30s"
      - "[ERROR] payment-service - Retry #4 also failed; backing off"
    rubric:
      - { description: "Identified the timing-out service", weight: 0.3,
          required_action: read_logs, required_target: payment-service }
      - { description: "Restarted the service", weight: 0.7,
          required_action: restart_service, required_target: payment-service }

The loader parses YAML (using a minimal parser if PyYAML isn't installed),
validates it, and synthesizes a `BaseScenario` subclass on the fly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from incident_commander_env.models import ActionRecord
from incident_commander_env.server.scenarios.base_scenario import BaseScenario, RubricCheck
from incident_commander_env.server.simulation.cluster import Cluster
from incident_commander_env.server.simulation.log_generator import normal_logs
from incident_commander_env.server.simulation.metrics_engine import ANOMALY_HANDLERS
from incident_commander_env.server.simulation.service import ServiceHealth


# ---------------------------------------------------------------------------
# YAML parsing - prefer PyYAML when available, fall back to a minimal parser.
# ---------------------------------------------------------------------------

def _parse_yaml(text: str) -> Dict[str, Any]:
    try:
        import yaml  # PyYAML
        return yaml.safe_load(text) or {}
    except ImportError:
        return _minimal_yaml_parse(text)


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    """Tiny subset YAML parser - handles flat key:value, nested dicts, and
    lists of strings/scalars. Sufficient for our scenario schema, but we
    encourage installing PyYAML for anything fancier.
    """
    import re

    def _coerce(value: str) -> Any:
        v = value.strip()
        if not v:
            return ""
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        if v.lower() == "null":
            return None
        # int / float
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        # quoted string
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        # inline list [a, b, c]
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                return []
            return [_coerce(x) for x in _split_csv(inner)]
        # inline dict {k: v, ...}
        if v.startswith("{") and v.endswith("}"):
            inner = v[1:-1].strip()
            d: Dict[str, Any] = {}
            for part in _split_csv(inner):
                if ":" in part:
                    k, vp = part.split(":", 1)
                    d[k.strip()] = _coerce(vp.strip())
            return d
        return v

    def _split_csv(s: str) -> List[str]:
        # comma split that respects nested {} and []
        out, depth, buf = [], 0, []
        for ch in s:
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
            if ch == "," and depth == 0:
                out.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf).strip())
        return out

    # Strip comments (we only support `# comment` outside quoted strings)
    cleaned: List[str] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            cleaned.append(raw)
            continue
        # find '#' not inside quotes
        in_quote = False
        out_chars: List[str] = []
        for ch in raw:
            if ch in ('"', "'"):
                in_quote = not in_quote
            if ch == "#" and not in_quote:
                break
            out_chars.append(ch)
        cleaned.append("".join(out_chars).rstrip())

    result: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(0, result)]  # (indent, container)

    i = 0
    lines = cleaned
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.lstrip(" ")
        # Pop the stack to the right level
        while stack and stack[-1][0] > indent:
            stack.pop()
        container = stack[-1][1] if stack else result
        if stripped.startswith("- "):
            # List item
            item = stripped[2:].strip()
            if isinstance(container, list):
                if ":" in item:
                    # list of dicts inline-ish
                    new_dict: Dict[str, Any] = {}
                    for part in _split_csv(item):
                        if ":" in part:
                            k, v = part.split(":", 1)
                            new_dict[k.strip()] = _coerce(v.strip())
                    container.append(new_dict)
                else:
                    container.append(_coerce(item))
            i += 1
            continue
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                # Look ahead - is the next non-empty line a list ('- ') or another nested key?
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    next_line = lines[j]
                    next_indent = len(next_line) - len(next_line.lstrip(" "))
                    if next_indent > indent:
                        if next_line.lstrip(" ").startswith("- "):
                            new_container: Any = []
                        else:
                            new_container = {}
                        if isinstance(container, dict):
                            container[key] = new_container
                        elif isinstance(container, list):
                            container.append({key: new_container})
                        stack.append((next_indent, new_container))
                        i += 1
                        continue
                if isinstance(container, dict):
                    container[key] = None
            else:
                if isinstance(container, dict):
                    container[key] = _coerce(val)
                elif isinstance(container, list):
                    # treat as inline dict on a list item we already pushed
                    if container and isinstance(container[-1], dict):
                        container[-1][key] = _coerce(val)
        i += 1
    return result


# ---------------------------------------------------------------------------
# Validation + scenario synthesis
# ---------------------------------------------------------------------------

class YAMLScenarioError(ValueError):
    pass


_REQUIRED_FIELDS = {
    "task_id", "difficulty", "description", "target_service",
    "max_steps", "alert", "root_cause", "correct_action",
}


def validate_scenario_dict(d: Dict[str, Any]) -> None:
    missing = _REQUIRED_FIELDS - set(d)
    if missing:
        raise YAMLScenarioError(f"missing required fields: {sorted(missing)}")
    if d["difficulty"] not in ("easy", "medium", "hard"):
        raise YAMLScenarioError(f"difficulty must be easy|medium|hard, got {d['difficulty']!r}")
    anomaly = d.get("anomaly")
    if anomaly and anomaly not in ANOMALY_HANDLERS:
        raise YAMLScenarioError(
            f"unknown anomaly {anomaly!r}; must be one of {sorted(ANOMALY_HANDLERS)}"
        )
    correct = d["correct_action"]
    if not isinstance(correct, dict) or "action_type" not in correct:
        raise YAMLScenarioError("correct_action must be a dict with at least 'action_type'")


def build_scenario_class(spec: Dict[str, Any]) -> type:
    """Build a BaseScenario subclass from a validated YAML spec.

    Methods are bound to the class via `type()` at construction time so that
    Python's ABC machinery sees them and doesn't flag the class as abstract.
    Assigning methods after `class X(BaseScenario): pass` does NOT clear the
    abstract-method set - `type()` does.
    """
    validate_scenario_dict(spec)
    spec = dict(spec)  # defensive copy
    task_id = spec["task_id"]
    target = spec["target_service"]
    anomaly = spec.get("anomaly")
    correct = spec["correct_action"]
    log_lines = spec.get("log_lines") or []
    rubric_specs = spec.get("rubric") or []
    base_max_steps = int(spec["max_steps"])

    def _init(self, seed: Optional[int] = None, difficulty: float = 0.5) -> None:
        self.target_service = target
        self.relevant_services = {target}
        self.max_steps = max(8, int(base_max_steps *
                                    (1.5 - max(0.0, min(1.0, difficulty)))))

    def _setup(self, cluster: Cluster) -> None:
        svc = cluster.get_service(target)
        if not svc:
            return
        if anomaly:
            svc.set_anomaly(anomaly)
            handler = ANOMALY_HANDLERS.get(anomaly)
            if handler:
                try:
                    handler(svc)
                except TypeError:
                    handler(svc, rng=None)
        if log_lines:
            svc.add_logs(list(log_lines))
        for name, other in cluster.services.items():
            if name != target:
                other.add_logs(normal_logs(name, count=6))

    def _check_resolved(self, cluster: Cluster) -> bool:
        svc = cluster.get_service(target)
        if not svc:
            return False
        return svc.health == ServiceHealth.HEALTHY and (
            anomaly is None or anomaly not in svc._anomalies
        )

    def _check_action_match(self, actions: List[ActionRecord], rs: Dict[str, Any]) -> bool:
        rt = rs.get("required_target")
        ra = rs.get("required_action")
        if not ra:
            return False
        for a in actions:
            if a.action_type != ra:
                continue
            if rt and a.target_service != rt:
                continue
            return True
        return False

    def _get_rubric(self) -> List[Tuple[str, RubricCheck, float]]:
        if not rubric_specs:
            def did_investigate(actions, cluster):
                return any(a.action_type in ("read_logs", "check_metrics", "describe_service")
                           for a in actions)

            def did_correct(actions, cluster):
                return self._check_action_match(actions, {
                    "required_action": correct["action_type"],
                    "required_target": correct.get("target_service") or target,
                })
            return [
                ("Investigated the failing service", did_investigate, 0.5),
                ("Took the correct remediation", did_correct, 0.5),
            ]
        out: List[Tuple[str, RubricCheck, float]] = []
        for rs in rubric_specs:
            if not isinstance(rs, dict):
                continue
            desc = rs.get("description") or rs.get("desc") or "rubric criterion"
            weight = float(rs.get("weight") or 0)
            spec_for_check = dict(rs)
            def _make(spec_for_check=spec_for_check):
                def _check(actions, cluster):
                    return self._check_action_match(actions, spec_for_check)
                return _check
            out.append((desc, _make(), weight))
        return out

    def _is_correct_op(self, action, cluster):
        if action.action_type != correct["action_type"]:
            return False
        wanted_target = correct.get("target_service") or target
        if action.target_service != wanted_target:
            return False
        return True

    def _compute_penalties(self, actions: List[ActionRecord], cluster: Cluster) -> float:
        penalty = 0.0
        for a in actions:
            if (
                a.action_type == "restart_service" and a.target_service
                and a.target_service != target
            ):
                penalty -= 0.05
        return penalty

    cls_dict = {
        "task_id": task_id,
        "difficulty": spec["difficulty"],
        "description": spec["description"],
        "alert_message": spec["alert"],
        "root_cause": spec["root_cause"],
        "root_cause_keywords": list(spec.get("root_cause_keywords") or []),
        "relevant_services": {target},
        "max_steps": base_max_steps,
        "__init__": _init,
        "setup": _setup,
        "check_resolved": _check_resolved,
        "_check_action_match": _check_action_match,
        "get_rubric": _get_rubric,
        "is_correct_op": _is_correct_op,
        "compute_penalties": _compute_penalties,
    }
    return type(f"YAMLScenario_{task_id}", (BaseScenario,), cls_dict)


def load_yaml_scenarios(yaml_dir: Path) -> Dict[str, type]:
    """Load every *.yaml/*.yml under `yaml_dir` and return {task_id: class}."""
    yaml_dir = Path(yaml_dir)
    out: Dict[str, type] = {}
    if not yaml_dir.exists():
        return out
    for path in sorted(list(yaml_dir.glob("*.yaml")) + list(yaml_dir.glob("*.yml"))):
        try:
            text = path.read_text(encoding="utf-8")
            spec = _parse_yaml(text)
            if not isinstance(spec, dict):
                continue
            cls = build_scenario_class(spec)
            out[cls.task_id] = cls
        except YAMLScenarioError as exc:
            print(f"[praetor] yaml scenario {path.name} skipped: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[praetor] yaml scenario {path.name} failed to load: {exc}")
    return out


__all__ = [
    "YAMLScenarioError",
    "validate_scenario_dict",
    "build_scenario_class",
    "load_yaml_scenarios",
]
