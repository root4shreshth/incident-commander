from pathlib import Path

from incident_commander_env.server.scenarios.scenario_oom_crash import OOMCrashScenario
from incident_commander_env.server.scenarios.scenario_db_pool import DBPoolScenario
from incident_commander_env.server.scenarios.scenario_bad_deploy import BadDeployScenario
from incident_commander_env.server.scenarios.scenario_disk_full import DiskFullScenario
from incident_commander_env.server.scenarios.scenario_slow_query import SlowQueryScenario
from incident_commander_env.server.scenarios.scenario_cert_expiry import CertExpiryScenario
from incident_commander_env.server.scenarios.yaml_loader import load_yaml_scenarios


# Built-in scenarios (Python classes)
SCENARIO_REGISTRY = {
    "oom_crash": OOMCrashScenario,
    "db_pool_exhaustion": DBPoolScenario,
    "bad_deployment_cascade": BadDeployScenario,
    "disk_full": DiskFullScenario,
    "slow_query": SlowQueryScenario,
    "cert_expiry": CertExpiryScenario,
}

# Phase 2 - load community / YAML-authored scenarios from scenarios/yaml/
# at import time. Failures are non-fatal: a malformed YAML file just gets
# skipped with a warning, the rest of the system keeps running.
_YAML_DIR = Path(__file__).parent / "yaml"
try:
    _yaml_scenarios = load_yaml_scenarios(_YAML_DIR)
    for tid, cls in _yaml_scenarios.items():
        if tid in SCENARIO_REGISTRY:
            print(f"[praetor] yaml scenario {tid!r} skipped - collides with built-in")
            continue
        SCENARIO_REGISTRY[tid] = cls
except Exception as _exc:  # pragma: no cover - defensive
    print(f"[praetor] yaml scenarios skipped: {_exc}")


__all__ = [
    "SCENARIO_REGISTRY",
    "OOMCrashScenario",
    "DBPoolScenario",
    "BadDeployScenario",
    "DiskFullScenario",
    "SlowQueryScenario",
    "CertExpiryScenario",
]
