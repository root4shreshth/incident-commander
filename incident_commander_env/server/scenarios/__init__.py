from incident_commander_env.server.scenarios.scenario_oom_crash import OOMCrashScenario
from incident_commander_env.server.scenarios.scenario_db_pool import DBPoolScenario
from incident_commander_env.server.scenarios.scenario_bad_deploy import BadDeployScenario

SCENARIO_REGISTRY = {
    "oom_crash": OOMCrashScenario,
    "db_pool_exhaustion": DBPoolScenario,
    "bad_deployment_cascade": BadDeployScenario,
}

__all__ = ["SCENARIO_REGISTRY", "OOMCrashScenario", "DBPoolScenario", "BadDeployScenario"]
