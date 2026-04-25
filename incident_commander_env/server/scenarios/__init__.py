from incident_commander_env.server.scenarios.scenario_oom_crash import OOMCrashScenario
from incident_commander_env.server.scenarios.scenario_db_pool import DBPoolScenario
from incident_commander_env.server.scenarios.scenario_bad_deploy import BadDeployScenario
from incident_commander_env.server.scenarios.scenario_disk_full import DiskFullScenario
from incident_commander_env.server.scenarios.scenario_slow_query import SlowQueryScenario
from incident_commander_env.server.scenarios.scenario_cert_expiry import CertExpiryScenario

SCENARIO_REGISTRY = {
    "oom_crash": OOMCrashScenario,
    "db_pool_exhaustion": DBPoolScenario,
    "bad_deployment_cascade": BadDeployScenario,
    "disk_full": DiskFullScenario,
    "slow_query": SlowQueryScenario,
    "cert_expiry": CertExpiryScenario,
}

__all__ = [
    "SCENARIO_REGISTRY",
    "OOMCrashScenario",
    "DBPoolScenario",
    "BadDeployScenario",
    "DiskFullScenario",
    "SlowQueryScenario",
    "CertExpiryScenario",
]
