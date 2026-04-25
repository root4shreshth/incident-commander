"""RealBackend — wraps a Docker Compose stack for the sim-to-real demo.

Skeleton implementation: the Backend Protocol surface is here so the env
can route through it cleanly today. Phase 6 fills in the Docker shell-outs
once the user's vibecoded site is ready.

Until then, RealBackend reports a clean cluster and rejects ops actions
with a typed error so a misconfigured `BACKEND=real` deployment doesn't
silently produce nonsense.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from incident_commander_env.models import IncidentAction, IncidentObservation
from incident_commander_env.server.backends.protocol import (
    BackendSnapshot,
    QuotaSnapshot,
    ServiceSnapshot,
)

if TYPE_CHECKING:
    from incident_commander_env.server.scenarios.base_scenario import BaseScenario


# Default service names the env expects to find in the user's vibecoded compose.
# These must match what `docker-compose.yml` declares.
DEFAULT_REAL_SERVICES = ("frontend", "api", "postgres")


class RealBackend:
    """Docker-Compose-backed real environment.

    Phase 6 fills in:
      - reset(): `docker compose up -d`, run chaos.py
      - execute(restart_service): `docker compose up -d --force-recreate <svc>`
      - execute(read_logs): `docker compose logs <svc> --tail N`
      - execute(check_metrics): `docker stats --format json`
      - execute(rollback_deployment): bump IMAGE_TAG env var, recreate
      - execute(update_config): set per-service env vars, recreate
      - check_resolved(): poll `<svc>/health` for 30s of 200s
      - teardown(): `docker compose down`

    For Phase 2 we ship a stub that lets the env import + boot under
    `BACKEND=real` without crashing, and surfaces a clear error on every
    action so misconfigured deployments are immediately diagnosable.
    """

    name = "real"

    def __init__(
        self,
        compose_root: str = "./targets/site",
        service_names: Optional[List[str]] = None,
    ) -> None:
        self.compose_root = Path(compose_root).resolve()
        self.service_names = list(service_names or DEFAULT_REAL_SERVICES)
        self._reset_done = False

    # ---- Phase 6 implementation slots ----------------------------------------

    def reset(self, scenario: "BaseScenario", seed: Optional[int] = None) -> None:
        # Phase 6: docker compose up -d, chaos inject
        if not self.compose_root.exists():
            # Tolerate missing compose root so the env can still boot for tests
            # and the user's vibecoded site can land later. We just skip the
            # actual Docker work.
            self._reset_done = True
            return
        compose_file = self.compose_root / "docker-compose.yml"
        if not compose_file.exists():
            self._reset_done = True
            return
        try:
            subprocess.run(
                ["docker", "compose", "up", "-d"],
                cwd=self.compose_root,
                check=False,
                timeout=60,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Docker not installed or hung; let downstream surface the issue.
            pass
        self._reset_done = True

    def execute(
        self, action: IncidentAction, scenario: "BaseScenario"
    ) -> IncidentObservation:
        # Phase 6 will populate the per-action shell-outs.
        return IncidentObservation(
            message=(
                f"RealBackend stub: action '{action.action_type}' not implemented yet. "
                "Phase 6 wires this to docker compose commands."
            ),
            error="RealBackend action not implemented in Phase 2 stub",
        )

    def snapshot(self) -> BackendSnapshot:
        # Phase 6: query docker stats / docker compose ps, parse JSON.
        # Phase 2 stub: report all configured services as healthy.
        services = {
            name: ServiceSnapshot(
                name=name,
                health="healthy",
                version="real-stub",
                replicas=1,
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_limit_mb=512.0,
                error_rate_percent=0.0,
            )
            for name in self.service_names
        }
        return BackendSnapshot(services=services, quota=QuotaSnapshot())

    def check_resolved(self, scenario: "BaseScenario") -> bool:
        # Phase 6: poll the configured /health endpoints until 30s of 200s.
        return False

    def tick(self) -> None:
        # Real Docker advances on its own clock; nothing for us to do here.
        pass

    def teardown(self) -> None:
        if not self._reset_done:
            return
        if not self.compose_root.exists():
            return
        try:
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=self.compose_root,
                check=False,
                timeout=30,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass


__all__ = ["RealBackend", "DEFAULT_REAL_SERVICES"]
