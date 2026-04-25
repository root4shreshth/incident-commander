"""Backend dispatch for IncidentCommanderEnv.

The env can run against several substrates with the same OpenEnv API:

  * SimulatedBackend   — fast in-memory Python cluster (used for training)
  * WebsiteBackend     — HTTP client for any deployed site exposing the
                         operator contract (sim-to-real demo, Phase 3)
  * RealBackend        — Docker Compose wrapper (legacy; kept for parity)
  * CodeAwareBackend   — reserved for Phase 2 post-hackathon roadmap

`get_backend()` reads the `BACKEND` env var and returns the matching instance.
Default is `sim`, which is what the HF Space runs (no Docker, no external
HTTP target required).
"""

from __future__ import annotations

import os
from typing import Optional

from incident_commander_env.server.backends.protocol import (
    Backend,
    BackendSnapshot,
    ServiceSnapshot,
    QuotaSnapshot,
)
from incident_commander_env.server.backends.real import RealBackend
from incident_commander_env.server.backends.sim import SimulatedBackend
from incident_commander_env.server.backends.website import WebsiteBackend


def get_backend(name: Optional[str] = None) -> Backend:
    """Return a backend instance keyed off the `BACKEND` env var (default `sim`).

    Args:
        name: explicit backend name; if omitted, reads `os.environ['BACKEND']`.

    Raises:
        ValueError if the requested name is unknown.
    """
    chosen = (name or os.getenv("BACKEND") or "sim").lower()
    if chosen == "sim":
        return SimulatedBackend()
    if chosen == "website":
        return WebsiteBackend(site_url=os.getenv("SITE_URL"))
    if chosen == "real":
        compose_root = os.getenv("COMPOSE_ROOT", "./targets/site")
        return RealBackend(compose_root=compose_root)
    if chosen == "code_aware":
        raise ValueError(
            "BACKEND=code_aware is reserved for the post-hackathon code-aware "
            "expansion and is not implemented in the current submission."
        )
    raise ValueError(f"Unknown BACKEND={chosen}. Expected one of: sim, website, real, code_aware.")


__all__ = [
    "Backend",
    "BackendSnapshot",
    "ServiceSnapshot",
    "QuotaSnapshot",
    "SimulatedBackend",
    "WebsiteBackend",
    "RealBackend",
    "get_backend",
]
