"""Reproducibility tests — same seed + same action sequence -> identical observations.

Required by the OpenEnv/Gymnasium contract. Previously the env used the global
`random` module which made trajectories non-reproducible across runs; we now
thread a `random.Random(seed)` through the cluster -> metrics engine.
"""

from __future__ import annotations

import pytest

from incident_commander_env.models import IncidentAction
from incident_commander_env.server.environment import IncidentCommanderEnv


def _run_episode(seed: int, actions: list[tuple[str, str | None, dict]]):
    """Run a fresh env episode with given seed and action sequence; return per-step trace."""
    env = IncidentCommanderEnv()
    env.reset(task_id="oom_crash", seed=seed)
    trace = []
    for action_type, target, params in actions:
        obs = env.step(IncidentAction(
            action_type=action_type,
            target_service=target,
            parameters=params,
        ))
        # Record only the deterministic-after-anomaly fields
        trace.append({
            "reward": round(obs.reward, 6),
            "done": obs.done,
            "message": obs.message,
            # Sample a few metric fields if metrics is present
            "cpu": obs.metrics.cpu_percent if obs.metrics else None,
            "memory_mb": obs.metrics.memory_mb if obs.metrics else None,
            "p99": obs.metrics.request_latency_p99_ms if obs.metrics else None,
        })
    return trace


SAMPLE_TRAJECTORY = [
    ("list_services", None, {}),
    ("read_logs", "auth-service", {"lines": 30}),
    ("check_metrics", "auth-service", {}),
    ("read_logs", "payment-service", {"lines": 50}),
    ("check_metrics", "payment-service", {}),
]


class TestSeedReproducibility:
    def test_same_seed_same_trajectory(self):
        trace_a = _run_episode(seed=42, actions=SAMPLE_TRAJECTORY)
        trace_b = _run_episode(seed=42, actions=SAMPLE_TRAJECTORY)
        assert trace_a == trace_b, (
            f"Seeded episodes diverged.\nA: {trace_a}\nB: {trace_b}"
        )

    def test_different_seeds_can_diverge(self):
        # Two different seeds *may* produce different metrics on healthy services
        # (anomalous services like crashed payment have deterministic metrics).
        # Run a longer trajectory that ticks the cluster a few times so
        # apply_healthy_baseline (the random path) gets exercised.
        trace_a = _run_episode(seed=1, actions=SAMPLE_TRAJECTORY * 3)
        trace_b = _run_episode(seed=999, actions=SAMPLE_TRAJECTORY * 3)
        # Healthy-service metrics use rng.uniform; with two different seeds they should differ
        # somewhere in the trace.
        assert any(a.get("cpu") != b.get("cpu") for a, b in zip(trace_a, trace_b)), (
            "Seeds produced identical traces — rng plumbing isn't actually using the seed"
        )

    def test_no_seed_still_works(self):
        """Backwards-compat: omitting seed is fine, just non-reproducible."""
        env = IncidentCommanderEnv()
        obs = env.reset(task_id="oom_crash")
        assert obs.done is False
        # Episode runs through fine
        for action_type, target, params in SAMPLE_TRAJECTORY:
            obs = env.step(IncidentAction(
                action_type=action_type,
                target_service=target,
                parameters=params,
            ))
        # Just sanity: no exceptions, last observation is well-formed
        assert obs is not None
