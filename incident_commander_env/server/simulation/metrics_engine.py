"""Realistic metric generation with anomaly injection for simulated services.

Every stochastic call here threads through an `rng: random.Random` argument
rather than the global `random` module. This is the precondition for
reproducible RL training: same seed + same action history -> identical
metrics, identical observations, identical reward.

The legacy callable signatures `apply_*_anomaly(service)` continue to work
(rng defaults to a fresh `random.Random()`), but the env always passes its
cluster's seeded rng through `apply_anomalies(service, rng=...)`.
"""

from __future__ import annotations

import random
from typing import Optional

from incident_commander_env.server.simulation.service import Service, ServiceHealth, ServiceMetrics


def _rng(rng: Optional[random.Random] = None) -> random.Random:
    """Return the rng or a fresh fallback so legacy unit tests still work."""
    return rng if rng is not None else random.Random()


def apply_healthy_baseline(service: Service, rng: Optional[random.Random] = None) -> None:
    """Set metrics to normal healthy baseline with slight noise."""
    r = _rng(rng)
    m = service.metrics
    m.cpu_percent = 10.0 + r.uniform(-3, 8)
    m.memory_mb = m.memory_limit_mb * r.uniform(0.20, 0.35)
    m.request_latency_p50_ms = r.uniform(8, 20)
    m.request_latency_p99_ms = r.uniform(30, 60)
    m.error_rate_percent = r.uniform(0.0, 0.5)
    m.active_connections = r.randint(5, 25)
    m.requests_per_second = r.uniform(80, 200)


def apply_oom_anomaly(service: Service, rng: Optional[random.Random] = None) -> None:
    """Set metrics for an OOM-crashed service. Deterministic - no rng usage."""
    service.health = ServiceHealth.CRASHED
    m = service.metrics
    m.cpu_percent = 0.0
    m.memory_mb = service.config.memory_limit_mb() * 1.15
    m.request_latency_p50_ms = 0.0
    m.request_latency_p99_ms = 0.0
    m.error_rate_percent = 100.0
    m.active_connections = 0
    m.requests_per_second = 0.0


def apply_db_pool_exhaustion(
    service: Service, pool_size: int = 20, rng: Optional[random.Random] = None
) -> None:
    """Set metrics for a DB with exhausted connection pool."""
    r = _rng(rng)
    service.health = ServiceHealth.DEGRADED
    m = service.metrics
    m.cpu_percent = r.uniform(60, 80)
    m.memory_mb = m.memory_limit_mb * r.uniform(0.55, 0.70)
    m.request_latency_p50_ms = r.uniform(2000, 4000)
    m.request_latency_p99_ms = r.uniform(8000, 15000)
    m.error_rate_percent = r.uniform(40, 60)
    m.active_connections = pool_size
    m.requests_per_second = r.uniform(10, 30)


def apply_connection_leak_anomaly(
    service: Service, rng: Optional[random.Random] = None
) -> None:
    """Set metrics for a service leaking DB connections."""
    r = _rng(rng)
    service.health = ServiceHealth.UNHEALTHY
    m = service.metrics
    m.cpu_percent = r.uniform(45, 65)
    m.memory_mb = m.memory_limit_mb * r.uniform(0.50, 0.65)
    m.request_latency_p50_ms = r.uniform(500, 1500)
    m.request_latency_p99_ms = r.uniform(5000, 10000)
    m.error_rate_percent = r.uniform(30, 50)
    m.active_connections = r.randint(40, 60)
    m.requests_per_second = r.uniform(20, 50)


def apply_cascade_degradation(
    service: Service, rng: Optional[random.Random] = None
) -> None:
    """Set metrics for a service degraded due to upstream failures."""
    r = _rng(rng)
    service.health = ServiceHealth.DEGRADED
    m = service.metrics
    m.cpu_percent = r.uniform(30, 50)
    m.memory_mb = m.memory_limit_mb * r.uniform(0.35, 0.50)
    m.request_latency_p50_ms = r.uniform(200, 800)
    m.request_latency_p99_ms = r.uniform(2000, 5000)
    m.error_rate_percent = r.uniform(15, 40)
    m.active_connections = r.randint(30, 50)
    m.requests_per_second = r.uniform(30, 70)


def apply_memory_leak_anomaly(
    service: Service, rng: Optional[random.Random] = None
) -> None:
    """Set metrics for a service with active memory leak (bad deployment)."""
    r = _rng(rng)
    service.health = ServiceHealth.UNHEALTHY
    m = service.metrics
    m.cpu_percent = r.uniform(70, 90)
    m.memory_mb = m.memory_limit_mb * r.uniform(0.95, 1.10)
    m.request_latency_p50_ms = r.uniform(150, 400)
    m.request_latency_p99_ms = r.uniform(2000, 5000)
    m.error_rate_percent = r.uniform(15, 30)
    m.active_connections = r.randint(40, 80)
    m.requests_per_second = r.uniform(40, 80)


def apply_resource_starved(
    service: Service, rng: Optional[random.Random] = None
) -> None:
    """Set metrics for a service starved of cluster resources."""
    r = _rng(rng)
    service.health = ServiceHealth.DEGRADED
    m = service.metrics
    m.cpu_percent = r.uniform(85, 98)
    m.memory_mb = m.memory_limit_mb * r.uniform(0.80, 0.95)
    m.request_latency_p50_ms = r.uniform(1000, 3000)
    m.request_latency_p99_ms = r.uniform(5000, 10000)
    m.error_rate_percent = r.uniform(20, 40)
    m.active_connections = r.randint(10, 20)
    m.requests_per_second = r.uniform(10, 30)


def apply_disk_full_anomaly(
    service: Service, rng: Optional[random.Random] = None
) -> None:
    """Disk space exhausted. Service is degraded - writes failing, reads ok."""
    r = _rng(rng)
    service.health = ServiceHealth.DEGRADED
    m = service.metrics
    m.cpu_percent = r.uniform(20, 45)
    m.memory_mb = m.memory_limit_mb * r.uniform(0.40, 0.55)
    m.request_latency_p50_ms = r.uniform(80, 180)
    m.request_latency_p99_ms = r.uniform(800, 2500)
    m.error_rate_percent = r.uniform(15, 35)  # writes fail, reads ok → mid error rate
    m.active_connections = r.randint(8, 18)
    m.requests_per_second = r.uniform(20, 50)


def apply_lock_contention_anomaly(
    service: Service, rng: Optional[random.Random] = None
) -> None:
    """Slow query / lock contention. Latency spikes, throughput drops."""
    r = _rng(rng)
    service.health = ServiceHealth.DEGRADED
    m = service.metrics
    m.cpu_percent = r.uniform(50, 75)
    m.memory_mb = m.memory_limit_mb * r.uniform(0.45, 0.65)
    m.request_latency_p50_ms = r.uniform(400, 900)   # massively elevated p50
    m.request_latency_p99_ms = r.uniform(3000, 12000)  # query lock waits dominate p99
    m.error_rate_percent = r.uniform(8, 22)
    m.active_connections = r.randint(15, 20)         # connections held by locked txns
    m.requests_per_second = r.uniform(5, 15)         # throughput collapses


def apply_cert_expired_anomaly(
    service: Service, rng: Optional[random.Random] = None
) -> None:
    """TLS certificate expired. Inbound HTTPS handshakes fail outright."""
    r = _rng(rng)
    service.health = ServiceHealth.UNHEALTHY
    m = service.metrics
    # Service process is fine; clients just can't connect over TLS.
    m.cpu_percent = r.uniform(5, 18)
    m.memory_mb = m.memory_limit_mb * r.uniform(0.25, 0.40)
    m.request_latency_p50_ms = r.uniform(0, 5)        # fails before reaching app code
    m.request_latency_p99_ms = r.uniform(5, 30)
    m.error_rate_percent = r.uniform(85, 99)          # nearly every TLS handshake fails
    m.active_connections = r.randint(0, 3)            # no clients can establish a session
    m.requests_per_second = r.uniform(0, 2)


ANOMALY_HANDLERS = {
    "oom": apply_oom_anomaly,
    "db_pool_exhaustion": apply_db_pool_exhaustion,
    "connection_leak": apply_connection_leak_anomaly,
    "cascade_degradation": apply_cascade_degradation,
    "memory_leak": apply_memory_leak_anomaly,
    "resource_starved": apply_resource_starved,
    "disk_full": apply_disk_full_anomaly,
    "lock_contention": apply_lock_contention_anomaly,
    "cert_expired": apply_cert_expired_anomaly,
}


def apply_anomalies(service: Service, rng: Optional[random.Random] = None) -> None:
    """Apply all active anomalies to a service's metrics. Threads `rng` through."""
    if not service._anomalies:
        apply_healthy_baseline(service, rng=rng)
        return

    for anomaly_type in service._anomalies:
        handler = ANOMALY_HANDLERS.get(anomaly_type)
        if handler:
            # All handlers accept rng kwarg now (oom ignores it; db_pool also takes pool_size)
            try:
                handler(service, rng=rng)
            except TypeError:
                # Backwards compat for any custom handler that hasn't been updated yet
                handler(service)
