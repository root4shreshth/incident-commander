"""Realistic metric generation with anomaly injection for simulated services."""

from __future__ import annotations

import random
from typing import Dict

from incident_commander_env.server.simulation.service import Service, ServiceHealth, ServiceMetrics


def apply_healthy_baseline(service: Service) -> None:
    """Set metrics to normal healthy baseline with slight noise."""
    m = service.metrics
    m.cpu_percent = 10.0 + random.uniform(-3, 8)
    m.memory_mb = m.memory_limit_mb * random.uniform(0.20, 0.35)
    m.request_latency_p50_ms = random.uniform(8, 20)
    m.request_latency_p99_ms = random.uniform(30, 60)
    m.error_rate_percent = random.uniform(0.0, 0.5)
    m.active_connections = random.randint(5, 25)
    m.requests_per_second = random.uniform(80, 200)


def apply_oom_anomaly(service: Service) -> None:
    """Set metrics for an OOM-crashed service."""
    service.health = ServiceHealth.CRASHED
    m = service.metrics
    m.cpu_percent = 0.0
    m.memory_mb = service.config.memory_limit_mb() * 1.15
    m.request_latency_p50_ms = 0.0
    m.request_latency_p99_ms = 0.0
    m.error_rate_percent = 100.0
    m.active_connections = 0
    m.requests_per_second = 0.0


def apply_db_pool_exhaustion(service: Service, pool_size: int = 20) -> None:
    """Set metrics for a DB with exhausted connection pool."""
    service.health = ServiceHealth.DEGRADED
    m = service.metrics
    m.cpu_percent = random.uniform(60, 80)
    m.memory_mb = m.memory_limit_mb * random.uniform(0.55, 0.70)
    m.request_latency_p50_ms = random.uniform(2000, 4000)
    m.request_latency_p99_ms = random.uniform(8000, 15000)
    m.error_rate_percent = random.uniform(40, 60)
    m.active_connections = pool_size
    m.requests_per_second = random.uniform(10, 30)


def apply_connection_leak_anomaly(service: Service) -> None:
    """Set metrics for a service leaking DB connections."""
    service.health = ServiceHealth.UNHEALTHY
    m = service.metrics
    m.cpu_percent = random.uniform(45, 65)
    m.memory_mb = m.memory_limit_mb * random.uniform(0.50, 0.65)
    m.request_latency_p50_ms = random.uniform(500, 1500)
    m.request_latency_p99_ms = random.uniform(5000, 10000)
    m.error_rate_percent = random.uniform(30, 50)
    m.active_connections = random.randint(40, 60)
    m.requests_per_second = random.uniform(20, 50)


def apply_cascade_degradation(service: Service) -> None:
    """Set metrics for a service degraded due to upstream failures."""
    service.health = ServiceHealth.DEGRADED
    m = service.metrics
    m.cpu_percent = random.uniform(30, 50)
    m.memory_mb = m.memory_limit_mb * random.uniform(0.35, 0.50)
    m.request_latency_p50_ms = random.uniform(200, 800)
    m.request_latency_p99_ms = random.uniform(2000, 5000)
    m.error_rate_percent = random.uniform(15, 40)
    m.active_connections = random.randint(30, 50)
    m.requests_per_second = random.uniform(30, 70)


def apply_memory_leak_anomaly(service: Service) -> None:
    """Set metrics for a service with active memory leak (bad deployment)."""
    service.health = ServiceHealth.UNHEALTHY
    m = service.metrics
    m.cpu_percent = random.uniform(70, 90)
    m.memory_mb = m.memory_limit_mb * random.uniform(0.95, 1.10)
    m.request_latency_p50_ms = random.uniform(150, 400)
    m.request_latency_p99_ms = random.uniform(2000, 5000)
    m.error_rate_percent = random.uniform(15, 30)
    m.active_connections = random.randint(40, 80)
    m.requests_per_second = random.uniform(40, 80)


def apply_resource_starved(service: Service) -> None:
    """Set metrics for a service starved of cluster resources."""
    service.health = ServiceHealth.DEGRADED
    m = service.metrics
    m.cpu_percent = random.uniform(85, 98)
    m.memory_mb = m.memory_limit_mb * random.uniform(0.80, 0.95)
    m.request_latency_p50_ms = random.uniform(1000, 3000)
    m.request_latency_p99_ms = random.uniform(5000, 10000)
    m.error_rate_percent = random.uniform(20, 40)
    m.active_connections = random.randint(10, 20)
    m.requests_per_second = random.uniform(10, 30)


ANOMALY_HANDLERS = {
    "oom": apply_oom_anomaly,
    "db_pool_exhaustion": apply_db_pool_exhaustion,
    "connection_leak": apply_connection_leak_anomaly,
    "cascade_degradation": apply_cascade_degradation,
    "memory_leak": apply_memory_leak_anomaly,
    "resource_starved": apply_resource_starved,
}


def apply_anomalies(service: Service) -> None:
    """Apply all active anomalies to a service's metrics."""
    if not service._anomalies:
        apply_healthy_baseline(service)
        return

    for anomaly_type in service._anomalies:
        handler = ANOMALY_HANDLERS.get(anomaly_type)
        if handler:
            handler(service)
