"""Template-based structured log synthesis for simulated services."""

from __future__ import annotations

import random
from typing import List

_TS_BASE = "2026-03-29T"


def _ts(hour: int, minute: int, second: int) -> str:
    return f"{_TS_BASE}{hour:02d}:{minute:02d}:{second:02d}.{random.randint(100, 999)}Z"


def normal_logs(service: str, count: int = 10, base_hour: int = 9) -> List[str]:
    """Generate normal operational log lines."""
    templates = [
        "[INFO]  {svc} - Request processed successfully in {lat}ms",
        "[INFO]  {svc} - Health check passed",
        "[DEBUG] {svc} - Connection pool stats: active={conn}, idle={idle}",
        "[INFO]  {svc} - Served GET /api/v1/status 200 in {lat}ms",
        "[INFO]  {svc} - Metrics exported successfully",
        "[DEBUG] {svc} - GC pause: {gc}ms, heap: {heap}MB",
        "[INFO]  {svc} - Incoming request from api-gateway processed",
        "[INFO]  {svc} - Cache hit ratio: {cache}%",
    ]
    logs = []
    for i in range(count):
        ts = _ts(base_hour, random.randint(0, 59), random.randint(0, 59))
        tmpl = random.choice(templates)
        line = f"{ts} {tmpl.format(svc=service, lat=random.randint(5, 50), conn=random.randint(5, 20), idle=random.randint(3, 10), gc=random.randint(5, 30), heap=random.randint(64, 256), cache=random.randint(85, 99))}"
        logs.append(line)
    return sorted(logs)


def oom_crash_logs(service: str, memory_limit_mb: int = 256) -> List[str]:
    """Generate logs showing an OOM crash sequence."""
    return [
        f"{_ts(3, 38, 10)} [INFO]  {service} - Request processed successfully in 23ms",
        f"{_ts(3, 38, 45)} [WARN]  {service} - Memory usage elevated: {int(memory_limit_mb * 0.75)}MB / {memory_limit_mb}MB (75% utilization)",
        f"{_ts(3, 39, 12)} [WARN]  {service} - GC overhead increasing: 45% of CPU time spent in garbage collection",
        f"{_ts(3, 39, 30)} [WARN]  {service} - Memory usage critical: {int(memory_limit_mb * 0.90)}MB / {memory_limit_mb}MB (90% utilization)",
        f"{_ts(3, 40, 1)} [ERROR] {service} - Memory allocation failed: requested 32MB, available 12MB",
        f"{_ts(3, 40, 2)} [ERROR] {service} - java.lang.OutOfMemoryError: Java heap space",
        f"{_ts(3, 40, 2)} [ERROR] {service} -     at com.payments.processing.TransactionHandler.process(TransactionHandler.java:142)",
        f"{_ts(3, 40, 2)} [ERROR] {service} -     at com.payments.api.PaymentController.handleRequest(PaymentController.java:87)",
        f"{_ts(3, 40, 3)} [FATAL] {service} - Process terminated due to OutOfMemoryError. Exit code: 137 (OOMKilled)",
        f"{_ts(3, 40, 3)} [ERROR] {service} - Health check failed: connection refused on port 8080",
        f"{_ts(3, 40, 15)} [ERROR] {service} - Liveness probe failed 3/3 times. Container marked as CRASHED.",
        f"{_ts(3, 42, 0)} [ERROR] {service} - PagerDuty alert triggered: SERVICE_DOWN severity=critical",
    ]


def db_pool_exhaustion_logs(service: str, pool_size: int = 20) -> List[str]:
    """Generate logs showing database connection pool exhaustion."""
    return [
        f"{_ts(14, 20, 0)} [INFO]  {service} - Connection pool stats: active={pool_size - 5}, idle=5, max={pool_size}",
        f"{_ts(14, 21, 15)} [WARN]  {service} - Connection pool utilization at 85%: active={int(pool_size * 0.85)}, max={pool_size}",
        f"{_ts(14, 22, 0)} [WARN]  {service} - Connection pool utilization at 95%: active={int(pool_size * 0.95)}, max={pool_size}",
        f"{_ts(14, 22, 30)} [ERROR] {service} - Connection pool exhausted: {pool_size}/{pool_size} active connections. New requests queued.",
        f"{_ts(14, 22, 31)} [ERROR] {service} - Timeout waiting for database connection: waited 30000ms",
        f"{_ts(14, 22, 45)} [ERROR] {service} - org.postgresql.util.PSQLException: Cannot acquire connection from pool - pool exhausted",
        f"{_ts(14, 23, 0)} [ERROR] {service} - 47 requests queued waiting for DB connection. Queue timeout: 30s",
        f"{_ts(14, 23, 10)} [WARN]  {service} - Potential connection leak detected: 3 connections held >60s by order-service",
        f"{_ts(14, 23, 30)} [ERROR] {service} - Health check degraded: DB connection acquisition time >5000ms",
    ]


def connection_leak_logs(service: str) -> List[str]:
    """Generate logs showing connection leak symptoms in a service."""
    return [
        f"{_ts(14, 18, 0)} [INFO]  {service} - Processing batch order #4521",
        f"{_ts(14, 19, 30)} [DEBUG] {service} - Acquired DB connection (pool active: 12/20)",
        f"{_ts(14, 20, 0)} [DEBUG] {service} - Acquired DB connection (pool active: 15/20)",
        f"{_ts(14, 20, 30)} [WARN]  {service} - Slow query detected: SELECT * FROM orders WHERE... took 2340ms",
        f"{_ts(14, 21, 0)} [DEBUG] {service} - Acquired DB connection (pool active: 18/20)",
        f"{_ts(14, 21, 30)} [ERROR] {service} - Transaction timeout: connection held for 45s without commit/rollback",
        f"{_ts(14, 22, 0)} [ERROR] {service} - Failed to acquire DB connection: pool exhausted (20/20)",
        f"{_ts(14, 22, 15)} [ERROR] {service} - HTTP 500 returned to client: database connection unavailable",
        f"{_ts(14, 22, 30)} [ERROR] {service} - 12 consecutive request failures due to DB connection unavailable",
    ]


def cascading_failure_logs(service: str, upstream_cause: str) -> List[str]:
    """Generate logs showing a service failing due to upstream dependency."""
    return [
        f"{_ts(14, 22, 45)} [WARN]  {service} - Elevated latency on requests to {upstream_cause}: p99=2500ms (normal: 45ms)",
        f"{_ts(14, 23, 0)} [ERROR] {service} - Request to {upstream_cause} failed: HTTP 503 Service Unavailable",
        f"{_ts(14, 23, 10)} [ERROR] {service} - Circuit breaker OPEN for {upstream_cause}: 15/20 requests failed in last 30s",
        f"{_ts(14, 23, 20)} [WARN]  {service} - Degraded mode: returning cached/fallback responses",
        f"{_ts(14, 23, 30)} [ERROR] {service} - Error rate elevated: 35% of requests failing (threshold: 5%)",
    ]


def bad_deployment_logs(service: str, version: str) -> List[str]:
    """Generate logs showing a bad deployment with memory leak."""
    return [
        f"{_ts(9, 10, 0)} [INFO]  {service} - Deployment started: rolling update to {version}",
        f"{_ts(9, 10, 30)} [INFO]  {service} - Deployment {version} health check passed (initial)",
        f"{_ts(9, 12, 0)} [INFO]  {service} - Deployment {version} active. All replicas healthy.",
        f"{_ts(9, 15, 0)} [WARN]  {service} - Memory usage trending upward: 256MB -> 384MB in 5 minutes",
        f"{_ts(9, 20, 0)} [WARN]  {service} - Memory usage: 520MB / 512MB - approaching limit",
        f"{_ts(9, 22, 0)} [ERROR] {service} - Memory leak suspected: heap growing at ~50MB/min since {version} deployment",
        f"{_ts(9, 23, 0)} [WARN]  {service} - Autoscaler triggered: scaling from 2 to 4 replicas due to memory pressure",
        f"{_ts(9, 24, 0)} [WARN]  {service} - Autoscaler: scaling from 4 to 6 replicas. Memory per replica still growing.",
        f"{_ts(9, 25, 0)} [ERROR] {service} - Autoscaler: cannot scale beyond 6 replicas - cluster resource quota reached",
        f"{_ts(9, 25, 30)} [ERROR] {service} - OOMKilled: replica-3 terminated. Memory: 540MB / 512MB",
        f"{_ts(9, 26, 0)} [ERROR] {service} - OOMKilled: replica-5 terminated. Remaining replicas under extreme memory pressure.",
    ]


def resource_quota_logs() -> List[str]:
    """Generate cluster-level resource quota exhaustion logs."""
    return [
        f"{_ts(9, 23, 30)} [WARN]  cluster-manager - Resource quota utilization: CPU 72%, Memory 78%",
        f"{_ts(9, 24, 15)} [WARN]  cluster-manager - Resource quota utilization: CPU 85%, Memory 91%",
        f"{_ts(9, 25, 0)} [ERROR] cluster-manager - Resource quota exceeded: Memory at 95% (7.6GB / 8.0GB)",
        f"{_ts(9, 25, 10)} [ERROR] cluster-manager - Rejected scale-up request for inventory-service: insufficient resources",
        f"{_ts(9, 25, 20)} [ERROR] cluster-manager - Rejected scale-up request for notification-service: insufficient resources",
        f"{_ts(9, 25, 30)} [WARN]  cluster-manager - Multiple services unable to maintain desired replica count",
    ]


def frontend_generic_error_logs(service: str) -> List[str]:
    """Generate generic error logs for frontend (hides root cause)."""
    return [
        f"{_ts(14, 23, 0)} [ERROR] {service} - GET /api/v1/orders returned 503 from upstream",
        f"{_ts(14, 23, 5)} [ERROR] {service} - GET /api/v1/orders returned 503 from upstream",
        f"{_ts(14, 23, 10)} [ERROR] {service} - POST /api/v1/checkout returned 500: Internal Server Error",
        f"{_ts(14, 23, 15)} [WARN]  {service} - Error rate spike: 42% of requests returning 5xx",
        f"{_ts(14, 23, 20)} [ERROR] {service} - User-facing error page served 23 times in last 60s",
    ]


def service_starved_logs(service: str) -> List[str]:
    """Generate logs for a service starved of resources due to quota exhaustion."""
    return [
        f"{_ts(9, 25, 30)} [WARN]  {service} - Replica count below desired: running 1/2 replicas",
        f"{_ts(9, 26, 0)} [ERROR] {service} - Scale-up request rejected by cluster: insufficient resources",
        f"{_ts(9, 26, 15)} [WARN]  {service} - Single replica under high load: CPU 92%, response times degraded",
        f"{_ts(9, 26, 30)} [ERROR] {service} - Request queue growing: 45 pending requests, avg wait 3200ms",
        f"{_ts(9, 27, 0)} [ERROR] {service} - Health check: DEGRADED. Error rate 28%, latency p99=5400ms",
    ]
