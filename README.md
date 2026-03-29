---
title: IncidentCommanderEnv
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
tags:
  - openenv
---

# IncidentCommanderEnv

**SRE/DevOps Cloud Incident Response & Diagnostics** — An OpenEnv environment where an AI agent acts as an on-call Site Reliability Engineer, diagnosing and remediating production incidents across a simulated microservices cluster.

## Motivation

Every tech company has on-call engineers who diagnose production outages under time pressure. This environment simulates that experience: the agent receives alerts, reads logs and metrics, traces dependency chains, and executes remediation actions. It tests multi-step causal reasoning, systematic diagnosis, and decision-making under uncertainty.

## Environment Description

The environment simulates a production microservices cluster with 9 services:

```
frontend-bff -> api-gateway -> order-service -> payment-service
                             -> user-service  -> auth-service
                             -> notification-service
                             -> inventory-service
                All DB-dependent services -> postgres-db
```

Each service has realistic:
- **Health states**: healthy, degraded, unhealthy, crashed, restarting
- **Metrics**: CPU%, memory, request latency (p50/p99), error rate, connections, RPS
- **Logs**: Timestamped structured logs with realistic error patterns
- **Deployment history**: Version tracking with rollback capability

## Action Space

| Action | Parameters | Description |
|---|---|---|
| `list_services` | — | Overview of all services with health and key metrics |
| `describe_service` | `target_service` | Detailed config, deployment history, dependencies |
| `read_logs` | `target_service`, `lines?`, `severity?` | Read structured log lines |
| `check_metrics` | `target_service` | CPU, memory, latency, error rate, connections |
| `restart_service` | `target_service`, `memory_limit?` | Restart with optional new config |
| `scale_service` | `target_service`, `replicas` | Change replica count |
| `rollback_deployment` | `target_service`, `to_version` | Rollback to previous version |
| `run_diagnostic` | `target_service`, `command` | Run: check_connectivity, check_health, check_resources, check_dns |
| `update_config` | `target_service`, `key`, `value` | Update runtime config |
| `resolve_incident` | `root_cause`, `resolution` | Declare incident resolved |

## Observation Space

Each observation includes:
- `message`: Human-readable text description of the action result
- `alert`: Active incident alert (on reset)
- `services_summary`: List of service health/metrics (from list_services)
- `service_detail`: Full service configuration (from describe_service)
- `logs`: Log lines (from read_logs)
- `metrics`: CPU, memory, latency snapshot (from check_metrics)
- `diagnostic_result`: Diagnostic output (from run_diagnostic)
- `dependency_graph`: Service dependency graph (on reset)
- `reward`: Per-step reward signal
- `done`: Episode termination flag

## Tasks

### Task 1: OOM Crash (Easy)
- **Scenario**: payment-service crashes due to OutOfMemoryError
- **Difficulty**: Easy | **Max Steps**: 15
- **Expected**: Identify service, read logs, restart with higher memory limit
- **Grading**: Service identification (0.20), log reading (0.20), diagnosis (0.20), correct fix (0.40)

### Task 2: DB Connection Pool Exhaustion (Medium)
- **Scenario**: order-service leaks DB connections, exhausting postgres-db pool, causing cascading 500s
- **Difficulty**: Medium | **Max Steps**: 25
- **Expected**: Trace from frontend symptoms through dependency chain to DB root cause
- **Grading**: Frontend investigation (0.10), trace to order-service (0.15), identify DB root (0.15), read DB logs (0.10), fix pool (0.20), restart order-service (0.20), resolve (0.10)

### Task 3: Bad Deployment Cascade (Hard)
- **Scenario**: Bad deployment causes memory leak, triggers autoscaler, exhausts cluster quota, starves secondary services
- **Difficulty**: Hard | **Max Steps**: 35
- **Expected**: Map blast radius, identify bad deployment, rollback (not restart), free resources, restart dependents in correct order
- **Grading**: 10 criteria including ordering correctness and efficiency

## Reward Design

- `+0.03` per diagnostic step on a relevant service
- `+0.02` per diagnostic step on any service
- `+0.15` per correct remediation action
- `-0.10` per harmful action (restarting healthy services)
- `-0.03` per redundant action (repeating same action)
- Multiplicative time decay: 0.995 per step

## Setup

### Prerequisites
- Python 3.10+
- Docker (for containerized execution)

### Local Development

```bash
# Install dependencies
pip install fastapi uvicorn pydantic openai requests

# Run the server
cd "OpenEV Hackathon"
PYTHONPATH=. uvicorn incident_commander_env.server.app:app --host 0.0.0.0 --port 8000

# Run inference (in another terminal)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:8000"
python inference.py
```

### Docker

```bash
docker build -t incident-commander-env .
docker run -p 8000:8000 incident-commander-env
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{"task_id": "oom_crash"}` |
| `/step` | POST | Execute action. Body: `{"action_type": "...", "target_service": "...", "parameters": {}}` |
| `/state` | GET | Get current episode state |
| `/health` | GET | Liveness check |
| `/tasks` | GET | List available tasks |

## Baseline Scores

| Task | Score | Steps | Status |
|---|---|---|---|
| oom_crash | ~0.80-1.00 | 4-8 | Resolved |
| db_pool_exhaustion | ~0.50-0.80 | 8-15 | Resolved |
| bad_deployment_cascade | ~0.30-0.60 | 12-25 | Partial |

*Scores vary by model. Tested with gpt-4o-mini.*
