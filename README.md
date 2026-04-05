---
title: IncidentCommanderEnv
emoji: "\U0001F6A8"
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
tags:
  - openenv
---

# IncidentCommanderEnv

**SRE/DevOps Cloud Incident Response & Diagnostics** -- An OpenEnv reinforcement learning environment where AI agents act as on-call Site Reliability Engineers, diagnosing and remediating production incidents across a simulated microservices cluster.

## Why This Environment?

Every tech company -- from startups to Meta-scale -- has on-call engineers who are woken at 3 AM to diagnose production outages. This is a **$45B+ observability/incident management market** (Datadog, PagerDuty, Splunk). Incidents are stressful, time-sensitive, and require systematic diagnostic reasoning -- exactly what AI agents need to learn.

**No OpenEnv environment exists for DevOps/infrastructure management.** This fills a real gap in the RL/agent ecosystem.

## How It Works

```
1. ALERT     -->  Agent receives PagerDuty-style incident alert
2. INVESTIGATE --> Read logs, check metrics, run diagnostics across 9 services
3. DIAGNOSE  -->  Trace dependency chains to identify root cause
4. REMEDIATE -->  Restart, rollback, scale, or reconfigure to fix the issue
5. RESOLVE   -->  Declare root cause and resolution
```

The agent interacts via the standard OpenEnv API: `POST /reset` to start an episode, `POST /step` to take actions, `GET /state` to check progress.

## Simulated Infrastructure

A production microservices cluster with **9 interconnected services**:

```
frontend-bff --> api-gateway --> order-service --> payment-service
                              --> user-service  --> auth-service
                              --> notification-service
                              --> inventory-service
                All DB-dependent services --> postgres-db
```

Each service has realistic:
- **Health states**: healthy, degraded, unhealthy, crashed, restarting
- **Live metrics**: CPU%, memory (MB), request latency (p50/p99), error rate, connections, RPS
- **Structured logs**: timestamped with realistic error patterns (OOM, connection pool, deployment failures)
- **Deployment history**: version tracking with rollback capability
- **Dependencies**: upstream/downstream failure cascading via a DAG topology

## Action Space (10 Actions)

| Action | Target | Parameters | Description |
|---|---|---|---|
| `list_services` | -- | -- | Overview of all 9 services with health and key metrics |
| `describe_service` | Required | -- | Full config, deployment history, dependencies |
| `read_logs` | Required | `lines`, `severity` | Structured log lines (ERROR, WARN, INFO) |
| `check_metrics` | Required | -- | CPU, memory, latency p50/p99, error rate, connections |
| `restart_service` | Required | `memory_limit` | Restart with optional new memory config |
| `scale_service` | Required | `replicas` | Change replica count |
| `rollback_deployment` | Required | `to_version` | Rollback to a previous version |
| `run_diagnostic` | Required | `command` | check_connectivity, check_health, check_resources, check_dns |
| `update_config` | Required | `key`, `value` | Change runtime config (e.g. DB pool size) |
| `resolve_incident` | -- | `root_cause`, `resolution` | Declare incident resolved with summary |

## Observation Space

Each observation returned by `/step` includes:

| Field | Type | Description |
|---|---|---|
| `message` | string | Human-readable description of action result |
| `done` | bool | Whether the episode has ended |
| `reward` | float | Per-step reward signal |
| `alert` | string (optional) | Incident alert text (on reset) |
| `services_summary` | list (optional) | All services health/metrics (from list_services) |
| `service_detail` | object (optional) | Full service config (from describe_service) |
| `logs` | list (optional) | Log lines (from read_logs) |
| `metrics` | object (optional) | CPU, memory, latency snapshot (from check_metrics) |
| `diagnostic_result` | string (optional) | Diagnostic output (from run_diagnostic) |
| `dependency_graph` | object (optional) | Service dependency graph (on reset) |
| `error` | string (optional) | Error message if action is invalid |

## 3 Tasks (Easy to Hard)

### Task 1: OOM Crash (Easy)
- **Scenario**: payment-service crashes with OutOfMemoryError (memory limit 256Mi, usage spiked to 300Mi)
- **Max Steps**: 15
- **Resolution**: Identify service, read logs, restart with higher memory limit (>256Mi)
- **Grading** (4 criteria): Service identification (0.20), log reading (0.20), OOM diagnosis (0.20), correct fix (0.40)
- **Penalties**: -0.10 per healthy service restart

### Task 2: DB Connection Pool Exhaustion (Medium)
- **Scenario**: order-service has a connection leak, exhausting postgres-db pool (20/20 connections), causing cascading 500s in order-service, payment-service, inventory-service. frontend-bff shows generic errors.
- **Max Steps**: 25
- **Resolution**: Trace from frontend symptoms through dependency chain to DB root cause. Fix pool config and restart order-service.
- **Grading** (7 criteria): Frontend investigation (0.10), trace to order-service (0.15), identify DB root (0.15), read DB logs (0.10), fix pool (0.20), restart order-service (0.20), resolve (0.10)
- **The trap**: Obvious symptom is in frontend-bff, but root cause is 2 layers deep in postgres-db

### Task 3: Bad Deployment Cascade (Hard)
- **Scenario**: Bad deployment (order-service v2.4.0) introduces memory leak. Autoscaler spawns 6 replicas, exhausting cluster quota (memory at 95%). inventory-service and notification-service starved.
- **Max Steps**: 35
- **Resolution**: Rollback order-service to v2.3.1 (NOT restart), then restart starved services. **Action ORDER matters** -- rollback must happen before dependent restarts.
- **Grading** (10 criteria): Blast radius mapping (0.10), identify origin (0.10), find bad deployment (0.10), rollback (0.15), address quota (0.10), restart inventory (0.10), restart notification (0.10), correct ordering (0.05), accurate resolution (0.10), efficiency (0.10)
- **The trap**: Multiple services failing simultaneously tempts restarting everything, but the root cause is a single bad deployment

## Reward Design

### Per-Step Rewards (guide agent exploration)
- `+0.03` per diagnostic step on a relevant service
- `+0.02` per diagnostic step on any service
- `+0.15` per correct remediation action
- `-0.10` per harmful action (restarting healthy services)
- `-0.03` per redundant action (repeating same action)
- Multiplicative time decay: 0.995 per step (incentivizes faster resolution)

### Episode-End Score (deterministic grader)
Each scenario defines a weighted rubric. The grader evaluates action history and final cluster state to produce a 0.0-1.0 score. Graders are fully deterministic -- same actions always produce the same score.

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized execution)

### Local Development

```bash
# Install dependencies
pip install fastapi uvicorn pydantic openai requests

# Run the environment server
PYTHONPATH=. uvicorn incident_commander_env.server.app:app --host 0.0.0.0 --port 8000

# Run baseline inference (in another terminal)
# Create a .env file with your API credentials:
#   API_BASE_URL=https://openrouter.ai/api/v1
#   MODEL_NAME=meta-llama/llama-3.3-70b-instruct
#   HF_TOKEN=your-api-key
#   ENV_URL=http://localhost:8000
python inference.py
```

### Docker

```bash
docker build -t incident-commander-env .
docker run -p 8000:8000 incident-commander-env
```

### HuggingFace Space

Live deployment: [hype4raj-incident-commander-env.hf.space](https://hype4raj-incident-commander-env.hf.space)

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive web UI with overview, demo, and API docs |
| `/reset` | POST | Start new episode. Body: `{"task_id": "oom_crash"}` |
| `/step` | POST | Execute action. Body: `{"action_type": "...", "target_service": "...", "parameters": {}}` |
| `/state` | GET | Get current episode state |
| `/health` | GET | Liveness check |
| `/tasks` | GET | List available tasks with difficulty and max steps |

## Baseline Scores

**Model:** Llama 3.3 70B Instruct (via OpenRouter)

| Task | Score | Steps | Status |
|---|---|---|---|
| oom_crash (Easy) | **1.00** | 4/15 | Resolved |
| db_pool_exhaustion (Medium) | **0.80** | 25/25 | Partial (missed order-service restart) |
| bad_deployment_cascade (Hard) | **1.00** | 18/35 | Resolved |
| **Average** | **0.93** | -- | -- |

*The hard task requires correct action ordering (rollback before dependent restarts). The agent successfully identified the bad deployment, rolled back, then restarted starved services in the correct sequence.*

## Project Structure

```
incident_commander_env/
  models.py              # Pydantic typed models (Action, Observation, State)
  openenv.yaml           # OpenEnv specification metadata
  server/
    app.py               # FastAPI routes (/reset, /step, /state)
    environment.py       # Core RL environment orchestrator
    actions/handlers.py  # 10 action handler implementations
    scenarios/           # 3 incident scenarios with rubric-based graders
    grading/             # Episode-end grader + per-step reward computation
    simulation/          # Cluster, services, dependency graph, metrics, logs
    static/index.html    # Interactive web UI
inference.py             # Baseline inference script (OpenAI client, structured logs)
Dockerfile               # Container spec (Python 3.11-slim)
.env                     # API credentials (gitignored)
baseline_results.json    # Latest baseline scores
```

## Technical Details

- **Pure Python simulation**: No external databases or services needed. All 9 services are simulated in-memory.
- **Deterministic grading**: Rubric-based scoring ensures reproducible evaluation.
- **Extensible architecture**: New scenarios can be added by implementing `BaseScenario` without touching core code.
- **Resource efficient**: Runs within 2 vCPU / 8 GB RAM constraint. Inference completes in under 10 minutes.
- **Structured logging**: Inference emits `[START]/[STEP]/[END]` logs per hackathon evaluation spec.
