# IncidentCommanderEnv - Core Ideation Document

## The Vision

**IncidentCommanderEnv** is a production-grade OpenEnv environment that simulates real-world cloud infrastructure incidents. An AI agent acts as an **on-call Site Reliability Engineer (SRE)**, receiving alerts, diagnosing root causes through log/metric analysis, and executing precise remediation actions across a simulated microservices cluster.

---

## Why This Wins

### The Problem We're Solving

Every tech company — from startups to Meta-scale — has on-call engineers who are woken at 3 AM to diagnose production outages. This is a **$45B+ observability/incident management market** (Datadog, PagerDuty, Splunk, New Relic). Incidents are stressful, time-sensitive, and require systematic diagnostic reasoning — exactly what AI agents need to learn.

**No OpenEnv environment exists for DevOps/infrastructure management.** Zero coverage. This is a wide-open gap in the ecosystem.

### Why Judges Will Love It

| Judge Perspective | Why It Resonates |
|---|---|
| **Meta Engineers** | They operate the largest infrastructure on the planet. Incident response is their daily reality. This environment models a problem they deeply understand and care about |
| **Hugging Face Engineers** | They manage GPU clusters and model serving infrastructure. On-call is real for them too |
| **RL/Agent Community** | Multi-step reasoning, partial observability, causal diagnosis, action ordering — this is a rich testbed for agent capabilities |

### Scoring Projection

| Category | Weight | Expected Score | Why |
|---|---|---|---|
| Real-world utility | 30% | 28-30 | Universal problem, $45B market, genuine evaluation value |
| Task & grader quality | 25% | 22-25 | 3 tasks with deterministic rubric-based graders, clear difficulty progression |
| Environment design | 20% | 17-20 | Rich state management, partial-credit rewards, clean episode boundaries |
| Code quality & spec | 15% | 13-15 | Full OpenEnv compliance, typed Pydantic models, Docker, tests |
| Creativity & novelty | 10% | 9-10 | Zero prior art in OpenEnv, clever reward design with time decay |
| **TOTAL** | **100%** | **89-100** | |

---

## The Simulated World

### Microservices Cluster (8 Services)

```
                    +-----------------+
                    |   api-gateway   |
                    +--------+--------+
                             |
              +--------------+--------------+------------------+
              |              |              |                  |
     +--------v------+ +----v--------+ +---v-----------+ +---v----------+
     | order-service  | | user-service| | notification- | | frontend-bff |
     +--------+-------+ +-----+------+ | service       | +--------------+
              |                |        +---------------+
     +--------v-------+  +----v------+
     | payment-service|  | auth-     |
     +----------------+  | service   |
                         +-----------+
              |
     +--------v--------+
     |   postgres-db   |  (shared database)
     +-----------------+
```

Each service is a fully simulated Python object with:
- **Health state**: healthy, degraded, unhealthy, crashed, restarting
- **Live metrics**: CPU%, memory (MB), request latency (p50/p99), error rate, active connections, RPS
- **Log buffer**: Timestamped structured logs with realistic patterns
- **Deployment history**: Version tracking with rollback capability
- **Configuration**: Memory limits, CPU limits, replica count, DB pool size

### Dependency Graph

Services have explicit dependencies. When a service fails, its dependents experience cascading effects — elevated latency, increased error rates, eventual degradation. The agent must trace symptoms upstream to find root causes.

---

## The Three Tasks

### Task 1: Single Service OOM Crash (Easy)
**Difficulty:** Easy | **Max Steps:** 15 | **Target Score:** Achievable by most LLMs

**Scenario:** The `payment-service` crashes due to an Out-of-Memory error. Its memory limit is set to 256Mi but actual usage spiked to 300Mi. The service is in `CRASHED` state.

**Alert:** `"CRITICAL: payment-service is DOWN. Multiple health check failures detected. PagerDuty alert triggered at 03:42 UTC."`

**What the agent must do:**
1. Survey the cluster (`list_services`) to find the crashed service
2. Inspect the service (`describe_service`) to see its resource configuration
3. Read logs (`read_logs`) — logs clearly show `java.lang.OutOfMemoryError: Java heap space`
4. Restart with increased memory (`restart_service` with `memory_limit: "512Mi"`)

**Grading Rubric:**
| Criterion | Weight | Condition |
|---|---|---|
| Identified failing service | 0.20 | Any action targeting `payment-service` |
| Read relevant logs | 0.20 | `read_logs` on `payment-service` with ERROR filter |
| Diagnosed OOM correctly | 0.20 | Logs containing OOM were retrieved |
| Applied correct fix | 0.40 | `restart_service` with memory_limit > 256Mi |
| Penalty: unnecessary restarts | -0.10 each | Restarting healthy services |

---

### Task 2: Database Connection Pool Exhaustion (Medium)
**Difficulty:** Medium | **Max Steps:** 25 | **Target Score:** Challenging for most LLMs

**Scenario:** The `order-service` has a connection leak bug causing it to exhaust the `postgres-db` connection pool (20/20 connections used). This causes cascading 500 errors in `order-service`, `payment-service`, and `inventory-service`. The `frontend-bff` shows generic "Service Unavailable" errors.

**Alert:** `"WARNING: Elevated 5xx error rates across multiple services. Customer-facing errors reported. Started at 14:23 UTC."`

**The trap:** The obvious symptom is in `frontend-bff`, but the root cause is 2 layers deep in `postgres-db`. The agent must resist the urge to restart the frontend and instead trace the dependency chain.

**What the agent must do:**
1. Investigate symptoms (check metrics on frontend-bff, order-service)
2. Trace to database layer (check metrics on postgres-db — 100% connection utilization)
3. Read postgres-db logs (shows "connection pool exhausted" warnings)
4. Run diagnostic to confirm connectivity issues
5. Increase pool size (`update_config` on postgres-db) or restart DB connections
6. Restart `order-service` to clear leaked connections
7. Resolve incident with accurate root cause summary

**Grading Rubric:**
| Criterion | Weight | Condition |
|---|---|---|
| Investigated frontend symptoms | 0.10 | Any diagnostic action on `frontend-bff` |
| Traced to order-service | 0.15 | Any diagnostic action on `order-service` |
| Identified postgres-db as root | 0.15 | Metrics or logs checked on `postgres-db` |
| Read DB pool exhaustion logs | 0.10 | `read_logs` on `postgres-db` |
| Fixed pool size or connections | 0.20 | `update_config` or `restart_service` on DB |
| Restarted order-service | 0.20 | `restart_service` on `order-service` |
| Resolved with correct root cause | 0.10 | `resolve_incident` called |
| Penalty: wrong restarts | -0.05 each | Restarting uninvolved services |

---

### Task 3: Bad Deployment Cascading Failure (Hard)
**Difficulty:** Hard | **Max Steps:** 35 | **Target Score:** Genuinely challenges frontier models

**Scenario:** A bad deployment (`order-service v2.4.0`) introduced a memory leak. As memory grows, the autoscaler kicks in, spawning more replicas. This exhausts the cluster's resource quota (CPU/memory limits). With no resources left, `inventory-service` and `notification-service` cannot maintain their replicas and begin failing. Multiple alarms fire simultaneously.

**Alert:** `"CRITICAL: Multiple services degraded. Cluster resource quota at 95%. Autoscaler events detected. Started at 09:15 UTC."`

**The trap:** Multiple services are failing simultaneously, making it tempting to restart everything. But the root cause is a single bad deployment, and the resolution ORDER matters — rollback first, then free resources, then restart dependents.

**What the agent must do (order matters):**
1. Map the blast radius (investigate multiple services)
2. Identify `order-service` as the origin (memory leak, high memory usage, recent deployment)
3. Find the bad deployment version (`v2.4.0`)
4. **Rollback** order-service to `v2.3.1` (NOT restart — rollback)
5. Address resource quota (scale down or clear quota)
6. Restart `inventory-service` (after resources are freed)
7. Restart `notification-service`
8. Resolve incident with complete root cause analysis

**Grading Rubric:**
| Criterion | Weight | Condition |
|---|---|---|
| Mapped blast radius | 0.10 | Checked 3+ services |
| Identified order-service as origin | 0.10 | Diagnostic actions on `order-service` |
| Found bad deployment version | 0.10 | Checked deployment history |
| Rolled back correctly | 0.15 | `rollback_deployment` to `v2.3.1` |
| Addressed resource quota | 0.10 | Scale/config action on resources |
| Restarted inventory-service | 0.10 | After rollback completed |
| Restarted notification-service | 0.10 | After rollback completed |
| Correct ordering | 0.05 | Rollback before dependent restarts |
| Accurate root cause in resolution | 0.10 | `resolve_incident` with deployment mention |
| Efficiency (within step budget) | 0.10 | No harmful or redundant actions |
| Penalty: restart instead of rollback | -0.10 | Restarting order-service |
| Penalty: unnecessary actions | -0.05 each | Actions on unrelated services |

---

## Action Space (10 Actions)

| Action | Parameters | Description |
|---|---|---|
| `list_services` | — | Get overview of all services: name, health, version, basic metrics |
| `describe_service` | `target_service` | Deep dive: full config, deployment history, resource limits |
| `read_logs` | `target_service`, `lines`, `severity` | Read log buffer (default 50 lines, filterable) |
| `check_metrics` | `target_service`, `metric`, `window` | Get CPU, memory, latency, error rate, connections |
| `restart_service` | `target_service`, `memory_limit?` | Restart with optional new config |
| `scale_service` | `target_service`, `replicas` | Change replica count |
| `rollback_deployment` | `target_service`, `to_version` | Rollback to a previous deployment version |
| `run_diagnostic` | `target_service`, `command` | Run check: connectivity, dns, resources, health |
| `update_config` | `target_service`, `key`, `value` | Update runtime config (e.g., pool size) |
| `resolve_incident` | `root_cause`, `resolution` | Declare incident resolved with summary |

---

## Reward Design

### Per-Step Rewards (Partial Credit Signal)

```
+0.02  — Useful diagnostic step (reading logs/metrics of a relevant service)
+0.01  — Investigating any service (encourages exploration)
+0.15  — Correct remediation action (restart with right config, rollback to right version)
-0.10  — Harmful action (restarting healthy service, wrong rollback version)
-0.05  — Redundant action (repeating exact same action)
-0.02  — Irrelevant action (checking metrics of a service unrelated to the incident)
```

### Episode-End Score (Grader)

The grader produces a deterministic 0.0-1.0 score by evaluating a weighted checklist of conditions against the action history and final cluster state. Each scenario defines its own rubric (detailed above).

### Time Decay

A multiplicative decay of 0.995 per step encourages efficient diagnosis. An agent that solves the incident in 8 steps scores higher than one that takes 25 steps, all else being equal.

---

## What Makes This Unique

1. **Zero prior art** — No OpenEnv, Gymnasium, or RL environment exists for infrastructure incident response
2. **Multi-step causal reasoning** — Agent must trace symptoms through a dependency graph to find root causes
3. **Action ordering matters** — Hard task requires correct sequence (rollback before restart dependents)
4. **Realistic observation space** — Structured logs, time-series metrics, service topology
5. **Rich reward signal** — Not binary; every diagnostic step provides partial credit
6. **Scalable** — Easy to add new scenarios (just implement BaseScenario) without touching core code
7. **Meta-relevant** — Directly models what the judges deal with daily at massive scale

---

## Technical Constraints Met

| Constraint | How We Meet It |
|---|---|
| 2 vCPU, 8GB RAM | All simulation is in-memory Python objects. No external databases or services |
| < 20 min inference | 3 tasks x max 35 steps = ~105 LLM calls max. At ~3s/call = ~5 min total |
| OpenAI client | inference.py uses `openai.OpenAI()` with env vars for `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` |
| Docker | Single-stage Dockerfile with Python 3.11-slim base |
| HF Spaces | Deploys as a Docker Space with FastAPI serving on port 8000 |
| `openenv validate` | Full spec compliance: typed models, openenv.yaml, step/reset/state endpoints |

---

## Competitive Edge Summary

Most hackathon participants will build:
- Email triage (obvious, low ceiling)
- Content moderation (simple classification)
- Data cleaning (well-understood problem)
- Customer support chatbots (tau-bench already exists in OpenEnv)

**We're building something that:**
- No one else will think of (requires SRE domain expertise)
- Directly impresses Meta/HF engineers (it's their world)
- Has genuinely complex multi-step reasoning (not just classification)
- Is technically elegant (Strategy pattern scenarios, dependency graph cascades)
- Fills a real gap in the RL/agent ecosystem
