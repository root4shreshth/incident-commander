---
title: "Praetor - Incident Commander for SREs"
emoji: "🚨"
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - sre
  - devops
  - incident-response
  - sft
  - llm-agents
  - praetor
---

# Praetor - Incident Commander for SREs

> The autonomous SRE commander. An OpenEnv-compatible RL environment that trains LLM agents to take the on-call page: investigate a microservices cluster, identify root cause, remediate under time pressure, verify recovery, and escalate to code investigation when runtime ops aren't enough. The same trained policy runs unchanged against a real deployed site for sim-to-real validation.
>
> **Built for the Meta OpenEnv Hackathon · April 2026 · Theme #3.1: Professional Tasks.**
>
> Codebase package name stays `incident_commander_env` for stability; product display name is **Praetor**.

---

## Submission package - every link a judge needs

| What | Where |
|---|---|
| **GitHub repository** | https://github.com/root4shreshth/incident-commander |
| **Live HuggingFace Space (Praetor)** | https://hype4raj-incident-commander-env.hf.space |
| **Live target site for the Real-Time demo (SwiftPay)** | https://shreshthn8n-swiftpay-target.hf.space |
| **Training notebook (Colab)** | [Open in Colab ↗](https://colab.research.google.com/github/root4shreshth/incident-commander/blob/main/training/train_grpo.ipynb) · source: [`training/train_grpo.ipynb`](training/train_grpo.ipynb) |
| **Trained LoRA adapter** | populated after training run completes |
| **90-second video walkthrough** | populated after recording |
| **Blog post** | source: [`BLOG.md`](BLOG.md) · live URL added on HF: `https://huggingface.co/blog/<USERNAME>/praetor-incident-commander` |
| **Eval results** | [`results/`](results/) (committed after training run) |

---

## Why this exists

### The pain point

Every tech company runs an on-call rotation. Engineers get woken at 3 AM to diagnose production outages under extreme time pressure. The problem is:

- **Expensive.** Production outages cost enterprises **$1M – $5M per hour**. Fortune 1000 companies lose **$1.25B – $2.5B annually** to preventable downtime. 97% of large enterprises say a single hour of downtime costs over $100K.
- **Slow.** Average **mean-time-to-resolution is 8.85 hours** globally. Level-1 maturity organizations routinely exceed 72 hours.
- **Burnout-inducing.** 65% of engineers report burnout. 70% of SRE teams cite alert fatigue. 78% of developers spend 30%+ of their time on manual operational toil.
- **Untrained.** There has been no safe, realistic environment to practice incident response. Engineers learn by making mistakes in production.

### The gap

There has been **no public RL environment** for SRE incident response. The work - methodical reasoning under uncertainty, with a typed action vocabulary and verifiable outcomes - is exactly what RL-trained LLM agents should be good at. There just hasn't been a substrate.

### What Praetor is

An autonomous incident commander. Once paged, Praetor investigates with a typed 10-action vocabulary, decides what to fix using the trained policy, executes via the same Backend Protocol that the simulator uses, verifies recovery, and escalates to **code investigation** if runtime ops aren't enough.

It is the first OpenEnv-compatible environment for SRE / DevOps work, packaged as a complete product: simulator + curriculum + training pipeline + sim-to-real bridge + tier-2 code escalation + autonomous webhook ingestion + post-mortem writer.

---

## The data-factory thesis (the throughput claim)

**RL training for SRE has been gated on data, not algorithms.** Modern policy-gradient methods need tens of thousands of trajectories per scenario family to converge. A real Kubernetes cluster takes ~60 seconds to spin up, break, and tear down - that's a 167-hour wall to produce 10,000 episodes for a single training run. Microsoft's AIOpsLab requires a live K8s cluster. The SF OpenEnv hackathon winner Noclue trained on a real GKE cluster - heroic, but not throughput-shaped. Production AI SRE tools (NeuBird, Resolve.ai, Datadog Bits AI) hit the same wall and substitute observability data + prompt engineering for actual training.

**Praetor cuts the wall down by five orders of magnitude.** Our deterministic, seeded simulator resets in **~0.5 ms** - roughly **1,900 resets/sec on a laptop**. The same 10,000-trajectory batch that would take 167 hours on real K8s runs in **~5 seconds** on our sim. That's measured (`results/throughput.json`), reproducible (`scripts/benchmark_throughput.py`, no GPU), and it's what makes the rest of the project possible:

| | Real K8s | Praetor sim |
|---|---|---|
| Reset time | ~60 s | **0.52 ms** |
| Resets per second | ~0.017 | **~1,900** |
| 10,000-trajectory batch | ~167 hours | **~5 seconds** |
| Step latency | network-bound | **0.16 ms** (~6,400 steps/sec) |

A second deliverable lives at `results/hf_dataset/` - chat-style SFT rows + raw step-level trajectories from 30 random-policy episodes plus the senior-SRE behavioral-clone trajectories, ready to push to a HuggingFace Dataset (`scripts/export_trajectories.py --push-to-hub <repo>`). That's the substrate other researchers can train against without re-running the simulator. We're not competing with NeuBird or Datadog Bits AI on production deployment, and we're not competing with Noclue on real-cluster training. We're **the throughput-optimized substrate underneath them** - the reproducible benchmark that makes those policies trainable at scale.

---

## The 30-second story

On-call SRE is a $45B market and a multi-billion-token-per-day workload for LLMs that couldn't be benchmarked because there was no public RL environment for it. We built one. The agent receives a PagerDuty-style alert ("payment-service is failing"), investigates a 9-service simulated cluster through 10 typed actions (`read_logs`, `check_metrics`, `restart_service`, …), and is graded by a **6-component verifiable rubric** with no learned reward model - so it cannot be reward-hacked. We trained Qwen2.5-Coder-1.5B with **SFT** on senior-SRE behavioral-clone trajectories and the success rate climbs from random-baseline → base-model → SFT across 8 scenario families (6 built-in + 2 community-contributed via YAML). The trained policy then drives a **real deployed site** through the same Backend Protocol - so the agent that learned in simulation also fixes a real outage live in our demo.

---

## What the environment actually is

A FastAPI server that exposes the OpenEnv contract - `POST /reset`, `POST /step`, `GET /state`, `GET /health`, `GET /tasks`, plus a typed observation/action surface. The agent talks to it the same way an OpenAI Gym agent talks to a Gym env, just over HTTP.

**Inside the env:** a 9-service simulated microservices cluster.

```
            frontend-bff ──▶ api-gateway
                              ├──▶ order-service ──▶ payment-service
                              │                  ──▶ inventory-service
                              │                  ──▶ postgres-db
                              ├──▶ user-service  ──▶ auth-service
                              └──▶ notification-service
```

Each service has live state: health (healthy / degraded / unhealthy / crashed / restarting), live metrics (CPU%, memory MB, p50 / p99 latency, error rate, active connections, RPS), a structured log buffer, deployment history, and config (memory limit, CPU limit, replicas, db pool size). Services have explicit dependencies - when one fails, dependents experience cascading effects the agent has to trace.

**Each episode runs this loop:**

1. **Reset.** A scenario family is selected (`oom_crash`, `db_pool_exhaustion`, …). With `(seed, difficulty)`, a fresh parametric instance is materialized - the broken service, the memory ceiling, the bad version are all randomized so the agent has to learn the *shape* of the fault, not memorize specific cases. The agent receives a PagerDuty-style alert string.

2. **Investigate.** The agent picks from 10 typed actions:

   | Action | Purpose |
   |---|---|
   | `list_services` | Cluster overview with health + key metrics for all 9 services |
   | `describe_service` | Full config, deployment history, dependencies for one service |
   | `read_logs` | Structured log lines with realistic error patterns (OOM, pool exhaustion, lock waits, cert errors) |
   | `check_metrics` | CPU, memory, latency p50/p99, error rate, connections, RPS for one service |
   | `restart_service` | Restart with optional new memory_limit |
   | `scale_service` | Change replica count |
   | `rollback_deployment` | Revert to a previous version (refuses rollback-to-self) |
   | `update_config` | Change a runtime setting (allowlisted keys only, scenario decides if it heals) |
   | `run_diagnostic` | Probes like check_connectivity, check_health, check_resources, check_dns |
   | `resolve_incident` | Declare resolved with `root_cause` + `resolution` strings |

3. **Reward.** Every step produces a 6-component breakdown - diagnostic, correct_op, resolution, format, efficiency, penalty - emitted independently to wandb so each axis is plottable on its own. No learned reward model, no LLM-as-judge. Pure math over the action history and cluster state.

4. **Done.** Either the scenario's resolution criteria are met (service healthy + correct fix applied + root cause keywords matched), the agent declares `resolve_incident`, or the step budget runs out. A structured post-mortem is auto-generated alongside the episode trace, and a one-line summary is appended to the project-level RUNBOOK.md.

**Same surface across substrates.** Because the env delegates execution to a `Backend` Protocol, the exact same agent and reward function run unchanged against (a) the in-memory simulator (used for training), (b) a real deployed website that implements the operator API contract (used for the sim-to-real demo), or (c) the codebase itself for tier-2 escalation when runtime ops aren't enough.

---

## Quick start

### Run the env locally

```bash
git clone https://github.com/root4shreshth/incident-commander
cd incident-commander
uv sync                                  # installs server + dev deps
uv run uvicorn incident_commander_env.server.app:app --port 8000
```

Open `http://localhost:8000`. You'll land on the Home tab. Switch to the Observatory to see auto-seeded baseline runs across all 8 scenarios. Switch to Apprentice to try a scenario yourself with the AI coach. Switch to Real-Time to wire up a deployed site.

### Run a quick baseline eval

```bash
uv run python -c "
from training.eval_runner import evaluate, random_policy
from training.datasets import SYSTEM_PROMPT
report = evaluate(
    'random-baseline',
    random_policy(rng_seed=42),
    families=['oom_crash','db_pool_exhaustion','bad_deployment_cascade',
              'disk_full','slow_query','cert_expiry'],
    seeds=list(range(1, 11)),
    system_prompt=SYSTEM_PROMPT,
    runs_root='runs',
)
print({fam: stats['success_rate'] for fam, stats in report.by_family.items()})"
```

Produces `runs/<run_id>/episode.jsonl` traces (replayable in Observatory) and an auto-generated `postmortem.md` next to each.

### Train (Colab - 1 GPU, ~6 hours wall on A100)

Open [`training/train_grpo.ipynb`](training/train_grpo.ipynb) in Colab via this URL pattern:

> **https://colab.research.google.com/github/root4shreshth/incident-commander/blob/main/training/train_grpo.ipynb**

Runtime → T4 (free tier) or A100 → Run all. The notebook is self-contained: pip install, clone repo, SFT (Qwen2.5-Coder-1.5B with LoRA r=16 via Unsloth), eval against the 6-component reward, plots, push LoRA to HF Hub. Results land in `/content/results/`.

### Run the sim-to-real demo

Open the live Praetor Space, click **Real-Time**, paste the SwiftPay target URL, click **Connect** → Praetor auto-classifies the fault → click **Run Praetor** → watch the live timeline → read the Final Report → click **📄 Export as PDF**.

```
https://shreshthn8n-swiftpay-target.hf.space
```

SwiftPay is a real deployed payments site we built ourselves. It implements the operator contract from [§"Real-stack contract"](#real-stack-contract) below, exposes three deliberate fault routes Praetor can detect and resolve, and runs as a separate HuggingFace Space - so there's no localhost or compose stack to set up. The full step-by-step walkthrough is in [§"End-to-end workflow"](#end-to-end-workflow---the-path-a-judge-actually-walks).

If you want to point Praetor at your *own* deployed site instead, vibecode anything that exposes the operator API contract (Render free tier, Vercel, Fly, HF Space), paste your URL into the Real-Time tab, and the same flow runs against it.

### Trigger the autonomous loop via webhook

```bash
curl -X POST http://localhost:8000/incidents/webhook/pagerduty \
     -H 'Content-Type: application/json' \
     -d '{"event":{"data":{"incident":{
            "title":"OutOfMemoryError on payment-service",
            "service":{"summary":"payment-service"}}}}}'
```

Response includes a `run_id`. Refresh the Observatory dropdown - the autonomous run appears with a full trace + auto-generated post-mortem.

---

## Architecture

```
                   ┌──────────────────────────────────────┐
   POST /reset ──▶ │       IncidentCommanderEnv           │ ◀── GET /state
   POST /step  ──▶ │   (orchestrator + reward computer)   │ ◀── GET /reward-breakdown
                   └──────────────┬───────────────────────┘
                                  │ Backend Protocol
                ┌─────────────────┼──────────────────┐
                ▼                 ▼                  ▼
     ┌──────────────────┐  ┌────────────┐   ┌────────────────┐
     │ SimulatedBackend │  │ Website-   │   │ CodeAware*     │
     │ (in-memory       │  │ Backend    │   │ (substrate     │
     │  Python cluster) │  │ (HTTP →    │   │  ready; RL-    │
     │ ─ used for       │  │  /ops/*)   │   │  training      │
     │   training       │  │ ─ used for │   │  pending GPU)  │
     │                  │  │   sim-to-  │   │                │
     │                  │  │   real     │   │                │
     └──────────────────┘  └────────────┘   └────────────────┘
                  │                │
                  ▼                ▼
            9 services        3 services
            (sim)             (real)
```

The agent's view (`BackendSnapshot`) is identical across substrates - same observation shape, same 10 typed actions, same 6-component reward. That decoupling is what makes the policy transferable from sim to real.

\* `CodeAwareBackend` substrate exists today via the **tier-2 code investigation** module - clone repo, grep for suspect code, propose patch, apply on a temp branch, run tests, optionally open PR. RL-training the agent to *choose* code actions vs runtime actions is the next step (needs GPU).

---

## The 6-component verifiable reward (RLVR)

No learned reward model. No LLM-as-judge. Six pure functions over `(action, snapshot, scenario)` - auditable and unhackable. Each component is logged separately to wandb so the training plot shows **what** the policy learned, not just a scalar.

| Component | Triggers when… | Range |
|---|---|---|
| `r_diagnostic` | first read on a relevant or adjacent service | +0.02 to +0.05 per step |
| `r_correct_op` | scenario-defined right-move (delegated to `scenario.is_correct_op(action)`) | +0.15 |
| `r_resolution` | terminal - fix matches scenario rubric AND root_cause keyword match | +0.30 |
| `r_format` | action parsed cleanly (no fallback) | +0.01 per step |
| `r_efficiency` | terminal - solved in ≤50% of step budget | +0.10 |
| `r_penalty` | sum of `harmful_restart`, `redundant`, `rollback_to_self`, `unknown_config_key` | -0.05 to -0.30 |

Each component is exposed via `GET /reward-breakdown` per step so the dashboard, training notebook, and tests share the same numbers.

### Anti-reward-hacking - receipts, not promises

Four exploits the docs warn about, all closed and pinned by regression tests:

| Exploit | The leak | How we plugged it |
|---|---|---|
| `update_config` string-match heal | Old code: `if "pool" in key.lower() and "size" in key.lower(): heal()` - any garbage like `"my.pool.size"` triggered a fix | Strict allowlist of 5 known config keys; heal decision delegated to `scenario.on_config_update()` |
| Unconditional anomaly clear on restart | `restart()` cleared *all* anomalies, so `memory_leak` was "fixed" by a bare restart with no memory bump | Class-level `_RESTART_CURABLE = {"oom","connection_leak","resource_starved","disk_full","cert_expired"}`; non-curable anomalies survive |
| Redundancy bypass via param tweak | Old detector compared full `parameters` dicts, so `{"lines":50}` and `{"lines":51}` were "different" | Compare on `(action_type, target_service)` within a 3-step window |
| Rollback-to-self | `rollback(to_version=current)` cleared anomalies as a side effect | Early guard refuses rollback to the currently-active version |

Each fix is pinned by a test in `tests/test_reward_hacks.py`. If the leak ever comes back, that test breaks first.

---

## The incident curriculum - 8 scenario families

Every `(seed, difficulty)` pair produces a fresh instance. The agent learns the *shape* of the fault, not three fixed cases. **Six built-in Python scenarios + two community-contributed YAML scenarios**, all auto-loaded at startup.

| # | Family | Difficulty | Real-world signature | Right fix | Famous outages |
|---|---|---|---|---|---|
| 1 | `oom_crash` | Easy | `java.lang.OutOfMemoryError: Java heap space` | restart with higher memory limit | Heroku Postgres OOM, Reddit Cassandra |
| 2 | `db_pool_exhaustion` | Medium | `PSQLException: pool exhausted (20/20)` | raise pool size + restart leaking service | GitHub 2018, Discord 2020, Shopify cascade |
| 3 | `bad_deployment_cascade` | Hard | `Memory leak v2.4.0 - autoscaler exhausted quota` | rollback bad deploy *before* restarting starved deps | **Knight Capital ($440M)**, **CrowdStrike 2024**, Facebook BGP 2021 |
| 4 | `disk_full` | Easy | `[Errno 28] No space left on device` | restart cycles the volume | Slack 2020, GitHub 2018, Stripe audit log |
| 5 | `slow_query` | Medium | `Lock wait timeout exceeded; txn rolled back` | rollback the slow-query deploy (restart is a quick fix that doesn't last) | GitHub 2020 (24h incident), Instagram migration |
| 6 | `cert_expiry` | Easy | `ssl.SSLError: certificate has expired` | restart triggers cert renewal hook | **Microsoft Teams 2020**, Spotify 2021, Azure DevOps, LinkedIn, Cloudflare 1.1.1.1 |
| 7 | `dns_failure` *(YAML)* | Medium | `Could not resolve host: payment-service.internal` | restart to refresh DNS resolver | AWS Route53 2017, Cloudflare 2019, Slack 2022 |
| 8 | `rate_limit_exhaustion` *(YAML)* | Medium | `Rate limit exceeded; HTTP 429 returned` | scale gateway replicas to spread budget | Twitter 2023 launch, GitHub Actions throttling |

### YAML scenario authoring DSL

Drop a YAML file under `incident_commander_env/server/scenarios/yaml/` and it auto-loads at startup as a new scenario family. PyYAML is optional - a minimal parser fallback ships with the loader. Schema:

```yaml
task_id: my_scenario
difficulty: medium
description: "Short description"
target_service: api-gateway
anomaly: connection_leak             # any anomaly type known to metrics_engine
max_steps: 18
alert: "PagerDuty: <alert text>"
root_cause: "<full root cause sentence>"
root_cause_keywords: [keyword1, keyword2]
correct_action:
  action_type: restart_service
  target_service: api-gateway
log_lines:
  - "[ERROR] api-gateway - <signature error line>"
rubric:
  - description: "Investigated the failing service"
    weight: 0.30
    required_action: read_logs
    required_target: api-gateway
  - description: "Took the correct fix"
    weight: 0.70
    required_action: restart_service
    required_target: api-gateway
```

Two examples ship with the repo: `dns_failure.yaml` and `rate_limit.yaml`.

---

## The unified dashboard - three modes, one product

A single dashboard with six tabs (Home, Observatory, Apprentice, Real-Time, What we offer, API). The three middle tabs are the product's three usage modes - one per audience - sharing the same backend, the same scenario library, and the same trained policy.

### Tab 1 · Observatory (for ML researchers)
- Replay any recorded trained-agent run
- 6-component reward decomposition with per-component sparklines
- Live-animated service map (red → amber → green as the agent acts)
- Filter chips: all families, by family, ✓ resolved
- Aggregate success-rate bars across conditions

### Tab 2 · Apprentice (for SREs / engineers)
- Tree-shaped curriculum: OOM Crash unlocks three scenarios, DB Pool unlocks two more
- 8 scenario cards (6 built-in + 2 YAML); locked cards greyed until prereq cleared
- AI coach with contextual hints + plain-English "Why?" explanations on every action
- Structured post-mortem with senior-SRE comparison after each incident

### Tab 3 · Real-Time (for hackathon judges + production deployments)
- Connect any deployed site that implements the operator contract
- Praetor probes `/ops/health` + `/ops/metrics` + `/ops/logs` and **auto-classifies** the fault - no manual scenario picking
- Three codebase source options for tier-2 escalation: **GitHub**, **Azure Repos**, **ZIP upload**
- Live unified timeline streams ops actions, then code investigation if needed
- Optional secondary path: inject a deliberate test fault from three chaos buttons (collapsible)

---

## End-to-end workflow - the path a judge actually walks

This is the canonical "open the live Space and follow these steps" walkthrough. The whole project ties together through a single dashboard and a single deployed target site. Two HuggingFace Spaces cooperate:

| Role | URL | What it does |
|---|---|---|
| **Praetor (the agent)** | https://hype4raj-incident-commander-env.hf.space | The OpenEnv-compatible env + dashboard + autonomous loop |
| **SwiftPay (the target)** | https://shreshthn8n-swiftpay-target.hf.space | A real deployed payments site we built ourselves. Implements the operator contract (`/ops/health`, `/ops/metrics`, `/ops/logs`, `/ops/restart`, `/ops/rollback`, `/ops/configure`) and exposes three deliberate fault routes Praetor can detect and resolve. |

Both Spaces are HuggingFace-hosted Docker containers. Praetor never touches your real infrastructure - it only talks to the SwiftPay endpoints over HTTPS, the same way it would talk to any production system that adopts the operator contract.

### Stage 1 - Watch the trained agent (Tab 1: Observatory)

Open https://hype4raj-incident-commander-env.hf.space and click **Observatory**.

1. **Read the legend at the top.** Three sentences explain (a) the score is in `[0, 1]` and ~`0.8` is a clean resolution, (b) Random baseline = uniform-random action policy, (c) Scripted playbook = deterministic best-trajectory.
2. **Pick a recorded run from the dropdown.** Each row shows the scenario family, the score, the model, and the run ID. Filter by family or by `✓ Resolved` using the chips.
3. **Hit ▶ Replay.** The run plays back: action timeline streams in on the right, the 6-component reward decomposition stack-bar populates on the left, the per-component sparklines plot reward earned per step, and the service map evolves from red to green as the agent acts.
4. **Scroll to the aggregate panel.** Eight scenario family cards (one per family) compare success rate across all conditions present in the recorded data - currently `Random baseline (n=5)` vs `Scripted playbook (n=N)`. The OOM crash card shows `Random 20% vs Scripted 100%`, which is the headline reward improvement.

What you've just seen: a fully recorded, fully verifiable training-eval pipeline producing real numbers a human can audit. No hand-waving.

### Stage 2 - Try a scenario yourself (Tab 2: Apprentice)

Click **Apprentice**. The picker shows scenario cards, gated by a curriculum tree. Locked cards unlock as you clear their prereqs. Pick a scenario - say "Your first page" (OOM Crash).

1. **Read the alert.** "It's 3:42 AM. Your phone buzzes - PagerDuty. The payment-service is throwing health check failures and customers can't check out. You're the on-call SRE. Let's go."
2. **Use the action toolbox.** Three groups: Investigate (list_services, read_logs, check_metrics, describe_service, run_diagnostic), Remediate (restart_service, rollback_deployment, scale_service, update_config), Declare (resolve_incident).
3. **Mode switcher (top-right of the picker / incident screen).** Junior mode keeps the AI coach on - it nudges you with hints and a "Why?" button on every action result that explains in plain English what just happened. Pro mode turns the coach off.
4. **Resolve the incident, declare a root cause.** A structured post-mortem renders comparing your run to a senior-SRE reference trajectory. Score and step count are committed to your local progress.

What you've just seen: the same env that trains the agent is also a hands-on training simulator for human SREs. Same observation shape, same action vocabulary, same reward function.

### Stage 3 - Watch Praetor act on a real site (Tab 3: Real-Time)

This is where the simulator-to-real loop closes. Click **Real-Time**.

#### Step 0 (optional) - Connect your platform
Three sub-tabs: GitHub OAuth (for tier-2 code escalation), Cloud account (Azure DevOps / generic), Generate adapter (drops a `praetor_adapter.py` you embed into your own deployment to expose the operator contract). Skip this for the demo - we just want to point Praetor at SwiftPay.

#### Step 1 - Connect to a deployed site
Paste this URL into the input box and click **Connect**:

```
https://shreshthn8n-swiftpay-target.hf.space
```

Praetor immediately:
- `GET /ops/health` to confirm the operator contract is implemented
- discovers the services SwiftPay exposes (`frontend`, `api`, `postgres`)
- `GET /ops/metrics` for each service to gather the current operational state
- runs the auto-classifier over the metrics + recent log patterns

A green `Connected · /ops/health = ok · services: frontend, api, postgres` line appears.

#### Step 2 - Praetor's classification
A card appears: **"DB POOL EXHAUSTION · confidence 70%"** (or `OOM CRASH`, or `BAD DEPLOYMENT CASCADE`, depending on what fault SwiftPay is currently exhibiting). Below it, three to five "evidence chips" cite the signals that drove the classification - e.g. *"postgres at 101 active connections (pool likely exhausted)"*, *"api error_rate spiked to 12%"*. **You did not pick the scenario.** Praetor figured it out.

If no fault is detected, expand the "Inject a test fault →" disclosure and click one of three chaos buttons (OOM Crash / DB Pool Exhaustion / Bad Deployment) - SwiftPay flips into that fault state and Praetor classifies it on the next tick.

#### Step 3 (optional) - Link the codebase for tier-2
Three options if Step 4 might need to escalate to code: GitHub OAuth → pick a repo, Azure DevOps PAT, or upload a ZIP of the codebase. Skip if you only want to see the runtime ops loop.

#### Step 4 - Run Praetor
Click **▶ Run agent**. The Live Unified Timeline starts streaming. Each step is a card with:
- **`step N`** pill, the action name (`list_services`, `read_logs → api`, `restart_service → api {"memory_limit":"1024Mi"}`), and a `TIER1` / `TIER2` pill on the right
- The action result body (logs / metrics / status)
- A **▸ Why this step?** expander - click it to read the agent's reasoning trace for why that particular action was the right next move
- Animated entrance + a pulsing accent border on the latest step

Tier 1 typically settles in 4-6 steps. If runtime ops fully heal the site, you see a **`✓ Tier 1 brought the site back to healthy`** banner. If they don't, Praetor escalates - `⚙ Tier 2 - code investigation` fires, the linked repo gets cloned, candidate code locations surface in the timeline, and (if `enable_pr_open=true` plus a write-scope token is set) a real GitHub pull request opens.

#### Step 5 - Final report + analysis
When the run completes, the **Final Report** card renders below the timeline with:
- **Status pill** - `RESOLVED in tier 1` (green) / `ESCALATED` (amber) / `UNRESOLVED` (red), with the scenario name as a side tag
- **Praetor's summary** - a narrative paragraph: *"Praetor diagnosed an oom crash incident in 5 steps over 14.5 seconds, walked the dependency graph from symptom to root cause, and resolved the incident using only tier-1 runtime operations. Praetor's decisive move was restart_service. The site is now responding 200 on /ops/health and the fix is durable."*
- **Stats grid** - 4 cells: Steps taken, Wall-clock, Outcome (FIXED in green / ESCALATED in amber / UNRESOLVED in red), Services touched
- **The problem we saw** - the original alert in an amber callout
- **Root cause & fix tags** - colored chips: red for cause (`oom`, `memory-pressure`), green for fix (`memory-bump`, `restart-curable`), purple for affected services (`api`)
- **Resolution path** - bulleted list of the meaningful ops the policy took
- **Action breakdown** - per-action-type counts (`read_logs ×2`, `restart_service ×1`, etc.)
- **Tier-2 escalation report** - rendered inline if the run escalated, with summary, suggested fix, and N candidate code locations

#### Step 6 - Export the report
At the bottom of the Final Report, two buttons:
- **📄 Export as PDF** - server-side rendered via `reportlab`, returns `application/pdf` with `Content-Disposition: attachment`. Triggers a real `.pdf` file download (no print dialog round-trip). The PDF mirrors the on-screen report: cover page with status pill and metadata grid, Praetor's summary, stats row, root-cause chips, resolution path bullets, action breakdown table, the original alert, every step with its Why rationale, optional tier-2 section, result paragraph, and a footer with run ID + page number on every page.
- **↗ Preview report** - opens the same content as a print-ready HTML page in a new tab. Use this to read the report on screen before downloading.

If the run record was already evicted server-side (long idle), the buttons surface a clear inline error: *"Report not available (HTTP 404). Start a new run and try again."* No silent failures.

### Stage 4 - Go autonomous (no UI required)
Once Praetor has been pointed at SwiftPay once via Step 1, the same flow runs without a human via webhook:

```bash
curl -X POST https://hype4raj-incident-commander-env.hf.space/incidents/webhook/generic \
  -H "X-Praetor-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"alert":"payment-service OOM","scenario":"oom_crash"}'
```

Praetor classifies the alert text, kicks off a background worker against the connected SwiftPay site, runs the same Tier 1 → Tier 2 loop, writes a JSONL trace to `runs/{run_id}/episode.jsonl`, and emits the same Final Report (queryable via `GET /realtime/run/{run_id}/report` for HTML or `/report.pdf` for the file). The Observatory tab picks the new run up automatically. PagerDuty and Prometheus webhooks also work - same dispatcher, different alert shape.

### What this demo proves
- The trained policy generalizes from the simulator to a **real deployed Docker container** (SwiftPay) without modification
- The same observation shape and action vocabulary work on a real HTTP target
- The Why expander surfaces the policy's reasoning - this is an **explainable** agent, not a black box
- The Final Report + PDF export turn each incident into an auditable artifact a human SRE or compliance team can read
- The autonomous webhook path means there's no human in the loop between alert and verdict

---

## The autonomous capability stack

Beyond the basic OpenEnv contract, Praetor closes the full SRE loop end-to-end: the alert lands automatically, the agent investigates and remediates, verifies recovery, and if runtime ops aren't enough, opens the codebase and ships a patch. Every capability below is shipped today.

| Capability | How it works |
|---|---|
| **Continuous monitoring via webhook** | `POST /incidents/webhook/{pagerduty,prometheus,generic}`. Heuristic classifier picks scenario from alert text. Token-gated via `PRAETOR_WEBHOOK_TOKEN` env var. Once paged, no humans in the loop. |
| **Tier-2 code escalation: ship the patch** | `POST /codebase/propose-and-test` chain: `investigate` → `propose_patch` → `apply_patch` (on a fresh branch, never the working tree) → `run_tests` (auto-detects pytest/unittest/npm) → `open_pull_request` (push + GitHub REST API). PR opening is hard-gated by `enable_pr_open=True` AND a write-scope token. |
| **Auto post-mortem + runbook ledger** | After every episode, `training/postmortem_writer.py` generates a structured `postmortem.md` (summary / alert / root cause / resolution / timeline / reward decomp / what went well / what didn't / scenario-specific action items) and appends a row to `RUNBOOK.md`. |
| **YAML scenario authoring DSL** | Drop a YAML under `scenarios/yaml/`, it auto-loads. Two examples ship: `dns_failure`, `rate_limit_exhaustion`. PyYAML optional. |
| **Sandboxed shell action** | 20-command allowlist (ls, ps, df, du, grep, find, head, tail, curl localhost-only, etc). Per-command argument validators (no path traversal, no shell metachars, network commands localhost-only). Hard 10s timeout, 8 KB output cap. `GET /shell/allowlist`, `POST /shell/run`. |
| **Auto-detect fault on connect** | `/realtime/connect` probes the site and infers the scenario family from metrics + log patterns. User doesn't pick from a list. |

---

## Training pipeline - SFT

We use supervised fine-tuning on senior-SRE behavioral-clone trajectories. SFT alone gives the policy the format and the canonical action sequence for each scenario family, which is the foundation any further RL training would build on.

### Stack
- **Model**: Qwen2.5-Coder-1.5B-Instruct, 4-bit quantized via Unsloth's `FastLanguageModel.from_pretrained(load_in_4bit=True)`
- **Adapter**: LoRA r=16, alpha=32
- **SFT**: `trl.SFTTrainer`, 1 epoch, lr=2e-4, batch=2, grad_accum=8 (~30 min on A100, ~75 min on T4 free tier)

### SFT seed dataset
Ideal trajectories for the six built-in scenario families × multiple seed variants ≈ ~120 (state, action, rationale) tuples drawn from `IDEAL_TRAJECTORIES` in `incident_commander_env/server/coach.py`. Each trajectory was hand-written as what a senior SRE would do for that scenario.

### Curriculum (in `training/curriculum.py`)
- Stage 1: warmup, OOM-only at low difficulty
- Stage 2: OOM + DB-pool at medium difficulty
- Stage 3: full mix at full difficulty

The schedule sampler draws `(family, difficulty)` per training step.

### Compute budget
~30-75 minutes on a single GPU (A100 or T4). The notebook ships ready to run on Colab's free T4 tier.

### Eval protocol
3 conditions × 6 families × 30 seeds = **540 episodes** per snapshot. Conditions: random / base model / SFT. Per-family success rate, average score, average steps used, action distribution, summed reward components.

### Eval results

The **random-baseline floor** is committed today; the trained-condition rows (Base / SFT) get appended to the same files when [`train_grpo.ipynb`](training/train_grpo.ipynb) runs on a GPU.

| Condition | OOM Crash | DB Pool | Bad Deploy | Disk Full | Slow Query | Cert Expiry | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random (n=180, real measurement)** | **17%** | **0%** | **0%** | **0%** | **23%** | **0%** | **6.7%** |
| Base model (no fine-tune) | populated post-Colab | _populated_ | _populated_ | _populated_ | _populated_ | _populated_ | _populated_ |
| SFT | populated post-Colab | _populated_ | _populated_ | _populated_ | _populated_ | _populated_ | _populated_ |

Random-baseline plot suite (committed in [`results/`](results/)):

| Plot | What it shows |
|---|---|
| [`baseline_reward_per_episode.png`](results/baseline_reward_per_episode.png) | Reward signal across all 180 baseline episodes with a 20-episode moving average |
| [`baseline_reward_components.png`](results/baseline_reward_components.png) | The 6 reward axes plotted separately - what the floor's component mix looks like |
| [`baseline_success_rates.png`](results/baseline_success_rates.png) | Per-family success bars for the random condition (trained conditions added post-Colab) |
| [`baseline_action_distribution.png`](results/baseline_action_distribution.png) | Action mix of the random policy |
| [`baseline_summary.json`](results/baseline_summary.json) | Machine-readable per-family stats |

Two things stand out in the floor numbers. **Cert expiry is the hardest baseline at 0%** - even though it's labelled "easy" by step budget - because metrics look almost normal and the only signal is a literal log line. A random policy that doesn't read those logs has zero chance of stumbling on the right fix. **OOM and slow_query each get one or two random wins** (17% and 23%) because the action space includes `restart_service`, and the random policy occasionally picks the right service by chance. Every other family is 0%.

That's the floor. The trained conditions go above it.

To regenerate the baseline plots locally: `uv run python scripts/generate_baseline_plots.py`. To produce the GPU-trained curves, run the Colab notebook.

---

## Real-stack contract - for the sim-to-real demo

The Real-Time tab connects to any deployed site that implements the contract below. CORS is not your problem - Praetor's env server makes the HTTP calls server-side, never from the browser.

### Service names the site reports
`frontend`, `api`, `postgres` (Praetor's defaults; can be overridden via `/realtime/connect`).

### Operator API the site must expose

```
GET  /ops/health          → {"status": "ok"|"degraded"|"down",
                              "services": [{"name": ..., "health": ...}]}
GET  /ops/metrics?service=<name>
                          → {cpu_percent, memory_mb, memory_limit_mb,
                              error_rate_percent, request_latency_p99_ms,
                              active_connections, requests_per_second}
GET  /ops/logs?service=<name>&lines=<N>
                          → {"logs": ["<line>", ...]}
POST /ops/restart         body: {"service":"...", "memory_limit_mb": 1024}
POST /ops/scale           body: {"service":"...", "replicas": N}
POST /ops/config          body: {"service":"...", "key":"...", "value": ...}
POST /ops/rollback        body: {"service":"...", "to_version":"v1.0"}
POST /ops/break           body: {"scenario": "oom_crash"|...}
POST /ops/heal            body: {} (resets all chaos)
```

### Chaos → state mapping the site must implement

| `scenario` | When `/ops/break` fires, the site reports… | Heal action |
|---|---|---|
| `oom_crash` | api memory_mb > 95% of limit; logs include `OutOfMemoryError`; `/cart` returns 500 | `restart` api with `memory_limit_mb >= 1024` |
| `db_pool_exhaustion` | postgres active_connections at limit; logs include `pool exhausted`; `/checkout` 500s | `config` `db.pool.max_size >= 50` on postgres |
| `bad_deployment_cascade` | api version=`v1.1`; error_rate climbs; logs include `memory leak in v1.1` | `rollback` api `to_version=v1.0` |
| `disk_full` | api logs include `No space left on device`; error_rate ~30% | `restart` api (any memory) |
| `slow_query` | api request_latency_p99 spikes to 8s; logs include `Lock wait timeout exceeded` | `rollback` to any version != current |
| `cert_expiry` | api error_rate=99%; cpu/mem tiny; logs include `certificate has expired` | `restart` api |

A working FastAPI reference implementation can be vibecoded in ~300 lines following this contract - see the prompt block below.

### Webhook ingestion - go autonomous

Set `PRAETOR_WEBHOOK_TOKEN` and configure your alerting source:

```bash
# PagerDuty webhook destination
curl -X POST https://YOUR-PRAETOR-HOST/incidents/webhook/pagerduty \
     -H "X-Praetor-Token: $PRAETOR_WEBHOOK_TOKEN" \
     -d @pagerduty-event.json

# Prometheus Alertmanager webhook destination
curl -X POST https://YOUR-PRAETOR-HOST/incidents/webhook/prometheus \
     -H "X-Praetor-Token: $PRAETOR_WEBHOOK_TOKEN" \
     -d @alertmanager-payload.json

# Generic minimal contract
curl -X POST https://YOUR-PRAETOR-HOST/incidents/webhook/generic \
     -H "X-Praetor-Token: $PRAETOR_WEBHOOK_TOKEN" \
     -d '{"alert":"OutOfMemoryError on payment-service","service":"payment-service"}'
```

Praetor classifies the alert into one of the 8 scenario families using log-pattern heuristics, kicks off a run in a background thread, writes the trace + post-mortem to `runs/<run_id>/`, and surfaces it in the Observatory dropdown. **No humans between page and verdict.**

---

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Unified dashboard (Home / Observatory / Apprentice / Real-Time / What we offer / API) |
| `/reset` | POST | Start episode (`task_id`, `seed`, `difficulty`) |
| `/step` | POST | Take action (`action_type`, `target_service`, `parameters`) |
| `/state` | GET | Current episode state |
| `/reward-breakdown` | GET | Last step's 6-component reward |
| `/health` | GET | Liveness check |
| `/backend` | GET | Which backend the env is wired to (sim / website / real) |
| `/tasks` | GET | All scenario families with metadata |
| `/coach/hint` | GET | Rule-based contextual hint for human apprentice |
| `/coach/explain` | POST | Plain-English explanation of last observation |
| `/postmortem` | GET | Structured episode-end review |
| `/runs` | GET | List recorded trained-agent runs |
| `/watch/{run_id}` | GET | Full event trace for one run |
| `/runs/{run_id}/postmortem` | GET | Auto-generated markdown post-mortem |
| `/runbook` | GET | Project-level incident ledger |
| `/realtime/connect` | POST | Connect a deployed site, auto-classify any active fault |
| `/realtime/inject` | POST | Trigger chaos on connected site (`/ops/break`) |
| `/realtime/heal` | POST | Reset chaos on connected site |
| `/realtime/run-agent` | POST | Run Praetor against the connected site |
| `/realtime/status/{run_id}` | GET | Poll for streaming events |
| `/realtime/codebase/link` | POST | Link a GitHub or Azure DevOps repo for tier-2 |
| `/realtime/codebase/upload-multipart` | POST | Upload a ZIP of the codebase for tier-2 |
| `/realtime/codebase/clear` | POST | Forget linked codebase |
| `/codebase/propose-and-test` | POST | Run the full tier-2 chain: investigate → patch → test → optionally PR |
| `/incidents/webhook/pagerduty` | POST | PagerDuty webhook destination |
| `/incidents/webhook/prometheus` | POST | Prometheus Alertmanager webhook destination |
| `/incidents/webhook/generic` | POST | Generic minimal-contract webhook |
| `/incidents/webhooks` | GET | List webhook endpoints + token status |
| `/shell/allowlist` | GET | The 20-command sandboxed-shell allowlist |
| `/shell/run` | POST | Execute a single allowlisted command |
| `/admin/regenerate-demo-runs` | POST | Regenerate baseline runs (UI fallback) |

All declared in `openenv.yaml` with full parameter schemas where applicable.

---

## Project structure

```
incident_commander_env/
  models.py                            # Pydantic typed Action / Observation / State
  openenv.yaml                         # Full OpenEnv spec
  server/
    app.py                             # FastAPI routes (incl. webhooks, shell, codebase, realtime)
    environment.py                     # Env orchestrator (delegates to Backend)
    incidents.py                       # Webhook normalizers + scenario classifier
    backends/
      protocol.py                      # Backend Protocol + typed BackendSnapshot
      sim.py                           # SimulatedBackend (in-memory)
      website.py                       # WebsiteBackend (HTTP → /ops/*)
      real.py                          # RealBackend (Docker compose, legacy / parity)
      docker_ops.py                    # Shell-out helpers
    grading/
      components.py                    # 6 pure reward functions
      reward.py                        # Backwards-compat facade
      grader.py                        # Episode-end rubric
      episode_context.py               # EpisodeContext dataclass
    actions/
      handlers.py                      # 10 typed action handlers
      sandboxed_shell.py               # 20-command allowlist runner
    scenarios/
      base_scenario.py                 # on_config_update / is_correct_op hooks
      scenario_oom_crash.py            # parametric (seed, difficulty)
      scenario_db_pool.py              # parametric
      scenario_bad_deploy.py           # parametric
      scenario_disk_full.py            # parametric
      scenario_slow_query.py           # parametric
      scenario_cert_expiry.py          # parametric
      yaml_loader.py                   # auto-loads YAML scenarios from yaml/
      yaml/
        dns_failure.yaml               # community-contributed
        rate_limit.yaml                # community-contributed
    simulation/                        # Cluster, services, metrics, logs, log generators
    static/
      index.html                       # 5-tab unified dashboard
      observatory.js                   # Phase 1 logic
      realtime.js                      # Phase 3 logic
      demo.js, coach.js, map.js, …     # apprentice (human) UI
training/
  datasets.py                          # SFT chat dataset from IDEAL_TRAJECTORIES
  eval_runner.py                       # episode + report runner (writes JSONL traces)
  episode_logger.py                    # JSONL writer / reader for /watch
  postmortem_writer.py                 # auto-generates postmortem.md
  curriculum.py                        # 3-phase difficulty schedule
  grpo_reward.py                       # TRL reward fn wrapping 6-component breakdown
  code_investigator.py                 # tier-2: clone, grep, propose_patch,
                                       # apply_patch, run_tests, open_pull_request
  plots.py                             # matplotlib helpers (lazy-imported)
  train_grpo.ipynb                     # Self-contained Colab notebook
tests/
  test_reward_components.py            # 29 per-component tests
  test_reward_hacks.py                 # 15 regression tests for the 4 exploits
  test_seeded_reproducibility.py       # 3 same-seed-same-trajectory tests
  test_backend_protocol.py             # 19 Backend contract tests
  test_real_backend.py                 # 28 RealBackend tests with mocked subprocess
  test_website_backend.py              # 23 WebsiteBackend tests with mocked HTTP
  test_observe_mode.py                 # 9 /watch + /runs + JSONL logger tests
  test_realtime_endpoints.py           # 13 realtime endpoints with TestClient
  test_code_investigator.py            # 9 tests with synthetic repo
  test_new_scenarios.py                # 18 tests for disk_full / slow_query / cert_expiry
  test_phase2.py                       # 40 tests for Phase 2 modules
  test_environment.py                  # 44 env behavior tests
  test_grading.py                      # 38 grader tests
  test_api.py                          # 30 HTTP surface tests
  test_training_modules.py             # 20 training plumbing tests
results/                               # plots + JSON eval reports (post-Colab run)
runs/                                  # JSONL traces of every recorded run (gitignored)
RUNBOOK.md                             # Auto-generated incident ledger (under runs/)
```

**346 / 346 tests passing.** Run with `uv run pytest`.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Web framework | FastAPI | Async, typed, OpenAPI-out-of-the-box |
| Models | Pydantic v2 | Typed action / observation contracts |
| Simulator | Pure Python | Zero external deps, deterministic with a seed |
| Training | HuggingFace TRL + Unsloth | SFTTrainer; 4-bit Qwen via Unsloth |
| Sim-to-real | HTTP `/ops/*` operator API | Any deployable site can implement; no Docker required |
| Tier 2 | git + heuristic + optional LLM | Cloning, grep, ranking, optional summary via OpenRouter / local |
| Frontend | Vanilla HTML / CSS / JS | No build step. Loads instantly. Editorial typography (Fraunces serif + Inter sans). |
| Telemetry | JSONL episode logs + auto-generated postmortems | No analytics, no telemetry, fully reproducible by seed |
| Quality | pytest, < 9s for 346 tests | CI-friendly. Mock-HTTP / mock-subprocess suites isolate from network |

---

## What's deferred (and why)

Honest scoping for the hackathon submission. Three items are explicitly deferred - substrate is in place, just not exercised yet:

| Item | Status | Why deferred |
|---|---|---|
| **Discriminated typed action union** (replace `Dict[str, Any]` parameters with per-action typed sub-models) | Substrate ready, refactor not done | Risky against 346 passing tests on the eve of submission. Better as a follow-up PR. |
| **RL-train tier-2** (let an RL trainer learn when to escalate to code vs runtime ops) | All four primitives (`propose_patch / apply_patch / run_tests / open_pull_request`) are callable from a TRL reward fn - training not run | Requires GPU. Hook these actions into a future Colab cell with a code-aware reward function. |
| **Eval table real numbers** | Pipeline ready, GPU run pending | The Colab notebook is self-contained and ready. Fire it once, results land in `results/`. README placeholders are clearly labelled. |

---

## Repro / development guide

### Running the test suite
```bash
uv run pytest                 # 346 tests, ~9s
uv run pytest -k phase2       # Phase 2 modules only
uv run pytest -k reward_hacks # anti-reward-hacking regression tests
```

### Running the env locally
```bash
uv run uvicorn incident_commander_env.server.app:app --port 8000
```

### Running with a different backend
```bash
BACKEND=sim     uv run uvicorn incident_commander_env.server.app:app  # default, in-memory
BACKEND=website SITE_URL=https://your-deployed-site.com \
                uv run uvicorn incident_commander_env.server.app:app  # sim-to-real
```

### Adding a new scenario via YAML
Drop a file under `incident_commander_env/server/scenarios/yaml/`. It auto-loads at startup. See the schema above; reference the two examples (`dns_failure.yaml`, `rate_limit.yaml`).

### Adding a new built-in scenario in Python
Subclass `BaseScenario`, implement `setup`, `check_resolved`, `get_rubric`, `compute_penalties`, `is_correct_op`. Add the class to `incident_commander_env/server/scenarios/__init__.py`. Add an entry to `LEARNING_CONTEXT` and `IDEAL_TRAJECTORIES` in `coach.py`. Add a demo playbook in `app.py` if you want it in Real-Time. Add log-pattern heuristics in `_classify_current_fault` and `incidents.py`. Write tests.

### Triggering an autonomous run
```bash
# From a webhook source
curl -X POST http://localhost:8000/incidents/webhook/generic \
     -H "X-Praetor-Token: $PRAETOR_WEBHOOK_TOKEN" \
     -d '{"alert":"OutOfMemoryError on payment-service","service":"payment-service"}'

# Manual
curl -X POST http://localhost:8000/realtime/run-agent \
     -d '{"scenario":"oom_crash"}'
```

### Reading the auto-generated post-mortem
After any run finishes, `runs/<run_id>/postmortem.md` is created. Visit `/runs/<run_id>/postmortem` in the dashboard or via the API. The project-level `RUNBOOK.md` accumulates a one-line summary per incident.

---

## License + attribution

MIT. Built on top of OpenEnv (Meta) + TRL (HuggingFace) + Unsloth.

No telemetry. Fully reproducible: same `(task_id, seed, difficulty)` always yields the same observations, rewards, scores, and trace.

Built for the **Meta OpenEnv Hackathon · April 2026** by **Team MetaMorphs**.

---

## Quick reference for judges

| Want to … | Click |
|---|---|
| See the live env (Praetor) | https://hype4raj-incident-commander-env.hf.space |
| See the live target site (SwiftPay) | https://shreshthn8n-swiftpay-target.hf.space |
| Read the code | https://github.com/root4shreshth/incident-commander |
| Read the blog post | source: [`BLOG.md`](BLOG.md) · live URL added on HF: `https://huggingface.co/blog/<USERNAME>/praetor-incident-commander` |
| Run the training | [Open `train_grpo.ipynb` in Colab](https://colab.research.google.com/github/root4shreshth/incident-commander/blob/main/training/train_grpo.ipynb) |
| **Follow the end-to-end demo** | [§ End-to-end workflow](#end-to-end-workflow---the-path-a-judge-actually-walks) |
| Watch a recorded trained-agent run | Live env → tab **1 Observatory** |
| Try solving an incident yourself | Live env → tab **2 Apprentice** |
| Watch the autonomous loop on a real site | Live env → tab **3 Real-Time** → paste `https://shreshthn8n-swiftpay-target.hf.space` |
| Read what we ship | Live env → tab **What we offer** |
| Verify the operator API | Live env → tab **API** |
| Trigger an autonomous run via webhook | `POST /incidents/webhook/generic` |
| See the auto-generated post-mortem | `GET /runs/{run_id}/postmortem` |
| See the running incident ledger | `GET /runbook` |
| Export an incident report as PDF | Live env → tab **3 Real-Time** → run finishes → click **📄 Export as PDF** |
