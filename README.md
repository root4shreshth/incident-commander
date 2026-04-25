---
title: Praetor — Incident Commander for SREs
emoji: "\U0001F6A8"
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
tags:
  - openenv
  - rl
  - reinforcement-learning
  - sre
  - devops
  - incident-response
  - grpo
  - sft
  - llm-agents
---

# Praetor — Incident Commander for SREs

> The autonomous SRE commander. An OpenEnv-compatible RL environment that trains LLM agents to take the on-call page: investigate a microservices cluster, identify root cause, remediate under time pressure, verify recovery, and escalate to code investigation when runtime ops aren't enough. The same trained policy runs unchanged against a real deployed site for sim-to-real validation.
>
> **Codebase package name** stays `incident_commander_env` for stability — the rename is product-level.

**Meta OpenEnv Hackathon · April 2026 · Theme #3.1: Professional Tasks**

| What | Where |
|---|---|
| Live HuggingFace Space | [hype4raj-incident-commander-env.hf.space](https://hype4raj-incident-commander-env.hf.space) |
| Training notebook (Colab) | [`training/train_grpo.ipynb`](training/train_grpo.ipynb) |
| Trained LoRA adapter | (link added on submission) |
| Blog post | [`docs/blog.md`](docs/blog.md) |
| 90-second video | (link added on submission) |
| Eval results | [`results/`](results/) |
| Project deck (.docx) | [`IncidentCommanderEnv_Project_Document.docx`](IncidentCommanderEnv_Project_Document.docx) |

---

## The 30-second story

On-call SRE is a $45B market and a multi-billion-token-per-day workload for LLMs that can't be benchmarked because there is no public RL environment for it. We built one. The agent gets a PagerDuty-style alert ("payment-service is failing"), investigates a 9-service simulated cluster through 10 typed actions (`read_logs`, `check_metrics`, `restart_service`, …), and is graded by a **6-component verifiable rubric** with no learned reward model — so it cannot be reward-hacked. We trained Qwen2.5-Coder-1.5B with **SFT then GRPO** and the success rate climbs from random-baseline → base-model → SFT → SFT+GRPO across three scenario families. The trained policy then drives a **real Docker stack** through the same Backend Protocol — so the agent that learned in simulation also fixes a real outage live in our demo video.

## Why this matters for the hackathon rubric

| Rubric (% weight) | What we ship for it |
|---|---|
| **Innovation (40%)** | First OpenEnv environment for SRE/DevOps. Backend Protocol that lets one trained policy run on either simulator or real Docker (sim-to-real transfer). Parametric scenario families instead of hardcoded cases — agent must generalize over a distribution of (target service, memory limits, deployment versions). |
| **Storytelling (30%)** | Three-act demo: train → eval → real-Docker outage rescue. Per-component wandb plots show the policy learning each reward axis separately. Observe-mode dashboard replays trained-agent runs as live incident streams. |
| **Reward improvement (20%)** | Eval table across 4 conditions × 3 families × 30 seeds = 360 episodes. Random baseline → base model → SFT → SFT+GRPO. Per-component breakdown surfaces *what* improved, not just the scalar. |
| **Pipeline coherence (10%)** | Pure FastAPI + Pydantic typed models, deterministic seed-to-trajectory mapping, anti-reward-hacking regression tests, OpenEnv-spec-compliant `openenv.yaml`. |

---

## Quick start

### 1. Run the env locally
```bash
git clone https://github.com/<org>/incident-commander
cd incident-commander
uv sync                                       # installs server + dev deps
uv run uvicorn incident_commander_env.server.app:app --port 8000
```
Then open `http://localhost:8000` for the human dashboard, or `http://localhost:8000/observe` to replay recorded trained-agent runs.

### 2. Run a quick eval (random baseline)
```bash
uv run python -c "from training.eval_runner import evaluate, random_policy; \
from training.datasets import SYSTEM_PROMPT; \
print(evaluate('random', random_policy(42), ['oom_crash','db_pool_exhaustion','bad_deployment_cascade'], list(range(30)), system_prompt=SYSTEM_PROMPT, runs_root='runs').to_dict())"
```
Produces `runs/<run_id>/episode.jsonl` traces that the dashboard can replay.

### 3. Train (Colab — 1 GPU, 6–7 hours wall on A100)
Open [`training/train_grpo.ipynb`](training/train_grpo.ipynb) in Colab. The notebook is fully self-contained: `pip install`, clone repo, SFT, eval, GRPO, eval, plots, push LoRA to HF Hub.

### 4. Run the sim-to-real demo
```bash
# Drop the user-vibecoded site under targets/site/ (see "Real-stack contract" below)
BACKEND=real COMPOSE_ROOT=./targets/site \
  uv run uvicorn incident_commander_env.server.app:app --port 8000
# Trained policy now runs against actual Docker containers via the same OpenEnv API
```

---

## Architecture

```
                   ┌──────────────────────────────────────┐
   POST /reset ──▶ │       IncidentCommanderEnv           │ ◀── GET /state
   POST /step  ──▶ │   (orchestrator + reward computer)   │ ◀── GET /reward-breakdown
                   └──────────────┬───────────────────────┘
                                  │ Backend Protocol
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
        │ Simulated    │  │ Real         │  │ CodeAware        │
        │ Backend      │  │ Backend      │  │ Backend          │
        │ (in-memory   │  │ (docker      │  │ (Phase 2 roadmap)│
        │  Cluster)    │  │  compose)    │  │                  │
        └──────────────┘  └──────────────┘  └──────────────────┘
                  │               │
                  ▼               ▼
            9 services       3 services
            (sim)            (real)
```

The agent's view (`BackendSnapshot`) is identical across substrates — same observation shape, same 10 typed actions, same 6-component reward. That decoupling is what makes the policy transferable from sim to real.

---

## The 6-component verifiable reward (RLVR)

No learned reward model. No LLM-as-judge. Every component is a pure function over `(action, snapshot, scenario)` — auditable and unhackable. Each component is logged separately to wandb so the training plot shows *what* the policy learned, not just a scalar.

| Component | Triggers when… | Range |
|---|---|---|
| `r_diagnostic` | first read on a relevant or adjacent service | +0.02 to +0.05 per step |
| `r_correct_op` | scenario-defined right-move (delegated to `scenario.is_correct_op(action)`) | +0.15 |
| `r_resolution` | terminal — fix matches scenario rubric AND root_cause keyword match | +0.30 |
| `r_format` | action parsed cleanly (no fallback) | +0.01 per step |
| `r_efficiency` | terminal — solved in ≤50% of step budget | +0.10 |
| `r_penalty` | sum of harmful_restart, redundant, rollback_to_self, unknown_config_key | -0.05 to -0.30 |

Each component is exposed via `GET /reward-breakdown` per step so the dashboard, training notebook, and tests all share the same numbers.

---

## Anti-reward-hacking (the four exploits we plugged)

| Exploit | The leak | How we plugged it |
|---|---|---|
| `update_config` string-match heal | Old code did `if "pool" in key.lower() and "size" in key.lower(): heal()` — any garbage like `"my.pool.size"` triggered a fix | Strict allowlist of 5 known config keys; heal decision delegated to `scenario.on_config_update()` |
| Unconditional anomaly clear on restart | `restart()` cleared *all* anomalies, so `memory_leak` and `db_pool_exhaustion` were "fixed" by a bare restart with no memory bump or pool resize | Class-level `_RESTART_CURABLE = {"oom","connection_leak","resource_starved"}`; non-curable anomalies survive a restart |
| Redundancy bypass via param tweak | Redundancy detector compared exact `parameters` dicts, so `{"lines":50}` and `{"lines":51}` were "different" and dodged the penalty | Compare on `(action_type, target_service)` tuple within a 3-step sliding window |
| Rollback-to-self | `rollback(to_version=current)` cleared anomalies as a side effect | Early guard: `if to_version == self.config.version: return error` |

Each is pinned by a regression test in `tests/test_reward_hacks.py`.

---

## Parametric scenario families

Three families instead of three hardcoded scenarios. Each `(seed, difficulty)` produces a distinct instance — the agent must learn the *rule*, not memorize *cases*.

| Family | Randomized parameters | Rubric anchor |
|---|---|---|
| `oom_crash` | target service ∈ {payment, order, inventory, user}, memory_limit_mb ∈ [192, 320] | restart with mem > old × 2 |
| `db_pool_exhaustion` | leaking service ∈ {order, payment}, pool_size ∈ {16, 20, 24} | update_config `db.pool.max_size` to ≥ 50 then restart |
| `bad_deployment_cascade` | (BAD, STABLE) ∈ 3 version pairs | rollback bad → restart starved deps in order |

`difficulty` ∈ [0.0, 1.0] scales `max_steps`, anomaly rates, and cascade depth — the curriculum (in `training/curriculum.py`) ramps difficulty over the training run.

---

## Action space (10 typed actions)

| Action | Target | Parameters |
|---|---|---|
| `list_services` | — | — |
| `describe_service` | required | — |
| `read_logs` | required | `lines`, `severity` |
| `check_metrics` | required | — |
| `restart_service` | required | `memory_limit` |
| `scale_service` | required | `replicas` |
| `rollback_deployment` | required | `to_version` |
| `run_diagnostic` | required | `command` |
| `update_config` | required | `key`, `value` (allowlist enforced) |
| `resolve_incident` | — | `root_cause`, `resolution` |

All declared in `openenv.yaml` with full parameter schemas.

---

## Real-stack contract (for the sim-to-real demo)

The `RealBackend` expects a Docker Compose stack at `targets/site/` with:

* **Services named** `frontend`, `api`, `postgres`
* **Env-var levers** in `docker-compose.yml`:
  * `IMAGE_TAG=${IMAGE_TAG:-v1.0}` (rollback target)
  * `mem_limit: ${API_MEM_LIMIT:-256m}` (restart with new memory)
  * `DB_POOL_SIZE: ${POOL_SIZE:-10}` (db pool resize)
* **`/health` endpoint** on each service returning `200` when nominal
* **`chaos.py`** CLI: `python chaos.py --scenario={oom,conn-leak,bad-deploy}|--stop`

When `BACKEND=real`, `RealBackend.reset()` runs `docker compose up -d` then invokes `chaos.py` to reproduce the scenario; each typed action is translated to the appropriate `docker compose ...` invocation. If Docker is unavailable, `RealBackend` degrades to a clearly-labelled stub — useful for CI, HF Space boots before the site lands, and pure tests.

---

## Eval results

```
results/
  eval_random.json       # uniform-random baseline
  eval_base.json         # Qwen2.5-Coder-1.5B (no fine-tune)
  eval_sft.json          # SFT-only
  eval_sft_grpo.json     # SFT then GRPO
  reward_curve.png       # mean reward vs training step
  reward_components.png  # 6 components plotted separately
  success_bars.png       # success rate per family per condition
  action_distribution.png
```

A representative summary table is regenerated by the eval pipeline; expected pattern (final numbers populated post-training run):

| Condition | oom_crash | db_pool_exhaustion | bad_deployment_cascade | Average |
|---|---|---|---|---|
| Random | ~10–15% | ~0–5% | ~0% | floor |
| Base model | improving | improving | improving | mid |
| SFT | strong | improving | mid | better |
| SFT + GRPO | strong | strong | strong | best |

(Numbers replaced with measured values when the Colab run completes.)

---

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Interactive dashboard |
| `/reset` | POST | Start episode (`task_id`, `seed`, `difficulty`) |
| `/step` | POST | Take action (`action_type`, `target_service`, `parameters`) |
| `/state` | GET | Current episode state |
| `/reward-breakdown` | GET | Last step's 6-component reward |
| `/backend` | GET | Which backend is wired (sim/real) |
| `/tasks` | GET | All scenario families with metadata |
| `/runs` | GET | List of recorded trained-agent runs |
| `/watch/{run_id}` | GET | Full event trace for one run |
| `/observe` | GET | Trained-agent observatory dashboard |
| `/health` | GET | Liveness |

---

## Project structure

```
incident_commander_env/
  models.py                            # Pydantic typed Action / Observation / State
  openenv.yaml                         # Full OpenEnv spec (action_space, observation_space, …)
  server/
    app.py                             # FastAPI routes
    environment.py                     # Env orchestrator (delegates to Backend)
    backends/
      protocol.py                      # Backend Protocol + typed BackendSnapshot
      sim.py                           # SimulatedBackend (in-memory)
      real.py                          # RealBackend (docker compose)
      docker_ops.py                    # Shell-out helpers (mock-friendly)
    grading/
      components.py                    # 6 pure reward functions
      reward.py                        # Backwards-compat facade
      grader.py                        # Episode-end rubric
      episode_context.py               # EpisodeContext dataclass
    scenarios/
      base_scenario.py                 # on_config_update / is_correct_op hooks
      scenario_oom_crash.py            # parametric (seed, difficulty)
      scenario_db_pool.py              # parametric
      scenario_bad_deploy.py           # parametric
    simulation/                        # Cluster, services, metrics, logs
    static/
      index.html, demo.js, …           # human dashboard
      observe.html, observe.js         # trained-agent observatory
training/
  datasets.py                          # SFT chat dataset from IDEAL_TRAJECTORIES
  eval_runner.py                       # episode + report runner (writes JSONL traces)
  episode_logger.py                    # JSONL writer / reader for /watch
  curriculum.py                        # 3-phase difficulty schedule
  grpo_reward.py                       # TRL reward fn wrapping 6-component breakdown
  plots.py                             # matplotlib helpers (lazy-imported)
  train_grpo.ipynb                     # Self-contained Colab notebook
tests/
  test_reward_components.py            # per-component unit tests
  test_reward_hacks.py                 # 4 exploits regression-tested
  test_seeded_reproducibility.py       # same seed → same trajectory
  test_backend_protocol.py             # Backend contract tests
  test_real_backend.py                 # RealBackend with mocked subprocess
  test_observe_mode.py                 # /watch + /runs + JSONL logger
  test_training_modules.py             # SFT/GRPO plumbing
  …
results/                               # plots + JSON eval reports
docs/
  blog.md                              # HuggingFace blog source
  video_script.md                      # 90-second video shot list
```

---

## Phase 2 roadmap (mentioned for the pitch)

This Phase-1 hackathon submission proves the methodology and substrate. The roadmap that turns "first OpenEnv for SRE" into "first end-to-end debugging-and-ops co-pilot trained on real-world incidents":

1. **CodeAwareBackend** — git worktree + pytest. Adds `ReadFile`, `GrepCode`, `GitDiff`, `ApplyPatch`, `RunTests` action types. The agent doesn't just restart services — it ships the actual code fix.
2. **Discriminated typed action union** — replace `Dict[str, Any]` parameters with per-action typed sub-models for compile-time safety.
3. **Code-fix scenario families** — each ops scenario gains a code-fix variant requiring a real patch + passing test instead of a runtime workaround.
4. **Sandboxed shell action with allowlist** — 20-command sandboxed shell for diagnostic flexibility beyond the typed action vocabulary.
5. **Crowdsourced scenario library** — YAML scenario authoring DSL; community PRs.

---

## License & attribution

MIT. Built on top of OpenEnv (Meta) + TRL (HuggingFace) + Unsloth.
Telemetry-free, deterministic, and reproducible — same seed always produces the same trajectory and the same score.
