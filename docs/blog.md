# Teaching an LLM to be the on-call SRE: IncidentCommanderEnv

*An OpenEnv environment for incident response, trained with SFT + GRPO, and verified against a real Docker stack.*

---

## The 3 AM problem

Every tech company has an on-call rotation. Someone's phone goes off at 3 AM, they open a laptop, see a PagerDuty alert that says *"payment-service is failing,"* and they have to figure out what's wrong before customers notice. That work — read logs, check metrics, identify root cause, ship a fix — is methodical, time-pressured, and largely *systematic reasoning under uncertainty*. It is exactly what RL-trained LLM agents should be good at.

It is also unrepresented in the OpenEnv ecosystem. Until now.

This blog walks through how we built **IncidentCommanderEnv**, an OpenEnv-compatible RL environment for SRE incident response, trained an LLM to navigate it with **SFT + GRPO**, then ran the same trained policy unchanged against a **real Docker Compose stack** to fix a real outage live.

## What the environment looks like

An episode begins with a typed `IncidentObservation` carrying a PagerDuty-style alert and a snapshot of a 9-service simulated cluster. The agent picks one of 10 typed actions (`read_logs`, `check_metrics`, `restart_service`, `update_config`, `rollback_deployment`, …) and gets a typed observation back. Reward arrives as a 6-component breakdown.

Three scenario *families* — not three hardcoded scenarios. Each `(seed, difficulty)` pair produces a distinct instance of OOM crash, DB connection pool exhaustion, or bad deployment cascade. The agent must learn the *rule*, not memorize the *case*.

```python
env = IncidentCommanderEnv()
obs = env.reset(task_id="oom_crash", seed=42, difficulty=0.5)
# alert: "payment-service crashed with OutOfMemoryError"
obs = env.step(IncidentAction(action_type="read_logs",
                              target_service="payment-service",
                              parameters={"lines": 50}))
# diagnostic step → r_diagnostic = +0.05, r_format = +0.01
```

## Reward design — verifiable, not learned

We use **RLVR** (reward from verifiable rubric): no learned reward model, no LLM-as-judge. Six pure functions over `(action, snapshot, scenario)`:

| Component | What it scores |
|---|---|
| `r_diagnostic` | first read on a relevant or adjacent service |
| `r_correct_op` | scenario-defined right-move |
| `r_resolution` | terminal — fix matches scenario rubric AND root_cause text matches |
| `r_format` | action parsed cleanly (no JSON fallback) |
| `r_efficiency` | terminal — solved in ≤50% of step budget |
| `r_penalty` | sum of harmful_restart, redundant, rollback_to_self, unknown_config_key |

Every component is logged separately to wandb. The training plot doesn't just show "reward went up" — it shows *which axes the policy improved on*, in what order. That is the storytelling-grade visual.

## Anti-reward-hacking — receipts, not promises

We pinned each of these as a regression test before we trained:

* **String-match heal.** Old `update_config` ran `if "pool" in key.lower() and "size" in key.lower(): heal()`. We replaced it with a strict allowlist of 5 known config keys and delegated the heal decision to `scenario.on_config_update()` — the scenario, not the handler, decides what counts as a fix.
* **Unconditional anomaly clear.** `restart()` cleared all anomalies, so `memory_leak` was "fixed" by a bare restart. We added `_RESTART_CURABLE = {"oom","connection_leak","resource_starved"}`; non-curable anomalies survive.
* **Redundancy bypass.** Old detector compared full `parameters` dicts, so `{"lines":50}` and `{"lines":51}` were "different." We compare on `(action_type, target_service)` within a 3-step window.
* **Rollback-to-self.** `rollback(to_version=current)` cleared anomalies as a side effect. Early guard now refuses.

Each fix has a `tests/test_reward_hacks.py` test that breaks if the leak comes back.

## The Backend Protocol — sim-to-real for free

Here's the trick that makes the demo work: the env orchestrator never touches `Cluster` directly. It talks to a `Backend` Protocol. Two implementations share it:

```python
class Backend(Protocol):
    def reset(self, scenario, seed): ...
    def execute(self, action, scenario) -> IncidentObservation: ...
    def snapshot(self) -> BackendSnapshot: ...
    def check_resolved(self, scenario) -> bool: ...
    def tick(self): ...
    def teardown(self): ...
```

`SimulatedBackend` wraps the in-memory cluster (fast, fully reproducible — same seed always produces the same trajectory). `RealBackend` translates each typed action into a `docker compose ...` shell-out and reads container state via `docker stats` + `docker compose ps`. Both produce the same typed `BackendSnapshot`. The same trained policy and the same 6-component reward function run unchanged across both.

That decoupling is what makes sim-to-real cheap: we trained on the simulator (fast, deterministic, parallelizable), and the policy works on the real Docker stack because, as far as the agent is concerned, *both substrates expose the same API*.

## Training: SFT then GRPO

We seeded supervised fine-tuning with `IDEAL_TRAJECTORIES` (16 expert traces × 8 seed variants per family ≈ 128 (state, action, rationale) tuples), then ran GRPO on top of that.

**Stack**: Qwen2.5-Coder-1.5B-Instruct (fits T4, also runs on A100), Unsloth `FastLanguageModel.from_pretrained(load_in_4bit=True)`, LoRA r=16 / alpha=32. SFT for 1 epoch (lr=2e-4, batch=2, grad_accum=8, ~30 minutes on A100). GRPO with TRL's `GRPOTrainer`, 4 rollouts per prompt, KL=0.04, lr=5e-6, 400 steps, ~5 hours on A100.

**Curriculum**: phase 1 (steps 0–100) is OOM-only at low difficulty; phase 2 (100–200) mixes OOM and pool exhaustion at medium difficulty; phase 3 (200–400) is the full mix at full difficulty. The schedule lives in `training/curriculum.py` and the curriculum class draws `(family, difficulty)` per training step.

**Eval**: 4 conditions (random / base / SFT / SFT+GRPO) × 3 families × 30 seeds = 360 episodes per snapshot. Plots in `results/`: mean reward curve, 6-component decomposition over time, per-family success bars, action-distribution histograms.

## The sim-to-real demo

We wrote an "observatory" page (`/observe`) that streams recorded trained-agent runs as live incident timelines — the same render code the human dashboard uses, just driven by the episode trace instead of a live env.

The 90-second video walks through three acts:

1. **Train** — wandb plot of all 6 components rising at different rates.
2. **Eval on sim** — the table of conditions × families showing reward improvement.
3. **Real outage** — start the user-vibecoded site under `BACKEND=real`, trigger `python chaos.py --scenario=oom`, point the trained agent at it, watch the service map turn red, watch the agent's actions stream in, watch it turn green.

## What's next (Phase 2)

This is Phase 1: the methodology + substrate. Phase 2 turns the env into a code-fixing co-pilot:

* `CodeAwareBackend` — git worktree + pytest, adding `ReadFile`, `GrepCode`, `GitDiff`, `ApplyPatch`, `RunTests` action types. The agent doesn't just restart — it ships the patch.
* Discriminated typed action union for compile-time safety.
* Code-fix scenario variants requiring a real patch + passing test.
* Sandboxed shell with a 20-command allowlist for diagnostic flexibility.
* Crowdsourced YAML scenario library.

## Try it

```bash
git clone https://github.com/<org>/incident-commander
cd incident-commander
uv sync
uv run uvicorn incident_commander_env.server.app:app --port 8000
# open http://localhost:8000/observe
```

Or open [`training/train_grpo.ipynb`](https://github.com/<org>/incident-commander/blob/main/training/train_grpo.ipynb) in Colab and reproduce the training run.

The deterministic seed-to-trajectory mapping means everything is reproducible: same `(task_id, seed, difficulty)` always produces the same observations, same rewards, same scores. Train once, replay forever.
