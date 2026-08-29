# Praetor - Project Overview

*A clean, 5-minute read for someone new to the project. For the deep-dive tech reference see [`README.md`](README.md); for the narrative build story see [`BLOG.md`](BLOG.md).*

---

## In one sentence

**Praetor is an OpenEnv-compatible reinforcement-learning environment that trains LLM agents to run payments-industry SRE incident response** — investigate a simulated microservices cluster, identify the fault, remediate under time pressure, verify recovery — and the same trained policy runs unchanged against a real deployed payments service for sim-to-real validation.

---

## The problem

Every fintech company runs an on-call rotation. When something breaks in production, an engineer gets woken up and has ten minutes to figure out what's wrong before the outage turns into real money. The problem is that this job is:

- **Expensive.** Production outages cost enterprises **$1M–$5M per hour**. For a payments company where every minute of downtime is transactions that never happen, the number is higher.
- **Slow.** Mean-time-to-resolution averages **~8.85 hours** globally. Level-1 orgs regularly exceed 72 hours.
- **Burnout-inducing.** 65% of engineers report burnout. 70% of SRE teams cite alert fatigue as a top-three concern.
- **Untrained.** There has been no safe, realistic environment for engineers to practice incident response. They learn by making mistakes in production.

And it's **exactly the shape of task that RL-trained LLM agents should be great at** — methodical reasoning under uncertainty, small typed action vocabulary, verifiable outcomes. There just hasn't been a public benchmark to train and evaluate against. Existing academic environments (Microsoft's AIOpsLab) require live Kubernetes clusters, which caps training throughput at roughly one trajectory every 60 seconds. That's a **167-hour wall to produce 10,000 training episodes** — before you even start tuning hyperparameters. Production AI SRE tools (NeuBird, Datadog Bits AI, Resolve.ai) hit the same wall and substitute observability data + prompt engineering for actual RL training.

---

## What we built

Praetor is four things stacked on top of each other:

### 1. A deterministic, high-throughput simulator

A 14-service simulated microservices cluster (9 core e-commerce services + 5 payments-industry services: `payment-gateway`, `webhook-consumer`, `fraud-check`, `refund-service`, `ledger-service`) with real dependency edges, live metrics, structured logs with realistic error patterns, deployment history, and config. **Resets in ~0.5 ms** — roughly **1,900 resets/sec on a laptop** — versus real Kubernetes at ~60 seconds. That's a **~114,000× speedup**, which is what makes RL training on this substrate actually possible.

Every scenario is parametric and seeded: `(family, seed, difficulty)` reproduces the same incident byte-for-byte, which is required for stable RL training and for the OpenEnv/Gymnasium contract.

### 2. 12 incident scenario families

**7 built-in Python scenarios** + **5 community-contributable YAML scenarios**. The library covers:

- **Generic SRE incidents (8):** OOM crash, DB pool exhaustion, bad deployment cascade, disk full, slow-query lock contention, TLS cert expiry, DNS failure, rate-limit exhaustion
- **Payments-industry incidents (4):**
  - `payment_gateway_timeout` — upstream processor 5xx spike; correct fix is `scale_service` to spread the outbound connection pool
  - `webhook_delivery_backlog` — merchant callback delivery stalled; correct fix is `restart` to drain stuck connections
  - `fraud_check_memory_blowup` — feature-cache heap growth; correct fix is preemptive restart with more memory
  - `refund_race_deadlock` — **ordering-sensitive** lock-acquisition-order bug that deadlocks refund-service with ledger-service. Correct fix is `rollback refund-service to v3.2.0` THEN `restart ledger-service`. Bare restart of either service leaves the bug in place and gets penalised.

### 3. A 6-component verifiable reward

No learned reward model. No LLM-as-judge. Six pure functions over `(action, snapshot, scenario)`:

| Component | Fires when… |
|---|---|
| `r_diagnostic` | First read on a relevant or adjacent service |
| `r_correct_op` | Scenario-defined right move (delegated to `scenario.is_correct_op()`) |
| `r_resolution` | Terminal — fix matches scenario rubric AND root-cause keywords match |
| `r_format` | Action parsed cleanly (no JSON fallback) |
| `r_efficiency` | Terminal — solved in ≤50% of step budget |
| `r_penalty` | Redundancy, harmful actions, handler errors |

Every component is logged separately, so the training plot shows *which* axis the policy is improving on. Four classic reward-hacking exploits are closed and pinned by regression tests in [`tests/test_reward_hacks.py`](tests/test_reward_hacks.py).

### 4. A training pipeline that actually produces a trained model

Two Colab notebooks against the vanilla `transformers>=4.46 + trl==0.15.2 + peft + bitsandbytes` stack:

- **`train_sft.ipynb`** — supervised fine-tuning on **Qwen2.5-Coder-1.5B**, 4-bit quantised, LoRA r=16, using ~200 hand-written senior-SRE trajectories replayed under multiple seeds. ~30–60 min on a Colab A100/L40S. Pushes to `<HF_USER>/praetor-incident-commander-sft`.
- **`train_grpo.ipynb`** — Group Relative Policy Optimization on top of the SFT adapter, using [`training/grpo_reward.py`](training/grpo_reward.py) that spins up a fresh env per completion and returns the 6-component reward total. Curriculum-scheduled prompt distribution via [`training/curriculum.py`](training/curriculum.py). 60 steps × 4 rollouts/prompt, ~90–120 min on A100/L40S. Pushes to `<HF_USER>/praetor-incident-commander-grpo`.

Full training pipeline: **367 tests pass locally**, both notebooks are one-click "Run all" from Colab.

### 5. Sim-to-real bridge (the demo that actually lands)

Praetor's environment doesn't talk to the simulator directly. It talks to a **`Backend` Protocol** with three implementations: `SimulatedBackend` (training), `WebsiteBackend` (HTTP to any deployed site implementing a small operator API), and `RealBackend` (Docker Compose). The trained policy can't tell which one it's running against — same observation shape, same 10 actions, same 6-component reward.

To prove this we built **SwiftPay** — a second HuggingFace Space, a real deployed payments-target site that implements the operator contract. In the Praetor dashboard's Real-Time tab, you paste `https://shreshthn8n-swiftpay-target.hf.space`, Praetor probes it, auto-classifies the active fault from log signatures, and runs the trained policy. The same model that learned in the simulator fixes a real outage on a real container, translating each typed action into a real `POST /ops/restart` call.

---

## How it works — one end-to-end run

1. **Alert.** PagerDuty (or a webhook, or a demo button) sends a JSON alert to Praetor: `"payment-gateway p99 at 8200ms, outbound pool at 92%, error rate 41%"`.

2. **Classify.** Praetor auto-classifies the fault into one of 12 scenario families using log-pattern heuristics on the target's `/ops/logs`. For this alert: `payment_gateway_timeout`.

3. **Reset.** The Praetor env orchestrator calls `SimulatedBackend.reset()` (or `WebsiteBackend.reset()` for the sim-to-real demo). The scenario's `setup()` runs, injecting the fault. The agent gets the alert text as its first observation.

4. **Loop.** For each of up to 20 steps:
   - The agent picks one of 10 typed actions: `list_services`, `describe_service`, `read_logs`, `check_metrics`, `restart_service`, `scale_service`, `rollback_deployment`, `update_config`, `run_diagnostic`, `resolve_incident`.
   - The env executes it via the current backend (mutating the sim's Python objects, or issuing `POST /ops/restart` to the real target).
   - The env computes the 6-component reward for this step and stashes it as `env._last_breakdown`.
   - The agent gets back a typed observation, picks the next action.

5. **Resolve.** When the scenario's `check_resolved()` returns true (service healthy + correct fix applied + root-cause keywords match), the episode terminates. A structured Markdown post-mortem is auto-written to `runs/<run_id>/postmortem.md`, and a one-line summary appended to `RUNBOOK.md`.

6. **Verdict.** The dashboard's Final Report card renders with the status pill, the per-step reasoning trace ("Why this step?" expanders on every action), and a **📄 Export as PDF** button that produces a compliance-friendly artifact — cover page with status, every step with rationale, run ID in every footer. Compliance teams have asked for this on every incident for years.

Nowhere in that loop does a human touch anything after the initial alert.

---

## Try it

- **Live demo, no setup:** [hype4raj-incident-commander-env.hf.space](https://hype4raj-incident-commander-env.hf.space) — Real-Time tab, paste `https://shreshthn8n-swiftpay-target.hf.space`, Connect, Run agent.
- **Try a scenario yourself:** same Space, Apprentice tab. Pick "Your first page," solve an OOM crash with the AI coach explaining each step.
- **Watch a trained-agent run:** same Space, Observatory tab. Pick a run from the dropdown, hit Replay.
- **Read the code:** [github.com/root4shreshth/incident-commander](https://github.com/root4shreshth/incident-commander)
- **Reproduce the training:** `training/train_grpo.ipynb` in Colab, A100/L40S runtime, Run All, ~90 min.
- **Pull the trajectory dataset:** committed under [`results/hf_dataset/`](results/hf_dataset/) — 760 senior-SRE behavioral-clone rows + 712 raw step-level rows.

---

## Why this project exists

Originally built for the **Meta OpenEnv Hackathon (April 2026, Theme #3.1: Professional Tasks)**. Post-hackathon it was extended into a portfolio release with:

- The payments-industry scenario library (4 new fintech-shaped incidents)
- Working GRPO training pipeline against the RLVR reward
- Cleaner README + BLOG framing that leads with the payments angle

The design philosophy is that a fintech infrastructure company (like Razorpay) cares about three things a candidate can plausibly build in their own time: **reliability** (things must work), **observability** (you can't debug what you can't see), and **correctness** (money is involved). Praetor is a project that lets a candidate demonstrate they've thought about all three — the deterministic simulator + verifiable reward is reliability, the per-step reasoning trace + PDF export is observability, the anti-reward-hacking test suite + ordering-sensitive `refund_race_deadlock` scenario is correctness.

---

## What's next

The single largest remaining piece is a **`KubernetesBackend` implementation** — a fourth Backend Protocol adapter that runs against a real local `kind` cluster with a proper manifests directory and per-scenario chaos overlays. When that ships, Praetor goes from "cool RL demo" to "you could actually deploy this in the environment we're hiring you to work in." The scoping doc for that lives in [`~/.claude/plans/gentle-doodling-bee.md`](../.claude/plans/gentle-doodling-bee.md); it's roughly a week of work and needs Docker Desktop + `kind` installed locally.

Beyond that, the roadmap in [`BLOG.md`](BLOG.md) ("What we'd build next") lists five more extensions: multi-region topologies, a learned fault classifier, a typed-action-union refactor, an AWS/Cloud Run adapter alongside the Kubernetes one, and a continuous fleet-monitoring watch mode that turns Praetor from incident commander into always-on duty officer.

---

*Built by [Shreshth](https://github.com/root4shreshth) and [Yasin](https://github.com/hype4raj) · Team MetaMorphs · April 2026 hackathon origin, August 2026 payments-industry extension.*
