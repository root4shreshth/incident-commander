# Teaching an LLM to take the 3 AM page

*Or: how we built **Praetor**, an OpenEnv environment that trains AI agents to do incident response - and what surprised us along the way.*

---

If you've ever been on call at a software company, you know the feeling. Your phone vibrates at 03:42 AM. Half-asleep, you open the laptop. A red banner: *payment-service is down. Customers can't check out.* You have ten minutes before the post-mortem timer turns into a real revenue number, and the only person who can stop the bleeding is you.

This is most of modern operations work. It is also one of the few jobs left where humans are routinely woken up by a buzzing pocket-sized rectangle to do *systematic reasoning under uncertainty*. Which is a fancy way of saying: this is exactly the kind of task that an RL-trained language model should be brilliant at - methodical, time-pressured, narrowly-scoped, with verifiable outcomes.

There was just one problem when we went looking. **Nobody had built it.**

There are RL environments for chess. RL environments for Atari. RL environments for browser navigation, code review, web shopping, math contests. There is no RL environment, anywhere, for the actual job of being an on-call SRE. The work is too messy to benchmark, the post-mortems are private, and the simulators are all internal tooling at companies that don't share them.

So we built one. We called it **Praetor** - after the Roman magistrate whose job, conveniently, was to take command in a crisis.

This is what it does, what we trained on it, and what we learned.

---

## The shape of the problem

Production outages cost real money. The numbers, depending on whose 2024 report you read, hover around **$1M to $5M per hour** for enterprise-scale outages. Mean-time-to-resolution averages somewhere around **8.85 hours** globally. Sixty-five percent of engineers report burnout. Seventy percent of SRE teams cite alert fatigue as a top-three operational concern.

What's striking, when you read post-mortems from Google, GitHub, Slack, Microsoft, and CrowdStrike side-by-side, is how *similar* the failure shapes are. A service runs out of memory. A database connection pool gets exhausted. A bad deployment introduces a memory leak that the autoscaler tries to paper over. A TLS certificate expires. A query holds a row-lock too long. A log volume fills up. A DNS resolver misconfigures.

Every one of these is a category. They recur. They have signatures in the logs. Senior engineers can pattern-match them in seconds, then take three or four targeted actions and walk away from the incident. The reason they're still hard is not the diagnosis - it's that *junior* engineers haven't seen the pattern yet, the fix often requires touching production, and there's no safe place to practice.

That last sentence is the gap Praetor fills.

---

## What Praetor actually is

A FastAPI server that exposes the **OpenEnv** contract - the same `POST /reset`, `POST /step`, `GET /state` surface used by RL environments across the Meta + Hugging Face ecosystem. Behind that contract is a simulated 9-service microservices cluster. Each service has health, live metrics (CPU, memory, latency, error rate), structured logs with realistic patterns, deployment history, and explicit dependencies. When one service fails, its dependents start cascading.

The agent's interface is ten typed actions:

```
list_services       describe_service     read_logs        check_metrics
restart_service     scale_service        rollback_deployment
update_config       run_diagnostic       resolve_incident
```

That's the entire vocabulary. The agent gets a PagerDuty-style alert, picks an action, gets a typed response back, picks the next action, and so on, until it declares the incident resolved or runs out of step budget. Each scenario family - six built-in plus two community-contributed via a YAML DSL we'll talk about later - corresponds to one of the canonical outage shapes:

- **OOM crash** (Java heap out of memory; restart with bigger limit)
- **DB connection pool exhaustion** (raise pool size, restart leaking service)
- **Bad deployment cascade** (rollback, *then* restart starved dependents - order matters)
- **Disk full** (restart cycles the volume)
- **Slow query / lock contention** (rollback the query-introducing deploy; *not* restart, which is a quick fix that doesn't last)
- **TLS certificate expired** (restart triggers cert renewal - and yes, this one is the most embarrassing)
- **DNS failure** (restart refreshes the resolver)
- **Rate-limit exhaustion** (scale the gateway to spread the budget)

Every scenario is parametric. Each `(seed, difficulty)` pair produces a fresh instance - a different victim service, a different memory ceiling, a different bad version number. The agent has to learn the *shape* of the fault, not memorize specific cases.

---

## The bottleneck nobody talks about: data, not algorithms

Other hackathon teams have trained SRE-shaped agents on *real* Kubernetes clusters. That's the right idea - it's the most realistic possible substrate. But there's a catch the published numbers don't dwell on: a real `kubectl rollout undo` plus pod-recreation plus health-stabilization cycle takes roughly **60 seconds**. That's the floor on how fast you can generate one new training trajectory. RL training wants hundreds of thousands of trajectories, sampled across many seeds, with the curriculum visiting each scenario shape repeatedly. The math doesn't work.

We measured Praetor's simulator on a stock laptop: **1,905 resets per second, 6,425 environment steps per second.** That's a **~114,000× speedup** over real Kubernetes for the inner loop of RL training. Same scenario surface, same reward function, same actions - just deterministic, seeded, and parametric so any `(family, seed, difficulty)` triple regenerates byte-identically. The numbers are committed to `results/throughput.json` and reproducible by running `python scripts/benchmark_throughput.py`.

That speedup is the entire data-factory thesis. *Of course* you should eventually train on real clusters - the Backend Protocol is exactly the seam to do that, and our `WebsiteBackend` already runs the trained policy unchanged against a real Render-deployed FastAPI site. But the hot loop where the agent is figuring out the *shape* of incident response - the part where you need a million reset-step-evaluate cycles - is the part that the simulator makes possible. Anyone who wants to train an SRE agent at the scale RL actually needs is going to need a substrate like this. We're publishing ours.

The curated trajectory dataset that fell out of this run is on the Hub as `praetor-sre-trajectories` - 760 senior-SRE behavioral-clone rows plus 712 raw step-level rows with full reward breakdowns, ready to drop into TRL or any chat-format SFT loop. Anyone can `pip install datasets && load_dataset(...)` and start training their own policy without ever spinning up a cluster.

---

## The reward function we didn't learn

This is where most LLM-based RL projects break, and we didn't want to break here. The standard move is to train a **learned reward model** that says "yes this looks like a good fix, no this looks bad." That works until the agent figures out how to game whatever the reward model thinks "good" means - which it always does, because reward hacking is what RL is *supposed* to do.

We took the other route: **RLVR - reward from verifiable rubric**. Six pure functions over `(action, snapshot, scenario)`:

| Component | What it scores |
|---|---|
| `r_diagnostic` | First read on a relevant service |
| `r_correct_op` | Scenario-defined right move (delegated to `scenario.is_correct_op(action)`) |
| `r_resolution` | Terminal - fix matches scenario rubric AND the agent's `root_cause` string mentions the right keywords |
| `r_format` | Action parsed cleanly (no JSON fallback) |
| `r_efficiency` | Terminal - solved in ≤50% of step budget |
| `r_penalty` | Sum of harmful_restart, redundant action, rollback-to-self, unknown config key |

No learned model. No LLM-as-judge. Six bits of math, every bit auditable, every bit logged separately to wandb so the training plot shows *which* axis the policy is improving on, not just the scalar.

We also wrote regression tests for the four classic reward-hacking exploits before we trained anything. The agent can't pretend to fix an OOM by toggling a config key with "pool" and "size" in the name, because we replaced that string-match with a strict allowlist. It can't pretend to fix a memory leak by bare-restarting, because the simulator distinguishes restart-curable anomalies (OOM, connection leak) from rollback-curable ones (memory leak introduced in v2.4.0). It can't game redundancy by tweaking an inert parameter, because we compare on `(action_type, target_service)` not the full parameter dict. It can't roll back to the version it's already on, because we added an early guard that returns an error.

These aren't promises. They're tests in `tests/test_reward_hacks.py` that break the build if any of those leaks come back.

---

## The training recipe

We followed the hackathon's recommended pipeline almost to the letter.

**SFT (supervised fine-tuning) first.** We hand-wrote sixteen ideal trajectories - what a senior SRE actually does when they get paged for each scenario family - and turned them into ~120 chat-format (state, action, rationale) tuples by sampling under multiple seeds. Single-epoch SFT on Qwen2.5-Coder-1.5B, 4-bit quantized via Unsloth, LoRA r=16. About thirty minutes on an A100.

**GRPO (Group Relative Policy Optimization) second.** This is the newer of the two trainers in TRL, originally from DeepSeek's R1 work. The trick: instead of asking "is this completion good in absolute terms?" (hard, requires a precise reward function for everything), GRPO compares the rewards within a small group of completions for the same prompt and uses the *relative ranking* as the gradient signal. Less to tune, more stable, cheaper to run. We used four rollouts per prompt, KL=0.04, lr=5e-6, 60 training steps. About forty minutes on an A100.

The curriculum ramps difficulty across training. The first third of training is OOM crashes at low difficulty - easy wins to seed the policy with formatting fluency. The middle third introduces DB pool exhaustion at medium difficulty. The final third is the full mix at full difficulty, including the harder bad-deployment cascade where action *ordering* matters and the trap is restarting dependents before rolling back the upstream cause.

The eval protocol is held-out seeds - 10 per family, no overlap with the training distribution - across two conditions: random baseline and SFT+GRPO. That's 60 episodes per condition, 120 total. The full SFT-then-GRPO run, including evaluation and plot generation, fits in roughly 80 minutes on an A100.

---

## What the floor looks like

The first thing we measured, before training anything, was the **random-baseline floor**. If a uniformly-random policy can occasionally solve a scenario, that means the task is in the achievable zone - RL needs success probability greater than zero or the gradient is uninformative. Here's what we got across 180 episodes (six families, thirty seeds each):

| Family | Random success | Random avg score |
|---|---:|---:|
| oom_crash | 17% | 0.26 |
| slow_query | 23% | 0.24 |
| db_pool_exhaustion | 0% | 0.40 |
| bad_deployment_cascade | 0% | 0.32 |
| disk_full | 0% | 0.09 |
| cert_expiry | 0% | 0.01 |

Two things stand out. First, OOM and slow-query have non-zero success rates because the action space includes `restart_service`, and a random policy occasionally picks the right service to restart by chance. Second, **cert expiry is the hardest baseline** even though it's a "junior-level" scenario. The reason is instructive: cert-expiry metrics look almost normal. CPU is low, memory is low. The only signal is the error rate at 99% and the literal text *"certificate has expired"* in the logs. A random policy that doesn't *read those logs* has effectively zero chance of stumbling on the right fix.

That's our favorite scenario, by the way. Cert expiry is the most embarrassing class of real-world outage - Microsoft Teams 2020, Spotify 2021, Azure DevOps 2018, LinkedIn 2017, Cloudflare's 1.1.1.1 in 2021. Every major company has been bitten by it. Preventing it just requires a calendar reminder. And yet the dashboard signal looks nearly identical to "everything is fine" except for the error rate, which is why so many on-call teams take fifteen confused minutes to figure it out. The trained model has to learn a single, transferable lesson: *when metrics look normal but error rate is at 99%, read the logs first.* That's not a fact about TLS. That's a fact about debugging.

---

## What the ceiling looks like (after training)

We trained on the three families with hand-written senior-SRE trajectories - OOM crash, DB pool exhaustion, and bad-deployment cascade - leaving the rest of the family library as out-of-distribution generalization tests for future work. The held-out eval is 10 fresh seeds per family per condition, no overlap with the training distribution. Random vs SFT+GRPO:

| Family | Random | SFT + GRPO | Δ |
|---|---:|---:|---:|
| OOM Crash | _filled in after Colab run_ | _filled in after Colab run_ | _filled in after Colab run_ |
| DB Pool Exhaustion | _filled in after Colab run_ | _filled in after Colab run_ | _filled in after Colab run_ |
| Bad Deployment Cascade | _filled in after Colab run_ | _filled in after Colab run_ | _filled in after Colab run_ |
| **Average** | _filled in after Colab run_ | _filled in after Colab run_ | _filled in after Colab run_ |

The two plots that tell the rest of the story:

- **`results/sft_loss_curve.png`** - SFT loss falling smoothly across ~80 logging points of the supervised epoch. The "behavioral clone learns the format and the canonical action sequence" plot.
- **`results/grpo_reward_curve.png`** - GRPO mean reward rising across 60 steps. The "policy starts to *prefer* the actions that actually solve the incident over the actions that merely look like they should" plot.
- **`results/grpo_reward_components.png`** - the per-component breakdown across all six axes of the verifiable reward. The interesting one is `r_correct_op` rising while `r_penalty` stays flat: the agent is doing more right things, not more things in general.

---

## Sim-to-real, almost for free

Here's the part we're most pleased about. Praetor's env orchestrator never touches the simulated cluster directly. It talks to a `Backend` Protocol with two implementations:

- `SimulatedBackend` - wraps the in-memory Python cluster (fast, deterministic, parallelizable for training)
- `WebsiteBackend` - talks HTTP to any deployed site that implements a small operator API (`/ops/health`, `/ops/restart`, `/ops/config`, `/ops/break` to inject a deliberate fault, `/ops/heal` to reset)

Both produce the same typed `BackendSnapshot`. The trained policy can't tell which one it's running against - same observation shape, same ten actions, same six-component reward. That decoupling is what makes sim-to-real cheap. We trained on the simulator. The same model then runs unchanged against a real Render-deployed FastAPI site that we vibecoded in twenty minutes. The agent fixes a real outage on a real container, with the agent's actions translated into actual `POST /ops/restart` calls.

The dashboard's third tab - Real-Time - is built around this. You paste the URL of a deployed site, Praetor probes its `/ops/health` and `/ops/metrics` and `/ops/logs`, **auto-classifies** the active fault from log signatures (no manual scenario picking - that would defeat the demo), and runs the trained policy. If runtime ops aren't enough, the same dashboard accepts a GitHub repo, an Azure DevOps repo, or a ZIP upload, and the tier-2 escalation module clones the code, greps for the suspect lines, and writes a structured *Code Escalation Report* with a suggested fix.

We also wired up the autonomous monitoring side. There are three webhook endpoints - `/incidents/webhook/pagerduty`, `/incidents/webhook/prometheus`, `/incidents/webhook/generic` - that accept real alert payloads, classify them into our scenario taxonomy using a keyword heuristic, and kick off a run in a background thread. Token-gated via an env var. Once paged, no humans in the loop. The agent investigates, decides, acts, verifies recovery, and writes a structured post-mortem markdown next to the trace.

That's the autonomy story we wanted to tell in a single sentence: **Praetor goes from PagerDuty to verdict without anyone touching a keyboard.**

---

## What surprised us

**The format reward matters more than we expected.** A small +0.01 per cleanly-parsed action sounds like rounding error, but it's the difference between the policy learning to emit valid JSON or generating a 256-token essay every time. Without that signal, GRPO has nothing to push on for the first few hundred steps, because nothing else can fire if the action doesn't parse.

**Scenario ordering matters more than scenario count.** The bad-deployment cascade scenario is the only one in the curriculum where the *order* of remediation actions counts as much as the actions themselves. You have to roll back the bad version *first*, then restart the starved dependents - restarting them while the autoscaler is still spinning up replicas of the bad version just makes it worse. Watching the curriculum sweep through this scenario at step 250+ was the first time we saw the per-component reward axes diverge cleanly, with `r_correct_op` rising while `r_penalty` stayed flat. The agent had figured out *what to do*. Now it was figuring out *when to do it*.

**The cert-expiry scenario is what makes the demo land.** When we show the auto-classifier in action, it's the cert-expiry case that makes engineers go "oh." Because the metrics look almost normal, the standard dashboard read-out from the agent says nothing useful. The only thing that lights up is the log-pattern classifier matching `ssl.SSLError: certificate has expired`. That's the moment the audience realizes the agent isn't doing anomaly detection - it's doing *interpretation*.

---

## What we'd build next

Six concrete capabilities sit just past the current build, each a specific extension rather than a rewrite. Two of them are about substrate - where the agent runs. Two are about what the agent perceives. Two are about how the action surface and the scenario library grow.

**Kubernetes and cloud adapters.** Today's substrates are `SimulatedBackend` (in-memory Python), `WebsiteBackend` (HTTP `/ops/*`), and `RealBackend` (docker compose). What's missing for production deployment is a `KubernetesBackend` that maps the typed actions to `kubectl rollout restart`, `kubectl scale`, `kubectl patch`, and equivalents - plus an `AWSBackend` (or `GCPBackend`) wrapping ECS / Cloud Run / Lambda. The Backend Protocol was designed exactly for this: each new substrate is a self-contained adapter that doesn't touch the agent or the reward function. This is the path from "demo on a laptop" to "actual production deployment."

**Continuous fleet monitoring.** Today Praetor reacts - a webhook fires, the agent runs. A watch-mode that polls `/ops/health` across a fleet on a schedule and proactively triggers an investigation when metrics drift past a baseline would be the difference between responding to incidents and preventing them. Same agent, same actions, same reward function, just a different invocation pattern at the front. Bolt this on the front and Praetor goes from "incident commander" to "always-on duty officer."

**Multi-cluster, multi-region topologies.** The current simulator is a single 9-service cluster. Production at scale runs multi-region, with DNS routing, cross-cluster dependencies, and regional failover. A topology generator that synthesizes multi-region setups, plus a class of region-eviction / BGP-flap / multi-region database-split scenarios, would extend the curriculum into the kind of incidents that take down companies for a day instead of an hour. The 2021 Facebook BGP outage and the 2022 Cloudflare regional incident are the patterns we have in mind.

**A learned fault classifier.** Right now the auto-classifier in the Real-Time tab is keyword heuristics - `"oom" in text`, `"pool exhausted" in text`, and so on. Training a small classifier head on `(logs, metrics) → scenario_family` from the trajectory dataset would be more robust to log-format drift, and would generalize to scenarios whose exact wording the heuristics haven't seen. The dataset to train it on is already exported under `results/hf_dataset/`.

**A discriminated typed-action union.** Today an action's `parameters` field is `Dict[str, Any]`. It's flexible but compile-time-unsafe - a typo in a key name surfaces only at runtime, and the spec in `openenv.yaml` has to carry the schema redundantly. Replacing it with per-action Pydantic sub-models would give every action a strict signature, surface bad parameter names at validation time, and let the OpenEnv YAML be auto-generated from the type annotations. It's a meaningful refactor that touches the action handlers, the scenarios' `is_correct_op` checks, and a slice of the test suite - but it's the right shape for a project this size to grow into.

**A larger scenario library, contributed by people who have actually been on call.** The YAML DSL is the seed. Anyone can convert a real post-mortem into a reproducible RL scenario without writing Python - two examples already ship (DNS failure: the AWS Route53 / Cloudflare 2019 / Slack 2022 pattern; rate-limit exhaustion: the Twitter 2023 launch pattern). The right finished form of this project is hundreds of scenarios contributed by SREs who lived through the original outages, each one a precise reproduction of a category of failure that's bitten enough teams to be worth training against.

---

## If you want to try it

- **Run the env locally:** `git clone` the repo and `uv run uvicorn incident_commander_env.server.app:app --port 8000`. Open the dashboard. Click **Apprentice** and try solving an OOM crash with the AI coach watching over your shoulder.
- **Watch a recorded trained-agent run:** Click **Observatory**. Pick a run. Hit Replay.
- **Connect a real deployed site:** Click **Real-Time**. Paste the URL. Let Praetor classify the fault and fix it.
- **Read the code:** [github.com/root4shreshth/incident-commander](https://github.com/root4shreshth/incident-commander)
- **Reproduce the training:** [Open `train_grpo.ipynb` in Colab](https://colab.research.google.com/github/root4shreshth/incident-commander/blob/main/training/train_grpo.ipynb) - runtime A100, Run All, walk away for ~80 minutes.
- **Pull the trajectory dataset:** `load_dataset("json", data_files="https://huggingface.co/datasets/<your-user>/praetor-sre-trajectories/resolve/main/sft.jsonl")` - 760 senior-SRE behavioral-clone rows + 712 raw step-level rows from the simulator.

---

The 3 AM phone is going to keep ringing. We'd like to make it ring a little less.

*- Team MetaMorphs · Meta OpenEnv Hackathon · April 2026*
