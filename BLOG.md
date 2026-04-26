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

Other hackathon teams have trained SRE-shaped agents on *real* Kubernetes clusters. That's the realistic substrate - but there's a catch the published numbers don't dwell on. A real `kubectl rollout undo` plus pod-recreation plus health-stabilization cycle takes roughly **60 seconds**. RL training wants hundreds of thousands of trajectories. The math doesn't work.

Praetor's simulator measured on a stock laptop: **1,905 resets/second, 6,425 environment steps/second** - a **~114,000× speedup** over real Kubernetes for the inner loop of RL. Same scenario surface, same reward function, same actions, just deterministic and seeded so any `(family, seed, difficulty)` triple regenerates byte-identically. Numbers committed to `results/throughput.json`, reproducible via `python scripts/benchmark_throughput.py`.

That speedup is the entire data-factory thesis. *Of course* you should eventually train on real clusters - the Backend Protocol is exactly the seam to do that, and `WebsiteBackend` already runs the trained policy unchanged against a real deployed site (more on that below). But the hot loop where the agent figures out the *shape* of incident response - the part where you need millions of reset-step-evaluate cycles - is the part the simulator makes possible. We're publishing ours so the next team doesn't have to rebuild it.

The trajectory dataset that fell out of this run is committed under `results/hf_dataset/` - 760 senior-SRE behavioral-clone rows plus 712 raw step-level rows with full reward breakdowns, ready to drop into TRL or any chat-format SFT loop without ever spinning up a cluster.

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

**SFT (supervised fine-tuning) first.** We hand-wrote senior-SRE ideal trajectories for the six built-in scenario families - what an experienced engineer actually does when they get paged - and turned them into ~120 chat-format (state, action, rationale) tuples by replaying under multiple seeds. Single-epoch SFT on Qwen2.5-Coder-1.5B, 4-bit quantized via Unsloth, LoRA r=16. Roughly thirty minutes on an A100.

**GRPO (Group Relative Policy Optimization) second.** The newer trainer in TRL, originally from DeepSeek's R1 work. Instead of scoring completions in absolute terms, GRPO compares rewards within a small group of completions for the same prompt and uses the *relative ranking* as the gradient signal. Less to tune, more stable, cheaper to run. Four rollouts per prompt, KL=0.04, lr=5e-6, 60 steps. About forty minutes on an A100.

The curriculum ramps difficulty across training - OOM crashes first (easy wins to seed formatting fluency), then DB pool exhaustion, then the full mix including the bad-deployment cascade where action *ordering* matters as much as action choice. The held-out eval is 10 fresh seeds per family with no overlap with training. The full SFT-then-GRPO run, including evaluation and plots, fits in roughly 80 minutes on an A100.

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

The Colab run is in flight as of submission. The headline numbers populate `results/eval_summary.json` and the three plots below drop into `results/` once it finishes - the public ones are committed to the repo and visible from the Observatory tab as soon as they exist.

The interesting plot to watch is `results/grpo_reward_components.png` - the per-component breakdown across all six reward axes. The signal we expect (and saw on shorter trial runs) is `r_correct_op` rising while `r_penalty` stays flat: the agent doing *more right things*, not *more things in general*. That divergence is the difference between a policy that learned and a policy that just got busier.

Until then, the random-baseline floor table above is the honest comparison anchor - 0% success on cert-expiry, 0% on disk-full, 17% on OOM crash by accidental restart. Anything the trained policy beats that floor on, it earned.

---

## Sim-to-real, almost for free

Praetor's env orchestrator never touches the simulated cluster directly. It talks to a `Backend` Protocol with two implementations - `SimulatedBackend` (in-memory, deterministic, parallelizable for training) and `WebsiteBackend` (HTTP to any deployed site that implements a small operator API). Both produce the same typed `BackendSnapshot`. The trained policy can't tell which one it's running against, which is what makes sim-to-real cheap.

To prove this we built a second HuggingFace Space called **SwiftPay** ([shreshthn8n-swiftpay-target.hf.space](https://shreshthn8n-swiftpay-target.hf.space)) - a real deployed payments site that implements the operator contract and exposes three deliberate fault routes. Praetor's third dashboard tab, *Real-Time*, asks for a target URL. You paste SwiftPay's, Praetor probes its `/ops/health` and `/ops/metrics`, **auto-classifies** the active fault from log signatures (no manual scenario picking - that would defeat the demo), and runs the policy. The same trained model fixes a real outage on a real container, with each typed action translated into an actual `POST /ops/restart` or `POST /ops/configure` call.

Three things we built to make the result legible to a human operator. First, every step in the live timeline has a **Why this step?** expander that surfaces the policy's reasoning trace - this is an explainable agent, not a black box. Second, when the run completes, a **Final Report** card renders with the status pill, a one-paragraph Praetor narrative summary, a stats grid, color-coded root-cause / fix / service tags, the resolution path, and a per-action-type breakdown. Third, a **📄 Export as PDF** button hits a server-side reportlab pipeline and downloads the whole report as a real `.pdf` - cover page with status, every step with its rationale, footer with run ID on every page. Compliance teams have asked for that artifact for years. Now they get it on every incident.

The autonomous side is wired to the same machinery. Three webhook endpoints - PagerDuty, Prometheus, and a generic minimal contract - accept real alert payloads, classify them, and kick off a run in a background thread. Token-gated. Once paged, no humans in the loop. **Praetor goes from PagerDuty to verdict without anyone touching a keyboard**, and the verdict is a downloadable PDF next to a JSONL trace next to a markdown post-mortem.

---

## What surprised us

**The format reward matters more than we expected.** A small +0.01 per cleanly-parsed action sounds like rounding error, but it's the difference between the policy learning to emit valid JSON or generating a 256-token essay every time. Without that signal, GRPO has nothing to push on for the first few hundred steps, because nothing else can fire if the action doesn't parse.

**Scenario ordering matters more than scenario count.** The bad-deployment cascade scenario is the only one in the curriculum where the *order* of remediation actions counts as much as the actions themselves. You have to roll back the bad version *first*, then restart the starved dependents - restarting them while the autoscaler is still spinning up replicas of the bad version just makes it worse. Watching the curriculum sweep through this scenario at step 250+ was the first time we saw the per-component reward axes diverge cleanly, with `r_correct_op` rising while `r_penalty` stayed flat. The agent had figured out *what to do*. Now it was figuring out *when to do it*.

**The cert-expiry scenario is what makes the demo land.** When we show the auto-classifier in action, it's the cert-expiry case that makes engineers go "oh." Because the metrics look almost normal, the standard dashboard read-out from the agent says nothing useful. The only thing that lights up is the log-pattern classifier matching `ssl.SSLError: certificate has expired`. That's the moment the audience realizes the agent isn't doing anomaly detection - it's doing *interpretation*.

---

## What we'd build next

Six concrete extensions, none a rewrite. Two are about *substrate*, two about *perception*, two about *the action surface and curriculum*.

**Substrate.** A `KubernetesBackend` mapping the typed actions to `kubectl rollout restart` / `scale` / `patch`, and an `AWSBackend` wrapping ECS / Cloud Run / Lambda. The Backend Protocol was designed for exactly this - each new substrate is a self-contained adapter that doesn't touch the agent or the reward. Plus a watch-mode that polls `/ops/health` across a fleet and triggers proactive investigations when metrics drift, turning Praetor from incident commander into always-on duty officer.

**Perception.** A learned fault classifier replacing today's keyword heuristics, trained on the `(logs, metrics) → scenario_family` rows already exported under `results/hf_dataset/`. And multi-cluster / multi-region topologies for the curriculum - region-eviction, BGP-flap, cross-region database-split - the shapes that took down Facebook in 2021 and Cloudflare in 2022.

**Action surface and curriculum.** Replacing `parameters: Dict[str, Any]` with per-action Pydantic sub-models so the OpenEnv YAML auto-generates from type annotations and bad keys surface at validation time. And a hundred more scenarios contributed via the YAML DSL by people who have actually been on call - the right finished form of this project is a library of precise reproductions of every category of failure that's bitten enough teams to be worth training against.

---

## If you want to try it

- **Live demo, no setup:** open [hype4raj-incident-commander-env.hf.space](https://hype4raj-incident-commander-env.hf.space) → tab **3 Real-Time** → paste `https://shreshthn8n-swiftpay-target.hf.space` → Connect → Run agent. Watch the timeline, read the Final Report, click 📄 Export as PDF.
- **Try a scenario yourself:** same Space, tab **2 Apprentice**. Pick "Your first page," solve an OOM crash with the AI coach watching over your shoulder.
- **Replay a trained-agent run:** same Space, tab **1 Observatory**. Pick a run from the dropdown, hit Replay.
- **Read the code:** [github.com/root4shreshth/incident-commander](https://github.com/root4shreshth/incident-commander) - the [End-to-end workflow](https://github.com/root4shreshth/incident-commander#end-to-end-workflow---the-path-a-judge-actually-walks) section in the README is the full step-by-step.
- **Reproduce the training:** [Open `train_grpo.ipynb` in Colab](https://colab.research.google.com/github/root4shreshth/incident-commander/blob/main/training/train_grpo.ipynb), A100 runtime, Run All, walk away for ~80 minutes.
- **Pull the trajectory dataset:** 760 senior-SRE behavioral-clone rows + 712 raw step-level rows under `results/hf_dataset/` - drop into TRL or any chat-format SFT loop.

---

The 3 AM phone is going to keep ringing. We'd like to make it ring a little less.

*- Team MetaMorphs · Meta OpenEnv Hackathon · April 2026*
