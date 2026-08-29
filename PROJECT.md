# Praetor Playbook Verifier

*A pre-production QA layer for SRE auto-remediation policies. Catch a bad policy before it reaches production, in ~8 seconds of CI.*

*For the deep-dive tech reference see [`README.md`](README.md); for the narrative build story see [`BLOG.md`](BLOG.md).*

---

## In one sentence

**Praetor is a Playbook Verifier for SRE auto-remediation policies.** You write a new `restart-on-OOM` or `rollback-on-deadlock` policy in YAML, run `praetor verify`, and get back a per-scenario pass/fail report telling you whether the policy actually fixes what it claims to fix, whether it misfires on unrelated incidents, and whether it produces net-negative reward (breaks things). Backed by a 12-scenario incident library covering both generic-SRE and payments-industry faults, plus a deterministic simulator that reproduces each incident byte-for-byte.

---

## The problem

Every payments-scale engineering team eventually builds runbook automation - "if X alert fires, take Y remediation action, page a human only if it fails." The classic examples: auto-restart on OOM, auto-scale on latency spike, auto-rollback on error-rate breach. These policies live in PagerDuty/Rundeck/StackStorm YAML, or in Kubernetes operator code, or in a `runbooks/` folder in the platform-eng repo.

**And they're almost never tested before rollout.**

- A policy that restarts on any CRITICAL alert also restarts `postgres-db` when the DB throws a slow-query warning. Nobody noticed until 3 AM on a Saturday.
- A policy that auto-rolls-back a service when its error rate exceeds 5% rolled back a `refund-service` deploy that was fixing a data-integrity bug. The rollback re-introduced the bug for two hours.
- A policy that scales up `payment-gateway` when the outbound-pool alert fires kept scaling during a partial upstream outage, blowing through the AWS ECS task budget in 12 minutes.

Every one of these is a real shape. The industry answer today is either **manual review by a senior SRE** (bottleneck, doesn't scale) or **ship and hope** (works until it doesn't). There's no `pytest` for ops policies.

The industry also lacks a shared benchmark to test against. Chaos-engineering tools (Chaos Mesh, Litmus, Gremlin) let you inject faults, but they're for QA'ing your **infrastructure**, not your **remediation policies**. You still write the tests yourself, per company, from scratch.

---

## What Praetor is

Praetor closes both gaps in one product.

### The Playbook Verifier

A CLI + GitHub Actions plugin that runs any auto-remediation policy against a canonical 12-scenario incident library, and produces a per-scenario verdict. Sample output:

```
Praetor Playbook Verifier | oom_auto_restart v1.0.0
Claims to fix: oom_crash

  [PASS]  oom_crash                     C trig res   steps=2   R=+0.32
  [PASS]  bad_deployment_cascade        - ---- ----  steps=0   R=+0.00
  [PASS]  webhook_delivery_backlog      - ---- ----  steps=0   R=+0.00
  ... (12 total)

Overall: PASS   pass=12  warn=0  fail=0   (100% pass rate on 12 scenarios)
```

Each scenario answers three questions the operator needs to know before shipping:

1. **Does the policy trigger on incidents it claims to fix?** If not: `FAIL claimed_but_not_triggered`.
2. **Does the policy trigger on incidents it doesn't claim?** If yes and it did no harm: `WARN false_positive`. If yes and it produced net-negative reward: `FAIL false_positive_negative_reward`.
3. **When triggered, does the policy actually resolve the incident?** If not: `FAIL triggered_but_no_resolve`.

The reward metric is the sim's 6-component `RewardBreakdown` (r_diagnostic, r_correct_op, r_resolution, r_format, r_efficiency, r_penalty) - a fully verifiable rubric, no learned reward model, so nothing to game.

### The substrate that makes it possible

- **A deterministic, seeded simulator** of a 14-service microservices cluster (9 core e-commerce + 5 payments-industry: payment-gateway, webhook-consumer, fraud-check, refund-service, ledger-service). Resets in ~0.5 ms per scenario, so verifying a policy against all 12 scenarios takes ~8 seconds on a laptop.
- **12 incident scenario families** covering generic SRE (OOM, DB pool exhaustion, bad-deploy cascade, disk full, slow query, cert expiry, DNS failure, rate-limit exhaustion) and payments-specific (`payment_gateway_timeout`, `webhook_delivery_backlog`, `fraud_check_memory_blowup`, `refund_race_deadlock` - the ordering-sensitive lock-order-bug one).
- **A YAML DSL for policy authoring** with trigger matching (alert content, service pattern, severity), templated action sequences (`{trigger.service}`), and safeguards (rate limits, require-human-confirmation-if).
- **A trained LLM agent** (Qwen2.5-Coder-1.5B, SFT + GRPO on the 6-component reward) that can serve as a baseline "what a competent operator would do here" for policy comparison.
- **Sim-to-real bridge** via the `Backend` Protocol - the same policy can be verified against the sim, then executed against a real deployed payments service (SwiftPay - a second HuggingFace Space that implements the operator contract) for staged rollout.

### The GitHub Actions integration

Every PR that touches `policies/*.yaml` triggers `praetor verify`. The workflow posts a Markdown report as a PR comment (per-scenario table) and blocks merge if any policy in the diff produces a FAIL verdict.

---

## How it works - one end-to-end flow

You're a platform engineer at a payments company. You want to add an auto-remediation policy for the webhook-backlog case: whenever `webhook-consumer` queue depth alerts, restart to drain stuck connections. You write:

```yaml
# policies/webhook_backlog_drain.yaml
name: webhook_backlog_drain
version: 1.0.0
owner: payments-platform@example.com
scenarios_claimed: [webhook_delivery_backlog]

trigger:
  event: alert
  match:
    alert_severity: WARNING
    service_pattern: "webhook-consumer"
    message_contains: [queue depth, backlog, delivery lagging]

actions:
  - action_type: read_logs
    target_service: webhook-consumer
    parameters: {lines: 60, severity: WARN}
  - action_type: restart_service
    target_service: webhook-consumer
  - action_type: resolve_incident
    parameters:
      root_cause: "webhook-consumer delivery workers blocked on stuck merchant HTTP connections"
      resolution: "Restarted to drain stuck connections"

safeguards:
  max_actions_per_hour: 6
```

You open a PR. GitHub Actions kicks off `praetor verify policies/webhook_backlog_drain.yaml` and posts this back as a comment within ~10 seconds:

> ## Praetor Playbook Verifier — `webhook_backlog_drain` v1.0.0
>
> **Verdict:** `PASS` — 12 pass / 0 warn / 0 fail across 12 scenarios (100% pass rate).
>
> **Claims to handle:** `webhook_delivery_backlog`
>
> | Scenario | Verdict | Claimed | Triggered | Resolved | Steps | Reward |
> |---|:-:|:-:|:-:|:-:|---:|---:|
> | `oom_crash` | **PASS** | - | - | - | 0 | +0.00 |
> | `webhook_delivery_backlog` | **PASS** | yes | yes | yes | 2 | +0.32 |
> | *(10 more rows)* | | | | | | |
>
> Generated by `praetor verify`.

Merge is unblocked. The policy ships.

Now imagine your co-worker later submits a *different* policy that also matches on "WARNING" alerts but restarts `payment-gateway`. The verifier catches the collision - both trigger on `webhook_delivery_backlog`, but the second one is a false positive that restarts an unrelated service. FAIL. PR blocked. Bug never reaches production.

That's the whole product.

---

## Why this is the right shape for Razorpay

Payments infrastructure teams have three concerns that the verifier speaks to directly:

1. **Reliability.** Policies are code; code without tests will break in production. Praetor is `pytest` for policies.
2. **Auditability.** Every verdict is a structured JSON report you can attach to a compliance ticket. "Before we deployed this rule, we verified it passed against the 12 canonical incident families" is a defensible answer to an auditor.
3. **Correctness under money-flow stakes.** The `refund_race_deadlock` scenario is deliberately ordering-sensitive: the policy has to rollback THEN restart, not the other way around. If the verifier catches a rollback-before-restart bug in your policy before it hits your ledger, you just avoided a two-hour reconciliation nightmare.

The payments-industry scenarios were designed against real war stories: Stripe subscription proration 2017, Adyen double-entry 2020, PayPal webhook lag, Razorpay upstream-processor timeouts during peak sale windows. A policy that passes all 12 of these has been stress-tested against 12 patterns that have actually put payments teams on-call.

---

## Try it

- **Verify a shipped example policy:**
  ```bash
  git clone https://github.com/root4shreshth/incident-commander
  cd incident-commander
  uv sync
  uv run python scripts/verify_policy.py policies/oom_auto_restart.yaml
  ```
- **Author your own policy:** copy any file in [`policies/`](policies/) and edit. See the schema in [`praetor_verify/policy.py`](praetor_verify/policy.py) or the four commented examples in that folder.
- **Wire it into CI:** the workflow at [`.github/workflows/policy-verify.yml`](.github/workflows/policy-verify.yml) is copy-paste-ready for any repo that stores policies as YAML.
- **See what a FAIL looks like:** [`policies/_bad_example_trigger_happy.yaml`](policies/_bad_example_trigger_happy.yaml) is an intentionally over-broad policy kept in the repo as a regression test. Run `praetor verify` on it and watch every scenario FAIL.
- **Play with the sim directly:** the live demo at [hype4raj-incident-commander-env.hf.space](https://hype4raj-incident-commander-env.hf.space) lets you replay recorded runs (Observatory tab), solve scenarios by hand with an AI coach (Apprentice), or point a trained agent at a real deployed payments target (Real-Time, with [SwiftPay](https://shreshthn8n-swiftpay-target.hf.space) as the built-in target).

---

## What's under the hood

- **Simulator + 12 scenarios** — deterministic, seeded, ~1,900 resets/sec on a laptop. The verifier's speed comes from this: one policy against 12 scenarios × 1 seed each = ~8 seconds wall-clock.
- **6-component verifiable reward** — the metric the verifier uses to detect "policy did nothing wrong but net-negatively reward'd because it restarted a healthy service" (harmful-restart penalty).
- **Trained baseline agent** — Qwen2.5-Coder-1.5B, SFT + GRPO, LoRA r=16. Useful as a "what a competent operator would do" reference; not part of the verifier itself but shipped alongside as a peer artifact.
- **Sim-to-real bridge** — the `Backend` Protocol lets the same policy that passed verification also be executed against a real deployed target site (SwiftPay). Staged rollout: verify in sim, canary against SwiftPay, then production.
- **367+ tests** across the sim, the scenario library, the reward pipeline, the policy DSL, and the verifier runner. `pytest` in ~10 seconds.

---

## Why this project exists

Originally built for the **Meta OpenEnv Hackathon (April 2026, Theme #3.1: Professional Tasks)** as a generic-SRE RL environment. Extended in August 2026 with the payments-industry scenario library and working GRPO training pipeline. Then pivoted from "portfolio piece" to "industrial-perspective solution" by wrapping the sim + scenarios + reward pipeline in the Playbook Verifier - the product a real platform team would actually deploy.

The design philosophy is that fintech infrastructure teams don't need another dashboard or another AI agent. They need **the CI system for their ops policies** - a way to catch a bad automation before it fires on real traffic at 3 AM. The verifier is that.

---

## What's next

The single largest remaining piece is a **`KubernetesBackend`** implementation so the verifier can also execute policies against a real local `kind` cluster (not just the sim). The scoping doc lives at [`~/.claude/plans/gentle-doodling-bee.md`](../.claude/plans/gentle-doodling-bee.md); it's roughly a week of work and needs Docker + `kind` installed. When that ships, `praetor verify` becomes "run the policy against the sim first, then against a real cluster, then decide."

Beyond that: a **learned fault classifier** to replace the keyword-heuristic in the verifier's trigger-matching (would use the trajectory dataset at `results/hf_dataset/` to train), **multi-cluster / multi-region topologies** in the scenario library (to model BGP-flap and region-eviction outages), and **integration with a real policy engine** (StackStorm sensors, Rundeck jobs, or Kubernetes operator controllers) so the same YAML that passes verification is what the runtime executes.

---

*Built by [Shreshth](https://github.com/root4shreshth) and [Yasin](https://github.com/hype4raj) · Team MetaMorphs · April 2026 hackathon origin, August 2026 payments-industry + Playbook Verifier extension.*
