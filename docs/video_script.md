# IncidentCommanderEnv — 90-second video shot list

**Total runtime:** 90s (hard cap — submission requirement).
**Format:** 1080p 30fps screen recording, voiceover narration.
**Tone:** confident, fast-paced, technical-but-accessible.

---

## Pre-record checklist

- [ ] HF Space green: `curl https://hype4raj-incident-commander-env.hf.space/health` returns `{"status":"ok"}`
- [ ] Trained LoRA pushed to HF Hub
- [ ] `runs/` has at least one trained-agent run on each family (used for observe-mode replay)
- [ ] User's vibecoded site running under `targets/site/` with `chaos.py` ready
- [ ] OBS / screen recorder configured for 1080p, no system notifications
- [ ] Voiceover script printed alongside the keyboard
- [ ] Two browser tabs ready: (a) wandb dashboard, (b) `localhost:8000/observe`

---

## Act 1 — The hook (0–10s)

**Visual:** Black screen, terminal opens, `curl` hits a service health endpoint and returns `503`. Cut to a Slack channel with a fake `#oncall-alerts` PagerDuty message: *"payment-service is failing — investigating?"*

**Voiceover (10s):**
> "On-call SRE work is methodical, time-pressured, and unpresentable as an RL benchmark — until now. We built IncidentCommanderEnv, an OpenEnv environment for incident response. And we trained an LLM to do the job."

---

## Act 2 — The training (10–25s, ~15s)

**Visual:** Jump-cut into the wandb dashboard. Show the 6-component reward curves rising at different rates. Pin the cursor on `r_resolution` first, then `r_correct_op`, then `r_efficiency`. End on a side-by-side: SFT-only vs SFT+GRPO success bars across the three scenario families.

**Voiceover (15s):**
> "Six independent reward components, every one a pure verifiable function — no learned reward model, no LLM-as-judge, no exploits. SFT seeded the policy from sixteen expert traces; GRPO took it the rest of the way. Watch each component rise at its own rate — diagnostic first, correct ops next, efficient resolution last."

---

## Act 3a — Eval on simulator (25–40s, ~15s)

**Visual:** Open `/observe` in the browser. Pick one of the trained-agent runs from the dropdown (one that resolved a `bad_deployment_cascade`). Hit "Replay." Steps stream in: `list_services` → `read_logs order-service` → `describe_service order-service` → `rollback_deployment to_version=v2.3.1` → `restart_service inventory-service` → `restart_service notification-service` → `resolve_incident`. Service map: red → orange → green.

**Voiceover (15s):**
> "Here's the trained agent on a hard scenario in simulation — bad deployment with cascading failures. It maps the blast radius, finds the bad version, rolls back, then restarts the starved dependents in the right order. Eighteen steps to resolved."

---

## Act 3b — Sim-to-real (40–80s, ~40s — the big one)

**Visual:**
- Cut to terminal: `BACKEND=real COMPOSE_ROOT=./targets/site uv run uvicorn …`
- Split-screen: left = `docker compose ps`, right = browser at the user-vibecoded site, `/cart` page.
- Run `python chaos.py --scenario=oom` in a third pane. The cart page starts returning 500s. `docker stats` shows `api` memory climbing past its 256MB limit.
- Now run the trained agent against `BACKEND=real`: `python -m training.eval_runner --condition=trained --task=oom_crash --backend=real`
- Cut to `/observe` showing the new run streaming in real time.
- Trained agent: `read_logs api` → `check_metrics api` → "OOM detected" → `restart_service api memory_limit=1024Mi`
- Split-screen: cart page recovers (200s), `docker stats` shows api at 256MB now under a 1024MB limit, `/observe` shows resolved=true.

**Voiceover (40s, paced):**
> "Now the same trained policy on a real Docker stack. Same OpenEnv API — `reset`, `step`, `state` — different substrate. We trigger a real out-of-memory crash on a real container. The agent picks it up through the same Backend Protocol, runs the same diagnostic actions, and ships the same fix — `restart_service` with a higher memory limit — translated into a `docker compose up --force-recreate` shell-out. The cart page recovers. No retraining. No fine-tuning. The simulator and the real stack are interchangeable from the agent's point of view."

---

## Act 4 — The close (80–90s)

**Visual:** Slide with three bullets:
- 🚨 First OpenEnv environment for SRE/DevOps
- 🧠 SFT + GRPO with 6-component verifiable rubric
- 🔁 Backend Protocol → sim-to-real for free

Then the submission links:
- 🤗 `hype4raj-incident-commander-env.hf.space`
- 📓 `train_grpo.ipynb` (Colab)
- 📝 Blog: `docs/blog.md`
- 📦 LoRA adapter on HF Hub

**Voiceover (10s):**
> "First OpenEnv for SRE. Six-component verifiable reward. Sim-to-real for free through a typed Backend Protocol. IncidentCommanderEnv — links in the description."

---

## Editing notes

- **Music**: low ambient pad, no drum line. Drop volume during voiceover, let `chaos.py` triggering the outage be a small audible "hit."
- **Captions**: hardcoded, white on black, top-aligned. Always show what action the agent took.
- **Cuts**: never sit on a screen for more than 4 seconds without zooming or panning.
- **Watermarks**: bottom-right, project logo + "Meta OpenEnv Hackathon · April 2026."

## Backup if Real-stack demo fails on recording day

Drop Act 3b to 20s using a pre-recorded successful run from `runs/<run_id>/episode.jsonl` replayed in `/observe`. Use the saved 5s before/after browser screenshots of the cart page. Voiceover stays the same — just the live execution is replaced with a polished replay.
