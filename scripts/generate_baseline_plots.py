"""Generate the baseline plot suite for the README + blog.

We don't have GPU access here, so the *trained* condition curves can only be
produced by running `training/train_grpo.ipynb` on Colab. What we CAN produce
right now - and what's worth committing - is a real measurement of the
random-policy floor across all 6 scenario families × 30 seeds.

That gives the README + blog four useful artifacts:

  results/baseline_reward_per_episode.png   - reward signal across 180 eps
  results/baseline_reward_components.png    - the 6 reward axes per ep
  results/baseline_success_rates.png        - per-family success bars (single
                                              random condition; trained
                                              conditions added post-Colab)
  results/baseline_action_distribution.png  - action mix random uses
  results/baseline_summary.json             - machine-readable numbers

Honest framing: these are the FLOOR every trained condition has to beat.
When `train_grpo.ipynb` runs on A100, the same plot files get re-rendered
with the SFT and SFT+GRPO conditions appended.

Usage:
  uv run python scripts/generate_baseline_plots.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on sys.path so imports work regardless of CWD
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.datasets import SYSTEM_PROMPT
from training.eval_runner import evaluate, random_policy
from training.plots import (
    make_action_distribution,
    make_reward_components,
    make_reward_curve,
    make_success_bars,
    save_figure,
)


FAMILIES = [
    "oom_crash",
    "db_pool_exhaustion",
    "bad_deployment_cascade",
    "disk_full",
    "slow_query",
    "cert_expiry",
]
SEEDS = list(range(4000, 4030))  # 30 held-out seeds, no overlap with eval seed ranges


def main() -> None:
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] running random-policy eval - {len(FAMILIES)} families × {len(SEEDS)} seeds = {len(FAMILIES)*len(SEEDS)} episodes")
    n_done = [0]
    def _progress(idx, total, family, seed, ep):
        n_done[0] += 1
        if n_done[0] % 20 == 0 or n_done[0] == total:
            print(f"  [{idx:>3}/{total}] {family:<25s} seed={seed} steps={ep.steps_used:>2} resolved={ep.resolved}")

    report = evaluate(
        "random-baseline",
        random_policy(rng_seed=42),
        families=FAMILIES,
        seeds=SEEDS,
        system_prompt=SYSTEM_PROMPT,
        on_episode=_progress,
    )

    # ──────────────────── Plot 1: reward per episode ────────────────────
    # X-axis is "episode index" (1..180) since we don't have a training step
    # axis. Sums each episode's per-component reward into a single scalar.
    rewards_per_ep = [
        sum(ep.breakdown_totals.values()) for ep in report.episodes
    ]
    fig = make_reward_curve(
        steps=list(range(1, len(rewards_per_ep) + 1)),
        rewards=rewards_per_ep,
        window=20,
    )
    # Override title - these aren't training rewards, they're per-episode.
    fig.axes[0].set_title("Random-baseline floor: reward per episode (180 episodes)")
    fig.axes[0].set_xlabel("episode index (across 6 scenario families)")
    save_figure(fig, str(out_dir / "baseline_reward_per_episode.png"))
    print("[baseline] wrote baseline_reward_per_episode.png")

    # ──────────────────── Plot 2: 6 components over episodes ────────────
    components_over_time = {
        k: [ep.breakdown_totals.get(k, 0.0) for ep in report.episodes]
        for k in ("diagnostic", "correct_op", "resolution",
                  "format", "efficiency", "penalty")
    }
    fig = make_reward_components(
        steps=list(range(1, len(report.episodes) + 1)),
        components_over_time=components_over_time,
    )
    fig.axes[0].set_title(
        "Random-baseline 6-axis breakdown - the floor every trained run beats"
    )
    fig.axes[0].set_xlabel("episode index")
    save_figure(fig, str(out_dir / "baseline_reward_components.png"))
    print("[baseline] wrote baseline_reward_components.png")

    # ──────────────────── Plot 3: success rates by family ───────────────
    fig = make_success_bars(
        reports_by_condition={"random": report.by_family},
        families=FAMILIES,
    )
    fig.axes[0].set_title("Random-policy success rate per scenario (n=30 seeds each)")
    save_figure(fig, str(out_dir / "baseline_success_rates.png"))
    print("[baseline] wrote baseline_success_rates.png")

    # ──────────────────── Plot 4: action distribution ───────────────────
    actions_random: dict = defaultdict(int)
    for ep in report.episodes:
        for a, _t in ep.actions:
            actions_random[a] += 1
    fig = make_action_distribution(
        actions_by_condition={"random": dict(actions_random)},
    )
    fig.axes[0].set_title(
        "Action distribution: random-policy floor "
        "(trained mix added post-Colab)"
    )
    save_figure(fig, str(out_dir / "baseline_action_distribution.png"))
    print("[baseline] wrote baseline_action_distribution.png")

    # ──────────────────── JSON summary ──────────────────────────────────
    summary = {
        "condition": "random-baseline",
        "n_families": len(FAMILIES),
        "n_seeds": len(SEEDS),
        "n_episodes": report.n_episodes,
        "by_family": {
            fam: {
                "success_rate": stats["success_rate"],
                "avg_score": stats["avg_score"],
                "avg_steps_used": stats["avg_steps_used"],
            }
            for fam, stats in report.by_family.items()
        },
        "note": (
            "Random-policy floor across all 6 scenario families. "
            "The trained conditions (SFT, SFT+GRPO) are produced by running "
            "training/train_grpo.ipynb on Colab - those numbers replace this "
            "file's contents post-run."
        ),
    }
    with (out_dir / "baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[baseline] wrote baseline_summary.json")

    # ──────────────────── Print headline ────────────────────────────────
    print("\n" + "=" * 60)
    print("Random baseline floor (per family):")
    for fam, stats in report.by_family.items():
        print(f"  {fam:<28s}  success={stats['success_rate']*100:>4.0f}%  "
              f"score={stats['avg_score']:.2f}  steps={stats['avg_steps_used']:.1f}")
    print("=" * 60)
    print(f"\nArtifacts written to {out_dir}/")


if __name__ == "__main__":
    main()
