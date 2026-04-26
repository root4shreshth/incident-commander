"""Matplotlib plot helpers for the storytelling artifacts.

Every function takes raw data and returns a matplotlib Figure; the Colab
saves them to `results/<name>.png` with `dpi=120` so they embed cleanly in
the README and HF blog.

Plot inventory (the four images that anchor the submission's
"Showing Improvement in Rewards" judging axis):

  1. make_reward_curve          - total reward over training steps
  2. make_reward_components     - six reward components on the same axes (the killer image)
  3. make_success_bars          - per-scenario success rate per condition (bar chart)
  4. make_action_distribution   - pre/post action histogram
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _lazy_mpl():
    """Lazy-import matplotlib so this module imports without it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433
    return plt


_COMPONENT_COLORS = {
    "diagnostic": "#3b82f6",   # blue
    "correct_op": "#22c55e",   # green
    "resolution": "#a855f7",   # purple
    "format":     "#94a3b8",   # gray
    "efficiency": "#f59e0b",   # orange
    "penalty":    "#ef4444",   # red
}


def make_reward_curve(steps: List[int], rewards: List[float], window: int = 20):
    """Total reward per training step + a smoothed moving-average overlay."""
    plt = _lazy_mpl()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, rewards, alpha=0.25, color="#3b82f6", label="per-step total reward")
    if len(rewards) >= window:
        smoothed = [
            sum(rewards[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(rewards))
        ]
        ax.plot(steps, smoothed, color="#1d4ed8", lw=2, label=f"moving avg ({window})")
    ax.set_xlabel("training step")
    ax.set_ylabel("episode total reward")
    ax.set_title("GRPO training reward")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def make_reward_components(
    steps: List[int],
    components_over_time: Dict[str, List[float]],
):
    """All six reward components plotted on the same axes.

    This is THE submission's killer image. Shows the agent learning to maximize
    the rewarded components (correct_op, resolution, efficiency) while reducing
    the penalty component (redundant + harmful actions). Anti-reward-hacking
    evidence: if any single component dominated, that'd be the gaming signal.
    """
    plt = _lazy_mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ["diagnostic", "correct_op", "resolution", "format", "efficiency", "penalty"]:
        values = components_over_time.get(name, [])
        if not values:
            continue
        ax.plot(steps[:len(values)], values, label=name, color=_COMPONENT_COLORS[name], lw=1.5)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("component reward (running avg)")
    ax.set_title("Reward components during GRPO training")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    fig.tight_layout()
    return fig


def make_success_bars(
    reports_by_condition: Dict[str, Dict[str, Dict[str, Any]]],
    families: List[str],
):
    """Grouped bar chart: rows are conditions, bars are per-scenario success rate.

    `reports_by_condition` shape:
        {"random": {"oom_crash": {"success_rate": 0.13, ...}, ...},
         "base":   {...},
         "sft":    {...},
         "sft+grpo": {...}}
    """
    plt = _lazy_mpl()
    conditions = list(reports_by_condition.keys())
    n_conds = len(conditions)
    n_fams = len(families)
    width = 0.8 / max(1, n_conds)

    fig, ax = plt.subplots(figsize=(9, 5))
    cond_colors = {
        "random":   "#94a3b8",
        "base":     "#3b82f6",
        "sft":      "#a855f7",
        "sft+grpo": "#22c55e",
    }
    for ci, cond in enumerate(conditions):
        rates = []
        for fam in families:
            stats = reports_by_condition[cond].get(fam, {})
            rates.append(stats.get("success_rate", 0.0))
        xs = [i + ci * width - 0.4 + width / 2 for i in range(n_fams)]
        ax.bar(xs, rates, width=width, label=cond, color=cond_colors.get(cond, "#888"))

    ax.set_xticks(range(n_fams))
    ax.set_xticklabels([f.replace("_", " ") for f in families], rotation=12)
    ax.set_ylabel("episode success rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Resolution rate per scenario, per training condition")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def make_action_distribution(
    actions_by_condition: Dict[str, Dict[str, int]],
    top_n: int = 10,
):
    """Stacked bar comparison of action distributions across conditions.

    Trained agents shift toward `restart_service`, `rollback_deployment`,
    `update_config` (remediation), away from random reads. This plot makes
    that shift visible.
    """
    plt = _lazy_mpl()
    # Union of action keys across conditions, ordered by total volume
    all_actions = sorted(
        set().union(*(d.keys() for d in actions_by_condition.values())),
        key=lambda a: -sum(d.get(a, 0) for d in actions_by_condition.values()),
    )[:top_n]

    fig, ax = plt.subplots(figsize=(9, 5))
    conditions = list(actions_by_condition.keys())
    bottom = [0] * len(conditions)
    palette = ["#3b82f6", "#22c55e", "#a855f7", "#f59e0b", "#ef4444",
               "#06b6d4", "#84cc16", "#ec4899", "#6366f1", "#14b8a6"]
    for i, action in enumerate(all_actions):
        counts = [actions_by_condition[c].get(action, 0) for c in conditions]
        # Normalize to percentage per condition
        totals = [sum(actions_by_condition[c].values()) for c in conditions]
        pcts = [
            (counts[j] / totals[j] * 100) if totals[j] else 0
            for j in range(len(conditions))
        ]
        ax.bar(conditions, pcts, bottom=bottom, label=action, color=palette[i % len(palette)])
        bottom = [bottom[j] + pcts[j] for j in range(len(conditions))]
    ax.set_ylabel("action share (%)")
    ax.set_title("Action distribution per condition")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str, *, dpi: int = 120) -> None:
    """Convenience: save a figure to disk and close it."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.clear()
    import matplotlib.pyplot as plt  # noqa: WPS433
    plt.close(fig)


__all__ = [
    "make_reward_curve",
    "make_reward_components",
    "make_success_bars",
    "make_action_distribution",
    "save_figure",
]
