"""Generate the random-vs-scripted-playbook comparison plot from runs/.

Reads every episode.jsonl under runs/, groups by (model, task_id), and
renders a grouped-bar chart of success rate per scenario family per
condition. This is the canonical "improvement evidence" plot for the
README's results section, anchored in real recorded data.

Two conditions are read from disk:
  * random-baseline    - uniform-random action policy (the floor)
  * scripted-playbook  - deterministic best-trajectory from coach.py
                         (the ceiling for any non-learned policy)

When the SFT model finishes training on Colab, append a third condition
by re-running this script after the new runs land under runs/.

Usage:
    uv run python scripts/generate_comparison_plot.py

Outputs:
    results/comparison_success_rates.png
    results/comparison_summary.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


# Stable family order + friendly labels for the x-axis
FAMILY_ORDER = [
    "oom_crash",
    "db_pool_exhaustion",
    "bad_deployment_cascade",
    "disk_full",
    "slow_query",
    "cert_expiry",
    "dns_failure",
    "rate_limit_exhaustion",
]
FAMILY_LABEL = {
    "oom_crash":              "OOM\ncrash",
    "db_pool_exhaustion":     "DB pool\nexhaustion",
    "bad_deployment_cascade": "Bad\ndeployment",
    "disk_full":              "Disk\nfull",
    "slow_query":             "Slow\nquery",
    "cert_expiry":            "Cert\nexpiry",
    "dns_failure":            "DNS\nfailure",
    "rate_limit_exhaustion":  "Rate\nlimit",
}

# Stable condition order + colors
CONDITION_ORDER = ["random-baseline", "scripted-playbook"]
CONDITION_LABEL = {
    "random-baseline":   "Random baseline",
    "scripted-playbook": "Scripted playbook",
}
CONDITION_COLOR = {
    "random-baseline":   "#94a3b8",  # slate-400
    "scripted-playbook": "#22c55e",  # green-500
    "sft":               "#3b82f6",  # blue-500 (when SFT runs are added)
}


def _load_runs(runs_root: Path) -> List[Dict]:
    """Read every episode.jsonl under runs_root and extract a per-run summary.

    Returns a list of {"model", "task_id", "resolved", "score"} dicts.
    """
    out = []
    for ep_path in sorted(runs_root.glob("*/episode.jsonl")):
        start = None
        end = None
        try:
            with ep_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if ev.get("type") == "start":
                        start = ev
                    elif ev.get("type") == "end":
                        end = ev
        except Exception:
            continue

        if not start:
            continue
        out.append({
            "run_id": ep_path.parent.name,
            "model": start.get("model") or "unknown",
            "task_id": start.get("task_id") or "unknown",
            "resolved": (end or {}).get("resolved", False),
            "score": (end or {}).get("score", 0.0),
        })
    return out


def _aggregate(runs: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """Group runs by (model, task_id) and compute success_rate + counts."""
    bucket: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        bucket[r["model"]][r["task_id"]].append(r)

    summary: Dict[str, Dict[str, Dict]] = {}
    for model, by_fam in bucket.items():
        summary[model] = {}
        for fam, eps in by_fam.items():
            n = len(eps)
            successes = sum(1 for e in eps if e["resolved"])
            summary[model][fam] = {
                "n": n,
                "success_rate": successes / max(1, n),
                "avg_score": sum(e["score"] for e in eps) / max(1, n),
            }
    return summary


def _make_plot(summary: Dict[str, Dict[str, Dict]], out_path: Path) -> None:
    """Grouped bar chart: x-axis = scenario family, bars = conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Keep only the conditions we actually have data for, in stable order
    conditions = [c for c in CONDITION_ORDER if c in summary]
    families = [f for f in FAMILY_ORDER
                if any(f in summary.get(c, {}) for c in conditions)]
    if not conditions or not families:
        print("[comparison] no conditions/families to plot - skipping")
        return

    n_conds = len(conditions)
    n_fams = len(families)
    bar_width = 0.8 / max(1, n_conds)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_facecolor("#fafbfc")

    # Bars
    for ci, cond in enumerate(conditions):
        rates = []
        ns = []
        for fam in families:
            stats = summary[cond].get(fam, {})
            rates.append(stats.get("success_rate", 0.0) * 100)
            ns.append(stats.get("n", 0))
        xs = [i + ci * bar_width - 0.4 + bar_width / 2 for i in range(n_fams)]
        bars = ax.bar(
            xs, rates, width=bar_width,
            label=f"{CONDITION_LABEL.get(cond, cond)}",
            color=CONDITION_COLOR.get(cond, "#888"),
            edgecolor="white", linewidth=0.6,
        )
        # Numeric labels on top of each bar
        for bar, rate, n in zip(bars, rates, ns):
            if n == 0:
                continue
            label = f"{rate:.0f}%" if rate > 0 else "0%"
            sub = f"\nn={n}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                label + sub,
                ha="center", va="bottom",
                fontsize=8, color="#475569",
                linespacing=1.05,
            )

    # Cosmetics
    ax.set_xticks(range(n_fams))
    ax.set_xticklabels([FAMILY_LABEL.get(f, f) for f in families], fontsize=10)
    ax.set_ylabel("Episode success rate (%)", fontsize=11, color="#1f2937")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(
        "Random baseline vs scripted playbook - per scenario family",
        fontsize=13, color="#1f2937", pad=14, weight="bold",
    )
    ax.legend(
        loc="upper right",
        frameon=True, framealpha=0.95,
        edgecolor="#cbd5e1", fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    fig.text(
        0.5, 0.01,
        "n = number of recorded episodes per (condition, family). "
        "Trained-model condition will be added once Colab SFT run completes.",
        ha="center", fontsize=8.5, color="#64748b", style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    runs_root = repo_root / "runs"
    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(runs_root)
    if not runs:
        print(f"[comparison] no runs found under {runs_root}")
        return 1

    summary = _aggregate(runs)

    # Save the JSON summary first - it's useful even if matplotlib is missing
    summary_path = out_dir / "comparison_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "n_runs_total": len(runs),
                "by_condition": {
                    cond: {
                        fam: stats for fam, stats in by_fam.items()
                    }
                    for cond, by_fam in summary.items()
                },
                "note": (
                    "Real measured success rates from runs/ directory. "
                    "Random-baseline = uniform-random policy; "
                    "scripted-playbook = deterministic best-trajectory from "
                    "coach.py. SFT condition appended after Colab training "
                    "writes new episodes to runs/."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[comparison] wrote {summary_path}  ({len(runs)} total runs)")

    # Print the table to stdout for the README
    print()
    print("Conditions found:", list(summary.keys()))
    for cond, by_fam in summary.items():
        print(f"\n{cond}:")
        for fam in FAMILY_ORDER:
            stats = by_fam.get(fam)
            if not stats:
                continue
            print(
                f"  {fam:<25s}  n={stats['n']:>2}  "
                f"success={stats['success_rate']*100:>5.1f}%  "
                f"avg_score={stats['avg_score']:.2f}"
            )

    # Generate the plot
    plot_path = out_dir / "comparison_success_rates.png"
    try:
        _make_plot(summary, plot_path)
        size_kb = plot_path.stat().st_size // 1024
        print(f"\n[comparison] wrote {plot_path}  ({size_kb} KB)")
    except ImportError as exc:
        print(f"[comparison] matplotlib not available: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
