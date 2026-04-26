"""Benchmark how fast the simulator generates training data.

The data-factory thesis: real Kubernetes resets in ~60s, our simulator resets in
~10ms. That's the bottleneck for RL training in this domain — without high
throughput you can't generate the trajectory volume RL needs.

This script produces a single number that anchors that pitch. It writes
`results/throughput.json` with raw metrics plus a markdown summary block
suitable for pasting into the README/blog.

Usage::

    uv run python scripts/benchmark_throughput.py
    uv run python scripts/benchmark_throughput.py --resets 500 --steps 5000

Reuses (no edits):
- IncidentCommanderEnv (incident_commander_env/server/environment.py) — reset/step
- random_policy() (training/eval_runner.py) — for step-rate measurement
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Allow `python scripts/benchmark_throughput.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from incident_commander_env.models import IncidentAction
from incident_commander_env.server.environment import IncidentCommanderEnv
from training.eval_runner import random_policy

# Real-world Kubernetes baseline. `kubectl rollout restart deployment` plus
# pod ready time is on the order of 30-60s for a typical deployment; cluster
# bring-up via `gcloud container clusters create` is several minutes. We
# anchor against 60s as a fair midpoint for "reset to a known-bad state" in
# a real GKE setup (Noclue's reported workflow).
REAL_K8S_RESET_SECONDS = 60.0

FAMILIES = ("oom_crash", "db_pool_exhaustion", "bad_deployment_cascade")


def _percentiles(values: List[float], pcts=(50.0, 95.0, 99.0)) -> Dict[str, float]:
    """Return p50/p95/p99 in milliseconds. Tolerates short lists."""
    if not values:
        return {f"p{int(p)}_ms": 0.0 for p in pcts}
    sorted_values = sorted(values)
    out: Dict[str, float] = {}
    for p in pcts:
        # Nearest-rank percentile — good enough for benchmark display.
        k = max(0, min(len(sorted_values) - 1, int(round((p / 100.0) * (len(sorted_values) - 1)))))
        out[f"p{int(p)}_ms"] = sorted_values[k] * 1000.0
    return out


def benchmark_resets(env: IncidentCommanderEnv, n_per_family: int) -> Dict[str, Any]:
    """Time `n_per_family` resets for each scenario family."""
    per_family: Dict[str, Dict[str, Any]] = {}
    all_durations: List[float] = []

    for family in FAMILIES:
        durations: List[float] = []
        for seed in range(n_per_family):
            t0 = time.perf_counter()
            env.reset(task_id=family, seed=seed, difficulty=0.5)
            durations.append(time.perf_counter() - t0)
        all_durations.extend(durations)

        per_family[family] = {
            "n": n_per_family,
            "total_seconds": sum(durations),
            "mean_ms": (sum(durations) / len(durations)) * 1000.0,
            "resets_per_sec": len(durations) / sum(durations) if sum(durations) > 0 else 0.0,
            **_percentiles(durations),
        }

    return {
        "per_family": per_family,
        "overall": {
            "n": len(all_durations),
            "total_seconds": sum(all_durations),
            "mean_ms": (sum(all_durations) / len(all_durations)) * 1000.0,
            "resets_per_sec": len(all_durations) / sum(all_durations) if sum(all_durations) > 0 else 0.0,
            **_percentiles(all_durations),
        },
    }


def benchmark_steps(env: IncidentCommanderEnv, n_steps: int) -> Dict[str, Any]:
    """Time `n_steps` env.step() calls under the random policy.

    Resets are amortized — we reset every ~10 steps to keep episodes short
    and to match the typical RL rollout shape. Pure step time is what we
    report, exclusive of the resets in between.
    """
    policy = random_policy(rng_seed=0)
    history: List[Dict[str, str]] = []
    step_durations: List[float] = []
    family_idx = 0
    seed = 0
    steps_since_reset = 0

    env.reset(task_id=FAMILIES[0], seed=seed, difficulty=0.5)

    while len(step_durations) < n_steps:
        action_dict = policy(history)
        action = IncidentAction(
            action_type=action_dict["action_type"],
            target_service=action_dict.get("target_service"),
            parameters=action_dict.get("parameters") or {},
        )
        t0 = time.perf_counter()
        obs = env.step(action)
        step_durations.append(time.perf_counter() - t0)
        steps_since_reset += 1

        if obs.done or steps_since_reset >= 10:
            family_idx = (family_idx + 1) % len(FAMILIES)
            seed += 1
            env.reset(task_id=FAMILIES[family_idx], seed=seed, difficulty=0.5)
            steps_since_reset = 0
            history = []

    total_seconds = sum(step_durations)
    return {
        "n": len(step_durations),
        "total_seconds": total_seconds,
        "mean_ms": (total_seconds / len(step_durations)) * 1000.0,
        "steps_per_sec": len(step_durations) / total_seconds if total_seconds > 0 else 0.0,
        **_percentiles(step_durations),
    }


def speedup_vs_real_k8s(reset_resets_per_sec: float) -> float:
    """How many times faster is our simulator than a real K8s reset cycle?"""
    real_resets_per_sec = 1.0 / REAL_K8S_RESET_SECONDS  # ≈ 0.0167
    if real_resets_per_sec == 0:
        return 0.0
    return reset_resets_per_sec / real_resets_per_sec


def render_markdown(result: Dict[str, Any]) -> str:
    """Render a copy-paste markdown block for the README/blog."""
    overall = result["resets"]["overall"]
    steps = result["steps"]
    speedup = result["speedup_vs_real_k8s_x"]

    lines = [
        "## Throughput",
        "",
        f"- Sim reset: **{overall['mean_ms']:.2f} ms** mean "
        f"(p95 **{overall['p95_ms']:.2f} ms**) — **{overall['resets_per_sec']:.0f} resets/sec**",
        f"- Sim step: **{steps['mean_ms']:.2f} ms** mean — **{steps['steps_per_sec']:.0f} steps/sec**",
        f"- Real K8s reset baseline: **~{REAL_K8S_RESET_SECONDS:.0f} s** (cluster + pod ready)",
        f"- Headline: **~{speedup:,.0f}× faster** than real K8s reset",
        "",
        "Per-family reset latency:",
        "",
        "| Family | mean (ms) | p95 (ms) | resets/sec |",
        "|---|---|---|---|",
    ]
    for family, stats in result["resets"]["per_family"].items():
        lines.append(
            f"| `{family}` | {stats['mean_ms']:.2f} | {stats['p95_ms']:.2f} | {stats['resets_per_sec']:.0f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resets", type=int, default=200,
                        help="Number of resets per family (default 200; 600 total).")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of env.step() calls under random policy (default 2000).")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "throughput.json",
                        help="Where to write throughput.json")
    args = parser.parse_args()

    print(f"Benchmarking sim throughput on {platform.platform()}", flush=True)
    print(f"  Python {sys.version.split()[0]}", flush=True)
    print(f"  Resets: {args.resets} per family x {len(FAMILIES)} families", flush=True)
    print(f"  Steps:  {args.steps}", flush=True)
    print()

    env = IncidentCommanderEnv()

    print("Measuring reset latency...", flush=True)
    reset_stats = benchmark_resets(env, args.resets)
    print(f"  -> {reset_stats['overall']['resets_per_sec']:.0f} resets/sec "
          f"({reset_stats['overall']['mean_ms']:.2f} ms mean)", flush=True)

    print("Measuring step latency...", flush=True)
    step_stats = benchmark_steps(env, args.steps)
    print(f"  -> {step_stats['steps_per_sec']:.0f} steps/sec "
          f"({step_stats['mean_ms']:.2f} ms mean)", flush=True)

    speedup = speedup_vs_real_k8s(reset_stats["overall"]["resets_per_sec"])
    print()
    print(f"==> Headline: ~{speedup:,.0f}x faster than real K8s reset (60s baseline)")
    print()

    result = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "real_k8s_reset_seconds": REAL_K8S_RESET_SECONDS,
        "speedup_vs_real_k8s_x": speedup,
        "resets": reset_stats,
        "steps": step_stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")
    print()
    print(render_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
