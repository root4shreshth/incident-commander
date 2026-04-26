"""Curriculum scheduler for GRPO training.

The hackathon docs explicitly recommend "easy tasks with short horizons,
medium with branching, harder only after the model gets non-zero reward".
This module implements that as a step-gated sampler that returns a list of
`(family, difficulty, weight)` tuples to draw from at each gradient step.

Phases (default schedule):
  Phase 1 (steps   0..100): 100% oom_crash @ difficulty=0.3 - easiest
  Phase 2 (steps 100..200): mix oom_crash + db_pool_exhaustion at moderate diff
  Phase 3 (steps 200..400): full mix across all 3 families at varied difficulty

The schedule can be advanced "early" by calling `try_advance(success_rate)` -
once the policy starts solving the current phase reliably, jump ahead.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CurriculumPhase:
    """One phase of the curriculum: which families, at which difficulties."""
    name: str
    start_step: int
    end_step: int
    mix: List[Tuple[str, float, float]]  # (family, difficulty, weight)


DEFAULT_SCHEDULE: List[CurriculumPhase] = [
    CurriculumPhase(
        name="warmup_oom_easy",
        start_step=0,
        end_step=100,
        mix=[("oom_crash", 0.3, 1.0)],
    ),
    CurriculumPhase(
        name="ops_mixed",
        start_step=100,
        end_step=200,
        mix=[
            ("oom_crash",          0.5, 0.5),
            ("db_pool_exhaustion", 0.4, 0.5),
        ],
    ),
    CurriculumPhase(
        name="full_mix",
        start_step=200,
        end_step=400,
        mix=[
            ("oom_crash",              0.6, 0.34),
            ("db_pool_exhaustion",     0.5, 0.33),
            ("bad_deployment_cascade", 0.4, 0.33),
        ],
    ),
]


class Curriculum:
    """Returns a (family, difficulty) draw at each step according to the schedule."""

    def __init__(
        self,
        schedule: List[CurriculumPhase] = None,
        rng_seed: int = 0,
    ) -> None:
        self.schedule = schedule or DEFAULT_SCHEDULE
        self._rng = _random.Random(rng_seed)

    def phase_at(self, step: int) -> CurriculumPhase:
        for phase in self.schedule:
            if phase.start_step <= step < phase.end_step:
                return phase
        return self.schedule[-1]  # cap at the final phase

    def draw(self, step: int) -> Tuple[str, float]:
        """Sample one (family, difficulty) for the next training rollout."""
        phase = self.phase_at(step)
        weights = [w for _, _, w in phase.mix]
        idx = self._rng.choices(range(len(phase.mix)), weights=weights, k=1)[0]
        family, difficulty, _ = phase.mix[idx]
        return family, difficulty

    def schedule_summary(self) -> List[Tuple[str, int, int, List[str]]]:
        """For logging at training start: phase name, step range, families."""
        return [
            (
                p.name,
                p.start_step,
                p.end_step,
                [f"{fam}@{diff}" for fam, diff, _ in p.mix],
            )
            for p in self.schedule
        ]


__all__ = ["Curriculum", "CurriculumPhase", "DEFAULT_SCHEDULE"]
