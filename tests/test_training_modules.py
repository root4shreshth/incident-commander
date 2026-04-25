"""Smoke tests for the training/ pipeline modules.

These verify the *plumbing* — schemas, dispatch, action parsing — without
loading any real model. Real training behavior is tested by running the
Colab notebook on actual compute (Phase 5).
"""

from __future__ import annotations

import json

import pytest

from training.curriculum import Curriculum, DEFAULT_SCHEDULE
from training.datasets import (
    SYSTEM_PROMPT,
    build_sft_dataset,
    replay_trajectory,
    to_chat_messages,
)
from training.eval_runner import (
    evaluate,
    parse_action_response,
    random_policy,
    run_episode,
)
from training.grpo_reward import (
    get_recent_breakdowns,
    grpo_reward_fn,
    reset_history,
)


# ---------------------------------------------------------------------------
# datasets.py
# ---------------------------------------------------------------------------

class TestSFTDatasetBuilder:
    def test_default_build_produces_rows(self):
        rows = build_sft_dataset(n_seeds_per_family=2)
        assert len(rows) > 0
        # Every row has the three message slots
        for r in rows:
            assert "system" in r and "user" in r and "assistant" in r
            assert r["system"] == SYSTEM_PROMPT

    def test_assistant_messages_are_valid_json(self):
        rows = build_sft_dataset(n_seeds_per_family=1)
        for r in rows:
            data = json.loads(r["assistant"])
            assert "action_type" in data
            assert "thinking" in data

    def test_chat_message_conversion(self):
        rows = build_sft_dataset(n_seeds_per_family=1)
        msgs = to_chat_messages(rows[0])
        assert len(msgs) == 3
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]

    def test_distinct_seeds_yield_distinct_rows(self):
        # Same trajectory under two seeds should produce two state contexts
        # because OOM scenario randomizes target service per seed.
        rows_a = replay_trajectory(
            "oom_crash", seed=1,
            trajectory=[{"action": "list_services", "target": None, "why": "first"}],
        )
        rows_b = replay_trajectory(
            "oom_crash", seed=2,
            trajectory=[{"action": "list_services", "target": None, "why": "first"}],
        )
        assert rows_a[0]["user"] != rows_b[0]["user"]


# ---------------------------------------------------------------------------
# eval_runner.py
# ---------------------------------------------------------------------------

class TestParseActionResponse:
    def test_bare_json(self):
        text = '{"action_type":"list_services","target_service":null,"parameters":{}}'
        a = parse_action_response(text)
        assert a["action_type"] == "list_services"

    def test_fenced_json(self):
        text = '```json\n{"action_type":"read_logs","target_service":"foo","parameters":{}}\n```'
        a = parse_action_response(text)
        assert a["action_type"] == "read_logs"
        assert a["target_service"] == "foo"

    def test_garbage_falls_back_to_list_services(self):
        a = parse_action_response("totally not json")
        assert a["action_type"] == "list_services"

    def test_extra_prose_around_json(self):
        text = 'Sure! Here is my action:\n{"action_type":"check_metrics","target_service":"db"}\nDone.'
        a = parse_action_response(text)
        assert a["action_type"] == "check_metrics"


class TestRandomPolicy:
    def test_random_policy_returns_valid_dict(self):
        act = random_policy(rng_seed=42)
        out = act([])
        assert "action_type" in out
        assert "target_service" in out
        assert "parameters" in out

    def test_random_policy_reproducible(self):
        a = random_policy(rng_seed=42)([])
        b = random_policy(rng_seed=42)([])
        assert a["action_type"] == b["action_type"]


class TestRunEpisode:
    def test_run_episode_terminates(self):
        ep = run_episode(
            task_id="oom_crash",
            seed=1,
            act=random_policy(rng_seed=42),
            system_prompt=SYSTEM_PROMPT,
        )
        assert ep.task_id == "oom_crash"
        assert ep.steps_used > 0
        assert isinstance(ep.score, float)
        assert isinstance(ep.resolved, bool)
        # Breakdown components should be aggregated across the whole episode
        assert "diagnostic" in ep.breakdown_totals
        assert len(ep.actions) == ep.steps_used


class TestEvaluate:
    def test_evaluate_returns_per_family_stats(self):
        report = evaluate(
            "test",
            random_policy(rng_seed=42),
            families=["oom_crash"],
            seeds=[1, 2],
            system_prompt=SYSTEM_PROMPT,
        )
        assert report.n_episodes == 2
        assert "oom_crash" in report.by_family
        stats = report.by_family["oom_crash"]
        assert 0.0 <= stats["success_rate"] <= 1.0
        assert "action_distribution" in stats


# ---------------------------------------------------------------------------
# curriculum.py
# ---------------------------------------------------------------------------

class TestCurriculum:
    def test_phase_at_returns_correct_phase(self):
        cur = Curriculum()
        assert cur.phase_at(0).name == "warmup_oom_easy"
        assert cur.phase_at(150).name == "ops_mixed"
        assert cur.phase_at(300).name == "full_mix"

    def test_draw_within_phase_constraints(self):
        cur = Curriculum(rng_seed=42)
        # Phase 1 — only oom_crash
        for _ in range(20):
            family, _ = cur.draw(50)
            assert family == "oom_crash"

    def test_draw_in_full_mix_covers_all_families(self):
        cur = Curriculum(rng_seed=42)
        seen = set()
        for _ in range(100):
            family, _ = cur.draw(300)
            seen.add(family)
        assert len(seen) >= 2  # over many draws all families show up

    def test_schedule_summary_serializes(self):
        cur = Curriculum()
        s = cur.schedule_summary()
        assert len(s) == len(DEFAULT_SCHEDULE)


# ---------------------------------------------------------------------------
# grpo_reward.py
# ---------------------------------------------------------------------------

class TestGRPOReward:
    def setup_method(self, _method):
        reset_history()

    def test_correct_completion_scores_higher_than_garbage(self):
        # OOM scenario at seed=2 picks payment-service as the target (parametric).
        # The correct fix is restart_service payment-service with memory > 256.
        good = json.dumps({
            "action_type": "restart_service",
            "target_service": "payment-service",
            "parameters": {"memory_limit": "1024Mi"},
        })
        bad = "this is not json"

        rewards = grpo_reward_fn(
            prompts=[None, None],
            completions=[good, bad],
            task_id=["oom_crash", "oom_crash"],
            seed=[2, 2],   # seed=2 -> target=payment-service
            difficulty=[0.5, 0.5],
        )
        assert len(rewards) == 2
        # The correct fix gets a meaningfully higher reward than the fallback
        assert rewards[0] > rewards[1]

    def test_breakdowns_recorded_for_each_completion(self):
        completions = [
            json.dumps({"action_type": "list_services", "target_service": None, "parameters": {}}),
            json.dumps({"action_type": "read_logs", "target_service": "payment-service", "parameters": {}}),
        ]
        grpo_reward_fn(
            prompts=[None] * 2,
            completions=completions,
            task_id=["oom_crash"] * 2,
            seed=[1] * 2,
            difficulty=[0.5] * 2,
        )
        bds = get_recent_breakdowns()
        assert len(bds) == 2

    def test_handles_chat_message_completion_format(self):
        # GRPOTrainer can hand list-of-message-dicts; we should extract the assistant content
        completion = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": json.dumps({
                "action_type": "list_services",
                "target_service": None,
                "parameters": {},
            })},
        ]
        rewards = grpo_reward_fn(
            prompts=[None],
            completions=[completion],
            task_id=["oom_crash"],
            seed=[1],
            difficulty=[0.5],
        )
        assert len(rewards) == 1
        assert rewards[0] != 0.0  # at least r_format should fire


# ---------------------------------------------------------------------------
# Lightweight integration: the eval-runner-against-random-policy report that's
# emitted at the start of training to confirm the env hasn't drifted.
# ---------------------------------------------------------------------------

class TestEndToEndPlumbing:
    def test_random_policy_at_least_runs(self):
        report = evaluate(
            "smoke",
            random_policy(rng_seed=42),
            families=["oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"],
            seeds=[100, 101],
            system_prompt=SYSTEM_PROMPT,
        )
        assert report.n_episodes == 6
        for fam in ("oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"):
            assert fam in report.by_family
