"""Tests for training.grpo_reward - the TRL sidecar.

These tests intentionally avoid every heavy ML dep (torch, trl, peft,
bitsandbytes) and every network call. The reward function is a pure adapter
over IncidentCommanderEnv + RewardBreakdown; both are already covered by the
env's own test suite. What we verify here:

  1. The TRL-shaped call signature (`prompts`, `completions`, `**kwargs`)
     produces a `list[float]` of the right length.
  2. The sidecar `_LAST_BREAKDOWNS` accumulates one entry per completion and
     is cleared by `reset_history()`.
  3. Malformed completions don't crash the trainer - they fall through to
     the `list_services` parser fallback.
  4. Missing `kwargs` (task_id/seed/difficulty) fall through to defaults.
  5. Both string-shaped and chat-message-shaped completions parse.
"""

from __future__ import annotations

from training.grpo_reward import (
    grpo_reward_fn,
    reset_history,
    get_recent_breakdowns,
)


class TestGRPORewardSignature:
    def test_returns_list_of_floats_matching_completion_count(self) -> None:
        reset_history()
        prompts = [[{"role": "user", "content": "unused"}]] * 3
        completions = [
            '{"action_type": "list_services"}',
            '{"action_type": "read_logs", "target_service": "payment-service"}',
            '{"action_type": "check_metrics", "target_service": "postgres-db"}',
        ]
        rewards = grpo_reward_fn(
            prompts,
            completions,
            task_id=["oom_crash", "oom_crash", "oom_crash"],
            seed=[1, 2, 3],
            difficulty=[0.5, 0.5, 0.5],
        )
        assert isinstance(rewards, list)
        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)

    def test_sidecar_records_one_breakdown_per_completion(self) -> None:
        reset_history()
        assert get_recent_breakdowns() == []
        grpo_reward_fn(
            [[]],
            ['{"action_type": "list_services"}'],
            task_id=["oom_crash"],
            seed=[1],
            difficulty=[0.5],
        )
        assert len(get_recent_breakdowns()) == 1

    def test_reset_history_clears_sidecar(self) -> None:
        grpo_reward_fn(
            [[]],
            ['{"action_type": "list_services"}'],
            task_id=["oom_crash"],
            seed=[1],
            difficulty=[0.5],
        )
        assert len(get_recent_breakdowns()) >= 1
        reset_history()
        assert get_recent_breakdowns() == []


class TestGRPORewardRobustness:
    def test_malformed_completion_returns_float_not_exception(self) -> None:
        reset_history()
        rewards = grpo_reward_fn(
            [[]],
            ["absolutely not json, just prose"],
            task_id=["oom_crash"],
            seed=[1],
            difficulty=[0.5],
        )
        assert isinstance(rewards[0], float)

    def test_missing_kwargs_fall_through_to_defaults(self) -> None:
        """Bare-bones dataset without task_id/seed/difficulty still runs."""
        reset_history()
        rewards = grpo_reward_fn(
            [[]],
            ['{"action_type": "list_services"}'],
        )
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_string_completion_shape(self) -> None:
        reset_history()
        rewards = grpo_reward_fn(
            [[]],
            ['{"action_type": "list_services"}'],
            task_id=["oom_crash"],
            seed=[1],
            difficulty=[0.5],
        )
        assert len(rewards) == 1

    def test_chat_message_completion_shape(self) -> None:
        """TRL sometimes passes completions as chat-message lists."""
        reset_history()
        chat_completion = [
            {"role": "assistant", "content": '{"action_type": "list_services"}'}
        ]
        rewards = grpo_reward_fn(
            [[]],
            [chat_completion],
            task_id=["oom_crash"],
            seed=[1],
            difficulty=[0.5],
        )
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)


class TestGRPORewardDifferentiation:
    def test_different_actions_produce_different_rewards_on_same_seed(self) -> None:
        """Sanity: the reward isn't constant across action choices.

        If it were, GRPO would have no gradient signal.
        """
        reset_history()
        rewards = grpo_reward_fn(
            [[], []],
            [
                '{"action_type": "list_services"}',
                '{"action_type": "restart_service", "target_service": "payment-service"}',
            ],
            task_id=["oom_crash", "oom_crash"],
            seed=[100, 100],
            difficulty=[0.5, 0.5],
        )
        assert rewards[0] != rewards[1], (
            f"Expected different rewards for different actions, got {rewards}. "
            "If they're equal, the reward function is degenerate."
        )
