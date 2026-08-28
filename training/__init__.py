"""IncidentCommander training pipeline.

Modules:
    datasets       - build SFT chat dataset from IDEAL_TRAJECTORIES
    grpo_reward    - GRPO reward function (wraps env's 6-component breakdown for TRL)
    eval_runner    - run N episodes against (model, env), return EvalReport
    curriculum     - phase-gated scenario sampler for GRPO prompt scheduling
    plots          - matplotlib helpers for the storytelling plots
    episode_logger - per-episode JSONL logger used by eval_runner
    postmortem_writer - markdown post-mortem generator for resolved episodes

Two Colab notebooks orchestrate these into:
    train_sft.ipynb  - SFT-only warm-start from senior-SRE trajectories,
                       then evaluation + plots
    train_grpo.ipynb - SFT (reuses adapter if present) + GRPO fine-tune with
                       curriculum + 6-component verifiable reward, then
                       evaluation + reward-curve plots + Hub push

Both notebooks write intermediate state to /content/checkpoints/ so a Colab
disconnect doesn't lose work.
"""
