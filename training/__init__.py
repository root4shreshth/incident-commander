"""IncidentCommander training pipeline.

Modules:
    datasets       - build SFT chat dataset from IDEAL_TRAJECTORIES
    eval_runner    - run N episodes against (model, env), return EvalReport
    curriculum     - phase-gated scenario sampler for GRPO
    plots          - matplotlib helpers for the storytelling plots
    grpo_reward    - reward function consumed by TRL's GRPOTrainer

The Colab notebook `train_grpo.ipynb` orchestrates these into:
    SFT (warm-start from senior-SRE trajectories) -> GRPO (verifiable rewards)
"""
