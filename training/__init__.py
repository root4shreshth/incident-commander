"""IncidentCommander training pipeline.

Modules:
    datasets       - build SFT chat dataset from IDEAL_TRAJECTORIES
    eval_runner    - run N episodes against (model, env), return EvalReport
    curriculum     - phase-gated scenario sampler
    plots          - matplotlib helpers for the storytelling plots

The Colab notebook `train_sft.ipynb` orchestrates these into:
    SFT (warm-start from senior-SRE trajectories) -> evaluation -> plots
"""
