"""Tests for the observe-mode pipeline.

  * EpisodeLogger writes valid JSONL with start/step/end events
  * eval_runner emits a JSONL trace when runs_root is provided
  * /runs lists recorded runs with summary metadata
  * /watch/<run_id> returns the full event list + summary
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from incident_commander_env.server import app as app_module
from incident_commander_env.server.app import app
from training.episode_logger import EpisodeLogger, iter_runs, read_episode
from training.datasets import SYSTEM_PROMPT
from training.eval_runner import (
    evaluate,
    random_policy,
    run_episode,
)


# ---------------------------------------------------------------------------
# EpisodeLogger primitives
# ---------------------------------------------------------------------------

class TestEpisodeLogger:
    def test_writes_jsonl_with_three_event_kinds(self, tmp_path):
        log = EpisodeLogger.for_run(tmp_path, "oom_crash", seed=42)
        with log:
            log.start({"task_id": "oom_crash", "seed": 42, "model": "smoke"})
            log.step(1, action={"action_type": "list_services"}, observation={"message": "..."}, reward_breakdown={"diagnostic": 0.05})
            log.end({"resolved": True, "score": 0.9, "steps_used": 1})
        events = read_episode(log.file_path)
        assert [e["type"] for e in events] == ["start", "step", "end"]
        assert events[0]["seed"] == 42
        assert events[1]["step"] == 1
        assert events[2]["resolved"] is True

    def test_run_id_is_unique_per_call(self, tmp_path):
        a = EpisodeLogger.for_run(tmp_path, "oom_crash", seed=1)
        b = EpisodeLogger.for_run(tmp_path, "oom_crash", seed=1)
        # Different short-uuid suffixes
        assert a.run_id != b.run_id  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# iter_runs aggregator
# ---------------------------------------------------------------------------

class TestIterRuns:
    def test_iter_runs_summarizes_directory(self, tmp_path):
        # Create two completed runs
        for seed in (1, 2):
            log = EpisodeLogger.for_run(tmp_path, "oom_crash", seed=seed)
            with log:
                log.start({"task_id": "oom_crash", "seed": seed, "model": "smoke"})
                log.step(1, action={}, observation={}, reward_breakdown={})
                log.end({"resolved": seed == 1, "score": 0.5 + 0.1 * seed, "steps_used": 1})
        runs = list(iter_runs(tmp_path))
        assert len(runs) == 2
        for r in runs:
            assert r["task_id"] == "oom_crash"
            assert r["model"] == "smoke"
            assert r["resolved"] in (True, False)


# ---------------------------------------------------------------------------
# eval_runner emits traces when runs_root is set
# ---------------------------------------------------------------------------

class TestEvalRunnerTracing:
    def test_run_episode_emits_jsonl(self, tmp_path):
        ep = run_episode(
            task_id="oom_crash",
            seed=42,
            act=random_policy(rng_seed=7),
            system_prompt=SYSTEM_PROMPT,
            runs_root=str(tmp_path),
            model_label="smoke-test",
        )
        # exactly one run dir was written
        runs = list(iter_runs(tmp_path))
        assert len(runs) == 1
        r = runs[0]
        assert r["task_id"] == "oom_crash"
        assert r["model"] == "smoke-test"
        assert r["seed"] == 42
        # Step count matches what the episode reported
        events = read_episode(tmp_path / r["run_id"] / "episode.jsonl")
        steps = [e for e in events if e["type"] == "step"]
        assert len(steps) == ep.steps_used

    def test_evaluate_emits_one_trace_per_episode(self, tmp_path):
        report = evaluate(
            "smoke",
            random_policy(rng_seed=42),
            families=["oom_crash"],
            seeds=[1, 2, 3],
            system_prompt=SYSTEM_PROMPT,
            runs_root=str(tmp_path),
        )
        assert report.n_episodes == 3
        runs = list(iter_runs(tmp_path))
        assert len(runs) == 3
        # Each run is tagged with the condition name as model
        assert all(r["model"] == "smoke" for r in runs)


# ---------------------------------------------------------------------------
# /watch + /runs HTTP surface
# ---------------------------------------------------------------------------

class TestObserveEndpoints:
    def test_runs_lists_recorded_runs(self, tmp_path, monkeypatch):
        # Point the app at our temp runs dir
        monkeypatch.setattr(app_module, "RUNS_ROOT", tmp_path)
        # Seed a run
        log = EpisodeLogger.for_run(tmp_path, "oom_crash", seed=99)
        with log:
            log.start({"task_id": "oom_crash", "seed": 99, "model": "test"})
            log.step(1, action={"action_type": "list_services"}, observation={"message": "ok"}, reward_breakdown={})
            log.end({"resolved": True, "score": 0.7, "steps_used": 1})

        client = TestClient(app)
        r = client.get("/runs")
        assert r.status_code == 200
        body = r.json()
        assert any(run["seed"] == 99 for run in body["runs"])

    def test_watch_returns_events(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module, "RUNS_ROOT", tmp_path)
        log = EpisodeLogger.for_run(tmp_path, "oom_crash", seed=42)
        rid = log.run_id  # type: ignore[attr-defined]
        with log:
            log.start({"task_id": "oom_crash", "seed": 42, "model": "test"})
            log.step(1, action={"action_type": "list_services"}, observation={"message": "ok"}, reward_breakdown={"diagnostic": 0.05})
            log.end({"resolved": True, "score": 0.7, "steps_used": 1})

        client = TestClient(app)
        r = client.get(f"/watch/{rid}")
        assert r.status_code == 200
        body = r.json()
        assert body["summary"]["task_id"] == "oom_crash"
        assert len(body["events"]) == 3
        assert body["events"][0]["type"] == "start"

    def test_watch_unknown_run_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module, "RUNS_ROOT", tmp_path)
        client = TestClient(app)
        r = client.get("/watch/nonexistent-run")
        assert r.status_code == 200
        body = r.json()
        assert "error" in body and body["events"] == []

    def test_watch_rejects_path_traversal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module, "RUNS_ROOT", tmp_path)
        client = TestClient(app)
        # Even a token without slashes that contains '..' should be sanitized
        # and treated as a non-existent run (not escape the sandbox).
        r = client.get("/watch/..secret")
        assert r.status_code == 200
        body = r.json()
        assert "error" in body
        assert body["events"] == []
