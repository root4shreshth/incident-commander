"""Episode logger - writes structured JSONL traces of agent runs.

Each evaluation episode emits one JSONL file at `runs/<run_id>/episode.jsonl`
with three event kinds, mirroring the format already used by `inference.py`:

  {"type": "start",    "task_id": "oom_crash", "seed": 42, ...}
  {"type": "step",     "step": 1, "action": {...}, "observation": {...}, "reward_breakdown": {...}}
  ...
  {"type": "end",      "resolved": true, "score": 0.81, "steps_used": 7, "breakdown_totals": {...}}

The dashboard's `/watch/{run_id}` endpoint streams these back to the browser
so the user can replay a trained-agent episode in observe mode while we
narrate the video. The same files are also the ground-truth artifact we
attach to the submission as evidence of "trained policy actually solved
real outages".
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of dataclasses / pydantic / mappings to JSON."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return asdict(obj)
        except Exception:
            pass
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    return repr(obj)


class EpisodeLogger:
    """Append-only JSONL writer scoped to one episode.

    Usage:
        with EpisodeLogger.for_run("results/sft_eval", "oom_crash", seed=42) as log:
            log.start({"task_id": "oom_crash", "seed": 42, "model": "qwen-1.5b"})
            for action, obs, breakdown in episode:
                log.step(action, obs, breakdown)
            log.end(resolved=True, score=0.81, steps_used=7)
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._closed = False

    @classmethod
    def for_run(
        cls,
        runs_root: str | Path,
        task_id: str,
        seed: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> "EpisodeLogger":
        """Create a logger under `runs_root/<run_id>/episode.jsonl`."""
        rid = run_id or _make_run_id(task_id, seed)
        path = Path(runs_root) / rid / "episode.jsonl"
        log = cls(path)
        log.run_id = rid  # type: ignore[attr-defined]
        return log

    def __enter__(self) -> "EpisodeLogger":
        self._fh = self.file_path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- writer methods ----------------------------------------------------

    def _write(self, event: Dict[str, Any]) -> None:
        if self._closed:
            return
        if self._fh is None:
            self._fh = self.file_path.open("a", encoding="utf-8")
        event = {"ts": round(time.time(), 3), **event}
        self._fh.write(json.dumps(_to_jsonable(event), ensure_ascii=False) + "\n")
        self._fh.flush()

    def start(self, payload: Dict[str, Any]) -> None:
        self._write({"type": "start", **payload})

    def step(
        self,
        step_num: int,
        action: Any,
        observation: Any,
        reward_breakdown: Any = None,
        message: Optional[str] = None,
    ) -> None:
        self._write({
            "type": "step",
            "step": step_num,
            "action": _to_jsonable(action),
            "observation": _to_jsonable(observation),
            "reward_breakdown": _to_jsonable(reward_breakdown),
            "message": message,
        })

    def end(self, payload: Dict[str, Any]) -> None:
        self._write({"type": "end", **payload})

    def close(self) -> None:
        if self._fh and not self._closed:
            self._fh.close()
        self._closed = True


def _make_run_id(task_id: str, seed: Optional[int]) -> str:
    """Generate a sortable, descriptive run id."""
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    short = uuid.uuid4().hex[:6]
    seed_part = f"s{seed}" if seed is not None else "snone"
    return f"{ts}-{task_id}-{seed_part}-{short}"


# ---------------------------------------------------------------------------
# Reader / iterator API used by /watch
# ---------------------------------------------------------------------------

def read_episode(path: str | Path) -> List[Dict[str, Any]]:
    """Load all events from a single episode.jsonl file."""
    p = Path(path)
    out: List[Dict[str, Any]] = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def iter_runs(runs_root: str | Path) -> Iterator[Dict[str, Any]]:
    """Enumerate runs under `runs_root` with summary metadata."""
    root = Path(runs_root)
    if not root.exists():
        return
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        ep = child / "episode.jsonl"
        if not ep.exists():
            continue
        events = read_episode(ep)
        if not events:
            continue
        start = next((e for e in events if e.get("type") == "start"), {})
        end = next((e for e in reversed(events) if e.get("type") == "end"), {})
        yield {
            "run_id": child.name,
            "task_id": start.get("task_id"),
            "seed": start.get("seed"),
            "model": start.get("model"),
            "n_events": len(events),
            "resolved": end.get("resolved"),
            "score": end.get("score"),
            "steps_used": end.get("steps_used"),
        }


__all__ = ["EpisodeLogger", "read_episode", "iter_runs"]
