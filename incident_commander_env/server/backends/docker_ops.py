"""Docker Compose shell-out helpers used by RealBackend.

Factored out so tests can mock the `_run` function and exercise the per-action
translation logic without needing a real Docker daemon. Every helper returns a
structured `DockerResult` instead of raising — RealBackend converts those into
typed IncidentObservation responses.

Conventions:
  * `compose_root` is the directory containing `docker-compose.yml`
  * service names match the env's logical names (frontend, api, postgres, ...)
  * env-var levers expected on the user's vibecoded compose:
      - IMAGE_TAG=v1.0|v1.1                 (rollback)
      - API_MEM_LIMIT=256m|1024m            (restart with increased memory)
      - POOL_SIZE=10|50                     (db pool resize)
      - REPLICAS_<svc>=N                    (scale)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.error
import urllib.request


@dataclass
class DockerResult:
    """Structured outcome of a docker shell-out."""
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None  # populated when ok=False


# ---------------------------------------------------------------------------
# low-level runner — the seam tests mock
# ---------------------------------------------------------------------------

def _run(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: int = 30,
    extra_env: Optional[Dict[str, str]] = None,
) -> DockerResult:
    """Run a shell command and return a typed result.

    Tests monkeypatch this function to inject deterministic outputs without
    spawning real subprocesses.
    """
    env = None
    if extra_env:
        env = os.environ.copy()
        env.update({k: str(v) for k, v in extra_env.items()})

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            timeout=timeout,
            capture_output=True,
            text=True,
            env=env,
        )
        return DockerResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            returncode=proc.returncode,
            elapsed_s=time.monotonic() - started,
            error=None if proc.returncode == 0 else (proc.stderr.strip() or f"exit {proc.returncode}"),
        )
    except FileNotFoundError as exc:
        return DockerResult(ok=False, error=f"command not found: {cmd[0]} ({exc})")
    except subprocess.TimeoutExpired:
        return DockerResult(ok=False, error=f"timeout after {timeout}s: {' '.join(cmd)}")
    except OSError as exc:
        return DockerResult(ok=False, error=f"OSError running {cmd[0]}: {exc}")


# ---------------------------------------------------------------------------
# Compose lifecycle
# ---------------------------------------------------------------------------

def compose_up(
    compose_root: Path, env_vars: Optional[Dict[str, str]] = None, timeout: int = 90
) -> DockerResult:
    """`docker compose up -d` with optional env overrides."""
    return _run(
        ["docker", "compose", "up", "-d"],
        cwd=compose_root,
        timeout=timeout,
        extra_env=env_vars,
    )


def compose_down(compose_root: Path, timeout: int = 60) -> DockerResult:
    return _run(["docker", "compose", "down", "-v"], cwd=compose_root, timeout=timeout)


def compose_restart(
    compose_root: Path,
    service: str,
    env_vars: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> DockerResult:
    """Restart a service. If env_vars are passed, recreate the container so the
    new env takes effect (e.g. raising memory limit)."""
    if env_vars:
        return _run(
            ["docker", "compose", "up", "-d", "--force-recreate", "--no-deps", service],
            cwd=compose_root,
            timeout=timeout,
            extra_env=env_vars,
        )
    return _run(["docker", "compose", "restart", service], cwd=compose_root, timeout=timeout)


def compose_scale(
    compose_root: Path, service: str, replicas: int, timeout: int = 60
) -> DockerResult:
    return _run(
        ["docker", "compose", "up", "-d", "--scale", f"{service}={replicas}", service],
        cwd=compose_root,
        timeout=timeout,
    )


def compose_logs(
    compose_root: Path, service: str, tail: int = 100, timeout: int = 15
) -> DockerResult:
    return _run(
        ["docker", "compose", "logs", "--no-color", "--tail", str(tail), service],
        cwd=compose_root,
        timeout=timeout,
    )


def compose_ps_json(compose_root: Path, timeout: int = 15) -> DockerResult:
    """`docker compose ps --format json` — one JSON object per line."""
    return _run(
        ["docker", "compose", "ps", "--format", "json"],
        cwd=compose_root,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Container introspection
# ---------------------------------------------------------------------------

def docker_stats_json(timeout: int = 10) -> DockerResult:
    """`docker stats --no-stream --format json` — one JSON object per line."""
    return _run(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
        ],
        timeout=timeout,
    )


def docker_inspect(container: str, timeout: int = 10) -> DockerResult:
    return _run(["docker", "inspect", container], timeout=timeout)


# ---------------------------------------------------------------------------
# Chaos orchestration — invokes the user's chaos.py CLI on the vibecoded site
# ---------------------------------------------------------------------------

CHAOS_FOR_TASK = {
    "oom_crash": "oom",
    "db_pool_exhaustion": "conn-leak",
    "bad_deployment_cascade": "bad-deploy",
    "disk_full": "disk-full",
    "slow_query": "lock-contention",
    "cert_expiry": "cert-expired",
}


def chaos_inject(
    compose_root: Path, task_id: str, timeout: int = 30
) -> DockerResult:
    """Invoke `python chaos.py --scenario=<name>` on the vibecoded site."""
    chaos_arg = CHAOS_FOR_TASK.get(task_id)
    if not chaos_arg:
        return DockerResult(ok=False, error=f"no chaos mapping for task '{task_id}'")
    chaos_script = compose_root / "chaos.py"
    if not chaos_script.exists():
        return DockerResult(ok=False, error=f"chaos.py not found at {chaos_script}")
    return _run(
        ["python", str(chaos_script), f"--scenario={chaos_arg}"],
        cwd=compose_root,
        timeout=timeout,
    )


def chaos_stop(compose_root: Path, timeout: int = 15) -> DockerResult:
    chaos_script = compose_root / "chaos.py"
    if not chaos_script.exists():
        return DockerResult(ok=False, error=f"chaos.py not found at {chaos_script}")
    return _run(
        ["python", str(chaos_script), "--stop"],
        cwd=compose_root,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------

def http_health(url: str, timeout: float = 2.0) -> Tuple[bool, int]:
    """Hit `url` and return (ok, status_code). Network errors → (False, 0)."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", 200)
            return (200 <= code < 300, int(code))
    except urllib.error.HTTPError as exc:
        return (False, int(exc.code))
    except (urllib.error.URLError, OSError, TimeoutError):
        return (False, 0)


# ---------------------------------------------------------------------------
# Parsers — pull out the bits we care about from docker JSON output
# ---------------------------------------------------------------------------

def parse_compose_ps(stdout: str) -> List[Dict[str, Any]]:
    """`docker compose ps --format json` may emit one JSON per line OR a JSON
    array depending on Compose v2.x. Handle both."""
    rows: List[Dict[str, Any]] = []
    s = stdout.strip()
    if not s:
        return rows
    # Try parsing as a single array first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    # Fallback: NDJSON (one object per line)
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def parse_docker_stats(stdout: str) -> List[Dict[str, Any]]:
    """`docker stats --format {{json .}}` emits NDJSON."""
    rows: List[Dict[str, Any]] = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _percent_to_float(value: Any) -> float:
    """Convert "12.34%" → 12.34. Returns 0.0 on any parsing issue."""
    if value is None:
        return 0.0
    s = str(value).strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _mem_to_mb(value: Any) -> float:
    """Convert "256MiB" / "1.2GiB" / "12kB" / "12MB" → MB float."""
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    units = [
        ("KiB", 1.0 / 1024.0),
        ("MiB", 1.0),
        ("GiB", 1024.0),
        ("TiB", 1024.0 * 1024.0),
        ("kB", 0.001),
        ("MB", 1.0),
        ("GB", 1000.0),
        ("KB", 0.001),
        ("B", 1.0 / (1024.0 * 1024.0)),
    ]
    for unit, factor in units:
        if s.endswith(unit):
            try:
                return float(s[: -len(unit)]) * factor
            except ValueError:
                return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def stats_to_service_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Roll up `docker stats` rows into per-service metrics.

    Keyed by container name (which Compose names like '<project>-<svc>-N').
    Returns {logical_service: {cpu_percent, memory_mb, memory_limit_mb}}.
    """
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        name = r.get("Name") or r.get("name") or r.get("Container")
        if not name:
            continue
        # Compose names are <project>-<service>-<replica>; pick the middle.
        parts = str(name).rsplit("-", 1)
        head = parts[0]
        if "-" in head:
            logical = head.split("-", 1)[1]
        else:
            logical = head
        # Parse CPU/mem
        cpu = _percent_to_float(r.get("CPUPerc") or r.get("cpu_percent"))
        mem_usage = r.get("MemUsage") or r.get("memory_usage") or ""
        mem_mb = 0.0
        mem_limit_mb = 0.0
        if " / " in str(mem_usage):
            used, limit = str(mem_usage).split(" / ", 1)
            mem_mb = _mem_to_mb(used)
            mem_limit_mb = _mem_to_mb(limit)
        out[logical] = {
            "cpu_percent": cpu,
            "memory_mb": mem_mb,
            "memory_limit_mb": mem_limit_mb,
        }
    return out


__all__ = [
    "DockerResult",
    "compose_up",
    "compose_down",
    "compose_restart",
    "compose_scale",
    "compose_logs",
    "compose_ps_json",
    "docker_stats_json",
    "docker_inspect",
    "chaos_inject",
    "chaos_stop",
    "http_health",
    "parse_compose_ps",
    "parse_docker_stats",
    "stats_to_service_metrics",
    "CHAOS_FOR_TASK",
]
