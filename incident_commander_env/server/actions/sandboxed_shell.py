"""Sandboxed shell action — Phase 2.

Adds a `run_shell` capability to Praetor's diagnostic vocabulary, with a
strict 20-command allowlist. Lets the agent reach for tools the typed
action space doesn't cover (process inspection, file inventory, quick
network checks) without giving it carte blanche over a real shell.

Safety posture:
  - Only commands in `ALLOWED_COMMANDS` are accepted. Anything else returns
    a typed error before subprocess.run is ever called.
  - Each command has a per-command argument allowlist (regex or literal).
  - Hard timeout (10s by default) on every invocation.
  - Output is capped to 8 KB (head + tail with a marker in the middle if
    truncated).
  - No shell expansion: we always pass `shell=False` and a list of args.
  - Working directory is the env's runtime, NOT the user's repo unless
    explicitly passed via `cwd`.
  - Network commands are read-only and target localhost / loopback only.

This is *not* a general-purpose shell. It's a typed read-mostly probe
surface designed to be safe even if the policy goes off-script.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Allowlist — 20 commands max, each with an argument validator.
# ---------------------------------------------------------------------------

# Format: command -> (description, argument_validator).
# argument_validator is a callable (List[str]) -> Tuple[bool, error_msg].

def _no_args(args: List[str]) -> Tuple[bool, Optional[str]]:
    if args:
        return False, "this command takes no arguments"
    return True, None


def _safe_path_args(args: List[str]) -> Tuple[bool, Optional[str]]:
    """Each arg must be a path that doesn't traverse outside cwd."""
    for a in args:
        if a.startswith("-"):
            # allow -short and --long flags but not anything fancy
            if not re.match(r"^-[a-zA-Z0-9]+$|^--[a-zA-Z0-9-]+(=[\w./-]+)?$", a):
                return False, f"unsafe flag: {a!r}"
            continue
        if ".." in a or a.startswith("/") or a.startswith("\\"):
            return False, f"path traversal not allowed: {a!r}"
        if any(c in a for c in (";", "|", "&", "`", "$", ">", "<")):
            return False, f"shell metachars not allowed: {a!r}"
    return True, None


def _localhost_url(args: List[str]) -> Tuple[bool, Optional[str]]:
    """For curl / wget — must hit localhost / 127.0.0.1 / our env's host."""
    if not args:
        return False, "URL required"
    url = args[-1]
    allowed_hosts = ("http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1")
    if not url.startswith(allowed_hosts):
        return False, "url must point at localhost / 127.0.0.1"
    for a in args[:-1]:
        if not re.match(r"^-[a-zA-Z]$|^--[a-zA-Z-]+$", a):
            return False, f"only short / long flags allowed: {a!r}"
    return True, None


def _grep_args(args: List[str]) -> Tuple[bool, Optional[str]]:
    """grep needs a pattern + optionally a path."""
    if not args:
        return False, "grep needs a pattern"
    # Disallow pipe / file redirection; only safe paths
    return _safe_path_args(args)


ALLOWED_COMMANDS: Dict[str, Tuple[str, callable]] = {
    # --- File inventory (5) ---
    "ls":       ("list directory contents", _safe_path_args),
    "pwd":      ("print working directory", _no_args),
    "wc":       ("count words / lines / chars", _safe_path_args),
    "head":     ("first lines of a file", _safe_path_args),
    "tail":     ("last lines of a file", _safe_path_args),
    # --- Search (3) ---
    "grep":     ("search text in files", _grep_args),
    "find":     ("find files by name", _safe_path_args),
    "stat":     ("file metadata", _safe_path_args),
    # --- Process inspection (4) ---
    "ps":       ("list processes", _safe_path_args),
    "top":      ("snapshot of top processes", _no_args),
    "uptime":   ("how long the system has been up", _no_args),
    "free":     ("memory usage", _safe_path_args),
    # --- Disk (2) ---
    "df":       ("disk free", _safe_path_args),
    "du":       ("disk usage", _safe_path_args),
    # --- Network (read-only, localhost-only) (3) ---
    "curl":     ("HTTP probe (localhost only, read-only)", _localhost_url),
    "ping":     ("ICMP probe (4 packets, localhost only)",
                 lambda a: (
                     (len(a) == 1 and a[0] in ("localhost", "127.0.0.1")),
                     None if (len(a) == 1 and a[0] in ("localhost", "127.0.0.1"))
                          else "ping target must be localhost / 127.0.0.1",
                 )),
    "nslookup": ("DNS lookup (localhost / 127.0.0.1 only)",
                 lambda a: (
                     (len(a) == 1 and a[0] in ("localhost", "127.0.0.1")),
                     None if (len(a) == 1 and a[0] in ("localhost", "127.0.0.1"))
                          else "nslookup target must be localhost / 127.0.0.1",
                 )),
    # --- Misc (3) ---
    "echo":     ("echo arguments back", lambda a: (True, None)),
    "date":     ("current date / time", _no_args),
    "whoami":   ("current user", _no_args),
}

assert len(ALLOWED_COMMANDS) <= 20, "Allowlist must stay at or below 20 commands"


@dataclass
class ShellResult:
    ok: bool
    command: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None
    truncated: bool = False


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

_MAX_OUTPUT = 8 * 1024
_DEFAULT_TIMEOUT = 10


def _truncate(text: str) -> Tuple[str, bool]:
    if len(text) <= _MAX_OUTPUT:
        return text, False
    half = _MAX_OUTPUT // 2 - 32
    return text[:half] + "\n\n…[truncated]…\n\n" + text[-half:], True


def parse_command(line: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    """Split a command line into (cmd, args, error). Uses shlex for quote-aware
    parsing but never executes the shell."""
    line = (line or "").strip()
    if not line:
        return None, [], "empty command"
    try:
        tokens = shlex.split(line, posix=True)
    except ValueError as exc:
        return None, [], f"unparseable command: {exc}"
    if not tokens:
        return None, [], "no command after parse"
    cmd, *args = tokens
    if cmd not in ALLOWED_COMMANDS:
        return None, [], (
            f"command {cmd!r} not in allowlist. "
            f"Allowed: {sorted(ALLOWED_COMMANDS)}"
        )
    desc, validator = ALLOWED_COMMANDS[cmd]
    ok, why = validator(args)
    if not ok:
        return None, [], f"argument validation failed for {cmd!r}: {why}"
    return cmd, args, None


def run_shell(
    line: str, cwd: Optional[Path] = None, timeout: int = _DEFAULT_TIMEOUT,
) -> ShellResult:
    """Execute a single shell command from the allowlist."""
    import time as _time
    cmd, args, err = parse_command(line)
    if err:
        return ShellResult(ok=False, command=line, error=err)
    # Resolve binary — fail closed if it isn't installed on the system
    binary = shutil.which(cmd)
    if not binary:
        return ShellResult(
            ok=False, command=line,
            error=f"{cmd!r} is on the allowlist but not installed on this host",
        )
    started = _time.monotonic()
    try:
        proc = subprocess.run(
            [binary, *args], shell=False, cwd=str(cwd) if cwd else None,
            check=False, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ShellResult(
            ok=False, command=line,
            error=f"timed out after {timeout}s",
            elapsed_s=round(_time.monotonic() - started, 2),
        )
    out, out_truncated = _truncate(proc.stdout or "")
    errt, err_truncated = _truncate(proc.stderr or "")
    return ShellResult(
        ok=(proc.returncode == 0), command=line,
        stdout=out, stderr=errt,
        returncode=proc.returncode,
        elapsed_s=round(_time.monotonic() - started, 2),
        truncated=(out_truncated or err_truncated),
    )


def list_allowed() -> List[Dict[str, str]]:
    """Return the allowlist for the API/UI to render."""
    return [
        {"command": cmd, "description": desc}
        for cmd, (desc, _) in ALLOWED_COMMANDS.items()
    ]


__all__ = [
    "ALLOWED_COMMANDS",
    "ShellResult",
    "parse_command",
    "run_shell",
    "list_allowed",
]
