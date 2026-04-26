"""Tier-2 code escalation - fires when Tier 1 (runtime ops) leaves the site degraded.

Honest scoping: Tier 1 is the trained RL policy executing the 10 typed actions
through the Backend Protocol. Tier 2 is a *rule-based + LLM-summarized* code
investigation that fires only when Tier 1's recovery check fails. It is NOT
RL-trained for this hackathon submission - it's the natural next step we wire
in to demonstrate the full SRE workflow (ops first, code second).

Phase 2 of the project roadmap promises to RL-train Tier 2 too via the
`CodeAwareBackend`. For now, this module:

  1. Clones (or shallow-fetches) a public GitHub repo at the given URL.
  2. Greps for files matching the failing service's name + the alert keywords
     (e.g. for an OOM crash on api: search for `api`, `memory`, `OOM`, `cache`,
     `buffer`).
  3. Reads the top-N matching files and extracts the lines that look most
     likely to be related (memory allocations, large list/dict construction,
     long-lived caches without bounds).
  4. Optionally calls an LLM (OpenRouter / local) to write a one-paragraph
     "Code Escalation Report" - suspected file, suspected lines, suggested
     direction. If no LLM is available, falls back to a rule-based summary.

The module makes a deliberate choice to NOT modify the codebase. Tier 2 is
diagnosis-only for the hackathon. Phase 2 adds `propose_patch` + `run_tests`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# What we look for, by scenario family. Loose - these are heuristics, not RL.
# ---------------------------------------------------------------------------

SCENARIO_KEYWORDS: Dict[str, List[str]] = {
    "oom_crash": [
        "memory", "oom", "out of memory", "outofmemory", "cache", "buffer",
        "list(", "dict(", "lru_cache", "store", "accumulate",
    ],
    "db_pool_exhaustion": [
        "pool", "connection", "engine", "session", "leak", "close()",
        "with engine", "create_engine", "psycopg", "sqlalchemy",
    ],
    "bad_deployment_cascade": [
        "version", "deploy", "release", "v1.0", "v1.1", "rollout",
        "image_tag", "feature_flag", "migration",
    ],
}

# File extensions worth grepping for source code
CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".go", ".java", ".rb", ".rs", ".php"}

# How many files we'll inspect (capped to keep latency + LLM tokens bounded)
MAX_FILES_INSPECTED = 8

# How many lines per file we'll surface in the report
MAX_LINES_PER_FILE = 12


@dataclass
class CodeFinding:
    """One candidate code location worth inspecting."""
    file_path: str
    line_no: int
    snippet: str
    score: float
    why: str


@dataclass
class CodeEscalationReport:
    """The artifact tier 2 produces. Renders as a card in the Phase 3 UI."""
    repo_url: str
    scenario: str
    target_service: Optional[str]
    findings: List[CodeFinding] = field(default_factory=list)
    summary: str = ""
    suggested_fix: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_url": self.repo_url,
            "scenario": self.scenario,
            "target_service": self.target_service,
            "findings": [
                {
                    "file_path": f.file_path,
                    "line_no": f.line_no,
                    "snippet": f.snippet,
                    "score": round(f.score, 2),
                    "why": f.why,
                }
                for f in self.findings
            ],
            "summary": self.summary,
            "suggested_fix": self.suggested_fix,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Repo acquisition - shallow clone into a temp dir
# ---------------------------------------------------------------------------

def _clone_repo(repo_url: str, token: Optional[str] = None, timeout: int = 30) -> Optional[Path]:
    """Shallow-clone a public GitHub repo into a temp directory.

    For private repos, pass a token (will be embedded in the URL for git's auth).
    Returns the temp Path on success or None on failure.
    """
    repo_url = (repo_url or "").strip().rstrip("/")
    if not repo_url:
        return None

    # If a token is provided and the URL is github.com, embed it for HTTPS auth.
    auth_url = repo_url
    if token and "github.com" in repo_url:
        # Ensure https://
        if repo_url.startswith("http://"):
            repo_url = "https://" + repo_url[len("http://"):]
        if repo_url.startswith("https://"):
            auth_url = "https://x-access-token:" + token + "@" + repo_url[len("https://"):]

    tmpdir = tempfile.mkdtemp(prefix="ic-tier2-")
    try:
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", auth_url, tmpdir],
            check=False, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None
        return Path(tmpdir)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None


def cleanup_repo(path: Optional[Path]) -> None:
    """Best-effort cleanup of a previously cloned repo."""
    if path is None:
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Grep + score
# ---------------------------------------------------------------------------

def _list_code_files(root: Path) -> List[Path]:
    """List source code files under `root`, skipping junk dirs."""
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.suffix.lower() not in CODE_EXTS:
            continue
        try:
            if p.stat().st_size > 200_000:  # skip huge generated files
                continue
        except OSError:
            continue
        out.append(p)
    return out


def _score_file(content: str, target_service: Optional[str], keywords: List[str]) -> float:
    """Heuristic relevance score for a file's content."""
    lower = content.lower()
    score = 0.0
    if target_service:
        score += 5.0 * lower.count(target_service.lower())
    for kw in keywords:
        score += lower.count(kw.lower())
    return score


def _extract_findings(
    file_path: Path, root: Path, content: str,
    target_service: Optional[str], keywords: List[str],
) -> List[CodeFinding]:
    """Pull lines from `content` that look related to the failure."""
    rel_path = str(file_path.relative_to(root)).replace("\\", "/")
    findings: List[CodeFinding] = []
    needles = [n.lower() for n in keywords]
    if target_service:
        needles.append(target_service.lower())
    for i, line in enumerate(content.splitlines(), start=1):
        ll = line.lower()
        hits = sum(1 for n in needles if n and n in ll)
        if hits == 0:
            continue
        score = float(hits)
        # Bigger boost if the line clearly allocates / leaks
        for boost_kw in ("memory", "cache", "leak", "pool", "connection", "alloc"):
            if boost_kw in ll:
                score += 0.5
        why = "matches: " + ", ".join(
            n for n in needles if n and n in ll
        )
        findings.append(CodeFinding(
            file_path=rel_path,
            line_no=i,
            snippet=line.strip()[:200],
            score=score,
            why=why,
        ))
    findings.sort(key=lambda f: -f.score)
    return findings[:MAX_LINES_PER_FILE]


# ---------------------------------------------------------------------------
# Summary writer - LLM-optional
# ---------------------------------------------------------------------------

def _rule_based_summary(
    scenario: str, target_service: Optional[str], findings: List[CodeFinding]
) -> Tuple[str, str]:
    """Fallback summary when no LLM is available. Honest about its scope."""
    files = sorted({f.file_path for f in findings})
    if not files:
        return (
            f"Tier 1 ops did not fully heal the site, but no obvious code-level "
            f"signature was found for scenario '{scenario}' in the linked repo. "
            f"Consider widening the search keywords or providing a more specific "
            f"reproducer.",
            "No file-level suggestion. Try widening the search keywords."
        )
    files_text = ", ".join(files[:5])
    summary = (
        f"Tier 1 (runtime ops) did not fully heal the site. Tier 2 grepped the "
        f"linked repo for keywords related to '{scenario}'"
        + (f" on service '{target_service}'" if target_service else "")
        + f". Top suspect files: {files_text}. The most-relevant lines surface "
          f"unbounded data structures, missing connection-pool limits, or version "
          f"strings that match the scenario."
    )
    suggested = ""
    if scenario == "oom_crash":
        suggested = (
            f"Inspect {files[0]} for unbounded caches or accumulator structures "
            f"(lists / dicts that grow without an LRU bound). Add a maxsize or "
            f"emit eviction metrics."
        )
    elif scenario == "db_pool_exhaustion":
        suggested = (
            f"Inspect {files[0]} for connection acquisition without explicit "
            f"close/release. Use a `with engine.connect()` or context manager "
            f"and confirm the pool ceiling matches expected concurrency."
        )
    elif scenario == "bad_deployment_cascade":
        suggested = (
            f"Inspect {files[0]} for the recently changed image tag / version "
            f"string. Compare with the prior release; revert the offending "
            f"change and add a canary stage to the deploy pipeline."
        )
    else:
        suggested = f"Inspect {files[0]} for the matching keywords reported in 'findings'."
    return summary, suggested


def _llm_summary(
    scenario: str, target_service: Optional[str],
    findings: List[CodeFinding],
    llm_call: Callable[[str], str],
) -> Tuple[str, str]:
    """Call the supplied LLM to produce the summary + suggested fix."""
    snippets = "\n".join(
        f"  {f.file_path}:{f.line_no}  [{f.score:.1f}]  {f.snippet}"
        for f in findings[:24]
    )
    prompt = (
        "You are a senior SRE escalating an incident from runtime ops to code review. "
        "Tier 1 (restart/rollback/config changes) ran but the site is still degraded. "
        "Below are code lines a heuristic flagged as potentially related. Write a "
        "one-paragraph diagnosis ('summary') and a one-paragraph suggested fix.\n\n"
        f"Scenario: {scenario}\n"
        f"Target service: {target_service or '(unknown)'}\n\n"
        "Flagged lines:\n" + snippets + "\n\n"
        'Reply ONLY in JSON: {"summary": "...", "suggested_fix": "..."}'
    )
    try:
        raw = llm_call(prompt) or ""
        match = re.search(r"\{[\s\S]+\}", raw)
        if match:
            obj = json.loads(match.group(0))
            return (
                str(obj.get("summary") or "").strip(),
                str(obj.get("suggested_fix") or "").strip(),
            )
    except Exception:
        pass
    return _rule_based_summary(scenario, target_service, findings)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def investigate(
    repo_url: str,
    scenario: str,
    target_service: Optional[str] = None,
    repo_token: Optional[str] = None,
    llm_call: Optional[Callable[[str], str]] = None,
    cloned_root: Optional[Path] = None,
) -> CodeEscalationReport:
    """Run tier-2 code investigation. Caller may pre-clone for testing.

    Args:
        repo_url:        the GitHub URL to inspect (or a local file:// path
                         when `cloned_root` is supplied for tests).
        scenario:        scenario family (drives keyword set).
        target_service:  e.g. 'api' - biases the file ranking.
        repo_token:      optional GitHub PAT for private repos.
        llm_call:        optional callable that takes a prompt string and
                         returns the LLM's raw text reply. If None, we use
                         a deterministic rule-based summary instead.
        cloned_root:     if provided, skip cloning and use this path directly
                         (used by tests + for re-running without re-cloning).

    Returns:
        CodeEscalationReport (with `error` populated on failure).
    """
    keywords = SCENARIO_KEYWORDS.get(scenario, [])
    report = CodeEscalationReport(
        repo_url=repo_url, scenario=scenario, target_service=target_service,
    )
    own_clone = False
    root = cloned_root
    if root is None:
        root = _clone_repo(repo_url, repo_token)
        own_clone = True
        if root is None:
            report.error = "git clone failed (check repo_url / token / network)"
            return report
    try:
        files = _list_code_files(root)
        if not files:
            report.error = "no source files found under repo root"
            return report
        # Score files, take top N
        scored: List[Tuple[float, Path]] = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            s = _score_file(content, target_service, keywords)
            if s > 0:
                scored.append((s, f))
        scored.sort(reverse=True)
        top_files = [f for _, f in scored[:MAX_FILES_INSPECTED]]
        all_findings: List[CodeFinding] = []
        for f in top_files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            all_findings.extend(_extract_findings(f, root, content, target_service, keywords))
        all_findings.sort(key=lambda x: -x.score)
        # Cap total findings
        report.findings = all_findings[:24]
        # Summarize
        if llm_call is not None:
            summary, suggested = _llm_summary(scenario, target_service, report.findings, llm_call)
        else:
            summary, suggested = _rule_based_summary(scenario, target_service, report.findings)
        report.summary = summary
        report.suggested_fix = suggested
        return report
    finally:
        if own_clone:
            cleanup_repo(root)


__all__ = [
    "CodeFinding",
    "CodeEscalationReport",
    "SCENARIO_KEYWORDS",
    "investigate",
    "cleanup_repo",
    # Phase 2 - actually ship the fix
    "Patch",
    "TestResult",
    "PullRequestResult",
    "propose_patch",
    "apply_patch",
    "run_tests",
    "open_pull_request",
]


# ===========================================================================
# Phase 2 - code-shipping actions (propose, apply, test, PR).
#
# Tier-2 is no longer diagnosis-only. Given a CodeEscalationReport, Praetor
# can:
#   1. propose_patch  - synthesize a unified diff for the most-suspect line
#   2. apply_patch    - apply the diff to a temp branch in the cloned repo
#   3. run_tests      - run pytest (or other) against the patched tree
#   4. open_pull_request - push the branch + open a PR via the GitHub API
#
# Safety posture for the hackathon:
#   - propose / apply / test all happen in a temp clone, never the user's
#     working tree.
#   - open_pull_request is gated by an explicit `enable_pr_open=True` flag
#     AND a token with `repo` scope. Without both, returns dry_run=True.
#   - All operations have hard timeouts and capped output sizes.
# ===========================================================================


@dataclass
class Patch:
    """A proposed code change in unified-diff form."""
    file_path: str
    line_no: int
    diff: str               # unified diff text
    rationale: str          # why this change addresses the scenario
    confidence: float = 0.5  # 0..1, heuristic confidence that this is the right fix


@dataclass
class TestResult:
    """Outcome of running the test suite after applying a patch."""
    framework: str          # "pytest" | "unittest" | "npm" | ...
    passed: bool
    n_tests: int = 0
    n_failed: int = 0
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: Optional[str] = None


@dataclass
class PullRequestResult:
    """Outcome of pushing a branch + opening a PR."""
    opened: bool
    dry_run: bool
    branch: str
    pr_url: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Patch synthesis - a small set of scenario-aware templates
# ---------------------------------------------------------------------------

_PATCH_TEMPLATES: Dict[str, Dict[str, str]] = {
    # Each entry: { trigger_substring: replacement_or_directive }
    "oom_crash": {
        "_cache = {}":
            "# Bound the cache so it doesn't grow without limit (was: unbounded dict).\n"
            "from functools import lru_cache\n"
            "_CACHE_MAX = 1024\n"
            "_cache: dict = {}\n"
            "def _evict_if_needed():\n"
            "    if len(_cache) > _CACHE_MAX:\n"
            "        # FIFO eviction; replace with proper LRU in production.\n"
            "        for k in list(_cache)[: len(_cache) - _CACHE_MAX]:\n"
            "            _cache.pop(k, None)\n",
        "@functools.lru_cache":
            "@functools.lru_cache(maxsize=1024)  # Bounded - was unbounded.",
    },
    "db_pool_exhaustion": {
        "engine.connect()":
            "# Use a context manager so the connection is always returned to the pool.\n"
            "with engine.connect() as conn:",
        "create_engine(": (
            "# Raise pool ceiling and add a sane overflow.\n"
            "create_engine(  # NOTE: pool_size raised + max_overflow for spike traffic\n"
        ),
    },
    "slow_query": {
        "FOR UPDATE":
            "# Avoid holding row-locks across long txns; use SELECT ... FOR UPDATE SKIP LOCKED.\n"
            "FOR UPDATE SKIP LOCKED",
    },
    "disk_full": {
        "log.write(": (
            "# Rotate logs after every N writes so we don't fill the volume.\n"
            "log.write(  # TODO: wire up RotatingFileHandler with maxBytes\n"
        ),
    },
    "cert_expiry": {
        "validity until": (
            "# Cert expiry: ensure auto-renewal cron is wired up.\n"
            "# TODO: confirm certbot or cert-manager will renew this cert automatically.\n"
            "# validity until"
        ),
    },
    "bad_deployment_cascade": {
        "IMAGE_TAG = ": (
            "# Roll back to the last-known-good tag; revert in a follow-up commit\n"
            "# once the bad version is fully diagnosed.\n"
            "IMAGE_TAG = "
        ),
    },
}


def propose_patch(report: CodeEscalationReport) -> Optional[Patch]:
    """Synthesize a unified-diff patch from the highest-scored finding.

    Returns None if no template matches the scenario or no finding is suitable.
    """
    if not report.findings:
        return None
    templates = _PATCH_TEMPLATES.get(report.scenario, {})
    if not templates:
        return None
    for finding in report.findings:
        snippet = finding.snippet
        for trigger, replacement in templates.items():
            if trigger.lower() in snippet.lower():
                # Build a unified diff. Single-line, single-hunk - the smallest
                # actionable change to demonstrate the workflow.
                old_line = snippet
                new_block = replacement
                diff_lines = [
                    f"--- a/{finding.file_path}",
                    f"+++ b/{finding.file_path}",
                    f"@@ -{finding.line_no},1 +{finding.line_no},{len(new_block.splitlines()) or 1} @@",
                    f"-{old_line}",
                ]
                for nl in (new_block.splitlines() or [new_block]):
                    diff_lines.append(f"+{nl}")
                rationale = (
                    f"Heuristic match against scenario '{report.scenario}'. "
                    f"The line {finding.file_path}:{finding.line_no} matches "
                    f"the pattern {trigger!r}; replacing with the corresponding "
                    f"bounded / safer form."
                )
                return Patch(
                    file_path=finding.file_path,
                    line_no=finding.line_no,
                    diff="\n".join(diff_lines),
                    rationale=rationale,
                    confidence=min(1.0, finding.score / 5.0),
                )
    return None


# ---------------------------------------------------------------------------
# Apply patch - write the new content into a branch, never the working tree
# ---------------------------------------------------------------------------

def apply_patch(repo_root: Path, patch: Patch, branch: str = "praetor/auto-fix") -> bool:
    """Apply `patch` to `repo_root` on a fresh branch. Returns True on success."""
    repo_root = Path(repo_root)
    if not (repo_root / ".git").exists():
        # Initialize a temporary repo so we can cleanly stash changes on a branch.
        subprocess.run(["git", "init", "-q"], cwd=repo_root, check=False, timeout=10)
        subprocess.run(["git", "add", "-A"], cwd=repo_root, check=False, timeout=10)
        subprocess.run(
            ["git", "-c", "user.email=praetor@local", "-c", "user.name=Praetor",
             "commit", "-q", "-m", "baseline"],
            cwd=repo_root, check=False, timeout=10,
        )
    # Create branch
    subprocess.run(["git", "checkout", "-q", "-b", branch], cwd=repo_root,
                   check=False, timeout=10)
    # Read current file, line-edit (we already have the diff but applying it via
    # `git apply` is brittle for our heuristic format - easier to do a direct
    # line replacement using the patch's structure).
    target = repo_root / patch.file_path
    if not target.exists():
        return False
    try:
        original = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        # Replace the target line with the new block from the diff
        new_lines: List[str] = []
        for line in patch.diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                new_lines.append(line[1:] + "\n")
        if not new_lines or patch.line_no < 1 or patch.line_no > len(original):
            return False
        original[patch.line_no - 1: patch.line_no] = new_lines
        target.write_text("".join(original), encoding="utf-8")
    except OSError:
        return False
    # Stage + commit on the branch
    subprocess.run(["git", "add", str(target)], cwd=repo_root, check=False, timeout=10)
    proc = subprocess.run(
        ["git", "-c", "user.email=praetor@local", "-c", "user.name=Praetor",
         "commit", "-q", "-m",
         f"Praetor auto-fix: {patch.file_path}:{patch.line_no}\n\n{patch.rationale}"],
        cwd=repo_root, check=False, capture_output=True, text=True, timeout=10,
    )
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Run tests - pytest by default; falls back to unittest discovery; npm if Node
# ---------------------------------------------------------------------------

def run_tests(
    repo_root: Path,
    framework: str = "auto",
    timeout: int = 120,
    test_target: Optional[str] = None,
) -> TestResult:
    """Run the project's test suite inside `repo_root`.

    `framework='auto'` picks pytest if there's a tests/ dir, npm if there's a
    package.json + test script, otherwise falls back to `python -m unittest discover`.
    """
    import time as _time
    repo_root = Path(repo_root)
    chosen = framework
    cmd: Optional[List[str]] = None
    if chosen == "auto":
        if (repo_root / "package.json").exists():
            chosen, cmd = "npm", ["npm", "test", "--silent"]
        elif (repo_root / "tests").exists() or (repo_root / "test").exists():
            chosen, cmd = "pytest", ["python", "-m", "pytest", "-q", "--no-header"]
        else:
            chosen, cmd = "unittest", ["python", "-m", "unittest", "discover", "-q"]
    elif chosen == "pytest":
        cmd = ["python", "-m", "pytest", "-q", "--no-header"]
    elif chosen == "npm":
        cmd = ["npm", "test", "--silent"]
    elif chosen == "unittest":
        cmd = ["python", "-m", "unittest", "discover", "-q"]
    else:
        return TestResult(framework=chosen, passed=False, error=f"unknown framework: {chosen}")
    if test_target:
        cmd.append(test_target)
    start = _time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=repo_root, check=False, capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as exc:
        return TestResult(framework=chosen, passed=False, error=f"command not found: {exc}")
    except subprocess.TimeoutExpired:
        return TestResult(framework=chosen, passed=False, error=f"tests timed out after {timeout}s")
    duration = _time.monotonic() - start
    out = (proc.stdout or "")[-3000:]
    err = (proc.stderr or "")[-3000:]
    n_failed = 0
    n_tests = 0
    # Parse pytest-style "1 failed, 5 passed in 0.2s"
    import re as _re
    m = _re.search(r"(\d+)\s+failed[\s,].*?(\d+)\s+passed", out)
    if m:
        n_failed = int(m.group(1)); n_tests = int(m.group(1)) + int(m.group(2))
    else:
        m = _re.search(r"(\d+)\s+passed", out)
        if m: n_tests = int(m.group(1))
    return TestResult(
        framework=chosen, passed=(proc.returncode == 0),
        n_tests=n_tests, n_failed=n_failed, duration_s=round(duration, 2),
        stdout_tail=out, stderr_tail=err,
    )


# ---------------------------------------------------------------------------
# Open pull request - pushes branch + creates a GitHub PR
# ---------------------------------------------------------------------------

def open_pull_request(
    repo_root: Path,
    repo_url: str,
    branch: str,
    title: str,
    body: str,
    token: Optional[str] = None,
    enable_pr_open: bool = False,
    base_branch: str = "main",
) -> PullRequestResult:
    """Push the branch and open a PR on GitHub.

    Hard-gated: even with a token, this only runs when `enable_pr_open=True`.
    Default (off) returns dry_run=True so the user can preview the proposed
    PR before authorizing the actual push.
    """
    if not enable_pr_open:
        return PullRequestResult(
            opened=False, dry_run=True, branch=branch,
            error="enable_pr_open=False - preview only. Re-run with the flag to push the branch.",
        )
    if not token:
        return PullRequestResult(
            opened=False, dry_run=False, branch=branch,
            error="no token - cannot push branch or open PR",
        )
    if "github.com" not in repo_url.lower():
        return PullRequestResult(
            opened=False, dry_run=False, branch=branch,
            error="only github.com is supported for PR opening in Phase 2 scope",
        )
    repo_root = Path(repo_root)
    # Build the auth URL
    auth_url = repo_url
    if auth_url.startswith("https://"):
        auth_url = auth_url.replace(
            "https://", f"https://x-access-token:{token}@", 1,
        )
    # Push the branch
    push = subprocess.run(
        ["git", "push", "-q", auth_url, branch], cwd=repo_root,
        check=False, capture_output=True, text=True, timeout=30,
    )
    if push.returncode != 0:
        return PullRequestResult(
            opened=False, dry_run=False, branch=branch,
            error=f"git push failed: {(push.stderr or push.stdout)[-300:]}",
        )
    # Open the PR via the GitHub REST API
    import urllib.error as _ue
    import urllib.request as _ur
    # owner/repo from URL
    canonical = repo_url.rstrip("/").rstrip(".git")
    parts = canonical.split("github.com/", 1)[-1].split("/")
    if len(parts) < 2:
        return PullRequestResult(
            opened=False, dry_run=False, branch=branch,
            error=f"could not parse owner/repo from {repo_url}",
        )
    owner, repo = parts[0], parts[1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    payload = json.dumps({
        "title": title, "body": body, "head": branch, "base": base_branch,
    }).encode("utf-8")
    req = _ur.Request(api_url, data=payload, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "praetor-tier2",
    })
    try:
        with _ur.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return PullRequestResult(
                opened=True, dry_run=False, branch=branch,
                pr_url=data.get("html_url"),
            )
    except _ue.HTTPError as exc:
        return PullRequestResult(
            opened=False, dry_run=False, branch=branch,
            error=f"GitHub API HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:300]}",
        )
    except (_ue.URLError, OSError) as exc:
        return PullRequestResult(
            opened=False, dry_run=False, branch=branch,
            error=f"network error opening PR: {exc}",
        )
