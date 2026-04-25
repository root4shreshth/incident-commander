"""Tier-2 code escalation — fires when Tier 1 (runtime ops) leaves the site degraded.

Honest scoping: Tier 1 is the trained RL policy executing the 10 typed actions
through the Backend Protocol. Tier 2 is a *rule-based + LLM-summarized* code
investigation that fires only when Tier 1's recovery check fails. It is NOT
RL-trained for this hackathon submission — it's the natural next step we wire
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
     "Code Escalation Report" — suspected file, suspected lines, suggested
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
# What we look for, by scenario family. Loose — these are heuristics, not RL.
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
# Repo acquisition — shallow clone into a temp dir
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
# Summary writer — LLM-optional
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
        target_service:  e.g. 'api' — biases the file ranking.
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
]
