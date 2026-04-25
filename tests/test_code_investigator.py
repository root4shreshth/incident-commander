"""Tests for the tier-2 code investigator.

Bypasses git clone by providing `cloned_root` directly. Verifies the heuristic
ranks the right files, surfaces relevant lines, and produces a sensible
rule-based summary. Also tests the optional LLM hook.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.code_investigator import (
    CodeEscalationReport,
    CodeFinding,
    SCENARIO_KEYWORDS,
    investigate,
)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Build a small fake repo with files clearly linked to each scenario."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "api.py").write_text(
        "# Service: api\n"
        "import functools\n"
        "\n"
        "_cache = {}  # unbounded cache — memory grows forever\n"
        "\n"
        "@functools.lru_cache  # uses memory\n"
        "def lookup(key):\n"
        "    _cache[key] = compute(key)\n"
        "    return _cache[key]\n"
    )
    (tmp_path / "src" / "db.py").write_text(
        "# postgres connection helpers\n"
        "from sqlalchemy import create_engine\n"
        "engine = create_engine('postgresql://...', pool_size=10)  # tiny pool\n"
        "\n"
        "def query(sql):\n"
        "    conn = engine.connect()  # leak: no close\n"
        "    return conn.execute(sql)\n"
    )
    (tmp_path / "src" / "deploy.py").write_text(
        "IMAGE_TAG = 'v1.1'\n"
        "PREVIOUS = 'v1.0'\n"
        "# rollout strategy: blue-green\n"
    )
    # Add some noise that should not score
    (tmp_path / "README.md").write_text("# project\nA boring readme.\n")
    (tmp_path / "src" / "noise.js").write_text("function foo() { return 42; }\n")
    return tmp_path


# ---------------------------------------------------------------------------
# OOM crash investigation
# ---------------------------------------------------------------------------

class TestOOMCrash:
    def test_finds_api_py_at_top(self, fake_repo):
        report = investigate(
            repo_url="local://fake",
            scenario="oom_crash",
            target_service="api",
            cloned_root=fake_repo,
        )
        assert report.error is None
        # The cache + lru_cache lines from api.py should rank highly
        top_files = [f.file_path for f in report.findings]
        assert any("api.py" in p for p in top_files)
        assert "memory" in report.summary.lower() or "cache" in report.summary.lower() or "leak" in report.summary.lower() or "unbounded" in report.summary.lower()
        assert report.suggested_fix


# ---------------------------------------------------------------------------
# DB pool exhaustion
# ---------------------------------------------------------------------------

class TestDBPool:
    def test_finds_db_py_at_top(self, fake_repo):
        report = investigate(
            repo_url="local://fake",
            scenario="db_pool_exhaustion",
            target_service="postgres",
            cloned_root=fake_repo,
        )
        assert report.error is None
        top_files = [f.file_path for f in report.findings]
        assert any("db.py" in p for p in top_files)
        # Suggested fix should mention pool / connection / engine
        sf = report.suggested_fix.lower()
        assert any(kw in sf for kw in ("pool", "connection", "engine"))


# ---------------------------------------------------------------------------
# Bad deployment
# ---------------------------------------------------------------------------

class TestBadDeploy:
    def test_finds_deploy_py(self, fake_repo):
        report = investigate(
            repo_url="local://fake",
            scenario="bad_deployment_cascade",
            target_service="api",
            cloned_root=fake_repo,
        )
        assert report.error is None
        top_files = [f.file_path for f in report.findings]
        assert any("deploy.py" in p for p in top_files)


# ---------------------------------------------------------------------------
# LLM hook
# ---------------------------------------------------------------------------

class TestLLMHook:
    def test_llm_summary_used_when_provided(self, fake_repo):
        captured = []
        def fake_llm(prompt: str) -> str:
            captured.append(prompt)
            return json.dumps({
                "summary": "LLM-says: api.py has an unbounded cache.",
                "suggested_fix": "Bound _cache to 1024 entries with an LRU.",
            })
        report = investigate(
            repo_url="local://fake", scenario="oom_crash", target_service="api",
            llm_call=fake_llm, cloned_root=fake_repo,
        )
        assert report.error is None
        assert report.summary == "LLM-says: api.py has an unbounded cache."
        assert "1024 entries" in report.suggested_fix
        assert captured  # the LLM was actually called

    def test_llm_invalid_json_falls_back_to_rule_based(self, fake_repo):
        report = investigate(
            repo_url="local://fake", scenario="oom_crash", target_service="api",
            llm_call=lambda p: "not even json", cloned_root=fake_repo,
        )
        assert report.error is None
        # Falls back to rule-based — produces SOME summary
        assert report.summary
        assert report.suggested_fix


# ---------------------------------------------------------------------------
# Empty / failed cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_repo_reports_no_source_files(self, tmp_path):
        report = investigate(
            repo_url="local://empty",
            scenario="oom_crash",
            cloned_root=tmp_path,
        )
        assert report.error is not None
        assert "no source files" in report.error

    def test_unknown_scenario_uses_empty_keywords(self, fake_repo):
        report = investigate(
            repo_url="local://fake",
            scenario="not_a_real_scenario",
            target_service="api",
            cloned_root=fake_repo,
        )
        # No error — just thin findings (target_service still scores)
        assert report.error is None

    def test_scenario_keywords_table_complete(self):
        for fam in ("oom_crash", "db_pool_exhaustion", "bad_deployment_cascade"):
            assert fam in SCENARIO_KEYWORDS
            assert len(SCENARIO_KEYWORDS[fam]) >= 3


# ---------------------------------------------------------------------------
# CodeEscalationReport serialization
# ---------------------------------------------------------------------------

class TestReportSerialization:
    def test_to_dict_round_trip(self):
        r = CodeEscalationReport(
            repo_url="https://github.com/foo/bar",
            scenario="oom_crash",
            target_service="api",
            findings=[CodeFinding(
                file_path="src/api.py", line_no=12, snippet="cache = {}",
                score=4.5, why="matches: memory, cache",
            )],
            summary="s", suggested_fix="f",
        )
        d = r.to_dict()
        assert d["scenario"] == "oom_crash"
        assert d["findings"][0]["file_path"] == "src/api.py"
        assert d["findings"][0]["score"] == 4.5
