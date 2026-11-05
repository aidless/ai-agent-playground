"""Locks the eval-harness and gate behaviors that used to go green on no signal."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from eval_harness.runner import is_passing
from eval_harness.scorers import (
    compute_total_score,
    score_llm_judge,
    score_tool_success_rate,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_no_signal_is_not_a_pass():
    scores = compute_total_score("hello world", enabled_scorers=["contains", "llm_judge"])
    assert scores["total"] == 0.0
    assert is_passing("", scores["total"], 0.6) is False


def test_empty_keyword_lists_do_not_score_perfect():
    scores = compute_total_score(
        "hello",
        expected_keywords=[],
        forbidden_keywords=[],
        enabled_scorers=["contains"],
    )
    assert "contains" not in scores
    assert "forbidden" not in scores
    assert scores["total"] == 0.0


def test_judge_outage_scores_zero(monkeypatch):
    class Boom:
        def send(self, **kwargs):
            raise RuntimeError("offline")

    monkeypatch.setattr("eval_harness.scorers.get_client", lambda: Boom())
    assert score_llm_judge("q", "a", "be correct") == 0.0


def test_unparseable_judge_scores_zero(monkeypatch):
    class Junk:
        def send(self, **kwargs):
            return "not a number"

    monkeypatch.setattr("eval_harness.scorers.get_client", lambda: Junk())
    assert score_llm_judge("q", "a", "be correct") == 0.0


def test_runner_error_cannot_pass():
    assert is_passing("boom", 1.0, 0.6) is False
    assert is_passing("", 0.6, 0.6) is True


def test_missing_tool_rate_is_not_a_pass():
    assert score_tool_success_rate({}) == 0.0
    assert score_tool_success_rate({"tool_calls": 2}) == 0.0


def test_b3_label_is_not_a_defense():
    bench = _load("b3_security_bench", "scripts/b3_security_bench.py")
    assert bench.defense_held(injection_detected=False, path_blocked=False) is False
    assert bench.defense_held(injection_detected=True, path_blocked=False) is True
    assert bench.defense_held(injection_detected=False, path_blocked=True) is True


def test_eval_gate_ci_missing_baseline_fails_closed():
    gate = _load("eval_gate_mod", "scripts/eval_gate.py")
    assert gate.exit_code_when_baseline_missing(True) == 2
    assert gate.exit_code_when_baseline_missing(False) == 0


def test_placeholder_key_is_not_a_live_model(monkeypatch):
    e2e = _load("e2e_test_mod", "scripts/e2e_test.py")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ci-placeholder")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    assert e2e.live_model_configured() is False
