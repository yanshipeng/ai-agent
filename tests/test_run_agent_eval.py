"""run_agent_eval 报告/样例分布单测（不打 LLM）。"""

from __future__ import annotations

from pathlib import Path

from scripts.run_agent_eval import (
    build_report,
    is_clarified,
    is_tool_fail_error,
    load_samples,
    validate_sample_distribution,
)


def test_agent_eval_samples_distribution():
    samples = load_samples(Path("agent_eval_samples.jsonl"))
    counts = validate_sample_distribution(samples)
    assert len(samples) >= 30
    assert counts["tool"] >= 20
    assert counts["clarify"] >= 8
    assert all(s.get("category") == "A" for s in samples)


def test_build_agent_report_metrics_shape():
    results = [
        {
            "tag": "tool",
            "ok": True,
            "had_tool_calls": True,
            "citations_nonempty": True,
            "expect_clarify": False,
            "clarified": None,
            "tool_failed": False,
            "latency_ms_total": 100,
            "agent_steps": 2,
            "error_code": None,
        },
        {
            "tag": "tool",
            "ok": True,
            "had_tool_calls": True,
            "citations_nonempty": False,
            "expect_clarify": False,
            "clarified": None,
            "tool_failed": False,
            "latency_ms_total": 200,
            "agent_steps": 3,
            "error_code": None,
        },
        {
            "tag": "clarify",
            "ok": True,
            "had_tool_calls": False,
            "citations_nonempty": False,
            "expect_clarify": True,
            "clarified": True,
            "tool_failed": False,
            "latency_ms_total": 150,
            "agent_steps": 1,
            "error_code": None,
        },
        {
            "tag": "tool",
            "ok": False,
            "had_tool_calls": False,
            "citations_nonempty": False,
            "expect_clarify": False,
            "clarified": None,
            "tool_failed": True,
            "latency_ms_total": 80,
            "agent_steps": None,
            "error_code": "TOOL_TIMEOUT",
        },
    ]
    report = build_report(results, label="t")
    assert report["tool_call_rate"] == 0.5
    assert report["citation_coverage"] == 0.25
    assert report["clarify_rate"] == 1.0
    assert report["tool_fail_rate"] == 0.25
    assert report["latency_ms_total"]["p50"] is not None
    assert report["latency_ms_total"]["p95"] is not None
    assert report["avg_steps"] is not None
    assert report["p95_steps"] is not None
    assert report["top_errors"][0]["error_code"] == "TOOL_TIMEOUT"


def test_clarify_and_tool_fail_helpers():
    assert is_clarified(answer="请补充机型与复现路径", stop_reason=None, finish_reason=None)
    assert is_clarified(answer="随便", stop_reason="clarify", finish_reason=None)
    assert not is_clarified(answer="先看 traces.txt", stop_reason="final_answer", finish_reason="stop")
    assert is_tool_fail_error("TOOL_TIMEOUT")
    assert is_tool_fail_error("AGENT_NO_ANSWER")
    assert not is_tool_fail_error("UPSTREAM_TIMEOUT")
