"""Day10 run_rag_eval 报告/分布逻辑单测（不打 LLM）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_rag_eval import (
    build_report,
    compare_reports,
    load_samples,
    validate_sample_distribution,
)


def test_eval_samples_rag_distribution():
    samples = load_samples(Path("eval/eval_samples_rag.jsonl"))
    counts = validate_sample_distribution(samples)
    assert counts["normal"] >= 30
    assert counts["insufficient"] >= 10
    assert counts["sensitive"] >= 10
    assert len(samples) >= 50


def test_build_report_metrics_shape():
    results = [
        {
            "tag": "normal",
            "ok": True,
            "citations_nonempty": True,
            "latency_ms_total": 100,
            "retrieve_ms": 10,
            "insufficient_handled": None,
            "sensitive_handled": None,
            "error_code": None,
        },
        {
            "tag": "insufficient",
            "ok": True,
            "citations_nonempty": True,
            "latency_ms_total": 200,
            "retrieve_ms": 20,
            "insufficient_handled": True,
            "sensitive_handled": None,
            "error_code": None,
        },
        {
            "tag": "sensitive",
            "ok": True,
            "citations_nonempty": False,
            "latency_ms_total": 150,
            "retrieve_ms": 15,
            "insufficient_handled": None,
            "sensitive_handled": True,
            "error_code": None,
        },
        {
            "tag": "normal",
            "ok": False,
            "citations_nonempty": False,
            "latency_ms_total": 50,
            "retrieve_ms": None,
            "insufficient_handled": None,
            "sensitive_handled": None,
            "error_code": "UPSTREAM_TIMEOUT",
        },
    ]
    report = build_report(results, label="t", top_k=5)
    assert report["citation_coverage"] == 0.5
    assert report["insufficient_handling_rate"] == 1.0
    assert report["sensitive_handling_rate"] == 1.0
    assert report["latency_ms_total"]["p50"] is not None
    assert report["retrieve_ms"]["p95"] is not None
    assert report["top_errors"][0]["error_code"] == "UPSTREAM_TIMEOUT"


def test_compare_reports_delta():
    a = build_report(
        [
            {
                "tag": "normal",
                "ok": True,
                "citations_nonempty": True,
                "latency_ms_total": 100,
                "retrieve_ms": 10,
                "insufficient_handled": None,
                "sensitive_handled": None,
            }
        ],
        label="A",
        top_k=3,
        ab_var="top_k",
        ab_value=3,
    )
    b = build_report(
        [
            {
                "tag": "normal",
                "ok": True,
                "citations_nonempty": True,
                "latency_ms_total": 80,
                "retrieve_ms": 8,
                "insufficient_handled": None,
                "sensitive_handled": None,
            }
        ],
        label="B",
        top_k=5,
        ab_var="top_k",
        ab_value=5,
    )
    cmp = compare_reports(a, b)
    assert cmp["deltas"]["latency_total_p50"]["delta"] == -20.0
