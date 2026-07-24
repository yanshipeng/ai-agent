"""stats_requests.py 汇总指标单元测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_stats_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "stats_requests.py"
    spec = importlib.util.spec_from_file_location("stats_requests", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summarize_new_schema_with_tokens():
    mod = _load_stats_module()
    records = [
        {
            "ok": True,
            "latency_ms_total": 100,
            "retry_count": 0,
            "total_tokens": 10,
        },
        {
            "ok": True,
            "latency_ms_total": 200,
            "retry_count": 2,
            "total_tokens": 20,
            "error_code": "UPSTREAM_TIMEOUT",
        },
        {
            "ok": False,
            "latency_ms_total": 300,
            "retry_count": 0,
            "error_code": "UPSTREAM_UNAUTHORIZED",
        },
    ]
    summary = mod.summarize(records)
    assert summary["total"] == 3
    assert summary["ok"] == 2
    assert summary["fail"] == 1
    assert summary["ok_rate"] == round(2 / 3, 4)
    assert summary["p50_latency_ms"] == 200
    assert summary["p95_latency_ms"] is not None
    assert summary["max_latency_ms"] == 300
    assert summary["retry_rate"] == round(1 / 3, 4)
    assert summary["top_error_codes"][0]["error_code"] in {
        "UPSTREAM_TIMEOUT",
        "UPSTREAM_UNAUTHORIZED",
    }
    assert summary["avg_total_tokens"] == 15.0
    assert summary["max_total_tokens"] == 20
