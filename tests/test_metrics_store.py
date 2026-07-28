"""requests.jsonl 落盘字段验收测试。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core.logging import query_sha256_8
from app.main import create_app
from app.services.llm_client import LLMResult
from app.services.metrics_store import build_ask_metric


def test_build_ask_metric_shape_and_tokens():
    row = build_ask_metric(
        request_id="r1",
        path="/ask",
        ok=True,
        status_code=200,
        latency_ms_total=100,
        latency_ms_llm=80,
        llm_model="deepseek-v4-flash",
        retry_count=0,
        finish_reason="stop",
        query="hello",
        usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    )
    assert row["request_id"] == "r1"
    assert row["path"] == "/ask"
    assert row["ok"] is True
    assert row["status_code"] == 200
    assert row["latency_ms_total"] == 100
    assert row["latency_ms_llm"] == 80
    assert row["llm_model"] == "deepseek-v4-flash"
    assert row["retry_count"] == 0
    assert row["finish_reason"] == "stop"
    assert row["query_len"] == 5
    assert row["query_sha256_8"] == query_sha256_8("hello")
    assert row["prompt_tokens"] == 3
    assert row["completion_tokens"] == 5
    assert row["total_tokens"] == 8
    assert "query" not in row
    assert "answer" not in row


def test_build_ask_metric_includes_rag_fields():
    row = build_ask_metric(
        request_id="r2",
        ok=True,
        status_code=200,
        mode="rag",
        top_k=5,
        retrieve_ms=33,
        context_chunks=5,
        citations_count=5,
    )
    assert row["mode"] == "rag"
    assert row["top_k"] == 5
    assert row["retrieve_ms"] == 33
    assert row["context_chunks"] == 5
    assert row["citations_count"] == 5


def test_ask_writes_one_jsonl_line(tmp_path, monkeypatch):
    metrics_path = tmp_path / "requests.jsonl"
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(metrics_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    mock_client = MagicMock()
    mock_client.chat.return_value = LLMResult(
        answer="ok",
        model="deepseek-v4-flash",
        finish_reason="stop",
        latency_ms=42,
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        retry_count=0,
    )

    with TestClient(app) as client:
        client.app.state.llm_client = mock_client
        resp = client.post("/ask", json={"query": "不写进jsonl的原文"})
        assert resp.status_code == 200
        request_id = resp.json()["request_id"]

    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["request_id"] == request_id
    assert row["path"] == "/ask"
    assert row["ok"] is True
    assert row["status_code"] == 200
    assert row["latency_ms_llm"] == 42
    assert row["llm_model"] == "deepseek-v4-flash"
    assert row["finish_reason"] == "stop"
    assert row["retry_count"] == 0
    assert row["query_len"] == len("不写进jsonl的原文")
    assert row["query_sha256_8"] == query_sha256_8("不写进jsonl的原文")
    assert row["prompt_tokens"] == 1
    assert row["completion_tokens"] == 2
    assert row["total_tokens"] == 3
    assert row["mode"] == "llm"
    assert row["context_chunks"] == 0
    assert row["citations_count"] == 0
    assert "session_id" not in row
    assert "history_messages" not in row
    assert "query" not in row
    assert "answer" not in row
    assert "不写进jsonl的原文" not in lines[0]


def test_ask_session_writes_history_metrics(tmp_path, monkeypatch):
    metrics_path = tmp_path / "requests.jsonl"
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(metrics_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    from app.core.config import get_settings
    from app.services.session_store import clear_all_sessions

    get_settings.cache_clear()
    clear_all_sessions()
    app = create_app()
    mock_client = MagicMock()
    mock_client.chat.side_effect = [
        LLMResult(answer="收到约束", model="fake", finish_reason="stop", latency_ms=10),
        LLMResult(answer="只讨论推送", model="fake", finish_reason="stop", latency_ms=12),
    ]

    with TestClient(app) as client:
        client.app.state.llm_client = mock_client
        r1 = client.post(
            "/ask",
            json={
                "query": "只讨论 Android 推送",
                "session_id": "s-metrics-1",
                "mode": "llm",
            },
        )
        r2 = client.post(
            "/ask",
            json={
                "query": "我叫什么？",
                "session_id": "s-metrics-1",
                "mode": "llm",
            },
        )
        assert r1.status_code == 200
        assert r2.status_code == 200

    lines = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(lines) == 2
    assert lines[0]["session_id"] == "s-metrics-1"
    assert lines[0]["history_messages"] == 0
    assert lines[0]["history_chars"] == 0
    assert lines[1]["session_id"] == "s-metrics-1"
    assert lines[1]["history_messages"] == 2
    assert lines[1]["history_chars"] > 0
    clear_all_sessions()
