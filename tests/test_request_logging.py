"""请求事件日志字段与脱敏验收测试。

抽查：不出现 query 原文 / API key；含 request_start / request_success 等字段。
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core.logging import query_sha256_8, sanitize_log_text
from app.main import create_app
from app.services.llm_client import LLMResult


def test_query_sha256_8_stable_and_short():
    assert query_sha256_8("hello") == query_sha256_8("hello")
    assert len(query_sha256_8("hello")) == 8
    assert query_sha256_8("hello") != query_sha256_8("world")


def test_sanitize_redacts_api_key():
    text = "Authentication Fails, Your api key: sk-abcdef1234567890 is invalid"
    cleaned = sanitize_log_text(text) or ""
    assert "sk-abcdef" not in cleaned
    assert "[REDACTED]" in cleaned


def _parse_json_logs(stdout: str) -> list[dict]:
    rows: list[dict] = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return rows


def test_ask_logs_expected_events(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-secret-key-should-not-appear")
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    mock_client = MagicMock()
    mock_client.chat.return_value = LLMResult(
        answer="ok",
        model="deepseek-v4-flash",
        finish_reason="stop",
        latency_ms=12,
        retry_count=1,
    )

    with TestClient(app) as client:
        client.app.state.llm_client = mock_client
        resp = client.post("/ask", json={"query": "秘密问题不要进日志"})
        assert resp.status_code == 200
        body = resp.json()

    logs = _parse_json_logs(capsys.readouterr().out)
    ask_logs = [r for r in logs if r.get("request_id") == body["request_id"]]
    events = [r.get("event") for r in ask_logs]

    assert "request_start" in events
    assert "request_success" in events
    blob = json.dumps(ask_logs, ensure_ascii=False)
    assert "秘密问题不要进日志" not in blob
    assert "sk-test-secret" not in blob

    start = next(r for r in ask_logs if r.get("event") == "request_start")
    assert start["path"] == "/ask"
    assert start["method"] == "POST"
    assert start["query_len"] == len("秘密问题不要进日志")
    assert start["query_sha256_8"] == query_sha256_8("秘密问题不要进日志")
    assert "query" not in start
    assert "hint" in start and "mode=llm" in start["hint"]

    success = next(r for r in ask_logs if r.get("event") == "request_success")
    assert success["ok"] is True
    assert success["status_code"] == 200
    assert success["llm_provider"] == "deepseek"
    assert success["llm_model"] == "deepseek-v4-flash"
    assert success["retry_count"] == 1
    assert "hint" in success


def test_ask_rag_logs_retrieve_events(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_path / "missing_index"))
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        client.app.state.llm_client = MagicMock()
        resp = client.post("/ask?mode=rag", json={"query": "ANR"})
        assert resp.status_code == 503
        rid = resp.json()["request_id"]

    logs = _parse_json_logs(capsys.readouterr().out)
    ask_logs = [r for r in logs if r.get("request_id") == rid]
    events = [r.get("event") for r in ask_logs]
    assert "request_start" in events
    assert "retrieve_start" in events
    assert "request_error" in events
    start = next(r for r in ask_logs if r.get("event") == "request_start")
    assert "mode=rag" in start.get("hint", "")
