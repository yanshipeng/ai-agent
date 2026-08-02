"""Day22：API Key 鉴权 + tenant 审计字段。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.auth import parse_api_keys
from app.main import create_app
from app.services.llm_client import LLMResult
from scripts.stats_requests import summarize


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEYS", "admin-k:admin,reader-k:reader")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("APP_VERSION", "0.1.0ba")
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        yield client, tmp_path
    get_settings.cache_clear()


def test_parse_api_keys() -> None:
    assert parse_api_keys("a:admin,b") == {"a": "admin", "b": "reader"}


def test_health_public_without_token(auth_client) -> None:
    client, _ = auth_client
    assert client.get("/health").status_code == 200
    assert client.get("/v1/health").status_code == 200


def test_ask_without_token_rejected(auth_client) -> None:
    client, _ = auth_client
    resp = client.post("/v1/ask", json={"query": "你好"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["code"] == "UNAUTHORIZED"
    assert set(body.keys()) == {"request_id", "code", "message"}


def test_ask_with_token_and_tenant_audit(auth_client) -> None:
    client, tmp_path = auth_client
    mock = MagicMock()
    mock.chat.return_value = LLMResult(
        answer="ok",
        model="deepseek-v4-flash",
        finish_reason="stop",
        latency_ms=3,
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    )
    client.app.state.llm_client = mock
    resp = client.post(
        "/v1/ask",
        headers={
            "X-Api-Key": "reader-k",
            "X-Tenant-Id": "tenant-a",
            "X-User-Id": "u1",
        },
        json={"query": "什么是 ANR？"},
    )
    assert resp.status_code == 200
    meta = resp.json()["meta"]
    assert meta["tenant_id"] == "tenant-a"
    assert meta["user_id"] == "u1"
    assert meta["role"] == "reader"

    lines = (tmp_path / "requests.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines
    row = json.loads(lines[-1])
    assert row["tenant_id"] == "tenant-a"
    assert row["user_id"] == "u1"
    assert "query" not in row
    assert "什么是 ANR" not in json.dumps(row, ensure_ascii=False)


def test_reader_forbidden_ingest(auth_client) -> None:
    client, _ = auth_client
    resp = client.post(
        "/v1/ingest",
        headers={"X-Api-Key": "reader-k"},
        json={"action": "ingest", "incremental": True},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "FORBIDDEN"


def test_stats_by_tenant() -> None:
    rows = [
        {
            "ok": True,
            "status_code": 200,
            "tenant_id": "t1",
            "latency_ms_total": 100,
            "total_tokens": 10,
        },
        {
            "ok": False,
            "status_code": 500,
            "tenant_id": "t1",
            "latency_ms_total": 200,
            "error_code": "UPSTREAM_5XX",
            "total_tokens": 5,
        },
        {
            "ok": True,
            "status_code": 200,
            "tenant_id": "t2",
            "latency_ms_total": 50,
            "total_tokens": 20,
        },
    ]
    summary = summarize(rows)
    assert "by_tenant" in summary
    assert summary["by_tenant"]["t1"]["total"] == 2
    assert summary["by_tenant"]["t1"]["fail"] == 1
    assert summary["by_tenant"]["t2"]["sum_total_tokens"] == 20
