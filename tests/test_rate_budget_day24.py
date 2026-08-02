"""Day24：限流 + 请求预算降级。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.llm_client import LLMResult
from app.services.rate_limit import reset_rate_limit_state
from app.services.request_budget import plan_request_budget


@pytest.fixture
def limited_client(tmp_path, monkeypatch):
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("API_AUTH_ENABLED", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_RPM", "3")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    monkeypatch.setenv("REQUEST_TOKEN_BUDGET", "512")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    from app.core.config import get_settings

    get_settings.cache_clear()
    reset_rate_limit_state()
    app = create_app()
    with TestClient(app) as client:
        yield client
    reset_rate_limit_state()
    get_settings.cache_clear()


def test_plan_budget_caps_tokens() -> None:
    plan = plan_request_budget("什么是 ANR？", top_k=5, mode="rag")
    assert plan.max_tokens <= 2048
    assert "cap_max_tokens" in plan.actions


def test_plan_budget_vague_clarify() -> None:
    plan = plan_request_budget("App 又卡了", top_k=5, mode="rag")
    assert plan.clarify_short is True
    assert plan.force_flash is True


def test_rate_limit_returns_429(limited_client: TestClient) -> None:
    mock = MagicMock()
    mock.chat.return_value = LLMResult(
        answer="ok",
        model="deepseek-v4-flash",
        finish_reason="stop",
        latency_ms=1,
        usage={"total_tokens": 3},
    )
    limited_client.app.state.llm_client = mock
    headers = {"X-Tenant-Id": "t-burst"}
    codes = []
    for i in range(5):
        resp = limited_client.post(
            "/v1/ask",
            headers=headers,
            json={"query": f"hello {i}"},
        )
        codes.append(resp.status_code)
        if resp.status_code == 429:
            assert resp.json()["code"] == "RATE_LIMITED"
            assert "Retry-After" in resp.headers
    assert 429 in codes
    assert codes.count(200) == 3


def test_budget_clarify_short_circuit(limited_client: TestClient) -> None:
    reset_rate_limit_state()
    mock = MagicMock()
    limited_client.app.state.llm_client = mock
    resp = limited_client.post(
        "/v1/ask",
        headers={"X-Tenant-Id": "t-clarify"},
        json={"query": "App 又卡了"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["finish_type"] == "budget_clarify"
    assert "信息不足" in body["answer"]
    mock.chat.assert_not_called()
