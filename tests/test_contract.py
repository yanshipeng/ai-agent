"""Day 5.1 接口契约测试（必须 mock LLMClient）。

覆盖：
- /health：200 + 必备字段
- /ask 成功：200 + 必备字段齐全（citations 必须是 array）
- /ask 失败：空 query / 超长 query → 4xx + 统一错误结构 + request_id
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api import CODE_INVALID_ARGUMENT, QUERY_MAX_LENGTH
from app.main import create_app
from app.services.llm_client import LLMResult

# 成功响应必备字段（删掉任一字段，如 citations，本测试应失败）
ASK_SUCCESS_REQUIRED_FIELDS = {
    "request_id",
    "answer",
    "citations",
    "latency_ms",
    "model",
}


def assert_ask_success_contract(body: dict[str, Any]) -> None:
    """契约断言：必备字段存在，且 citations 必须是 array。"""
    missing = ASK_SUCCESS_REQUIRED_FIELDS - set(body.keys())
    assert not missing, f"ask success response missing fields: {sorted(missing)}"
    assert isinstance(body["request_id"], str) and body["request_id"]
    uuid.UUID(body["request_id"])
    assert isinstance(body["answer"], str)
    assert isinstance(body["citations"], list), "citations must be an array"
    assert isinstance(body["latency_ms"], int)
    assert isinstance(body["model"], str) and body["model"]
    assert body.get("meta") is None or isinstance(body["meta"], dict)


def assert_error_contract(body: dict[str, Any]) -> None:
    """统一错误体：request_id / code / message。"""
    assert set(body.keys()) == {"request_id", "code", "message"}
    assert isinstance(body["request_id"], str) and body["request_id"]
    uuid.UUID(body["request_id"])
    assert isinstance(body["code"], str) and body["code"]
    assert isinstance(body["message"], str) and body["message"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("APP_VERSION", "0.1.0ba")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_health_contract(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"status", "version"}
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0ba"


def test_ask_success_contract_with_mocked_llm(client: TestClient):
    mock_client = MagicMock()
    mock_client.chat.return_value = LLMResult(
        answer="北京是中国的首都。",
        model="deepseek-v4-flash",
        finish_reason="stop",
        latency_ms=321,
        usage={"prompt_tokens": 10, "completion_tokens": 8},
    )
    client.app.state.llm_client = mock_client

    resp = client.post(
        "/ask",
        json={
            "query": "中国的首都是哪里？",
            "session_id": "s-1",
            "client_tag": "web",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert_ask_success_contract(body)
    assert body["answer"] == "北京是中国的首都。"
    assert body["citations"] == []
    assert body["latency_ms"] == 321
    assert body["model"] == "deepseek-v4-flash"

    mock_client.chat.assert_called_once()
    args, kwargs = mock_client.chat.call_args
    assert args == ([{"role": "user", "content": "中国的首都是哪里？"}],)
    assert body["request_id"] == kwargs["request_id"]


def test_ask_success_contract_requires_citations_field():
    """契约保护：若响应缺少 citations，断言必须失败。"""
    body = {
        "request_id": str(uuid.uuid4()),
        "answer": "x",
        # "citations" 故意缺失
        "latency_ms": 1,
        "model": "deepseek-v4-flash",
    }
    with pytest.raises(AssertionError, match="citations"):
        assert_ask_success_contract(body)


def test_ask_success_contract_requires_citations_be_array():
    body = {
        "request_id": str(uuid.uuid4()),
        "answer": "x",
        "citations": "not-an-array",
        "latency_ms": 1,
        "model": "deepseek-v4-flash",
    }
    with pytest.raises(AssertionError, match="citations must be an array"):
        assert_ask_success_contract(body)


@pytest.mark.parametrize(
    ("payload", "message_substr"),
    [
        ({"query": ""}, "empty"),
        ({"query": "   "}, "empty"),
        ({"query": "a" * (QUERY_MAX_LENGTH + 1)}, str(QUERY_MAX_LENGTH)),
    ],
)
def test_ask_invalid_query_contract(client: TestClient, payload: dict, message_substr: str):
    resp = client.post("/ask", json=payload)
    assert 400 <= resp.status_code < 500
    body = resp.json()
    assert_error_contract(body)
    assert body["code"] == CODE_INVALID_ARGUMENT
    assert message_substr.lower() in body["message"].lower()
