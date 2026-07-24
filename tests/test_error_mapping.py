"""Day 5.2 错误映射测试（mock 上游异常）。

至少覆盖：
- 401 → UPSTREAM_UNAUTHORIZED → HTTP 401
- 429 → UPSTREAM_RATE_LIMITED → HTTP 429
- timeout → UPSTREAM_TIMEOUT → HTTP 504
- 5xx → UPSTREAM_5XX → HTTP 502

并验证错误体结构一致，且 message 不含上游响应全文。
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api import CODE_INVALID_ARGUMENT, QUERY_MAX_LENGTH
from app.main import create_app
from app.services.llm_client import (
    FALLBACK_ANSWER,
    FALLBACK_MODEL,
    LLMClient,
    LLMError,
    UPSTREAM_5XX,
    UPSTREAM_BAD_REQUEST,
    UPSTREAM_RATE_LIMITED,
    UPSTREAM_TIMEOUT,
    UPSTREAM_UNAUTHORIZED,
    UPSTREAM_UNKNOWN,
    map_llm_error_to_http,
    map_status_to_error_code,
    public_error_message,
)


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, UPSTREAM_UNAUTHORIZED),
        (429, UPSTREAM_RATE_LIMITED),
        (400, UPSTREAM_BAD_REQUEST),
        (500, UPSTREAM_5XX),
        (502, UPSTREAM_5XX),
        (503, UPSTREAM_5XX),
        (418, UPSTREAM_UNKNOWN),
        (None, UPSTREAM_UNKNOWN),
    ],
)
def test_map_status_to_error_code(status: int | None, code: str):
    assert map_status_to_error_code(status) == code


@pytest.mark.parametrize(
    ("code", "expected_http"),
    [
        (UPSTREAM_UNAUTHORIZED, 401),
        (UPSTREAM_RATE_LIMITED, 429),
        (UPSTREAM_TIMEOUT, 504),
        (UPSTREAM_BAD_REQUEST, 400),
        (UPSTREAM_5XX, 502),
        (UPSTREAM_UNKNOWN, 502),
    ],
)
def test_map_llm_error_to_http(code: str, expected_http: int):
    http_status, mapped_code = map_llm_error_to_http(LLMError(code, "x"))
    assert http_status == expected_http
    assert mapped_code == code


def test_unknown_error_defaults_to_502():
    http_status, mapped_code = map_llm_error_to_http(LLMError("weird", "x"))
    assert http_status == 502
    assert mapped_code == "weird"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """注入 enable_fallback=False 的 client，便于直接断言上游错误映射。"""
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as test_client:
        # 关闭兜底，使可重试错误也以错误体返回（专门测 mapping）
        settings = get_settings()
        test_client.app.state.llm_client = LLMClient(
            settings,
            max_retries=0,
            enable_fallback=False,
        )
        yield test_client
    get_settings.cache_clear()


def _assert_upstream_error_body(body: dict, *, code: str) -> None:
    assert set(body.keys()) == {"request_id", "code", "message"}
    assert body["code"] == code
    assert body["message"] == public_error_message(code)
    uuid.UUID(body["request_id"])
    # 不包含上游响应全文特征
    assert "{" not in body["message"]
    assert "Error code:" not in body["message"]
    assert "sk-" not in body["message"].lower()


@pytest.mark.parametrize(
    ("code", "expected_http", "upstream_blob"),
    [
        (
            UPSTREAM_UNAUTHORIZED,
            401,
            "Error code: 401 - {'error': {'message': 'Authentication Fails, Your api key: sk-xxx'}}",
        ),
        (
            UPSTREAM_RATE_LIMITED,
            429,
            "Error code: 429 - {'error': {'message': 'Rate limit reached', 'type': 'rate_limit'}}",
        ),
        (
            UPSTREAM_TIMEOUT,
            504,
            "Request timed out. Full traceback and upstream dump: {...}",
        ),
        (
            UPSTREAM_5XX,
            502,
            "Error code: 503 - {'error': {'message': 'internal server error', 'body': 'LONG...'}}",
        ),
    ],
)
def test_ask_maps_upstream_errors(
    client: TestClient,
    code: str,
    expected_http: int,
    upstream_blob: str,
):
    """mock 上游异常：HTTP 策略一致 + 错误体不含上游全文。"""
    mock_client = MagicMock()
    mock_client.chat.side_effect = LLMError(code, upstream_blob)
    client.app.state.llm_client = mock_client

    resp = client.post("/ask", json={"query": "ping"})
    assert resp.status_code == expected_http
    body = resp.json()
    _assert_upstream_error_body(body, code=code)
    # 上游原文不得出现在响应中
    assert upstream_blob not in resp.text
    assert "Authentication Fails" not in resp.text
    assert "Rate limit reached" not in resp.text


@pytest.mark.parametrize(
    ("payload", "message_substr"),
    [
        ({}, "query"),
        ({"query": ""}, "empty"),
        ({"query": "   "}, "empty"),
        ({"query": "x" * (QUERY_MAX_LENGTH + 1)}, str(QUERY_MAX_LENGTH)),
    ],
)
def test_ask_invalid_argument_error_shape(
    client: TestClient,
    payload: dict,
    message_substr: str,
):
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert set(body.keys()) == {"request_id", "code", "message"}
    assert body["code"] == CODE_INVALID_ARGUMENT
    assert message_substr.lower() in body["message"].lower()


def test_chat_retries_then_fallback(monkeypatch):
    settings = MagicMock()
    settings.deepseek_api_key = "sk-test"
    llm = LLMClient(settings, max_retries=2, retry_backoff_seconds=0, enable_fallback=True)

    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise LLMError(UPSTREAM_TIMEOUT, "timeout", status_code=504)

    monkeypatch.setattr(llm, "_chat_once", boom)
    result = llm.chat([{"role": "user", "content": "hi"}])

    assert calls["n"] == 3
    assert result.fallback is True
    assert result.answer == FALLBACK_ANSWER
    assert result.model == FALLBACK_MODEL
    assert result.error_code == UPSTREAM_TIMEOUT
    assert result.retry_count == 2


def test_chat_no_retry_on_unauthorized(monkeypatch):
    settings = MagicMock()
    settings.deepseek_api_key = "sk-test"
    llm = LLMClient(settings, max_retries=2, retry_backoff_seconds=0, enable_fallback=True)

    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise LLMError(UPSTREAM_UNAUTHORIZED, "bad key", status_code=401)

    monkeypatch.setattr(llm, "_chat_once", boom)
    with pytest.raises(LLMError) as exc_info:
        llm.chat([{"role": "user", "content": "hi"}])

    assert calls["n"] == 1
    assert exc_info.value.code == UPSTREAM_UNAUTHORIZED


def test_chat_maps_5xx_status_error():
    settings = MagicMock()
    settings.deepseek_api_key = "sk-test"
    llm = LLMClient(settings, max_retries=0, enable_fallback=False)

    from openai import APIStatusError

    mock_openai = MagicMock()
    err = APIStatusError(
        message="server error",
        response=MagicMock(status_code=503, headers={}),
        body=None,
    )
    err.status_code = 503
    mock_openai.chat.completions.create.side_effect = err

    with patch.object(llm, "_get_client", return_value=mock_openai):
        with pytest.raises(LLMError) as exc_info:
            llm.chat([{"role": "user", "content": "hi"}])
    assert exc_info.value.code == UPSTREAM_5XX
