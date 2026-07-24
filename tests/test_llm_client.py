"""LLMClient.chat 单元测试（mock OpenAI，不打真实网络）。

覆盖成功路径、空 messages、空 answer、ask→chat 委托、固定调用参数。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings
from app.services.llm_client import (
    DEFAULT_BASE_URL,
    DEFAULT_EXTRA_BODY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_STREAM,
    DEFAULT_TIMEOUT_SECONDS,
    LLMClient,
    LLMError,
    UPSTREAM_BAD_REQUEST,
    UPSTREAM_UNKNOWN,
)


def _make_settings(**kwargs) -> Settings:
    data = {"DEEPSEEK_API_KEY": "sk-test", **kwargs}
    return Settings(_env_file=None, **data)


def _fake_response(
    *,
    content: str = "hello",
    finish_reason: str = "stop",
    model: str = DEFAULT_MODEL,
    usage: dict | None = None,
):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage_obj = None
    if usage is not None:
        usage_obj = SimpleNamespace(model_dump=lambda: usage)
    return SimpleNamespace(choices=[choice], model=model, usage=usage_obj)


def test_chat_success_returns_expected_fields():
    client = LLMClient(_make_settings())
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _fake_response(
        content="  北京  ",
        finish_reason="stop",
        usage={"prompt_tokens": 3, "completion_tokens": 2},
    )

    with patch.object(client, "_get_client", return_value=mock_openai):
        result = client.chat([{"role": "user", "content": "首都？"}])

    assert result.answer == "北京"
    assert result.model == DEFAULT_MODEL
    assert result.finish_reason == "stop"
    assert isinstance(result.latency_ms, int) and result.latency_ms >= 0
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 2}

    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["max_tokens"] == DEFAULT_MAX_TOKENS
    assert kwargs["stream"] is DEFAULT_STREAM
    assert kwargs["extra_body"] == DEFAULT_EXTRA_BODY
    assert kwargs["messages"] == [{"role": "user", "content": "首都？"}]


def test_chat_builds_openai_client_with_settings_timeout():
    client = LLMClient(_make_settings(LLM_TIMEOUT_SECONDS="1"))

    with patch("app.services.llm_client.OpenAI") as mock_cls:
        mock_cls.return_value = MagicMock(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=MagicMock(return_value=_fake_response()))
            )
        )
        client.chat([{"role": "user", "content": "hi"}])

    mock_cls.assert_called_once_with(
        api_key="sk-test",
        base_url=DEFAULT_BASE_URL,
        timeout=1.0,
    )


def test_chat_rejects_empty_messages():
    client = LLMClient(_make_settings())
    with pytest.raises(LLMError) as exc_info:
        client.chat([])
    assert exc_info.value.code == UPSTREAM_BAD_REQUEST


def test_chat_rejects_empty_answer_without_fallback():
    client = LLMClient(_make_settings(), max_retries=0, enable_fallback=False)
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _fake_response(content="   ")

    with patch.object(client, "_get_client", return_value=mock_openai):
        with pytest.raises(LLMError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
    assert exc_info.value.code == UPSTREAM_UNKNOWN


def test_ask_delegates_to_chat():
    client = LLMClient(_make_settings())
    with patch.object(client, "chat", return_value=_fake_as_result()) as mock_chat:
        client.ask("你好", system_prompt="你是助手")
    mock_chat.assert_called_once_with(
        [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
        ],
        request_id=None,
    )


def _fake_as_result():
    from app.services.llm_client import LLMResult

    return LLMResult(
        answer="ok",
        model=DEFAULT_MODEL,
        finish_reason="stop",
        latency_ms=1,
        usage=None,
    )
