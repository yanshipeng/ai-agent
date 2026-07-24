"""DeepSeek LLM Client：基于 OpenAI SDK 的封装（与 HTTP 路由解耦）。

职责：
- chat(messages) / ask(question)：发起上游调用
- 将 SDK / HTTP 错误映射为内部错误码（UPSTREAM_*）
- 对可重试错误做有限次重试；耗尽后可返回兜底文案

路由层只应依赖本模块的 LLMClient / LLMError / LLMResult，不要直接写 OpenAI SDK。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from app.core.config import Settings, get_settings
from app.core.logging import (
    EVENT_LLM_CALL_END,
    EVENT_LLM_CALL_START,
    LLM_PROVIDER,
    get_logger,
    log_event,
)

logger = get_logger(__name__)

# ---------- 调用默认值（与环境变量默认对齐；timeout 实际读 Settings） ----------
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_STREAM = False
DEFAULT_EXTRA_BODY: dict[str, Any] = {"thinking": {"type": "disabled"}}

# ---------- 内部错误码（最少集） ----------
UPSTREAM_UNAUTHORIZED = "UPSTREAM_UNAUTHORIZED"  # 401
UPSTREAM_RATE_LIMITED = "UPSTREAM_RATE_LIMITED"  # 429
UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"  # 超时 → HTTP 504
UPSTREAM_BAD_REQUEST = "UPSTREAM_BAD_REQUEST"  # 400
UPSTREAM_5XX = "UPSTREAM_5XX"  # 上游 >=500 → HTTP 502
UPSTREAM_UNKNOWN = "UPSTREAM_UNKNOWN"  # 其它 → HTTP 502

# 可重试：限流 / 超时 / 5xx / 未知（含连接失败等）
RETRYABLE_ERROR_CODES = frozenset(
    {
        UPSTREAM_RATE_LIMITED,
        UPSTREAM_TIMEOUT,
        UPSTREAM_5XX,
        UPSTREAM_UNKNOWN,
    }
)

# 重试：最多额外 2 次（共 3 次尝试）；兜底文案用于可重试错误耗尽后
LLM_MAX_RETRIES = 2
LLM_RETRY_BACKOFF_SECONDS = 0.4
FALLBACK_ANSWER = "抱歉，上游服务暂时不可用，请稍后重试。"
FALLBACK_MODEL = "fallback"

# 内部错误码 → 对外 HTTP status
_ERROR_HTTP_STATUS: dict[str, int] = {
    UPSTREAM_UNAUTHORIZED: 401,
    UPSTREAM_RATE_LIMITED: 429,
    UPSTREAM_TIMEOUT: 504,
    UPSTREAM_BAD_REQUEST: 400,
    UPSTREAM_5XX: 502,
    UPSTREAM_UNKNOWN: 502,
}

# 兼容旧常量名（指向新错误码）
ERR_MISSING_API_KEY = UPSTREAM_UNKNOWN
ERR_AUTH = UPSTREAM_UNAUTHORIZED
ERR_RATE_LIMIT = UPSTREAM_RATE_LIMITED
ERR_TIMEOUT = UPSTREAM_TIMEOUT
ERR_CONNECTION = UPSTREAM_UNKNOWN
ERR_BAD_REQUEST = UPSTREAM_BAD_REQUEST
ERR_UPSTREAM = UPSTREAM_UNKNOWN
ERR_EMPTY_RESPONSE = UPSTREAM_UNKNOWN
ERR_EMPTY_MESSAGES = UPSTREAM_BAD_REQUEST


class LLMError(Exception):
    """LLM 调用统一异常，携带内部错误码与可选上游 status_code。"""

    def __init__(self, code: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class LLMResult:
    """chat() / ask() 的统一返回结构。"""

    answer: str
    model: str
    finish_reason: str | None
    latency_ms: int  # 本次 LLM 段耗时（含重试等待）
    usage: dict[str, Any] | None = None
    fallback: bool = False  # True 表示已走兜底文案
    error_code: str | None = None  # 兜底或失败时的内部错误码
    retry_count: int = 0  # 实际重试次数（不含首次）


def map_llm_error_to_http(error: LLMError) -> tuple[int, str]:
    """将 LLMError 映射为 (HTTP status, error code)。"""
    return _ERROR_HTTP_STATUS.get(error.code, 502), error.code


def map_status_to_error_code(status: int | None) -> str:
    """按上游 HTTP status 映射内部错误码。"""
    if status == 401:
        return UPSTREAM_UNAUTHORIZED
    if status == 429:
        return UPSTREAM_RATE_LIMITED
    if status == 400:
        return UPSTREAM_BAD_REQUEST
    if status is not None and status >= 500:
        return UPSTREAM_5XX
    return UPSTREAM_UNKNOWN


def is_retryable(error: LLMError) -> bool:
    """判断该错误是否允许重试。"""
    return error.code in RETRYABLE_ERROR_CODES


# 对外错误文案（短且稳定，禁止把上游响应全文透出）
_PUBLIC_ERROR_MESSAGES: dict[str, str] = {
    UPSTREAM_UNAUTHORIZED: "upstream authentication failed",
    UPSTREAM_RATE_LIMITED: "upstream rate limited",
    UPSTREAM_TIMEOUT: "upstream request timed out",
    UPSTREAM_BAD_REQUEST: "upstream rejected the request",
    UPSTREAM_5XX: "upstream server error",
    UPSTREAM_UNKNOWN: "upstream request failed",
}


def public_error_message(code: str) -> str:
    """返回可对外暴露的短错误文案（不含上游响应全文）。"""
    return _PUBLIC_ERROR_MESSAGES.get(code, "request failed")


class LLMClient:
    """DeepSeek Chat Completions 客户端。

    路由 / 脚本应依赖 chat()/ask()，不要把 SDK 调用写死在路由里。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        max_retries: int = LLM_MAX_RETRIES,
        retry_backoff_seconds: float = LLM_RETRY_BACKOFF_SECONDS,
        enable_fallback: bool = True,
    ) -> None:
        self._settings = settings or get_settings()
        self._client: OpenAI | None = None
        # timeout 以 Settings.llm_timeout_seconds 为准（支持 .env 调小做超时演练）
        self._timeout_seconds = float(self._settings.llm_timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._enable_fallback = enable_fallback

    def _resolve_api_key(self) -> str:
        """读取并校验 API Key；缺失时抛 UPSTREAM_UNAUTHORIZED。"""
        api_key = (self._settings.deepseek_api_key or "").strip()
        if not api_key:
            raise LLMError(
                UPSTREAM_UNAUTHORIZED,
                "DEEPSEEK_API_KEY is not configured",
                status_code=401,
            )
        return api_key

    def _get_client(self) -> OpenAI:
        """懒加载 OpenAI 兼容客户端（base_url / timeout 在创建时固定）。"""
        if self._client is None:
            timeout = float(self._settings.llm_timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
            self._timeout_seconds = timeout
            self._client = OpenAI(
                api_key=self._resolve_api_key(),
                base_url=DEFAULT_BASE_URL,
                timeout=timeout,
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str | None = None,
    ) -> LLMResult:
        """调用 DeepSeek chat.completions（含重试）；可重试错误耗尽后可走兜底。

        Args:
            messages: OpenAI 风格 messages 列表
            request_id: 可选，写入 llm_call_* 日志便于链路追踪
        """
        if not messages:
            raise LLMError(UPSTREAM_BAD_REQUEST, "messages must not be empty", status_code=400)

        started = time.perf_counter()
        last_error: LLMError | None = None
        attempts = self._max_retries + 1
        retry_count = 0

        log_event(
            logger,
            EVENT_LLM_CALL_START,
            request_id=request_id,
            llm_provider=LLM_PROVIDER,
            llm_model=DEFAULT_MODEL,
        )

        for attempt in range(1, attempts + 1):
            try:
                result = self._chat_once(messages, started=started)
                retry_count = attempt - 1
                log_event(
                    logger,
                    EVENT_LLM_CALL_END,
                    request_id=request_id,
                    ok=True,
                    llm_provider=LLM_PROVIDER,
                    llm_model=result.model,
                    finish_reason=result.finish_reason,
                    retry_count=retry_count,
                    latency_ms=result.latency_ms,
                )
                return LLMResult(
                    answer=result.answer,
                    model=result.model,
                    finish_reason=result.finish_reason,
                    latency_ms=result.latency_ms,
                    usage=result.usage,
                    fallback=False,
                    error_code=None,
                    retry_count=retry_count,
                )
            except LLMError as exc:
                last_error = exc
                will_retry = is_retryable(exc) and attempt < attempts
                if will_retry:
                    retry_count = attempt
                    if self._retry_backoff_seconds > 0:
                        time.sleep(self._retry_backoff_seconds * attempt)
                    continue
                break

        assert last_error is not None
        retry_count = max(0, attempts - 1)
        latency_ms = int((time.perf_counter() - started) * 1000)

        # 可重试错误耗尽 → 兜底（HTTP 层仍可返回 200）
        if self._enable_fallback and is_retryable(last_error):
            log_event(
                logger,
                EVENT_LLM_CALL_END,
                request_id=request_id,
                ok=False,
                llm_provider=LLM_PROVIDER,
                llm_model=FALLBACK_MODEL,
                finish_reason="fallback",
                retry_count=retry_count,
                latency_ms=latency_ms,
                error_code=last_error.code,
            )
            return LLMResult(
                answer=FALLBACK_ANSWER,
                model=FALLBACK_MODEL,
                finish_reason="fallback",
                latency_ms=latency_ms,
                usage=None,
                fallback=True,
                error_code=last_error.code,
                retry_count=retry_count,
            )

        # 不可重试 / 未开启兜底 → 向上抛出，由路由映射为错误响应
        log_event(
            logger,
            EVENT_LLM_CALL_END,
            request_id=request_id,
            ok=False,
            llm_provider=LLM_PROVIDER,
            llm_model=DEFAULT_MODEL,
            retry_count=retry_count,
            latency_ms=latency_ms,
            error_code=last_error.code,
        )
        raise last_error

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        *,
        started: float,
    ) -> LLMResult:
        """单次上游调用（不含重试循环）；将 SDK 异常转为 LLMError。"""
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                max_tokens=DEFAULT_MAX_TOKENS,
                stream=DEFAULT_STREAM,
                extra_body=dict(DEFAULT_EXTRA_BODY),
            )
        except AuthenticationError as exc:
            raise LLMError(UPSTREAM_UNAUTHORIZED, str(exc), status_code=401) from exc
        except RateLimitError as exc:
            raise LLMError(UPSTREAM_RATE_LIMITED, str(exc), status_code=429) from exc
        except APITimeoutError as exc:
            raise LLMError(UPSTREAM_TIMEOUT, str(exc), status_code=504) from exc
        except APIConnectionError as exc:
            raise LLMError(UPSTREAM_UNKNOWN, str(exc), status_code=502) from exc
        except APIStatusError as exc:
            raise _map_api_status_error(exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                UPSTREAM_UNKNOWN,
                f"unexpected llm error: {exc}",
                status_code=502,
            ) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        return _parse_chat_response(response, latency_ms=latency_ms)

    def ask(
        self,
        question: str,
        *,
        system_prompt: str | None = None,
        request_id: str | None = None,
    ) -> LLMResult:
        """便捷单轮问答：组装 messages 后调用 chat()。"""
        if not question or not question.strip():
            raise LLMError(UPSTREAM_BAD_REQUEST, "question must not be empty", status_code=400)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question.strip()})
        return self.chat(messages, request_id=request_id)


# 兼容旧命名
DeepSeekClient = LLMClient


def _map_api_status_error(exc: APIStatusError) -> LLMError:
    """将 OpenAI APIStatusError 映射为带内部错误码的 LLMError。"""
    status = getattr(exc, "status_code", None)
    code = map_status_to_error_code(status)
    http_status = _ERROR_HTTP_STATUS.get(code, 502)
    return LLMError(code, str(exc), status_code=status if status is not None else http_status)


def _parse_chat_response(response: Any, *, latency_ms: int) -> LLMResult:
    """解析 chat.completions 响应；空内容视为 UPSTREAM_UNKNOWN。"""
    answer = ""
    finish_reason: str | None = None
    if response.choices:
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        answer = (content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)

    if not answer:
        raise LLMError(UPSTREAM_UNKNOWN, "llm returned empty content", status_code=502)

    usage: dict[str, Any] | None = None
    if getattr(response, "usage", None) is not None:
        usage_obj = response.usage
        usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else dict(usage_obj)

    model = getattr(response, "model", None) or DEFAULT_MODEL
    return LLMResult(
        answer=answer,
        model=model,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        usage=usage,
    )
