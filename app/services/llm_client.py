"""DeepSeek LLM Client：基于 OpenAI SDK 的封装（与 HTTP 路由解耦）。

==========================================================================
为什么单独拆这一层
==========================================================================
1) 路由只关心「给 messages，拿 answer / 错误码」，不要散落 OpenAI SDK。
2) 重试 / 错误映射 / 兜底集中在一处，契约测试可 mock LLMClient。
3) smoke_llm_client.py 可不启 FastAPI，直接验证 Key 与网络。

错误策略（简要）：
  - 401 / 400：不重试（配置或请求本身有问题）
  - 429 / 超时 / 5xx / 未知：有限重试；仍失败且开启兜底 → HTTP 200 + fallback 文案
  - 这样「可恢复的抖动」不把整站打成 5xx，同时日志仍能看到 error_code
==========================================================================
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
# 默认输出上限；实际请求优先用 Settings.llm_max_tokens（Day18 max_output_tokens）
DEFAULT_MAX_TOKENS = 2048
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
class ToolCall:
    """模型发起的单次 tool call（OpenAI / DeepSeek function calling 兼容）。

    id         —— 本轮唯一；Observe 写 role=tool 时必须回传同一 tool_call_id
    name       —— 工具名，如 kb_search
    arguments  —— JSON **字符串**（保持原样；由 tools.execute_tool 解析）
    """

    id: str
    name: str
    arguments: str  # JSON 字符串（原样回传 / 交给 Runner 解析）


@dataclass(frozen=True)
class LLMResult:
    """chat() 的统一返回：最终文本回答（不含 tool_calls 中间态）。"""

    answer: str
    model: str
    finish_reason: str | None
    latency_ms: int  # 本次 LLM 段耗时（含重试等待）
    usage: dict[str, Any] | None = None
    fallback: bool = False  # True 表示已走兜底文案
    error_code: str | None = None  # 兜底或失败时的内部错误码
    retry_count: int = 0  # 实际重试次数（不含首次）


@dataclass(frozen=True)
class LLMTurnResult:
    """单轮 chat.completions 结果：可能是终答，也可能是 tool_calls。

    Agent Runner 只认这一种结构：
      has_tool_calls → 进入 Act
      否则用 content 作为最终回答 → Final
    """

    content: str | None
    tool_calls: list[ToolCall]
    model: str
    finish_reason: str | None
    latency_ms: int
    usage: dict[str, Any] | None = None
    retry_count: int = 0
    fallback: bool = False
    error_code: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


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
        turn = self.chat_turn(messages, request_id=request_id, allow_fallback=True)
        if turn.has_tool_calls:
            raise LLMError(UPSTREAM_UNKNOWN, "unexpected tool_calls without tools", status_code=502)
        return LLMResult(
            answer=(turn.content or "").strip(),
            model=turn.model,
            finish_reason=turn.finish_reason,
            latency_ms=turn.latency_ms,
            usage=turn.usage,
            fallback=turn.fallback,
            error_code=turn.error_code,
            retry_count=turn.retry_count,
        )

    def chat_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        mode: str | None = None,
        agent_step: int | None = None,
        allow_fallback: bool = False,
    ) -> LLMTurnResult:
        """单轮 chat.completions（可选 tools）—— Agent 的「Plan」相位专用。

        与 chat() 的关系：
          chat() 只关心最终文本，内部也走 chat_turn，再把 content 提成 answer。
          Agent 必须用 chat_turn，才能拿到真实 tool_calls。

        行为：
          - 传入 tools=openai_tools_schema() 时，模型可能返回 tool_calls
          - 有 tool_calls：content 可为空；Runner 进入 Act
          - 无 tool_calls：content 为终答文本
          - 可重试错误默认抛 LLMError；allow_fallback=True 且无 tools 时可兜底文本
            （带 tools 的轮次不做文本兜底，避免「假装调过工具」）

        日志：llm_call_start/end 会带 tools / tool_calls_count，便于和 agent_step 对齐。
        """
        if not messages:
            raise LLMError(UPSTREAM_BAD_REQUEST, "messages must not be empty", status_code=400)

        started = time.perf_counter()
        last_error: LLMError | None = None
        attempts = self._max_retries + 1
        retry_count = 0
        tool_names = [
            str((t.get("function") or {}).get("name") or "")
            for t in (tools or [])
            if isinstance(t, dict)
        ]
        tool_names = [n for n in tool_names if n]

        log_event(
            logger,
            EVENT_LLM_CALL_START,
            request_id=request_id,
            llm_provider=LLM_PROVIDER,
            llm_model=DEFAULT_MODEL,
            message_count=len(messages),
            mode=mode,
            agent_step=agent_step,
            tools=tool_names or None,
            tools_count=len(tool_names) or None,
            hint=(
                f"开始调用 DeepSeek（model={DEFAULT_MODEL}，messages={len(messages)} 条"
                + (f"，tools={tool_names}" if tool_names else "")
                + "）"
            ),
        )

        for attempt in range(1, attempts + 1):
            try:
                turn = self._chat_once(messages, tools=tools, started=started)
                retry_count = attempt - 1
                log_event(
                    logger,
                    EVENT_LLM_CALL_END,
                    request_id=request_id,
                    ok=True,
                    llm_provider=LLM_PROVIDER,
                    llm_model=turn.model,
                    finish_reason=turn.finish_reason,
                    retry_count=retry_count,
                    latency_ms=turn.latency_ms,
                    mode=mode,
                    agent_step=agent_step,
                    tool_calls_count=len(turn.tool_calls),
                    tools_called=[tc.name for tc in turn.tool_calls] or None,
                    hint=(
                        f"大模型成功返回：耗时 {turn.latency_ms}ms，"
                        f"finish_reason={turn.finish_reason}，retry={retry_count}"
                        + (
                            f"，真实 tool_calls={len(turn.tool_calls)}"
                            if turn.tool_calls
                            else ""
                        )
                    ),
                )
                return LLMTurnResult(
                    content=turn.content,
                    tool_calls=turn.tool_calls,
                    model=turn.model,
                    finish_reason=turn.finish_reason,
                    latency_ms=turn.latency_ms,
                    usage=turn.usage,
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

        if allow_fallback and not tools and self._enable_fallback and is_retryable(last_error):
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
                mode=mode,
                hint=(
                    f"上游多次失败后走兜底文案：error_code={last_error.code}，"
                    f"retry={retry_count}"
                ),
            )
            return LLMTurnResult(
                content=FALLBACK_ANSWER,
                tool_calls=[],
                model=FALLBACK_MODEL,
                finish_reason="fallback",
                latency_ms=latency_ms,
                usage=None,
                retry_count=retry_count,
                fallback=True,
                error_code=last_error.code,
            )

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
            mode=mode,
            agent_step=agent_step,
            hint=f"大模型调用最终失败并将返回错误：error_code={last_error.code}",
        )
        raise last_error

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        *,
        started: float,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMTurnResult:
        """单次上游调用（不含重试循环）；将 SDK 异常转为 LLMError。"""
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "max_tokens": int(self._settings.llm_max_tokens or DEFAULT_MAX_TOKENS),
            "stream": DEFAULT_STREAM,
            "extra_body": dict(DEFAULT_EXTRA_BODY),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = client.chat.completions.create(**kwargs)
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
        return _parse_chat_turn(response, latency_ms=latency_ms)

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


def _extract_tool_calls(message: Any) -> list[ToolCall]:
    """从 assistant message 解析 tool_calls（兼容 SDK 对象与 dict）。"""
    raw = getattr(message, "tool_calls", None)
    if not raw:
        return []
    parsed: list[ToolCall] = []
    for item in raw:
        if isinstance(item, dict):
            fn = item.get("function") or {}
            call_id = str(item.get("id") or "")
            name = str(fn.get("name") or "")
            arguments = fn.get("arguments")
        else:
            fn = getattr(item, "function", None)
            call_id = str(getattr(item, "id", "") or "")
            name = str(getattr(fn, "name", "") or "") if fn is not None else ""
            arguments = getattr(fn, "arguments", None) if fn is not None else None
        if isinstance(arguments, dict):
            import json

            arguments = json.dumps(arguments, ensure_ascii=False)
        elif arguments is None:
            arguments = "{}"
        else:
            arguments = str(arguments)
        if not call_id or not name:
            continue
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return parsed


def _parse_chat_turn(response: Any, *, latency_ms: int) -> LLMTurnResult:
    """解析一轮响应：允许 content 为空，只要有 tool_calls。"""
    content: str | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = []
    if response.choices:
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        raw_content = getattr(message, "content", None) if message is not None else None
        if isinstance(raw_content, str):
            content = raw_content
        finish_reason = getattr(choice, "finish_reason", None)
        if message is not None:
            tool_calls = _extract_tool_calls(message)

    answer = (content or "").strip()
    if not answer and not tool_calls:
        raise LLMError(UPSTREAM_UNKNOWN, "llm returned empty content", status_code=502)

    usage: dict[str, Any] | None = None
    if getattr(response, "usage", None) is not None:
        usage_obj = response.usage
        usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else dict(usage_obj)

    model = getattr(response, "model", None) or DEFAULT_MODEL
    return LLMTurnResult(
        content=answer or None,
        tool_calls=tool_calls,
        model=model,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        usage=usage,
    )
