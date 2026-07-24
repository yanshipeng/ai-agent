"""HTTP 路由：GET /health、POST /ask。

/ask 流程概要：
1. 生成 request_id（UUID）
2. 打 request_start 日志（含 query_len / query_sha256_8，不含原文）
3. 调用 LLMClient.chat(...)
4. 成功 → request_success + 写 requests.jsonl；失败 → request_error + 写 jsonl
5. 参数校验失败由 ask_validation_exception_handler 统一返回 INVALID_ARGUMENT
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings
from app.core.logging import (
    EVENT_REQUEST_ERROR,
    EVENT_REQUEST_START,
    EVENT_REQUEST_SUCCESS,
    EVENT_VALIDATE_FAILED,
    LLM_PROVIDER,
    get_logger,
    log_event,
    query_sha256_8,
)
from app.services.llm_client import (
    LLMClient,
    LLMError,
    map_llm_error_to_http,
    public_error_message,
)
from app.services.metrics_store import append_request_metric, build_ask_metric

router = APIRouter()
logger = get_logger(__name__)

# query 校验：非空 + 最大长度
QUERY_MAX_LENGTH = 2000
CODE_INVALID_ARGUMENT = "INVALID_ARGUMENT"


class AskRequest(BaseModel):
    """POST /ask 请求体。"""

    query: str = Field(..., description="用户问题（必填）")
    session_id: str | None = Field(default=None, description="会话 ID，用于多轮")
    client_tag: str | None = Field(default=None, description="来源标识")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """去空白后校验非空与长度上限。"""
        if value is None:
            raise ValueError("query is required")
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        if len(stripped) > QUERY_MAX_LENGTH:
            raise ValueError(f"query must be at most {QUERY_MAX_LENGTH} characters")
        return stripped


class AskResponse(BaseModel):
    """POST /ask 成功响应。"""

    request_id: str
    answer: str
    citations: list[Any]  # 当前固定为空数组，预留引用扩展
    latency_ms: int
    model: str
    meta: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """GET /health 响应。"""

    status: str
    version: str


class ErrorBody(BaseModel):
    """统一错误体：request_id + code + message。"""

    request_id: str
    code: str
    message: str


def build_error_body(*, request_id: str, code: str, message: str) -> dict[str, Any]:
    """构造错误 JSON（供 JSONResponse 使用）。"""
    return ErrorBody(request_id=request_id, code=code, message=message).model_dump()


def format_validation_message(exc: RequestValidationError) -> str:
    """将 FastAPI/Pydantic 校验错误转成可读 message（不含 query 原文）。"""
    errors = exc.errors()
    if not errors:
        return "invalid request"
    parts: list[str] = []
    for err in errors:
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
        msg = err.get("msg", "invalid")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


def get_llm_client(request: Request) -> LLMClient:
    """从 app.state 取共享 LLMClient；缺失时懒创建。"""
    client = getattr(request.app.state, "llm_client", None)
    if client is None:
        client = LLMClient()
        request.app.state.llm_client = client
    return client


async def ask_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """请求体校验失败：validate_failed → request_error，HTTP 400 + INVALID_ARGUMENT。"""
    request_id = str(uuid.uuid4())
    message = format_validation_message(exc)
    common = {
        "request_id": request_id,
        "path": request.url.path,
        "method": request.method,
    }
    log_event(
        logger,
        EVENT_VALIDATE_FAILED,
        level=logging.WARNING,
        **common,
        status_code=400,
        ok=False,
        error_code=CODE_INVALID_ARGUMENT,
    )
    if request.url.path.rstrip("/") == "/ask":
        append_request_metric(
            build_ask_metric(
                request_id=request_id,
                path=request.url.path,
                ok=False,
                status_code=400,
                latency_ms_total=0,
                latency_ms_llm=None,
                error_code=CODE_INVALID_ARGUMENT,
                query_len=0,
                query_sha256_8=None,
            )
        )
    log_event(
        logger,
        EVENT_REQUEST_ERROR,
        level=logging.WARNING,
        **common,
        status_code=400,
        ok=False,
        error_code=CODE_INVALID_ARGUMENT,
        latency_ms_total=0,
    )
    return JSONResponse(
        status_code=400,
        content=build_error_body(
            request_id=request_id,
            code=CODE_INVALID_ARGUMENT,
            message=message,
        ),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """探活：不依赖 DeepSeek，返回 status + version。"""
    settings = get_settings()
    return HealthResponse(status="ok", version=settings.app_version)


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={400: {"model": ErrorBody}},
)
def ask(body: AskRequest, request: Request) -> AskResponse | JSONResponse:
    """问答接口：调用 LLMClient.chat()；citations 暂为空数组。"""
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    client = get_llm_client(request)
    q_len = len(body.query)
    q_hash = query_sha256_8(body.query)
    base_fields = {
        "request_id": request_id,
        "path": request.url.path,
        "method": request.method,
        "query_len": q_len,
        "query_sha256_8": q_hash,
    }

    log_event(logger, EVENT_REQUEST_START, **base_fields)

    try:
        result = client.chat(
            [{"role": "user", "content": body.query}],
            request_id=request_id,
        )
    except LLMError as exc:
        # 不可重试失败或未开启兜底：映射为 HTTP 错误体
        http_status, error_code = map_llm_error_to_http(exc)
        latency_ms_total = int((time.perf_counter() - started) * 1000)
        log_event(
            logger,
            EVENT_REQUEST_ERROR,
            level=logging.WARNING,
            **base_fields,
            status_code=http_status,
            ok=False,
            latency_ms_total=latency_ms_total,
            error_code=error_code,
            llm_provider=LLM_PROVIDER,
        )
        append_request_metric(
            build_ask_metric(
                request_id=request_id,
                path=request.url.path,
                ok=False,
                status_code=http_status,
                latency_ms_total=latency_ms_total,
                latency_ms_llm=None,
                error_code=error_code,
                query_len=q_len,
                query_sha256_8=q_hash,
            )
        )
        return JSONResponse(
            status_code=http_status,
            content=build_error_body(
                request_id=request_id,
                code=error_code,
                # 对外只返回稳定短文案，不回传上游响应全文
                message=public_error_message(error_code),
            ),
        )

    # 成功（含 fallback=True 的 HTTP 200 兜底）
    latency_ms_total = int((time.perf_counter() - started) * 1000)
    meta = _build_meta(
        finish_reason=result.finish_reason,
        usage=result.usage,
        fallback=result.fallback,
        error_code=result.error_code,
        retry_count=result.retry_count,
    )
    log_event(
        logger,
        EVENT_REQUEST_SUCCESS,
        **base_fields,
        status_code=200,
        ok=True,
        latency_ms_total=latency_ms_total,
        llm_provider=LLM_PROVIDER,
        llm_model=result.model,
        finish_reason=result.finish_reason,
        retry_count=result.retry_count,
        error_code=result.error_code if result.fallback else None,
    )
    append_request_metric(
        build_ask_metric(
            request_id=request_id,
            path=request.url.path,
            ok=True,
            status_code=200,
            latency_ms_total=latency_ms_total,
            latency_ms_llm=result.latency_ms,
            llm_model=result.model,
            retry_count=result.retry_count,
            finish_reason=result.finish_reason,
            error_code=result.error_code if result.fallback else None,
            query_len=q_len,
            query_sha256_8=q_hash,
            usage=result.usage,
        )
    )
    return AskResponse(
        request_id=request_id,
        answer=result.answer,
        citations=[],
        latency_ms=result.latency_ms,
        model=result.model,
        meta=meta,
    )


def _build_meta(
    *,
    finish_reason: str | None,
    usage: dict[str, Any] | None,
    fallback: bool = False,
    error_code: str | None = None,
    retry_count: int | None = None,
) -> dict[str, Any] | None:
    """组装响应 meta；全空则返回 None。"""
    meta: dict[str, Any] = {}
    if finish_reason is not None:
        meta["finish_reason"] = finish_reason
    if usage:
        meta["usage"] = usage
    if fallback:
        meta["fallback"] = True
    if error_code:
        meta["error_code"] = error_code
    if retry_count is not None:
        meta["retry_count"] = retry_count
    return meta or None


def error_response_schema() -> set[str]:
    """供测试引用的错误体字段约定。"""
    return {"request_id", "code", "message"}
