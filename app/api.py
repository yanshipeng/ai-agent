"""HTTP 路由：GET /health、POST /ask。

==========================================================================
/ask 在整条产品链路里的位置
==========================================================================
  用户问题
    →（可选）本地 retrieve TopK
    → 拼 messages（llm 仅 user；rag = system约束 + Context + 问题）
    → LLMClient.chat
    → 返回 answer + citations + meta
    → 写结构化日志 + requests.jsonl

==========================================================================
为什么 mode 同时支持 query (?mode=rag) 和 body.mode
==========================================================================
  curl 示例习惯写 /ask?mode=rag；前端也可能放 JSON body。
  约定：query 参数优先，便于调试时快速切换而不改 body。

为什么默认 mode=llm？
  兼容第一周契约与回归（citations=[]）；RAG 显式打开，避免无索引时全挂。

为什么成功路径也要写 metrics（即使 fallback）？
  评测与排障依赖「每次请求一行」；失败/兜底都要能统计。

流程步骤：
  1) 生成 request_id（UUID）——整条日志链的主键
  2) request_start（含 query_len / query_sha256_8，不含原文）
  3) mode=rag → retrieve_start/end；mode=llm → 跳过检索
  4) llm_call_*（在 LLMClient 内）
  5) request_success 或 request_error + 写 requests.jsonl
  6) 校验失败：validate_failed → request_error → 400 INVALID_ARGUMENT
==========================================================================
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings
from app.core.logging import (
    EVENT_REQUEST_ERROR,
    EVENT_REQUEST_START,
    EVENT_REQUEST_SUCCESS,
    EVENT_RETRIEVE_END,
    EVENT_RETRIEVE_START,
    EVENT_VALIDATE_FAILED,
    LLM_PROVIDER,
    get_logger,
    log_event,
    query_sha256_8,
)
from app.kb.rag import (
    ASK_MODE_LLM,
    ASK_MODE_RAG,
    ASK_MODES,
    run_rag_retrieve,
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
CODE_INDEX_NOT_READY = "INDEX_NOT_READY"

AskMode = Literal["llm", "rag"]


class AskRequest(BaseModel):
    """POST /ask 请求体。"""

    query: str = Field(..., description="用户问题（必填）")
    session_id: str | None = Field(default=None, description="会话 ID，用于多轮")
    client_tag: str | None = Field(default=None, description="来源标识")
    mode: AskMode = Field(
        default=ASK_MODE_LLM,
        description="llm=直连模型；rag=检索增强（也可用 query ?mode=rag）",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="RAG 检索条数；默认读 RAG_TOP_K",
    )

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

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        normalized = (value or ASK_MODE_LLM).strip().lower()
        if normalized not in ASK_MODES:
            raise ValueError(f"mode must be one of {list(ASK_MODES)}")
        return normalized


class Citation(BaseModel):
    """RAG 引用条目（mode=rag 时必须填充）。

    为什么这些字段是最小集？
      ref_id 对回答里的 [n]；chunk_id 对本地 index；url/title 给人点开；
      section_path / is_code 方便展示与评测「是否点到代码块」。
    """

    ref_id: int
    chunk_id: str | None = None
    url: str | None = None
    title: str | None = None
    section_path: str = ""
    is_code: bool = False


class AskResponse(BaseModel):
    """POST /ask 成功响应。"""

    request_id: str
    answer: str
    citations: list[Citation]
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


def resolve_ask_mode(
    *,
    query_mode: str | None,
    body_mode: str | None,
) -> AskMode:
    """query ?mode= 优先于 body.mode，默认 llm。

    理由：调试时改 URL 比改 JSON 更快；默认 llm 保证无索引也能问答。
    """
    for candidate in (query_mode, body_mode):
        if candidate is None:
            continue
        normalized = candidate.strip().lower()
        if normalized in ASK_MODES:
            return normalized  # type: ignore[return-value]
    return ASK_MODE_LLM  # type: ignore[return-value]


def resolve_top_k(body_top_k: int | None) -> int:
    """RAG top_k：请求体优先，否则读配置。"""
    if body_top_k is not None:
        return body_top_k
    return int(get_settings().rag_top_k)


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
        hint="参数不合法：请检查 query 是否为空、是否超过 2000 字符",
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
                mode=ASK_MODE_LLM,
                top_k=None,
                retrieve_ms=None,
                context_chunks=0,
                citations_count=0,
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
        hint="请求被拒绝：校验失败，未调用检索/大模型",
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
    responses={400: {"model": ErrorBody}, 503: {"model": ErrorBody}},
)
def ask(
    body: AskRequest,
    request: Request,
    mode: AskMode | None = Query(
        default=None,
        description="llm | rag；优先于 body.mode（例：/ask?mode=rag）",
    ),
) -> AskResponse | JSONResponse:
    """问答接口：mode=llm 直连模型；mode=rag 检索增强且 citations 必填。"""
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    client = get_llm_client(request)
    resolved_mode = resolve_ask_mode(query_mode=mode, body_mode=body.mode)
    top_k = resolve_top_k(body.top_k) if resolved_mode == ASK_MODE_RAG else None
    q_len = len(body.query)
    q_hash = query_sha256_8(body.query)
    base_fields = {
        "request_id": request_id,
        "path": request.url.path,
        "method": request.method,
        "query_len": q_len,
        "query_sha256_8": q_hash,
        "mode": resolved_mode,
    }

    log_event(
        logger,
        EVENT_REQUEST_START,
        **base_fields,
        top_k=top_k,
        hint=(
            "收到 /ask：mode=rag，将先本地检索再问大模型"
            if resolved_mode == ASK_MODE_RAG
            else "收到 /ask：mode=llm，直接把问题交给大模型（不查知识库）"
        ),
    )

    citations: list[dict[str, Any]] = []
    retrieve_ms: int | None = None
    context_chunks = 0
    messages: list[dict[str, str]]

    if resolved_mode == ASK_MODE_RAG:
        index_dir = get_settings().kb_index_dir
        resolved_top_k = top_k or resolve_top_k(None)
        log_event(
            logger,
            EVENT_RETRIEVE_START,
            **base_fields,
            top_k=resolved_top_k,
            index_dir=index_dir,
            hint=f"开始本地检索：index={index_dir}，准备取 Top{resolved_top_k} 片段",
        )
        try:
            rag_pack = run_rag_retrieve(
                body.query,
                top_k=resolved_top_k,
                index_dir=index_dir,
            )
        except FileNotFoundError:
            latency_ms_total = int((time.perf_counter() - started) * 1000)
            log_event(
                logger,
                EVENT_REQUEST_ERROR,
                level=logging.WARNING,
                **base_fields,
                status_code=503,
                ok=False,
                latency_ms_total=latency_ms_total,
                error_code=CODE_INDEX_NOT_READY,
                top_k=resolved_top_k,
                index_dir=index_dir,
                hint="知识库索引不存在：请先运行 python scripts/build_kb_index.py",
            )
            append_request_metric(
                build_ask_metric(
                    request_id=request_id,
                    path=request.url.path,
                    ok=False,
                    status_code=503,
                    latency_ms_total=latency_ms_total,
                    error_code=CODE_INDEX_NOT_READY,
                    query_len=q_len,
                    query_sha256_8=q_hash,
                    mode=resolved_mode,
                    top_k=top_k,
                    retrieve_ms=None,
                    context_chunks=0,
                    citations_count=0,
                )
            )
            return JSONResponse(
                status_code=503,
                content=build_error_body(
                    request_id=request_id,
                    code=CODE_INDEX_NOT_READY,
                    message="knowledge index not ready; run build_kb_index.py",
                ),
            )
        messages = rag_pack["messages"]
        citations = rag_pack["citations"]
        retrieve_ms = rag_pack["retrieve_ms"]
        context_chunks = rag_pack["context_chunks"]
        top_k = rag_pack["top_k"]
        top_chunk_ids = [c.get("chunk_id") for c in citations[:3]]
        log_event(
            logger,
            EVENT_RETRIEVE_END,
            **base_fields,
            top_k=top_k,
            retrieve_ms=retrieve_ms,
            context_chunks=context_chunks,
            citations_count=len(citations),
            top_chunk_ids=top_chunk_ids,
            hint=(
                f"检索完成：命中 {context_chunks} 条，耗时 {retrieve_ms}ms；"
                "已拼好 Context（含 [1][2]…），下一步调用大模型"
            ),
        )
    else:
        messages = [{"role": "user", "content": body.query}]

    try:
        result = client.chat(messages, request_id=request_id)
    except LLMError as exc:
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
            retrieve_ms=retrieve_ms,
            context_chunks=context_chunks,
            citations_count=len(citations),
            hint=f"大模型调用失败：error_code={error_code}，HTTP={http_status}",
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
                mode=resolved_mode,
                top_k=top_k,
                retrieve_ms=retrieve_ms,
                context_chunks=context_chunks,
                citations_count=len(citations),
            )
        )
        return JSONResponse(
            status_code=http_status,
            content=build_error_body(
                request_id=request_id,
                code=error_code,
                message=public_error_message(error_code),
            ),
        )

    latency_ms_total = int((time.perf_counter() - started) * 1000)
    meta = _build_meta(
        finish_reason=result.finish_reason,
        usage=result.usage,
        fallback=result.fallback,
        error_code=result.error_code,
        retry_count=result.retry_count,
        mode=resolved_mode,
        top_k=top_k,
        retrieve_ms=retrieve_ms,
        context_chunks=context_chunks,
        citations_count=len(citations),
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
        retrieve_ms=retrieve_ms,
        context_chunks=context_chunks,
        citations_count=len(citations),
        top_k=top_k,
        hint=(
            f"整单成功：mode={resolved_mode}，总耗时 {latency_ms_total}ms，"
            f"citations={len(citations)}；可用 trace_request.py 按 request_id 回放"
        ),
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
            mode=resolved_mode,
            top_k=top_k,
            retrieve_ms=retrieve_ms,
            context_chunks=context_chunks,
            citations_count=len(citations),
        )
    )
    return AskResponse(
        request_id=request_id,
        answer=result.answer,
        citations=[Citation(**c) for c in citations],
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
    mode: str | None = None,
    top_k: int | None = None,
    retrieve_ms: int | None = None,
    context_chunks: int | None = None,
    citations_count: int | None = None,
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
    if mode is not None:
        meta["mode"] = mode
    if top_k is not None:
        meta["top_k"] = top_k
    if retrieve_ms is not None:
        meta["retrieve_ms"] = retrieve_ms
    if context_chunks is not None:
        meta["context_chunks"] = context_chunks
    if citations_count is not None:
        meta["citations_count"] = citations_count
    return meta or None


def error_response_schema() -> set[str]:
    """供测试引用的错误体字段约定。"""
    return {"request_id", "code", "message"}
