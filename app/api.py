"""HTTP 路由：GET /health、POST /ask。

==========================================================================
/ask 在整条产品链路里的位置（三 mode）
==========================================================================
  mode=llm   ：拼 [history?] + user → LLMClient.chat
  mode=rag   ：hybrid retrieve → system+history+Context user → chat
  mode=agent ：load history → run_agent_loop（真实 tool_calls 状态机）

成功后：
  - 有 session_id → 写回 session_store（llm/rag 只存短 user/assistant）
  - 写结构化日志 + requests.jsonl（含 agent_* / session_id / history_*）
  - mode=agent 另写 traces.jsonl（Day18 逐步 agent_trace）

==========================================================================
为什么 mode 同时支持 query (?mode=) 和 body.mode
==========================================================================
  curl 习惯写 /ask?mode=agent；前端也可能放 JSON。
  约定：query 参数优先，便于调试时改 URL 不改 body。

为什么默认 mode=llm？
  兼容 Week1 契约与回归（citations=[]）；rag/agent 显式打开。

多轮 session（Week3）：
  load_session_history → compact（截断/滑窗/摘要）→ 注入 messages
  history_metric_fields → jsonl 的 session_id / history_messages / history_chars

Week4 护栏与可观测：
  Day16：retrieve_* 去重流进 meta / jsonl
  Day17：detect_prompt_injection 预检拒答；返回前引用门禁 + 泄密扫描
  Day18：reset_cache_counters；agent_trace / token 预算 / cache_hit|miss 落盘

流程步骤：
  1) 生成 request_id（UUID）；reset_cache_counters
  2) request_start（query_len / query_sha256_8 / session 指纹）
  3) 注入预检（命中则拒答，不调 LLM/工具）
  4) 按 mode 分支（rag 检索 / agent 状态机 / llm 直连）
  5) 成功：门禁 → session + meta + metrics（+ traces）；失败写 error_code 不崩服务
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

from app.agent.runner import (
    AGENT_MAX_STEPS,
    AGENT_NO_ANSWER,
    AGENT_TIMEOUT,
    AgentResult,
    run_agent_loop,
)
from app.agent.tools import (
    ASK_MODE_AGENT,
    TOOL_INVALID_ARGS,
    TOOL_NOT_FOUND,
    TOOL_TIMEOUT,
)
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
    ASK_MODES as RAG_ASK_MODES,
    run_rag_retrieve,
)
from app.core.safety import (
    build_injection_refusal,
    detect_prompt_injection,
    enforce_citation_consistency,
    enforce_no_leakage,
)
from app.kb.retriever import (
    get_cache_counters,
    reset_cache_counters,
    retrieve_stat_fields,
)
from app.services.conversation import (
    CompactStats,
    compact_messages,
    messages_for_storage,
    plain_chat_history,
    total_chars,
)
from app.services.llm_client import (
    LLMClient,
    LLMError,
    UPSTREAM_5XX,
    UPSTREAM_BAD_REQUEST,
    UPSTREAM_RATE_LIMITED,
    UPSTREAM_TIMEOUT,
    UPSTREAM_UNAUTHORIZED,
    UPSTREAM_UNKNOWN,
    map_llm_error_to_http,
    public_error_message,
)
from app.services.metrics_store import (
    append_request_metric,
    append_trace_metric,
    build_ask_metric,
)
from app.services.session_store import (
    get_session_messages,
    set_session_messages,
    touch_and_prune,
)

router = APIRouter()
logger = get_logger(__name__)

# query 校验：非空 + 最大长度
QUERY_MAX_LENGTH = 2000
CODE_INVALID_ARGUMENT = "INVALID_ARGUMENT"
CODE_INDEX_NOT_READY = "INDEX_NOT_READY"

ASK_MODES = (*RAG_ASK_MODES, ASK_MODE_AGENT)
AskMode = Literal["llm", "rag", "agent"]


def session_id_fingerprint(session_id: str | None) -> str | None:
    """日志/metrics 用：session_id 指纹，不落原文。"""
    sid = (session_id or "").strip()
    if not sid:
        return None
    return query_sha256_8(sid)


def _cache_metric_fields() -> dict[str, int]:
    """Day18：本请求累计的索引/BM25 缓存命中。"""
    return get_cache_counters()


def _agent_budget_fields(agent: AgentResult) -> dict[str, Any]:
    """Day18：token 预算 + agent_trace（写入 meta / requests.jsonl）。"""
    return {
        "agent_trace": list(agent.agent_trace or []),
        "max_context_tokens": agent.max_context_tokens,
        "context_tokens_used": agent.context_tokens_used,
        "max_output_tokens": agent.max_output_tokens,
        "budget_compressed": agent.budget_compressed,
    }


def _write_agent_trace_file(
    *,
    request_id: str,
    agent: AgentResult,
    ok: bool,
    status_code: int,
) -> None:
    """Day18：单独落盘 traces.jsonl（1 请求 1 行，含 steps）。"""
    append_trace_metric(
        {
            "request_id": request_id,
            "mode": ASK_MODE_AGENT,
            "ok": ok,
            "status_code": status_code,
            "stop_reason": agent.stop_reason,
            "agent_steps": agent.agent_steps,
            "max_steps": agent.max_steps,
            **_agent_budget_fields(agent),
            **_cache_metric_fields(),
        }
    )


def load_session_history(
    session_id: str | None,
    *,
    client: LLMClient | None = None,
    request_id: str | None = None,
    for_agent: bool = False,
) -> tuple[list[dict[str, Any]], CompactStats | None]:
    """读取并压缩同 session 历史；无 session 返回 ([], None)。

    for_agent=False（llm/rag）：
      只保留 user/assistant，避免把上一轮 agent 的 tool 轨迹塞进普通聊天。
    for_agent=True：
      保留截断后的 tool 轨迹，方便续轮排障。

    compact 顺序：单条截断 → 滑窗 →（可选）超预算摘要；统计写入日志 context_compact。
    """
    sid = (session_id or "").strip()
    if not sid:
        return [], None
    settings = get_settings()
    touch_and_prune(ttl_seconds=float(settings.session_ttl_seconds))
    raw = get_session_messages(sid)
    if not raw:
        return [], None
    compacted = compact_messages(
        raw,
        client=client if settings.session_summary_use_llm else None,
        request_id=request_id,
    )
    history = compacted.messages
    if not for_agent:
        history = plain_chat_history(history)
    return history, compacted.stats


def persist_plain_turn(
    session_id: str | None,
    *,
    history: list[dict[str, Any]],
    query: str,
    answer: str,
) -> None:
    """llm/rag 成功后：存「历史 + 本轮 user/assistant」（不含 RAG Context）。"""
    sid = (session_id or "").strip()
    if not sid:
        return
    settings = get_settings()
    to_store = messages_for_storage(
        list(history)
        + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
    )
    set_session_messages(
        sid,
        to_store,
        ttl_seconds=float(settings.session_ttl_seconds),
    )


def persist_agent_messages(
    session_id: str | None,
    messages: list[dict[str, Any]],
) -> None:
    """agent 成功后：覆盖写入本轮完整轨迹（已去 system、已截断）。"""
    sid = (session_id or "").strip()
    if not sid or not messages:
        return
    settings = get_settings()
    set_session_messages(
        sid,
        messages_for_storage(messages),
        ttl_seconds=float(settings.session_ttl_seconds),
    )


def _session_meta_fields(stats: CompactStats | None) -> dict[str, Any]:
    if stats is None:
        return {}
    return {
        "session_turns_kept": stats.turns_kept,
        "session_msgs": stats.output_messages,
        "session_chars": stats.output_chars,
        "session_summarized": stats.summarized,
        "session_truncated_msgs": stats.truncated_msgs,
    }


def history_metric_fields(
    session_id: str | None,
    history: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """requests.jsonl / meta：session_id + history_messages + history_chars（无原文）。"""
    sid = (session_id or "").strip()
    if not sid:
        return {}
    hist = history or []
    return {
        "session_id": sid,
        "history_messages": len(hist),
        "history_chars": total_chars(hist),
    }


# Agent / Tool 可控错误 → HTTP（不崩服务）
_AGENT_ERROR_HTTP: dict[str, int] = {
    TOOL_INVALID_ARGS: 400,
    TOOL_TIMEOUT: 504,
    TOOL_NOT_FOUND: 400,
    AGENT_MAX_STEPS: 504,
    AGENT_NO_ANSWER: 502,
    AGENT_TIMEOUT: 504,
}

_AGENT_PUBLIC_MESSAGES: dict[str, str] = {
    TOOL_INVALID_ARGS: "tool arguments invalid",
    TOOL_TIMEOUT: "tool execution timed out",
    TOOL_NOT_FOUND: "unknown tool",
    AGENT_MAX_STEPS: "agent exceeded max tool steps",
    AGENT_NO_ANSWER: "agent finished without an answer",
    AGENT_TIMEOUT: "agent exceeded max total time",
}

_ERROR_FROM_LLM = frozenset(
    {
        UPSTREAM_UNAUTHORIZED,
        UPSTREAM_RATE_LIMITED,
        UPSTREAM_TIMEOUT,
        UPSTREAM_BAD_REQUEST,
        UPSTREAM_5XX,
        UPSTREAM_UNKNOWN,
    }
)

class AskRequest(BaseModel):
    """POST /ask 请求体。"""

    query: str = Field(..., description="用户问题（必填）")
    session_id: str | None = Field(default=None, description="会话 ID，用于多轮")
    client_tag: str | None = Field(default=None, description="来源标识")
    mode: AskMode = Field(
        default=ASK_MODE_LLM,
        description="llm=直连；rag=检索增强；agent=工具调用循环（也可用 ?mode=）",
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
        description="llm | rag | agent；优先于 body.mode（例：/ask?mode=agent）",
    ),
) -> AskResponse | JSONResponse:
    """问答接口：mode=llm 直连；mode=rag 检索增强；mode=agent 真实 tool_calls 循环。"""
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    reset_cache_counters()  # Day18：按请求统计 cache_hit / cache_miss
    client = get_llm_client(request)
    resolved_mode = resolve_ask_mode(query_mode=mode, body_mode=body.mode)
    top_k = resolve_top_k(body.top_k) if resolved_mode in {ASK_MODE_RAG, ASK_MODE_AGENT} else None
    q_len = len(body.query)
    q_hash = query_sha256_8(body.query)
    sid_fp = session_id_fingerprint(body.session_id)
    base_fields = {
        "request_id": request_id,
        "path": request.url.path,
        "method": request.method,
        "query_len": q_len,
        "query_sha256_8": q_hash,
        "mode": resolved_mode,
        "session_id_sha256_8": sid_fp,
    }

    if resolved_mode == ASK_MODE_AGENT:
        hint = "收到 /ask：mode=agent，将走 Tool Runner（真实 tool_calls）"
    elif resolved_mode == ASK_MODE_RAG:
        hint = "收到 /ask：mode=rag，将先本地检索再问大模型"
    else:
        hint = "收到 /ask：mode=llm，直接把问题交给大模型（不查知识库）"
    if body.session_id:
        hint += f"；多轮 session（指纹 {sid_fp}）"

    log_event(
        logger,
        EVENT_REQUEST_START,
        **base_fields,
        top_k=top_k,
        hint=hint,
    )

    # Day17：明显提示注入 / 泄密请求 → 预检拒答（不调 LLM / 工具）
    injection_hit = detect_prompt_injection(body.query)
    if injection_hit:
        pack = build_injection_refusal()
        latency_ms_total = int((time.perf_counter() - started) * 1000)
        meta = _build_meta(
            finish_reason="injection_blocked",
            usage=None,
            mode=resolved_mode,
            top_k=top_k,
            context_chunks=0,
            citations_count=0,
            session_id_sha256_8=sid_fp,
            citation_guard="injection_precheck",
        )
        if meta is None:
            meta = {}
        meta["injection_blocked"] = True
        meta["injection_pattern"] = injection_hit.get("pattern")
        log_event(
            logger,
            EVENT_REQUEST_SUCCESS,
            **base_fields,
            status_code=200,
            ok=True,
            latency_ms_total=latency_ms_total,
            injection_blocked=True,
            hint="注入预检命中：已拒答，未调用检索/大模型/工具",
        )
        append_request_metric(
            build_ask_metric(
                request_id=request_id,
                path=request.url.path,
                ok=True,
                status_code=200,
                latency_ms_total=latency_ms_total,
                finish_reason="injection_blocked",
                query_len=q_len,
                query_sha256_8=q_hash,
                mode=resolved_mode,
                top_k=top_k,
                context_chunks=0,
                citations_count=0,
                session_id_sha256_8=sid_fp,
            )
        )
        return AskResponse(
            request_id=request_id,
            answer=str(pack["answer"]),
            citations=[],
            latency_ms=latency_ms_total,
            model=get_settings().llm_model,
            meta=meta,
        )

    if resolved_mode == ASK_MODE_AGENT:
        return _ask_agent(
            body=body,
            request=request,
            client=client,
            request_id=request_id,
            started=started,
            base_fields=base_fields,
            q_len=q_len,
            q_hash=q_hash,
            top_k=top_k,
            sid_fp=sid_fp,
        )

    history, session_stats = load_session_history(
        body.session_id,
        client=client,
        request_id=request_id,
        for_agent=False,
    )
    hist_fields = history_metric_fields(body.session_id, history)
    session_compact = _session_meta_fields(session_stats)
    citations: list[dict[str, Any]] = []
    retrieve_ms: int | None = None
    context_chunks = 0
    retrieve_stats: dict[str, Any] = {}
    messages: list[dict[str, Any]]

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
                    session_id_sha256_8=sid_fp,
                    **hist_fields,
                    **_cache_metric_fields(),
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
        rag_messages = rag_pack["messages"]
        # system + 历史 + 本轮（含 Context 的 user）；历史不含 RAG Context
        messages = [rag_messages[0], *history, rag_messages[-1]]
        citations = rag_pack["citations"]
        retrieve_ms = rag_pack["retrieve_ms"]
        context_chunks = rag_pack["context_chunks"]
        top_k = rag_pack["top_k"]
        retrieve_stats = retrieve_stat_fields(rag_pack)
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
            **retrieve_stats,
            hint=(
                f"检索完成：候选 {retrieve_stats.get('retrieve_candidates', '?')} → "
                f"去重前 {retrieve_stats.get('retrieve_before_dedup', '?')} → "
                f"去重后 {retrieve_stats.get('retrieve_after_dedup', '?')} → "
                f"保留 {retrieve_stats.get('retrieve_kept', context_chunks)}，"
                f"dedup_dropped={retrieve_stats.get('dedup_dropped', 0)}，"
                f"耗时 {retrieve_ms}ms；已拼好 Context，下一步调用大模型"
            ),
        )
    else:
        messages = [*history, {"role": "user", "content": body.query}]

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
            **retrieve_stats,
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
                session_id_sha256_8=sid_fp,
                **hist_fields,
                **retrieve_stats,
                **_cache_metric_fields(),
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

    persist_plain_turn(
        body.session_id,
        history=history,
        query=body.query,
        answer=result.answer,
    )

    latency_ms_total = int((time.perf_counter() - started) * 1000)
    # Day17：引用强约束 + 泄密扫描
    safe_answer, citation_meta = enforce_citation_consistency(
        result.answer,
        citations,
        mode=resolved_mode,
    )
    safe_answer, leak_meta = enforce_no_leakage(safe_answer)
    if leak_meta:
        citations = []
    cache_fields = _cache_metric_fields()
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
        session_id_sha256_8=sid_fp,
        **hist_fields,
        **session_compact,
        **retrieve_stats,
        **cache_fields,
        **citation_meta,
        **{k: v for k, v in leak_meta.items() if k in {"leakage_blocked"}},
    )
    if meta is not None and leak_meta.get("leakage_hits"):
        meta["leakage_hits"] = leak_meta["leakage_hits"]
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
        history_messages=hist_fields.get("history_messages"),
        history_chars=hist_fields.get("history_chars"),
        **session_compact,
        **retrieve_stats,
        **cache_fields,
        **{k: citation_meta[k] for k in ("citation_guard", "citation_invalid_refs") if k in citation_meta},
        hint=(
            f"整单成功：mode={resolved_mode}，总耗时 {latency_ms_total}ms，"
            f"history_messages={hist_fields.get('history_messages', 0)}，"
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
            session_id_sha256_8=sid_fp,
            **hist_fields,
            **session_compact,
            **retrieve_stats,
            **cache_fields,
        )
    )
    return AskResponse(
        request_id=request_id,
        answer=safe_answer,
        citations=[Citation(**c) for c in citations],
        latency_ms=result.latency_ms,
        model=result.model,
        meta=meta,
    )


def _ask_agent(
    *,
    body: AskRequest,
    request: Request,
    client: LLMClient,
    request_id: str,
    started: float,
    base_fields: dict[str, Any],
    q_len: int,
    q_hash: str,
    top_k: int | None,
    sid_fp: str | None = None,
) -> AskResponse | JSONResponse:
    """mode=agent：Plan→Act→Observe→Final；超步降级 RAG/澄清，超时兜底带 request_id。"""
    history, session_stats = load_session_history(
        body.session_id,
        client=client,
        request_id=request_id,
        for_agent=True,
    )
    hist_fields = history_metric_fields(body.session_id, history)
    agent = run_agent_loop(
        body.query,
        client=client,
        request_id=request_id,
        index_dir=get_settings().kb_index_dir,
        history_messages=history or None,
    )
    latency_ms_total = int((time.perf_counter() - started) * 1000)
    tools_used = list(agent.tools_used)
    citations = agent.citations
    session_compact = _session_meta_fields(session_stats)

    if agent.http_error_code:
        error_code = agent.http_error_code
        http_status = _AGENT_ERROR_HTTP.get(error_code, 502)
        if error_code in _ERROR_FROM_LLM:
            http_status, error_code = map_llm_error_to_http(
                LLMError(error_code, error_code)
            )
        message = _AGENT_PUBLIC_MESSAGES.get(error_code) or public_error_message(error_code)
        log_event(
            logger,
            EVENT_REQUEST_ERROR,
            level=logging.WARNING,
            **base_fields,
            status_code=http_status,
            ok=False,
            latency_ms_total=latency_ms_total,
            error_code=error_code,
            agent_steps=agent.agent_steps,
            tool_calls_count=agent.tool_calls_count,
            tools_used=tools_used,
            citations_count=len(citations),
            agent_final_phase=agent.final_phase,
            agent_phase_trace=agent.phase_trace,
            degraded_to=agent.degraded_to,
            stop_reason=agent.stop_reason,
            max_steps=agent.max_steps,
            history_messages=hist_fields.get("history_messages"),
            history_chars=hist_fields.get("history_chars"),
            **session_compact,
            hint=(
                f"Agent 失败（可控）：error_code={error_code}，"
                f"stop_reason={agent.stop_reason}，"
                f"steps={agent.agent_steps}/{agent.max_steps}，"
                f"tool_calls={agent.tool_calls_count}，phase={agent.final_phase}"
            ),
        )
        day18 = {**_agent_budget_fields(agent), **_cache_metric_fields()}
        append_request_metric(
            build_ask_metric(
                request_id=request_id,
                path=request.url.path,
                ok=False,
                status_code=http_status,
                latency_ms_total=latency_ms_total,
                latency_ms_llm=agent.latency_ms,
                llm_model=agent.model,
                retry_count=agent.retry_count,
                finish_reason=agent.finish_reason,
                error_code=error_code,
                query_len=q_len,
                query_sha256_8=q_hash,
                usage=agent.usage,
                mode=ASK_MODE_AGENT,
                top_k=top_k,
                retrieve_ms=agent.retrieve_ms,
                context_chunks=len(citations),
                citations_count=len(citations),
                agent_steps=agent.agent_steps,
                max_steps=agent.max_steps,
                tool_calls_count=agent.tool_calls_count,
                tools_used=tools_used,
                agent_final_phase=agent.final_phase,
                agent_phase_trace=agent.phase_trace,
                degraded_to=agent.degraded_to,
                stop_reason=agent.stop_reason,
                session_id_sha256_8=sid_fp,
                **hist_fields,
                **session_compact,
                **day18,
            )
        )
        _write_agent_trace_file(
            request_id=request_id,
            agent=agent,
            ok=False,
            status_code=http_status,
        )
        return JSONResponse(
            status_code=http_status,
            content=build_error_body(
                request_id=request_id,
                code=error_code,
                message=message,
            ),
        )

    persist_agent_messages(body.session_id, agent.session_messages)

    safe_answer, citation_meta = enforce_citation_consistency(
        agent.answer,
        citations,
        mode=ASK_MODE_AGENT,
    )
    safe_answer, leak_meta = enforce_no_leakage(safe_answer)
    if leak_meta:
        citations = []
    day18 = {**_agent_budget_fields(agent), **_cache_metric_fields()}
    meta = _build_meta(
        finish_reason=agent.finish_reason,
        usage=agent.usage,
        fallback=agent.fallback,
        error_code=agent.error_code,
        retry_count=agent.retry_count,
        mode=ASK_MODE_AGENT,
        top_k=top_k,
        retrieve_ms=agent.retrieve_ms,
        context_chunks=len(citations),
        citations_count=len(citations),
        agent_steps=agent.agent_steps,
        tool_calls_count=agent.tool_calls_count,
        tools_used=tools_used,
        agent_final_phase=agent.final_phase,
        agent_phase_trace=agent.phase_trace,
        degraded_to=agent.degraded_to,
        stop_reason=agent.stop_reason,
        max_steps=agent.max_steps,
        session_id_sha256_8=sid_fp,
        **hist_fields,
        **session_compact,
        **day18,
        **citation_meta,
        **{k: v for k, v in leak_meta.items() if k in {"leakage_blocked"}},
    )
    if meta is not None and leak_meta.get("leakage_hits"):
        meta["leakage_hits"] = leak_meta["leakage_hits"]
    log_event(
        logger,
        EVENT_REQUEST_SUCCESS,
        **base_fields,
        status_code=200,
        ok=True,
        latency_ms_total=latency_ms_total,
        llm_provider=LLM_PROVIDER,
        llm_model=agent.model,
        finish_reason=agent.finish_reason,
        retry_count=agent.retry_count,
        agent_steps=agent.agent_steps,
        max_steps=agent.max_steps,
        tool_calls_count=agent.tool_calls_count,
        tools_used=tools_used,
        citations_count=len(citations),
        top_k=top_k,
        agent_final_phase=agent.final_phase,
        agent_phase_trace=agent.phase_trace,
        degraded_to=agent.degraded_to,
        stop_reason=agent.stop_reason,
        error_code=agent.error_code if agent.fallback else None,
        history_messages=hist_fields.get("history_messages"),
        history_chars=hist_fields.get("history_chars"),
        context_tokens_used=agent.context_tokens_used,
        budget_compressed=agent.budget_compressed,
        **session_compact,
        **_cache_metric_fields(),
        **{k: citation_meta[k] for k in ("citation_guard", "citation_invalid_refs") if k in citation_meta},
        hint=(
            f"整单成功：mode=agent，stop_reason={agent.stop_reason}，"
            f"steps={agent.agent_steps}/{agent.max_steps}，"
            f"tool_calls={agent.tool_calls_count}，tools={tools_used}，"
            f"phase_trace={agent.phase_trace}，degraded_to={agent.degraded_to}，"
            f"trace_steps={len(agent.agent_trace)}，"
            f"ctx_tokens={agent.context_tokens_used}/{agent.max_context_tokens}，"
            f"history_messages={hist_fields.get('history_messages', 0)}，"
            f"总耗时 {latency_ms_total}ms"
        ),
    )
    append_request_metric(
        build_ask_metric(
            request_id=request_id,
            path=request.url.path,
            ok=True,
            status_code=200,
            latency_ms_total=latency_ms_total,
            latency_ms_llm=agent.latency_ms,
            llm_model=agent.model,
            retry_count=agent.retry_count,
            finish_reason=agent.finish_reason,
            error_code=agent.error_code if agent.fallback else None,
            query_len=q_len,
            query_sha256_8=q_hash,
            usage=agent.usage,
            mode=ASK_MODE_AGENT,
            top_k=top_k,
            retrieve_ms=agent.retrieve_ms,
            context_chunks=len(citations),
            citations_count=len(citations),
            agent_steps=agent.agent_steps,
            max_steps=agent.max_steps,
            tool_calls_count=agent.tool_calls_count,
            tools_used=tools_used,
            agent_final_phase=agent.final_phase,
            agent_phase_trace=agent.phase_trace,
            degraded_to=agent.degraded_to,
            stop_reason=agent.stop_reason,
            session_id_sha256_8=sid_fp,
            **hist_fields,
            **session_compact,
            **day18,
        )
    )
    _write_agent_trace_file(
        request_id=request_id,
        agent=agent,
        ok=True,
        status_code=200,
    )
    return AskResponse(
        request_id=request_id,
        answer=safe_answer,
        citations=[Citation(**c) for c in citations],
        latency_ms=agent.latency_ms,
        model=agent.model,
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
    retrieve_candidates: int | None = None,
    retrieve_before_dedup: int | None = None,
    retrieve_after_dedup: int | None = None,
    retrieve_kept: int | None = None,
    hybrid_weight: float | None = None,
    dedup_dropped: int | None = None,
    agent_steps: int | None = None,
    tool_calls_count: int | None = None,
    tools_used: list[str] | None = None,
    agent_final_phase: str | None = None,
    agent_phase_trace: list[str] | None = None,
    degraded_to: str | None = None,
    stop_reason: str | None = None,
    max_steps: int | None = None,
    session_id_sha256_8: str | None = None,
    session_id: str | None = None,
    history_messages: int | None = None,
    history_chars: int | None = None,
    session_turns_kept: int | None = None,
    session_msgs: int | None = None,
    session_chars: int | None = None,
    session_summarized: bool | None = None,
    session_truncated_msgs: int | None = None,
    citation_guard: str | None = None,
    citation_invalid_refs: list[int] | None = None,
    citation_missing_for_claims: bool | None = None,
    citation_refs_used: list[int] | None = None,
    leakage_blocked: bool | None = None,
    agent_trace: list[dict[str, Any]] | None = None,
    max_context_tokens: int | None = None,
    context_tokens_used: int | None = None,
    max_output_tokens: int | None = None,
    budget_compressed: bool | None = None,
    cache_hit: int | None = None,
    cache_miss: int | None = None,
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
    if retrieve_candidates is not None:
        meta["retrieve_candidates"] = retrieve_candidates
    if retrieve_before_dedup is not None:
        meta["retrieve_before_dedup"] = retrieve_before_dedup
    if retrieve_after_dedup is not None:
        meta["retrieve_after_dedup"] = retrieve_after_dedup
    if retrieve_kept is not None:
        meta["retrieve_kept"] = retrieve_kept
    if hybrid_weight is not None:
        meta["hybrid_weight"] = hybrid_weight
    if dedup_dropped is not None:
        meta["dedup_dropped"] = dedup_dropped
    if agent_steps is not None:
        meta["agent_steps"] = agent_steps
    if max_steps is not None:
        meta["max_steps"] = max_steps
    if tool_calls_count is not None:
        meta["tool_calls_count"] = tool_calls_count
    if tools_used is not None:
        meta["tools_used"] = tools_used
    if agent_final_phase is not None:
        meta["agent_final_phase"] = agent_final_phase
    if agent_phase_trace is not None:
        meta["agent_phase_trace"] = agent_phase_trace
    if degraded_to is not None:
        meta["degraded_to"] = degraded_to
    if stop_reason is not None:
        meta["stop_reason"] = stop_reason
    if session_id is not None:
        meta["session_id"] = session_id
    if session_id_sha256_8 is not None:
        meta["session_id_sha256_8"] = session_id_sha256_8
    if history_messages is not None:
        meta["history_messages"] = history_messages
    if history_chars is not None:
        meta["history_chars"] = history_chars
    if session_turns_kept is not None:
        meta["session_turns_kept"] = session_turns_kept
    if session_msgs is not None:
        meta["session_msgs"] = session_msgs
    if session_chars is not None:
        meta["session_chars"] = session_chars
    if session_summarized is not None:
        meta["session_summarized"] = session_summarized
    if session_truncated_msgs is not None:
        meta["session_truncated_msgs"] = session_truncated_msgs
    if citation_guard is not None:
        meta["citation_guard"] = citation_guard
    if citation_invalid_refs is not None:
        meta["citation_invalid_refs"] = citation_invalid_refs
    if citation_missing_for_claims is not None:
        meta["citation_missing_for_claims"] = citation_missing_for_claims
    if citation_refs_used is not None:
        meta["citation_refs_used"] = citation_refs_used
    if leakage_blocked is not None:
        meta["leakage_blocked"] = leakage_blocked
    if agent_trace is not None:
        meta["agent_trace"] = agent_trace
    if max_context_tokens is not None:
        meta["max_context_tokens"] = max_context_tokens
    if context_tokens_used is not None:
        meta["context_tokens_used"] = context_tokens_used
    if max_output_tokens is not None:
        meta["max_output_tokens"] = max_output_tokens
    if budget_compressed is not None:
        meta["budget_compressed"] = budget_compressed
    if cache_hit is not None:
        meta["cache_hit"] = cache_hit
    if cache_miss is not None:
        meta["cache_miss"] = cache_miss
    return meta or None


def error_response_schema() -> set[str]:
    """供测试引用的错误体字段约定。"""
    return {"request_id", "code", "message"}
