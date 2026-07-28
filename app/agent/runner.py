"""Agent 状态机：Plan → Act → Observe → Final。

==========================================================================
做什么
==========================================================================
把「模型调工具 → 执行 → 回传 → 再决策」固化成显式状态机，而不是
在一个大 for 循环里混写所有分支。

状态含义：
  Plan    —— 调 LLM（可带 tools），决定「再调工具」还是「终答」
  Act     —— 执行本轮 tool_calls（走 execute_tool）
  Observe —— 把工具结果以 role=tool 写回 messages（并截断过长结果）
  Final   —— 产出最终回答（含降级 RAG / 澄清 / 超时兜底）

为什么用状态机？
  1) 每步职责清晰，日志可按 agent_phase 回放
  2) max_steps / max_total_time_ms 在相位切换点统一检查
  3) 超限策略可配置：AGENT_ON_MAX_STEPS = rag | clarify | error
  4) 任意路径都写出 stop_reason，方便评测与排障

stop_reason（六选一，写入日志 + requests.jsonl + meta）：
  final_answer | clarify | degraded_to_rag | max_steps | timeout | upstream_error

多轮：
  history_messages 拼在 system 之后、本轮 user 之前；
  结束后 session_messages = 去 system + 截断，供 api 写入 session_store。
==========================================================================
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.agent.tools import (
    ASK_MODE_AGENT,
    execute_tool,
    openai_tools_schema,
)
from app.core.config import get_settings
from app.core.logging import get_logger, log_event
from app.kb.rag import run_rag_retrieve
from app.services.conversation import messages_for_storage, truncate_text
from app.services.llm_client import LLMClient, LLMError, LLMTurnResult, ToolCall

logger = get_logger(__name__)

EVENT_AGENT_STEP = "agent_step"
EVENT_AGENT_PHASE = "agent_phase"
EVENT_AGENT_STOP = "agent_stop"
EVENT_TOOL_CALL_START = "tool_call_start"
EVENT_TOOL_CALL_END = "tool_call_end"

AGENT_MAX_STEPS = "AGENT_MAX_STEPS"
AGENT_NO_ANSWER = "AGENT_NO_ANSWER"
AGENT_TIMEOUT = "AGENT_TIMEOUT"

DEFAULT_MAX_STEPS = 5
DEFAULT_MAX_TOTAL_TIME_MS = 20_000
# 超步数策略：rag=降级检索增强；clarify=澄清；error=硬终止 max_steps
ON_MAX_STEPS_RAG = "rag"
ON_MAX_STEPS_CLARIFY = "clarify"
ON_MAX_STEPS_ERROR = "error"
DEFAULT_ON_MAX_STEPS = ON_MAX_STEPS_RAG

# 验收：终止原因（写入日志 + requests.jsonl）
STOP_FINAL_ANSWER = "final_answer"
STOP_CLARIFY = "clarify"
STOP_DEGRADED_TO_RAG = "degraded_to_rag"
STOP_MAX_STEPS = "max_steps"
STOP_TIMEOUT = "timeout"
STOP_UPSTREAM_ERROR = "upstream_error"
STOP_REASONS = frozenset(
    {
        STOP_FINAL_ANSWER,
        STOP_CLARIFY,
        STOP_DEGRADED_TO_RAG,
        STOP_MAX_STEPS,
        STOP_TIMEOUT,
        STOP_UPSTREAM_ERROR,
    }
)
AGENT_SYSTEM_PROMPT = """你是「稳定性排障」助手，必须通过工具查阅本地知识库后再回答。

可用工具：
1) kb_search(query, top_k?)：检索相关片段（先用这个）
2) kb_get_chunk(chunk_id)：按 id 取片段全文（需要细节时再用）

硬性规则：
- 涉及 Android/ANR/OOM/Crash/卡顿/稳定性排查时，必须先调用 kb_search，禁止仅凭自身知识编造步骤。
- 若检索结果不足，可再 kb_search 换关键词，或 kb_get_chunk 读全文；仍不足则明确说明无法确定。
- 最终回答用中文，简洁可执行；提到的关键事实尽量带上来源标题。
- 不要假装已经调用过工具；需要证据时就发起 tool call。"""


class AgentPhase(str, Enum):
    """状态机相位。"""

    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    FINAL = "final"


@dataclass
class AgentResult:
    """Agent 状态机结束后的统一结果。"""

    answer: str
    model: str
    finish_reason: str | None
    latency_ms: int
    usage: dict[str, Any] | None = None
    fallback: bool = False
    error_code: str | None = None
    retry_count: int = 0
    agent_steps: int = 0
    max_steps: int = 0
    tool_calls_count: int = 0
    tools_used: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    # 硬失败：路由映射为 HTTP 错误体
    http_error_code: str | None = None
    # 状态机可观测字段
    final_phase: str = AgentPhase.FINAL.value
    phase_trace: list[str] = field(default_factory=list)
    degraded_to: str | None = None  # rag / clarify / timeout（兼容旧字段）
    # 验收字段：终止原因（六选一）
    stop_reason: str = STOP_FINAL_ANSWER
    retrieve_ms: int | None = None
    # 多轮：本轮结束后可写入 session 的消息（已去 system、已截断）
    session_messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _AgentRuntime:
    """单次请求的可变运行时状态。"""

    query: str
    client: LLMClient
    request_id: str | None
    index_dir: Path | str | None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_steps: int
    max_total_time_ms: int
    on_max_steps: str
    tool_timeout_seconds: float | None
    started: float
    phase: AgentPhase = AgentPhase.PLAN
    plan_rounds: int = 0
    pending_tool_calls: list[ToolCall] = field(default_factory=list)
    pending_observations: list[tuple[ToolCall, dict[str, Any]]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tool_calls_count: int = 0
    citations: list[dict[str, Any]] = field(default_factory=list)
    seen_chunk_ids: set[str] = field(default_factory=set)
    total_usage: dict[str, Any] | None = None
    total_retry: int = 0
    last_model: str = ""
    last_finish: str | None = None
    llm_latency_ms: int = 0
    phase_trace: list[str] = field(default_factory=list)
    retrieve_ms: int | None = None


def _usage_add(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    if not a and not b:
        return None
    out: dict[str, Any] = dict(a or {})
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        av = int(out.get(key) or 0)
        bv = int((b or {}).get(key) or 0)
        if av or bv:
            out[key] = av + bv
    return out or None


def _elapsed_ms(rt: _AgentRuntime) -> int:
    return int((time.perf_counter() - rt.started) * 1000)


def _is_timed_out(rt: _AgentRuntime) -> bool:
    return _elapsed_ms(rt) >= rt.max_total_time_ms


def _enter_phase(rt: _AgentRuntime, phase: AgentPhase, *, hint: str) -> None:
    rt.phase = phase
    rt.phase_trace.append(phase.value)
    log_event(
        logger,
        EVENT_AGENT_PHASE,
        request_id=rt.request_id,
        mode=ASK_MODE_AGENT,
        agent_phase=phase.value,
        agent_steps=rt.plan_rounds,
        agent_max_steps=rt.max_steps,
        elapsed_ms=_elapsed_ms(rt),
        max_total_time_ms=rt.max_total_time_ms,
        tool_calls_count=rt.tool_calls_count,
        hint=hint,
    )


def _assistant_message_from_turn(turn: LLMTurnResult) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": turn.content if turn.content else None,
    }
    if turn.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in turn.tool_calls
        ]
    return msg


def _merge_citations(rt: _AgentRuntime, payload: dict[str, Any]) -> None:
    if payload.get("results"):
        for hit in payload["results"]:
            cid = str(hit.get("chunk_id") or "")
            if cid and cid in rt.seen_chunk_ids:
                continue
            if cid:
                rt.seen_chunk_ids.add(cid)
            rt.citations.append(
                {
                    "ref_id": len(rt.citations) + 1,
                    "chunk_id": hit.get("chunk_id"),
                    "url": hit.get("url"),
                    "title": hit.get("title"),
                    "section_path": hit.get("section_path") or "",
                    "is_code": bool(hit.get("is_code")),
                }
            )
    chunk = payload.get("chunk")
    if isinstance(chunk, dict) and chunk.get("chunk_id"):
        cid = str(chunk.get("chunk_id") or "")
        if cid and cid not in rt.seen_chunk_ids:
            rt.seen_chunk_ids.add(cid)
            rt.citations.append(
                {
                    "ref_id": len(rt.citations) + 1,
                    "chunk_id": chunk.get("chunk_id"),
                    "url": chunk.get("url"),
                    "title": chunk.get("title"),
                    "section_path": chunk.get("section_path") or "",
                    "is_code": bool(chunk.get("is_code")),
                }
            )


def _clamp_steps(rt: _AgentRuntime) -> int:
    """验收：任意请求 agent_steps <= max_steps。"""
    return min(max(0, rt.plan_rounds), rt.max_steps)


def _log_stop(rt: _AgentRuntime, *, stop_reason: str, agent_steps: int) -> None:
    log_event(
        logger,
        EVENT_AGENT_STOP,
        request_id=rt.request_id,
        mode=ASK_MODE_AGENT,
        stop_reason=stop_reason,
        agent_steps=agent_steps,
        max_steps=rt.max_steps,
        tool_calls_count=rt.tool_calls_count,
        tools_used=list(dict.fromkeys(rt.tools_used)),
        agent_phase_trace=list(rt.phase_trace),
        elapsed_ms=_elapsed_ms(rt),
        hint=(
            f"Agent 终止：stop_reason={stop_reason}，"
            f"agent_steps={agent_steps}/{rt.max_steps}"
        ),
    )


def _base_result(rt: _AgentRuntime, *, stop_reason: str, **kwargs: Any) -> AgentResult:
    if stop_reason not in STOP_REASONS:
        raise ValueError(f"invalid stop_reason: {stop_reason}")
    steps = _clamp_steps(rt)
    assert steps <= rt.max_steps  # 验收硬约束
    defaults = dict(
        model=rt.last_model or get_settings().llm_model,
        latency_ms=_elapsed_ms(rt),
        usage=rt.total_usage,
        retry_count=rt.total_retry,
        agent_steps=steps,
        max_steps=rt.max_steps,
        tool_calls_count=rt.tool_calls_count,
        tools_used=list(dict.fromkeys(rt.tools_used)),
        citations=list(rt.citations),
        final_phase=rt.phase.value,
        phase_trace=list(rt.phase_trace),
        retrieve_ms=rt.retrieve_ms,
        stop_reason=stop_reason,
    )
    defaults.update(kwargs)
    defaults["agent_steps"] = min(int(defaults.get("agent_steps", steps)), rt.max_steps)
    defaults["stop_reason"] = stop_reason
    _log_stop(rt, stop_reason=stop_reason, agent_steps=int(defaults["agent_steps"]))
    return AgentResult(**defaults)


def _timeout_fallback(rt: _AgentRuntime) -> AgentResult:
    """总耗时超限：HTTP 仍可由路由返回 200 + 兜底文案（带 request_id）。"""
    _enter_phase(
        rt,
        AgentPhase.FINAL,
        hint=f"总耗时超过 {rt.max_total_time_ms}ms，进入超时兜底",
    )
    rid = rt.request_id or "unknown"
    answer = (
        f"抱歉，本次 Agent 处理超时（request_id={rid}，"
        f"已用时 {_elapsed_ms(rt)}ms / 上限 {rt.max_total_time_ms}ms）。"
        "请缩小问题范围后重试，或改用 mode=rag。"
    )
    return _base_result(
        rt,
        stop_reason=STOP_TIMEOUT,
        answer=answer,
        finish_reason="timeout",
        latency_ms=max(_elapsed_ms(rt), rt.llm_latency_ms),
        fallback=True,
        error_code=AGENT_TIMEOUT,
        degraded_to="timeout",
        http_error_code=None,
    )


def _clarify_fallback(rt: _AgentRuntime) -> AgentResult:
    _enter_phase(rt, AgentPhase.FINAL, hint="超过 max_steps，返回澄清提问")
    rid = rt.request_id or "unknown"
    answer = (
        f"目前证据仍不足，无法给出可靠排障结论（request_id={rid}，"
        f"已用 {_clamp_steps(rt)}/{rt.max_steps} 步）。"
        "请补充：机型/系统版本、复现路径、是否必现、相关日志片段（如 traces/tombstone），"
        "或改用更具体的关键词再问一次。"
    )
    return _base_result(
        rt,
        stop_reason=STOP_CLARIFY,
        answer=answer,
        finish_reason="clarify",
        latency_ms=max(_elapsed_ms(rt), rt.llm_latency_ms),
        fallback=True,
        error_code=AGENT_MAX_STEPS,
        degraded_to="clarify",
        http_error_code=None,
    )


def _max_steps_hard_stop(rt: _AgentRuntime) -> AgentResult:
    """超步硬终止（stop_reason=max_steps），不降级 RAG、不澄清。"""
    _enter_phase(rt, AgentPhase.FINAL, hint="超过 max_steps，硬终止")
    rid = rt.request_id or "unknown"
    answer = (
        f"已达到 Agent 最大步数上限（request_id={rid}，"
        f"{_clamp_steps(rt)}/{rt.max_steps}）。请缩小问题或改用 mode=rag。"
    )
    return _base_result(
        rt,
        stop_reason=STOP_MAX_STEPS,
        answer=answer,
        finish_reason="max_steps",
        latency_ms=max(_elapsed_ms(rt), rt.llm_latency_ms),
        fallback=True,
        error_code=AGENT_MAX_STEPS,
        degraded_to=None,
        http_error_code=None,
    )


def _degrade_to_rag(rt: _AgentRuntime) -> AgentResult:
    """超过 max_steps：降级到 RAG（固定检索 + 单次 LLM，不再走 tools）。"""
    _enter_phase(
        rt,
        AgentPhase.FINAL,
        hint="超过 max_steps，降级到 RAG（不再发起 tool_calls）",
    )
    if _is_timed_out(rt):
        return _timeout_fallback(rt)

    settings = get_settings()
    index_dir = rt.index_dir or settings.kb_index_dir
    top_k = int(settings.rag_top_k)
    try:
        pack = run_rag_retrieve(rt.query, top_k=top_k, index_dir=index_dir)
    except FileNotFoundError:
        # 步数已尽且无法降级 → 记为 max_steps（不是 upstream）
        result = _max_steps_hard_stop(rt)
        result.answer = (
            f"Agent 步数已用尽且知识库索引不可用（request_id={rt.request_id or 'unknown'}）。"
            "请先运行 python scripts/build_kb_index.py，或补充更多现场信息后再问。"
        )
        return result

    rt.retrieve_ms = int(pack["retrieve_ms"])
    rt.citations = list(pack["citations"])
    try:
        result = rt.client.chat(pack["messages"], request_id=rt.request_id)
    except LLMError as exc:
        return _base_result(
            rt,
            stop_reason=STOP_UPSTREAM_ERROR,
            answer="",
            finish_reason=None,
            error_code=exc.code,
            http_error_code=exc.code,
            degraded_to="rag",
        )

    rt.total_usage = _usage_add(rt.total_usage, result.usage)
    rt.total_retry += int(getattr(result, "retry_count", 0) or 0)
    rt.llm_latency_ms += int(getattr(result, "latency_ms", 0) or 0)
    return _base_result(
        rt,
        stop_reason=STOP_DEGRADED_TO_RAG,
        answer=str(getattr(result, "answer", "") or ""),
        model=str(getattr(result, "model", None) or rt.last_model),
        finish_reason="degraded_rag",
        latency_ms=max(_elapsed_ms(rt), rt.llm_latency_ms),
        usage=rt.total_usage,
        fallback=True,
        error_code=AGENT_MAX_STEPS,
        degraded_to="rag",
        http_error_code=None,
        retrieve_ms=rt.retrieve_ms,
    )


def _on_max_steps(rt: _AgentRuntime) -> AgentResult:
    policy = (rt.on_max_steps or DEFAULT_ON_MAX_STEPS).strip().lower()
    if policy == ON_MAX_STEPS_CLARIFY:
        return _clarify_fallback(rt)
    if policy == ON_MAX_STEPS_ERROR:
        return _max_steps_hard_stop(rt)
    return _degrade_to_rag(rt)


def _phase_plan(rt: _AgentRuntime) -> AgentResult | None:
    """Plan：调 LLM（带 tools），决定「再调工具」还是「终答」。

    返回非 None = 本相位已收口（超时 / 超步 / 上游错误 / 已有终答）。
    返回 None 且 pending_tool_calls 非空 → 进入 Act。
    """
    if _is_timed_out(rt):
        return _timeout_fallback(rt)
    if rt.plan_rounds >= rt.max_steps:
        return _on_max_steps(rt)

    _enter_phase(
        rt,
        AgentPhase.PLAN,
        hint=(
            f"Plan：第 {rt.plan_rounds + 1}/{rt.max_steps} 轮决策"
            f"（可发起 tool_calls 或直接终答）"
        ),
    )
    log_event(
        logger,
        EVENT_AGENT_STEP,
        request_id=rt.request_id,
        mode=ASK_MODE_AGENT,
        agent_step=rt.plan_rounds + 1,
        agent_max_steps=rt.max_steps,
        agent_phase=AgentPhase.PLAN.value,
        tool_calls_count=rt.tool_calls_count,
        tools_used=rt.tools_used,
        elapsed_ms=_elapsed_ms(rt),
        hint=f"Agent Plan 第 {rt.plan_rounds + 1}/{rt.max_steps} 步：调用大模型",
    )

    try:
        turn = rt.client.chat_turn(
            rt.messages,
            tools=rt.tools,
            request_id=rt.request_id,
            mode=ASK_MODE_AGENT,
            agent_step=rt.plan_rounds + 1,
        )
    except LLMError as exc:
        return _base_result(
            rt,
            stop_reason=STOP_UPSTREAM_ERROR,
            answer="",
            finish_reason=None,
            error_code=exc.code,
            http_error_code=exc.code,
        )

    rt.plan_rounds += 1
    # 防御：永不让 plan_rounds 超过 max_steps（正常路径在调用前已检查）
    if rt.plan_rounds > rt.max_steps:
        rt.plan_rounds = rt.max_steps
        return _on_max_steps(rt)

    rt.last_model = turn.model
    rt.last_finish = turn.finish_reason
    rt.total_retry += turn.retry_count
    rt.total_usage = _usage_add(rt.total_usage, turn.usage)
    rt.llm_latency_ms += turn.latency_ms

    if turn.tool_calls:
        rt.messages.append(_assistant_message_from_turn(turn))
        rt.pending_tool_calls = list(turn.tool_calls)
        rt.phase = AgentPhase.ACT
        return None

    answer = (turn.content or "").strip()
    if not answer:
        return _base_result(
            rt,
            stop_reason=STOP_UPSTREAM_ERROR,
            answer="",
            finish_reason=rt.last_finish,
            error_code=AGENT_NO_ANSWER,
            http_error_code=AGENT_NO_ANSWER,
        )

    _enter_phase(rt, AgentPhase.FINAL, hint="Plan 无 tool_calls，进入 Final 终答")
    return _base_result(
        rt,
        stop_reason=STOP_FINAL_ANSWER,
        answer=answer,
        finish_reason=rt.last_finish or "stop",
        latency_ms=max(_elapsed_ms(rt), rt.llm_latency_ms),
        http_error_code=None,
    )


def _phase_act(rt: _AgentRuntime) -> AgentResult | None:
    """Act：逐个 execute_tool；结果先放 pending_observations，由 Observe 写回。

    同时从成功的 kb_search / kb_get_chunk 结果里收集 citations（去重 chunk_id）。
    工具失败不中断整单：把 error_code 写进 tool 结果，让模型下一轮 Plan 自行处理。
    """
    if _is_timed_out(rt):
        return _timeout_fallback(rt)

    _enter_phase(
        rt,
        AgentPhase.ACT,
        hint=f"Act：执行 {len(rt.pending_tool_calls)} 个 tool_calls",
    )

    observed: list[tuple[ToolCall, dict[str, Any]]] = []
    for tc in rt.pending_tool_calls:
        if _is_timed_out(rt):
            return _timeout_fallback(rt)

        rt.tool_calls_count += 1
        rt.tools_used.append(tc.name)
        log_event(
            logger,
            EVENT_TOOL_CALL_START,
            request_id=rt.request_id,
            mode=ASK_MODE_AGENT,
            agent_step=rt.plan_rounds,
            agent_phase=AgentPhase.ACT.value,
            tool_name=tc.name,
            tool_call_id=tc.id,
            tool_calls_count=rt.tool_calls_count,
            hint=f"Act：执行真实 tool_calls → {tc.name}",
        )
        result = execute_tool(
            tc.name,
            tc.arguments,
            index_dir=rt.index_dir,
            timeout_seconds=rt.tool_timeout_seconds,
        )
        err_code = None if result.get("ok") else result.get("error_code")
        log_event(
            logger,
            EVENT_TOOL_CALL_END,
            request_id=rt.request_id,
            mode=ASK_MODE_AGENT,
            agent_step=rt.plan_rounds,
            agent_phase=AgentPhase.ACT.value,
            tool_name=tc.name,
            tool_call_id=tc.id,
            ok=bool(result.get("ok")),
            error_code=err_code,
            hint=(
                f"工具 {tc.name} 执行完成：ok={bool(result.get('ok'))}"
                + (f"，error_code={err_code}" if err_code else "")
            ),
        )
        if result.get("ok"):
            _merge_citations(rt, result)
        observed.append((tc, result))

    rt.pending_observations = observed
    rt.pending_tool_calls = []
    rt.phase = AgentPhase.OBSERVE
    return None


def _phase_observe(rt: _AgentRuntime) -> AgentResult | None:
    """Observe：把 Act 的工具结果以 role=tool 写回 messages，再回到 Plan。

    必须带上 tool_call_id，与上一轮 assistant.tool_calls[].id 对齐，
    否则上游会拒请求。结果过长时按 SESSION_TOOL_RESULT_MAX_CHARS 截断。
    """
    if _is_timed_out(rt):
        return _timeout_fallback(rt)

    _enter_phase(
        rt,
        AgentPhase.OBSERVE,
        hint=f"Observe：回传 {len(rt.pending_observations)} 条工具结果",
    )

    for tc, result in rt.pending_observations:
        raw = json.dumps(result, ensure_ascii=False)
        limit = int(get_settings().session_tool_result_max_chars)
        rt.messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": truncate_text(raw, limit),
            }
        )

    rt.pending_observations = []
    rt.phase = AgentPhase.PLAN
    return None


def run_agent_loop(
    query: str,
    *,
    client: LLMClient,
    request_id: str | None = None,
    index_dir: Path | str | None = None,
    max_steps: int | None = None,
    max_total_time_ms: int | None = None,
    on_max_steps: str | None = None,
    tool_timeout_seconds: float | None = None,
    history_messages: list[dict[str, Any]] | None = None,
) -> AgentResult:
    """状态机主循环：Plan → Act → Observe → Plan … → Final。

    参数要点：
      max_steps          —— Plan 轮数上限（默认 AGENT_MAX_STEPS）
      max_total_time_ms  —— 整单墙钟上限（默认 20s）
      on_max_steps       —— rag | clarify | error
      history_messages   —— 同 session 历史（不含本轮 system），插在 system 后

    返回 AgentResult：answer + 可观测字段 + session_messages（供落 session）。
    """
    settings = get_settings()
    steps_limit = max_steps if max_steps is not None else int(settings.agent_max_steps)
    steps_limit = max(1, steps_limit)
    time_limit = (
        max_total_time_ms
        if max_total_time_ms is not None
        else int(settings.agent_max_total_time_ms)
    )
    # 允许测试用更短预算；生产建议 15000–30000
    time_limit = max(50, time_limit)
    policy = (on_max_steps or settings.agent_on_max_steps or DEFAULT_ON_MAX_STEPS).lower()
    if policy not in {ON_MAX_STEPS_RAG, ON_MAX_STEPS_CLARIFY, ON_MAX_STEPS_ERROR}:
        policy = DEFAULT_ON_MAX_STEPS

    seed_messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
    ]
    if history_messages:
        seed_messages.extend(history_messages)
    seed_messages.append({"role": "user", "content": query.strip()})

    rt = _AgentRuntime(
        query=query.strip(),
        client=client,
        request_id=request_id,
        index_dir=index_dir,
        messages=seed_messages,
        tools=openai_tools_schema(),
        max_steps=steps_limit,
        max_total_time_ms=time_limit,
        on_max_steps=policy,
        tool_timeout_seconds=tool_timeout_seconds,
        started=time.perf_counter(),
        last_model=settings.llm_model,
    )

    # 安全上限：防止状态异常死循环（正常由 max_steps / timeout 退出）
    safety = steps_limit * 4 + 4
    for _ in range(safety):
        if rt.phase == AgentPhase.PLAN:
            done = _phase_plan(rt)
            if done is not None:
                done.session_messages = messages_for_storage(rt.messages)
                return done
            continue
        if rt.phase == AgentPhase.ACT:
            done = _phase_act(rt)
            if done is not None:
                done.session_messages = messages_for_storage(rt.messages)
                return done
            continue
        if rt.phase == AgentPhase.OBSERVE:
            done = _phase_observe(rt)
            if done is not None:
                done.session_messages = messages_for_storage(rt.messages)
                return done
            continue
        if rt.phase == AgentPhase.FINAL:
            break

    # 理论上不应落到这里；兜底为超步数策略
    done = _on_max_steps(rt)
    done.session_messages = messages_for_storage(rt.messages)
    return done
