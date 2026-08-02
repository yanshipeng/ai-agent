"""请求指标落盘：每次请求结束追加 1 行 JSON 到 requests.jsonl。

==========================================================================
做什么 / 为什么不跟「事件日志」混在一起
==========================================================================
- 事件日志（stdout JSON）：看过程（retrieve_start → llm_call_end …）
- requests.jsonl：看结果指标（一行一请求，方便 stats / 评测对比）

为什么不写 query / answer 原文？
  隐私与体积；用 query_len + query_sha256_8 足够关联排障。

为什么写入失败只打日志、不抛异常？
  指标是旁路；不能因为磁盘满了就让用户问答失败。

RAG 最小字段：mode / top_k / retrieve_ms / context_chunks / citations_count
  —— Day10 评测与 A/B 对比直接读这些列。
Day16 检索可观测：retrieve_candidates / before_dedup / after_dedup /
                 retrieve_kept / hybrid_weight / dedup_dropped
Day18：agent_trace / token budget / cache_hit|miss；另可写 traces.jsonl
==========================================================================
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger, query_sha256_8 as hash_query_sha256_8

logger = get_logger(__name__)

# 多 worker / 多线程追加时串行化写文件
_lock = threading.Lock()


def _default_path() -> Path:
    """从配置读取 JSONL 路径（REQUESTS_JSONL_PATH）。"""
    return Path(get_settings().requests_jsonl_path)


def _default_traces_path() -> Path:
    """Day18：逐步 trace 落盘路径（TRACES_JSONL_PATH）。"""
    return Path(get_settings().traces_jsonl_path)


def append_request_metric(
    record: dict[str, Any],
    *,
    path: str | Path | None = None,
) -> None:
    """将一条请求指标追加写入 JSONL（1 行 = 1 请求）。

    - 自动丢弃值为 None 的字段
    - 若调用方未提供 ts，自动补 UTC ISO 时间
    """
    payload = {k: v for k, v in record.items() if v is not None}
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())

    target = Path(path) if path is not None else _default_path()
    line = json.dumps(payload, ensure_ascii=False, default=str)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with target.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
    except OSError as exc:
        logger.error(
            "metrics_write_failed",
            extra={"event": "metrics_write_failed", "path": str(target), "error": str(exc)},
        )


def append_trace_metric(
    record: dict[str, Any],
    *,
    path: str | Path | None = None,
) -> None:
    """将一条请求的 agent_trace 追加到 traces.jsonl（1 行 = 1 请求）。"""
    payload = {k: v for k, v in record.items() if v is not None}
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
    target = Path(path) if path is not None else _default_traces_path()
    line = json.dumps(payload, ensure_ascii=False, default=str)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with target.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
    except OSError as exc:
        logger.error(
            "traces_write_failed",
            extra={"event": "traces_write_failed", "path": str(target), "error": str(exc)},
        )


def _extract_tokens(usage: dict[str, Any] | None) -> dict[str, int]:
    """从 usage 中提取可落盘的 token 计数字段。"""
    if not usage:
        return {}
    tokens: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            tokens[key] = value
    return tokens


def build_ask_metric(
    *,
    request_id: str,
    path: str = "/ask",
    ok: bool,
    status_code: int,
    latency_ms_total: int | None = None,
    latency_ms_llm: int | None = None,
    llm_model: str | None = None,
    retry_count: int | None = None,
    finish_reason: str | None = None,
    error_code: str | None = None,
    query: str | None = None,
    query_len: int | None = None,
    query_sha256_8: str | None = None,
    usage: dict[str, Any] | None = None,
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
    session_id: str | None = None,
    session_id_sha256_8: str | None = None,
    history_messages: int | None = None,
    history_chars: int | None = None,
    session_turns_kept: int | None = None,
    session_msgs: int | None = None,
    session_chars: int | None = None,
    session_summarized: bool | None = None,
    session_truncated_msgs: int | None = None,
    agent_trace: list[dict[str, Any]] | None = None,
    max_context_tokens: int | None = None,
    context_tokens_used: int | None = None,
    max_output_tokens: int | None = None,
    budget_compressed: bool | None = None,
    cache_hit: int | None = None,
    cache_miss: int | None = None,
) -> dict[str, Any]:
    """构造 /ask 请求结束指标记录（不含 query/answer 原文）。

    若传入 query，可自动补齐 query_len / query_sha256_8（仍不会把原文写入 record）。
    RAG 字段：mode / top_k / retrieve_ms / context_chunks / citations_count
             + retrieve_candidates / before_dedup / after_dedup /
               retrieve_kept / hybrid_weight / dedup_dropped。
    Agent 字段：agent_steps / max_steps / tool_calls_count / tools_used /
               agent_final_phase / agent_phase_trace / degraded_to / stop_reason
               + agent_trace / token budget。
    Day18：cache_hit / cache_miss。
    Session 字段：session_id / history_messages / history_chars（条数与字符，不含消息原文）
               + session_id_sha256_8 / 压缩统计。
    """
    resolved_len = query_len
    resolved_hash = query_sha256_8
    if query is not None:
        if resolved_len is None:
            resolved_len = len(query)
        if resolved_hash is None and query:
            resolved_hash = hash_query_sha256_8(query)

    sid = (session_id or "").strip() or None

    record: dict[str, Any] = {
        "request_id": request_id,
        "path": path,
        "ok": ok,
        "status_code": status_code,
        "latency_ms_total": latency_ms_total,
        "latency_ms_llm": latency_ms_llm,
        "llm_model": llm_model,
        "retry_count": retry_count,
        "finish_reason": finish_reason,
        "error_code": error_code,
        "query_len": resolved_len,
        "query_sha256_8": resolved_hash,
        "mode": mode,
        "top_k": top_k,
        "retrieve_ms": retrieve_ms,
        "context_chunks": context_chunks,
        "citations_count": citations_count,
        "retrieve_candidates": retrieve_candidates,
        "retrieve_before_dedup": retrieve_before_dedup,
        "retrieve_after_dedup": retrieve_after_dedup,
        "retrieve_kept": retrieve_kept,
        "hybrid_weight": hybrid_weight,
        "dedup_dropped": dedup_dropped,
        "agent_steps": agent_steps,
        "max_steps": max_steps,
        "tool_calls_count": tool_calls_count,
        "tools_used": tools_used,
        "agent_final_phase": agent_final_phase,
        "agent_phase_trace": agent_phase_trace,
        "degraded_to": degraded_to,
        "stop_reason": stop_reason,
        "agent_trace": agent_trace,
        "max_context_tokens": max_context_tokens,
        "context_tokens_used": context_tokens_used,
        "max_output_tokens": max_output_tokens,
        "budget_compressed": budget_compressed,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "session_id": sid,
        "session_id_sha256_8": session_id_sha256_8,
        "history_messages": history_messages,
        "history_chars": history_chars,
        "session_turns_kept": session_turns_kept,
        "session_msgs": session_msgs,
        "session_chars": session_chars,
        "session_summarized": session_summarized,
        "session_truncated_msgs": session_truncated_msgs,
    }
    record.update(_extract_tokens(usage))
    return record
