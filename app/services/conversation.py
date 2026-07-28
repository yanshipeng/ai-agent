"""多轮上下文控制：滑窗 + 截断 +（可选）摘要。

==========================================================================
做什么（进模型前 / 落 session 前都可能用到）
==========================================================================
  1) 截断：单条 tool 结果 / 长文限长（SESSION_TOOL_RESULT_MAX_CHARS 等）
  2) 滑窗：只保留最近 N 个 user 轮（SESSION_MAX_TURNS，建议 6–10）
  3) 摘要：总字符超 SESSION_MAX_CHARS 时，把更早对话压成一条 memory

为什么必须控上下文？
  多轮 + tool 轨迹会指数膨胀 → 贵、慢、触达 max_tokens、模型注意力稀释。
  在进模型前裁剪，比事后「模型自己忘掉」更可控。

摘要策略：
  - 默认抽取式（拼 user/assistant 要点），零额外 LLM 成本
  - SESSION_SUMMARY_USE_LLM=true 时才调一次 LLM；失败回落抽取式

memory 形态：
  user:    "[会话记忆/摘要]\\n..."
  assistant: "已理解历史背景，将基于记忆继续回答。"
  这样滑窗逻辑仍按「一轮 user」计数，且 llm/rag 的 plain_chat_history 能保留。
==========================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger, log_event

logger = get_logger(__name__)

EVENT_CONTEXT_COMPACT = "context_compact"

DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_CHARS = 20_000
DEFAULT_TOOL_RESULT_MAX_CHARS = 4_000
DEFAULT_CONTENT_MAX_CHARS = 8_000
MEMORY_USER_PREFIX = "[会话记忆/摘要]\n"
MEMORY_ASSISTANT_ACK = "已理解历史背景，将基于记忆继续回答。"


@dataclass
class CompactStats:
    """上下文压缩统计（写日志 / meta，不含原文）。"""

    input_messages: int = 0
    output_messages: int = 0
    input_chars: int = 0
    output_chars: int = 0
    turns_kept: int = 0
    truncated_msgs: int = 0
    summarized: bool = False
    max_turns: int = DEFAULT_MAX_TURNS
    max_chars: int = DEFAULT_MAX_CHARS


@dataclass
class CompactResult:
    messages: list[dict[str, Any]] = field(default_factory=list)
    stats: CompactStats = field(default_factory=CompactStats)


def _as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)


def message_chars(msg: dict[str, Any]) -> int:
    """估算单条消息字符数（含 tool arguments）。"""
    n = len(_as_text(msg.get("content")))
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                n += len(str(fn.get("name") or ""))
                n += len(_as_text(fn.get("arguments")))
            else:
                n += 64
    return n


def total_chars(messages: list[dict[str, Any]]) -> int:
    return sum(message_chars(m) for m in messages)


def truncate_text(text: str, limit: int) -> str:
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n…[truncated]"
    if limit <= len(marker):
        return text[:limit]
    return text[: limit - len(marker)].rstrip() + marker


def truncate_message(
    msg: dict[str, Any],
    *,
    tool_result_max_chars: int,
    content_max_chars: int,
) -> tuple[dict[str, Any], bool]:
    """按角色截断；返回 (新消息, 是否发生截断)。"""
    out = dict(msg)
    changed = False
    role = str(out.get("role") or "")
    content = _as_text(out.get("content"))
    limit = tool_result_max_chars if role == "tool" else content_max_chars
    if len(content) > limit:
        out["content"] = truncate_text(content, limit)
        changed = True
    # assistant.tool_calls 里的 arguments 也可能很长
    tool_calls = out.get("tool_calls")
    if isinstance(tool_calls, list):
        new_calls = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                new_calls.append(tc)
                continue
            tc2 = dict(tc)
            fn = dict(tc2.get("function") or {})
            args = _as_text(fn.get("arguments"))
            if len(args) > tool_result_max_chars:
                fn["arguments"] = truncate_text(args, tool_result_max_chars)
                changed = True
            tc2["function"] = fn
            new_calls.append(tc2)
        out["tool_calls"] = new_calls
    return out, changed


def truncate_messages(
    messages: list[dict[str, Any]],
    *,
    tool_result_max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
    content_max_chars: int = DEFAULT_CONTENT_MAX_CHARS,
) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    truncated = 0
    for msg in messages:
        m2, changed = truncate_message(
            msg,
            tool_result_max_chars=tool_result_max_chars,
            content_max_chars=content_max_chars,
        )
        if changed:
            truncated += 1
        out.append(m2)
    return out, truncated


def split_user_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """按 user 消息切轮：每一轮从 user 开始，直到下一个 user 之前。"""
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "")
        if role == "user":
            if current:
                turns.append(current)
            current = [msg]
        else:
            if not current:
                # 游离的 system/assistant/tool：挂到独立前缀轮
                current = [msg]
            else:
                current.append(msg)
    if current:
        turns.append(current)
    return turns


def apply_sliding_window(
    messages: list[dict[str, Any]],
    *,
    max_turns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """保留最近 max_turns 个 user 轮；返回 (kept, dropped, turns_kept)。"""
    max_turns = max(1, max_turns)
    turns = split_user_turns(messages)
    if len(turns) <= max_turns:
        kept = [m for t in turns for m in t]
        return kept, [], len(turns)
    dropped_turns = turns[: -max_turns]
    kept_turns = turns[-max_turns:]
    dropped = [m for t in dropped_turns for m in t]
    kept = [m for t in kept_turns for m in t]
    return kept, dropped, len(kept_turns)


def _extractive_summary(messages: list[dict[str, Any]], *, max_chars: int = 1500) -> str:
    """无 LLM 时的兜底摘要：抽取各轮 user/assistant 要点。"""
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        text = _as_text(msg.get("content")).strip()
        if not text or text.startswith(MEMORY_USER_PREFIX.strip()):
            continue
        # 跳过占位 ack
        if text == MEMORY_ASSISTANT_ACK:
            continue
        label = "用户" if role == "user" else "助手"
        parts.append(f"- {label}: {truncate_text(text.replace(chr(10), ' '), 180)}")
    blob = "\n".join(parts) if parts else "（较早对话已省略）"
    return truncate_text(blob, max_chars)


def _llm_summary(
    messages: list[dict[str, Any]],
    *,
    client: Any,
    request_id: str | None,
) -> str | None:
    """可选：用 LLM 总结旧对话；失败返回 None。"""
    if client is None:
        return None
    digest = _extractive_summary(messages, max_chars=6000)
    prompt = (
        "请将以下多轮对话要点总结为不超过 800 字的中文记忆，"
        "保留关键实体、结论与未决问题，不要编造：\n\n"
        f"{digest}"
    )
    try:
        result = client.chat(
            [{"role": "user", "content": prompt}],
            request_id=request_id,
        )
        text = (result.answer or "").strip()
        return text or None
    except Exception:  # noqa: BLE001
        return None


def build_memory_pair(summary: str) -> list[dict[str, Any]]:
    """把摘要变成可进 messages 的一轮 memory。"""
    return [
        {"role": "user", "content": MEMORY_USER_PREFIX + summary.strip()},
        {"role": "assistant", "content": MEMORY_ASSISTANT_ACK},
    ]


def compact_messages(
    messages: list[dict[str, Any]],
    *,
    max_turns: int | None = None,
    max_chars: int | None = None,
    tool_result_max_chars: int | None = None,
    content_max_chars: int | None = None,
    enable_summary: bool | None = None,
    client: Any | None = None,
    request_id: str | None = None,
) -> CompactResult:
    """对历史消息做：截断 → 滑窗 →（可选）超预算摘要。"""
    settings = get_settings()
    max_turns = int(max_turns if max_turns is not None else settings.session_max_turns)
    max_chars = int(max_chars if max_chars is not None else settings.session_max_chars)
    tool_limit = int(
        tool_result_max_chars
        if tool_result_max_chars is not None
        else settings.session_tool_result_max_chars
    )
    content_limit = int(
        content_max_chars
        if content_max_chars is not None
        else settings.session_content_max_chars
    )
    do_summary = (
        settings.session_enable_summary if enable_summary is None else bool(enable_summary)
    )

    stats = CompactStats(
        input_messages=len(messages),
        input_chars=total_chars(messages),
        max_turns=max_turns,
        max_chars=max_chars,
    )

    # 1) 先截断单条
    work, truncated_n = truncate_messages(
        messages,
        tool_result_max_chars=tool_limit,
        content_max_chars=content_limit,
    )
    stats.truncated_msgs = truncated_n

    # 2) 滑窗
    kept, dropped, turns_kept = apply_sliding_window(work, max_turns=max_turns)
    stats.turns_kept = turns_kept

    # 3) 超字符预算：把滑掉的旧对话压成 memory；kept 仍超则继续丢最旧轮
    if do_summary and dropped:
        summary = None
        if client is not None and settings.session_summary_use_llm:
            summary = _llm_summary(dropped, client=client, request_id=request_id)
        if not summary:
            summary = _extractive_summary(dropped)
        kept = build_memory_pair(summary) + kept
        stats.summarized = True

    while total_chars(kept) > max_chars:
        turns = split_user_turns(kept)
        if len(turns) <= 1:
            # 只剩一轮：强制更狠截断
            hard_tool = max(500, min(tool_limit, max_chars // 4))
            hard_content = max(500, min(content_limit, max_chars // 2))
            kept, extra_trunc = truncate_messages(
                kept,
                tool_result_max_chars=hard_tool,
                content_max_chars=hard_content,
            )
            stats.truncated_msgs += extra_trunc
            break

        drop_idx = 0
        first_content = _as_text(turns[0][0].get("content")) if turns[0] else ""
        if (
            len(turns) >= 3
            and first_content.startswith(MEMORY_USER_PREFIX)
        ):
            drop_idx = 1  # 保留 memory，丢掉次旧一轮

        dropped_more = turns.pop(drop_idx)
        if do_summary:
            extra = _extractive_summary(dropped_more)
            if turns and _as_text(turns[0][0].get("content")).startswith(MEMORY_USER_PREFIX):
                old_body = _as_text(turns[0][0].get("content")).replace(
                    MEMORY_USER_PREFIX, "", 1
                )
                merged = truncate_text((old_body + "\n" + extra).strip(), content_limit)
                turns[0] = build_memory_pair(merged)
            else:
                turns.insert(0, build_memory_pair(extra))
            stats.summarized = True

        kept = [m for t in turns for m in t]
        kept, extra_trunc = truncate_messages(
            kept,
            tool_result_max_chars=tool_limit,
            content_max_chars=content_limit,
        )
        stats.truncated_msgs += extra_trunc

    # 最终再截断一次兜底
    kept, extra_trunc = truncate_messages(
        kept,
        tool_result_max_chars=tool_limit,
        content_max_chars=content_limit,
    )
    stats.truncated_msgs += extra_trunc
    stats.output_messages = len(kept)
    stats.output_chars = total_chars(kept)
    stats.turns_kept = len(split_user_turns(kept))

    log_event(
        logger,
        EVENT_CONTEXT_COMPACT,
        request_id=request_id,
        input_messages=stats.input_messages,
        output_messages=stats.output_messages,
        input_chars=stats.input_chars,
        output_chars=stats.output_chars,
        turns_kept=stats.turns_kept,
        truncated_msgs=stats.truncated_msgs,
        summarized=stats.summarized,
        max_turns=stats.max_turns,
        max_chars=stats.max_chars,
        hint=(
            f"上下文压缩：msgs {stats.input_messages}→{stats.output_messages}，"
            f"chars {stats.input_chars}→{stats.output_chars}，"
            f"turns={stats.turns_kept}，summarized={stats.summarized}"
        ),
    )
    return CompactResult(messages=kept, stats=stats)


def strip_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """会话存储不保存主 system prompt（每次请求重新注入）。"""
    return [m for m in messages if str(m.get("role") or "") != "system"]


def plain_chat_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """llm/rag 注入用：只保留 user/assistant（含 memory），去掉 tool 轨迹。"""
    return [
        m
        for m in messages
        if str(m.get("role") or "") in {"user", "assistant"}
    ]


def messages_for_storage(
    messages: list[dict[str, Any]],
    *,
    tool_result_max_chars: int | None = None,
    content_max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """落 session 前：去 system + 截断。"""
    settings = get_settings()
    cleaned = strip_system_messages(messages)
    out, _ = truncate_messages(
        cleaned,
        tool_result_max_chars=int(
            tool_result_max_chars
            if tool_result_max_chars is not None
            else settings.session_tool_result_max_chars
        ),
        content_max_chars=int(
            content_max_chars
            if content_max_chars is not None
            else settings.session_content_max_chars
        ),
    )
    return out
