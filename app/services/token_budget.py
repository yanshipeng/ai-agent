"""Day18：Token 预算估算与超预算压缩。

==========================================================================
做什么
==========================================================================
- 用「字符启发式」估算 token（中文约 1 字 ≈ 1 token，英文约 4 字 ≈ 1 token）
- 检查 context 是否超 max_context_tokens
- 超预算时：压缩 tool/长文 → 仍超则建议澄清（由调用方决定）

为什么不用真实 tokenizer？
  教学项目少依赖；与 DeepSeek tokenizer 有误差，但对预算门禁足够。
==========================================================================
"""

from __future__ import annotations

import re
from typing import Any

from app.services.conversation import truncate_text

# 默认预算（可被 Settings / 入参覆盖）
DEFAULT_MAX_CONTEXT_TOKENS = 6000
DEFAULT_MAX_OUTPUT_TOKENS = 2048

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """启发式估算 token 数。

    中文按字计；ASCII 约 4 字符 ≈ 1 token。空串返回 0。
    不追求与 DeepSeek tokenizer 一致，只做预算门禁。
    """
    raw = text or ""
    if not raw:
        return 0
    cjk = len(_CJK_RE.findall(raw))
    other = max(0, len(raw) - cjk)
    # 中文按字；非中文按 ~4 chars/token
    return max(1, cjk + (other + 3) // 4)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """估算 messages 总 token（含 role 开销近似）。"""
    total = 0
    for msg in messages:
        total += 4  # 角色/分隔开销
        total += estimate_tokens(str(msg.get("content") or ""))
        # tool_calls 参数
        for tc in msg.get("tool_calls") or []:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            total += estimate_tokens(str(fn.get("name") or ""))
            total += estimate_tokens(str(fn.get("arguments") or ""))
    return total


def compress_messages_for_budget(
    messages: list[dict[str, Any]],
    *,
    max_context_tokens: int,
    tool_max_chars: int = 1200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """超预算时压缩：优先截断 role=tool / 过长 content。

    返回 (新 messages, 统计)。
    """
    before = estimate_messages_tokens(messages)
    if before <= max_context_tokens:
        return list(messages), {
            "compressed": False,
            "context_tokens_before": before,
            "context_tokens_after": before,
        }

    out: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "")
        row = dict(msg)
        content = str(row.get("content") or "")
        if role == "tool" and content:
            row["content"] = truncate_text(content, tool_max_chars)
        elif role in {"user", "assistant"} and len(content) > tool_max_chars * 2:
            # 保留 system；压缩过长 user/assistant
            if role != "system":
                row["content"] = truncate_text(content, tool_max_chars * 2)
        out.append(row)

    after = estimate_messages_tokens(out)
    # 仍超：再砍更狠（只留 system + 最近 2 条非 system）
    if after > max_context_tokens and len(out) > 3:
        head = [m for m in out if str(m.get("role") or "") == "system"][:1]
        tail = [m for m in out if str(m.get("role") or "") != "system"][-2:]
        out = head + tail
        after = estimate_messages_tokens(out)

    return out, {
        "compressed": True,
        "context_tokens_before": before,
        "context_tokens_after": after,
        "still_over_budget": after > max_context_tokens,
    }
