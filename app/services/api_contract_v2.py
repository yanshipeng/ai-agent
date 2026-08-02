"""Day21：API Contract v2 — 三模式统一 meta 核心字段。

==========================================================================
做什么
==========================================================================
任何 mode（llm / rag / agent）成功响应的 meta 必须具备：
  model / mode / latency / finish_type / tool_calls_count

兼容：保留 finish_reason（与 finish_type 双写），不破坏 Week1–5 旧字段。

latency：整单 wall-clock（ms），便于产品侧对齐成本与体验。
tool_calls_count：非 agent 固定为 0。
==========================================================================
"""

from __future__ import annotations

from typing import Any

# 验收用：三模式 meta 至少包含这些键
CORE_META_KEYS = (
    "model",
    "mode",
    "latency",
    "finish_type",
    "tool_calls_count",
)


def apply_meta_contract_v2(
    meta: dict[str, Any] | None,
    *,
    model: str,
    mode: str,
    latency_ms: int,
    finish_reason: str | None = None,
    tool_calls_count: int | None = None,
) -> dict[str, Any]:
    """把既有 meta 规范成 Contract v2（原地扩展后返回新 dict）。"""
    out: dict[str, Any] = dict(meta or {})
    finish = (
        finish_reason
        or out.get("finish_type")
        or out.get("finish_reason")
        or "unknown"
    )
    finish_s = str(finish)
    out["model"] = str(model or "")
    out["mode"] = str(mode or "")
    out["latency"] = int(latency_ms)
    out["finish_type"] = finish_s
    # 双写：旧客户端仍读 finish_reason
    out["finish_reason"] = finish_s
    if tool_calls_count is not None:
        out["tool_calls_count"] = int(tool_calls_count)
    else:
        raw = out.get("tool_calls_count")
        out["tool_calls_count"] = int(raw) if isinstance(raw, (int, float)) else 0
    return out


def assert_core_meta(meta: dict[str, Any] | None) -> list[str]:
    """返回缺失的核心字段名（空列表 = 通过）。"""
    if not isinstance(meta, dict):
        return list(CORE_META_KEYS)
    return [k for k in CORE_META_KEYS if k not in meta]
