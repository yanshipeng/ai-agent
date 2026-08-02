"""Day24：单次请求 token/成本预算与降级。

策略（简单可讲）：
1) 默认把 completion max_tokens 压到 REQUEST_TOKEN_BUDGET（控 p95）
2) 长/复杂问题：再降 TopK、强制 flash（避免 pro 成本失控）
3) 极短含糊问题：可走 clarify 短路（不调 LLM）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.services.cost_routing import (
    MODEL_FLASH,
    is_complex_query,
    is_long_procedure_query,
)

# 含糊短问：信息不足 → 澄清（省 token）
_VAGUE_HINTS = ("又卡了", "帮我看看", "怎么弄", "咋办", "挂了", "出问题了", "看看吧")


@dataclass(frozen=True)
class BudgetPlan:
    max_tokens: int
    top_k: int | None
    force_flash: bool
    clarify_short: bool
    reason: str
    actions: tuple[str, ...]

    def meta_fields(self) -> dict[str, Any]:
        return {
            "budget_max_tokens": self.max_tokens,
            "budget_force_flash": self.force_flash,
            "budget_clarify_short": self.clarify_short,
            "budget_reason": self.reason,
            "budget_actions": list(self.actions),
        }


def looks_vague_short(query: str) -> bool:
    q = (query or "").strip()
    if len(q) > 24:
        return False
    return any(h in q for h in _VAGUE_HINTS)


def plan_request_budget(
    query: str,
    *,
    top_k: int | None,
    mode: str,
) -> BudgetPlan:
    """根据 query 生成降级计划。"""
    settings = get_settings()
    hard_cap = max(
        64,
        int(getattr(settings, "request_token_budget", None) or 1024),
    )
    configured = max(64, int(getattr(settings, "llm_max_tokens", None) or 2048))
    max_tokens = min(configured, hard_cap)

    actions: list[str] = ["cap_max_tokens"]
    force_flash = False
    clarify = False
    reason = "default_budget"
    resolved_top_k = top_k

    vague = looks_vague_short(query)
    complex_q = is_complex_query(query) or is_long_procedure_query(query) or len(query) >= 80

    if vague and mode in {"llm", "rag", "agent"}:
        clarify = True
        force_flash = True
        max_tokens = min(max_tokens, 256)
        actions.extend(["clarify_short", "force_flash"])
        reason = "vague_short_clarify"
        if resolved_top_k is not None:
            resolved_top_k = min(int(resolved_top_k), 2)
            actions.append("reduce_top_k")
    elif complex_q:
        # 复杂题更容易撞高 token / 走 pro：主动压成本
        top_cap = max(1, int(getattr(settings, "request_budget_top_k_cap", 3) or 3))
        if resolved_top_k is not None and int(resolved_top_k) > top_cap:
            resolved_top_k = top_cap
            actions.append("reduce_top_k")
        force_flash = True
        actions.append("force_flash")
        max_tokens = min(max_tokens, hard_cap)
        reason = "complex_cost_guard"
    else:
        reason = "simple_budget"

    return BudgetPlan(
        max_tokens=int(max_tokens),
        top_k=resolved_top_k,
        force_flash=force_flash,
        clarify_short=clarify,
        reason=reason,
        actions=tuple(actions),
    )


def budget_clarify_answer() -> str:
    return (
        "信息不足，我先不展开长回答（成本预算保护）。"
        "请补充：机型/系统版本、卡顿或崩溃具体表现、是否有 ANR/OOM 日志或复现步骤。"
    )


def apply_force_flash(model: str, *, force: bool) -> tuple[str, str | None]:
    if not force:
        return model, None
    settings = get_settings()
    flash = (getattr(settings, "llm_model", None) or MODEL_FLASH).strip() or MODEL_FLASH
    if model == flash:
        return model, "already_flash"
    return flash, "budget_force_flash"
