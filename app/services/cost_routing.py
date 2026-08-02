"""Day20：成本/体验优化 + 多模型路由。

==========================================================================
做什么
==========================================================================
1) 动态 TopK：简单问题 3，复杂/步骤型 5（省 context）
2) chunk 合并去噪：同 doc 相邻片段合并，再砍近重复
3) 输出形态：长回答引导「分段 + 要点 + 引用」；max_tokens 仍用配置固定值
4) 多模型路由：默认 flash；低检索置信度 / 用户要高质量 / 长步骤规划 → pro

为什么单独成模块？
  api / rag / llm_client 都要读同一套启发式，避免三处复制。
==========================================================================
"""

from __future__ import annotations

import re
from typing import Any

from app.core.config import get_settings

# TopK
TOP_K_SIMPLE = 3
TOP_K_COMPLEX = 5

# 模型（DeepSeek）
MODEL_FLASH = "deepseek-v4-flash"
MODEL_PRO = "deepseek-v4-pro"

# 检索置信度：Top1 hybrid score 低于此 → 走 pro（可被 Settings 覆盖）
DEFAULT_ROUTE_PRO_MIN_SCORE = 0.35

# 长回答形态：超过此字符则做轻量整形提示结构
LONG_ANSWER_CHARS = 900

# 复杂问题信号
_COMPLEX_HINTS = (
    "怎么排查",
    "如何排查",
    "步骤",
    "checklist",
    "详细",
    "完整",
    "对比",
    "分析",
    "根因",
    "链路",
    "为什么会",
    "有哪些方案",
    "优化",
    "规划",
    "方案设计",
)

_QUALITY_HINTS = (
    "高质量",
    "仔细分析",
    "深度分析",
    "复杂推理",
    "用更好的模型",
    "pro",
    "详细总结",
    "全面总结",
    "请深入",
)

_LONG_PROCEDURE_HINTS = (
    "分步",
    "逐步",
    "完整步骤",
    "排查步骤",
    "操作步骤",
    "执行步骤",
    "长步骤",
    "规划一下",
    "给出方案",
    "落地计划",
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_REF_RE = re.compile(r"\[\d+\]")


def is_complex_query(query: str) -> bool:
    """简单 vs 复杂：步骤/排查/对比等 → 复杂。"""
    q = (query or "").strip()
    if not q:
        return False
    if len(q) >= 40:
        return True
    q_l = q.lower()
    return any(h in q or h in q_l for h in _COMPLEX_HINTS)


def is_long_procedure_query(query: str) -> bool:
    """长步骤规划 / 完整排查步骤。"""
    q = (query or "").strip()
    q_l = q.lower()
    return any(h in q or h in q_l for h in _LONG_PROCEDURE_HINTS)


def wants_high_quality(query: str) -> bool:
    """用户明确要求高质量/深度分析。"""
    q = (query or "").strip()
    q_l = q.lower()
    return any(h in q or h in q_l for h in _QUALITY_HINTS)


def resolve_dynamic_top_k(query: str, *, body_top_k: int | None = None) -> tuple[int, str]:
    """解析 TopK。

    返回 (top_k, reason)。
    body 显式传入时尊重用户；否则简单 3 / 复杂 5。
    """
    if body_top_k is not None:
        return int(body_top_k), "body_override"
    if is_complex_query(query) or is_long_procedure_query(query):
        return TOP_K_COMPLEX, "complex_query"
    return TOP_K_SIMPLE, "simple_query"


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def merge_context_hits(
    hits: list[dict[str, Any]],
    *,
    near_dup_jaccard: float = 0.80,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """检索后二次去噪：同 doc 合并正文 + 再砍近重复。

    Day16 dedupe 已在 retrieve 内；这里针对拼 Context 再压一层成本。

    tokenize 懒加载，避免 cost_routing ↔ kb.rag 循环导入。
    """
    from app.kb.embedder import tokenize

    if not hits:
        return [], {"merged_same_doc": 0, "dropped_near_dup": 0, "before": 0, "after": 0}

    merged_same = 0
    buckets: list[dict[str, Any]] = []
    for hit in hits:
        doc_id = str(hit.get("doc_id") or "")
        text = str(hit.get("text") or hit.get("text_snippet") or "")
        if (
            buckets
            and doc_id
            and str(buckets[-1].get("doc_id") or "") == doc_id
        ):
            prev = buckets[-1]
            prev_text = str(prev.get("text") or "")
            # 拼接未重叠尾部
            if text and text not in prev_text:
                prev["text"] = (prev_text + "\n" + text).strip()
                if hit.get("score") is not None:
                    prev["score"] = max(float(prev.get("score") or 0), float(hit["score"]))
                merged_same += 1
            continue
        row = dict(hit)
        if text and not row.get("text"):
            row["text"] = text
        buckets.append(row)

    kept: list[dict[str, Any]] = []
    kept_tok: list[set[str]] = []
    dropped_near = 0
    for hit in buckets:
        tokens = set(tokenize(str(hit.get("text") or hit.get("text_snippet") or "")))
        if any(_jaccard(tokens, prev) >= near_dup_jaccard for prev in kept_tok):
            dropped_near += 1
            continue
        kept.append(hit)
        kept_tok.append(tokens)

    stats = {
        "merged_same_doc": merged_same,
        "dropped_near_dup": dropped_near,
        "before": len(hits),
        "after": len(kept),
    }
    return kept, stats


def top1_score(hits: list[dict[str, Any]] | None) -> float | None:
    if not hits:
        return None
    score = hits[0].get("score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def resolve_route_model(
    query: str,
    *,
    top1: float | None = None,
    min_score: float | None = None,
) -> tuple[str, str]:
    """多模型路由：默认 flash。

    触发 pro（任一即可，便于讲解成本策略）：
      1) 用户明确要求高质量
      2) 检索 Top1 分低于阈值（主推荐条件）
      3) 长步骤规划类问题
    """
    settings = get_settings()
    flash = (getattr(settings, "llm_model", None) or MODEL_FLASH).strip() or MODEL_FLASH
    pro = (getattr(settings, "llm_model_pro", None) or MODEL_PRO).strip() or MODEL_PRO
    threshold = (
        float(min_score)
        if min_score is not None
        else float(
            getattr(settings, "rag_route_pro_min_score", None)
            or DEFAULT_ROUTE_PRO_MIN_SCORE
        )
    )

    if wants_high_quality(query):
        return pro, "user_quality_request"
    if top1 is not None and top1 < threshold:
        return pro, "low_retrieve_confidence"
    if is_long_procedure_query(query):
        return pro, "long_procedure"
    return flash, "default_flash"


def long_answer_system_addon() -> str:
    """拼进 RAG system：控制长答形态，不改 max_tokens。"""
    return (
        "7. 输出长度：默认简洁；若必须展开，使用「分段小标题 + 要点列表 + 引用 [n]」，"
        "避免大段散文；单次回答控制在可读长度内。"
    )


def shape_long_answer(answer: str, *, limit: int = LONG_ANSWER_CHARS) -> tuple[str, bool]:
    """过长且无结构时，包一层「要点」骨架（不删引用）。

    返回 (新文本, 是否改写)。
    """
    text = (answer or "").strip()
    if len(text) <= limit:
        return text, False
    # 已有明显分段/列表则不动
    if re.search(r"(?m)^(#{1,3}\s+|[-*•]\s+|\d+[\.、)]\s+)", text):
        return text, False
    refs = sorted({m.group(0) for m in _REF_RE.finditer(text)})
    head = text[: limit - 80].rstrip()
    # 尽量在句号处切断
    cut = max(head.rfind("。"), head.rfind("\n"), head.rfind("."))
    if cut > limit // 2:
        head = head[: cut + 1]
    ref_line = ("引用：" + " ".join(refs)) if refs else ""
    shaped = (
        "## 要点摘要\n"
        f"{head}…\n\n"
        "## 说明\n"
        "原文较长，已按成本策略截断展示要点；完整细节请缩小问题范围或指定章节再问。\n"
    )
    if ref_line:
        shaped += f"\n{ref_line}\n"
    return shaped.strip(), True
