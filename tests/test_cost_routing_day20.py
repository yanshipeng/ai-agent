"""Day20：动态 TopK / 路由 / chunk 合并 / 长答整形。"""

from __future__ import annotations

from app.services.cost_routing import (
    MODEL_FLASH,
    MODEL_PRO,
    TOP_K_COMPLEX,
    TOP_K_SIMPLE,
    merge_context_hits,
    resolve_dynamic_top_k,
    resolve_route_model,
    shape_long_answer,
)


def test_dynamic_top_k_simple_vs_complex() -> None:
    k, reason = resolve_dynamic_top_k("什么是 ANR？")
    assert k == TOP_K_SIMPLE
    assert reason == "simple_query"

    k2, reason2 = resolve_dynamic_top_k("Android ANR 怎么排查？请给出完整步骤")
    assert k2 == TOP_K_COMPLEX
    assert reason2 == "complex_query"

    k3, reason3 = resolve_dynamic_top_k("任意问题", body_top_k=7)
    assert k3 == 7
    assert reason3 == "body_override"


def test_route_model_triggers() -> None:
    model, reason = resolve_route_model("你好", top1=0.9)
    assert model == MODEL_FLASH or "flash" in model
    assert reason == "default_flash"

    model2, reason2 = resolve_route_model("你好", top1=0.1)
    assert model2 == MODEL_PRO or "pro" in model2
    assert reason2 == "low_retrieve_confidence"

    model3, reason3 = resolve_route_model("请做高质量深度分析 ANR", top1=0.9)
    assert "pro" in model3
    assert reason3 == "user_quality_request"

    model4, reason4 = resolve_route_model("请给出完整步骤规划排查方案", top1=0.9)
    assert "pro" in model4
    assert reason4 == "long_procedure"


def test_merge_context_hits_same_doc() -> None:
    hits = [
        {
            "chunk_id": "a",
            "doc_id": "d1",
            "score": 0.9,
            "text": "第一段 ANR 内容。",
            "url": "u1",
        },
        {
            "chunk_id": "b",
            "doc_id": "d1",
            "score": 0.8,
            "text": "第二段 traces 内容。",
            "url": "u1",
        },
        {
            "chunk_id": "c",
            "doc_id": "d2",
            "score": 0.7,
            "text": "另一文档 OOM。",
            "url": "u2",
        },
    ]
    merged, stats = merge_context_hits(hits)
    assert stats["merged_same_doc"] >= 1
    assert stats["after"] <= stats["before"]
    assert any("traces" in str(h.get("text") or "") for h in merged)


def test_shape_long_answer() -> None:
    short, changed = shape_long_answer("短回答 [1]")
    assert changed is False
    assert short == "短回答 [1]"

    long_plain = "句。" * 400 + "详见 [1][2]"
    shaped, changed2 = shape_long_answer(long_plain, limit=200)
    assert changed2 is True
    assert "要点摘要" in shaped
    assert "[1]" in shaped
