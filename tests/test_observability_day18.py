"""Day18：Agent Trace / Token Budget / Cache 计数。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.agent.runner import STOP_CLARIFY, run_agent_loop
from app.kb.index_store import build_index, save_index
from app.kb.retriever import (
    clear_index_cache,
    get_cache_counters,
    get_index,
    reset_cache_counters,
)
from app.services.llm_client import LLMTurnResult, ToolCall
from app.services.metrics_store import append_trace_metric, build_ask_metric
from app.services.token_budget import (
    compress_messages_for_budget,
    estimate_messages_tokens,
    estimate_tokens,
)


def _sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "A-1:0:0",
            "doc_id": "A-1",
            "category": "A",
            "category_name": "ANR",
            "title": "Android ANR 排查指南",
            "url": "https://example.com/anr",
            "source": "test",
            "tags": ["ANR"],
            "section_path": "分析 traces",
            "is_code": False,
            "char_len": 40,
            "text": "发生 ANR 时先看 /data/anr/traces.txt 主线程堆栈。",
        }
    ]


@pytest.fixture()
def tmp_index(tmp_path: Path) -> Path:
    clear_index_cache()
    index_dir = tmp_path / "index"
    save_index(build_index(_sample_chunks(), dim=64), index_dir)
    yield index_dir
    clear_index_cache()


def test_estimate_tokens_cjk_and_ascii() -> None:
    assert estimate_tokens("安卓") == 2
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("") == 0


def test_compress_messages_for_budget_truncates_tool() -> None:
    long_tool = "x" * 5000
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "tool", "content": long_tool},
    ]
    before = estimate_messages_tokens(messages)
    compressed, stats = compress_messages_for_budget(
        messages,
        max_context_tokens=max(80, before // 3),
        tool_max_chars=200,
    )
    assert stats["compressed"] is True
    assert stats["context_tokens_after"] < before
    tool_msg = next(m for m in compressed if m.get("role") == "tool")
    assert len(str(tool_msg.get("content") or "")) <= 200 + 20  # truncate 可能带省略标记


def test_agent_trace_shape_on_tool_loop(tmp_index, monkeypatch) -> None:
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_index))
    from app.core.config import get_settings

    get_settings.cache_clear()
    clear_index_cache()
    reset_cache_counters()

    client = MagicMock()
    client.chat_turn.side_effect = [
        LLMTurnResult(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="kb_search",
                    arguments=json.dumps({"query": "ANR", "top_k": 2}),
                )
            ],
            model="deepseek-v4-flash",
            finish_reason="tool_calls",
            latency_ms=8,
        ),
        LLMTurnResult(
            content="先查 traces。[1]",
            tool_calls=[],
            model="deepseek-v4-flash",
            finish_reason="stop",
            latency_ms=9,
        ),
    ]

    result = run_agent_loop("ANR 怎么排查", client=client, index_dir=tmp_index)
    assert result.http_error_code is None
    assert result.agent_trace
    actions = [step["action"] for step in result.agent_trace]
    assert "plan" in actions
    assert "tool_call" in actions
    assert actions[-1] == "final"

    tool_steps = [s for s in result.agent_trace if s["action"] == "tool_call"]
    assert tool_steps
    assert tool_steps[0]["tool_name"] == "kb_search"
    assert "tool_latency_ms" in tool_steps[0]
    assert tool_steps[0]["tool_ok"] is True
    assert result.max_context_tokens is not None
    assert result.context_tokens_used is not None
    assert result.max_output_tokens is not None

    cache = get_cache_counters()
    assert cache["cache_hit"] + cache["cache_miss"] >= 1


def test_token_budget_clarify_when_still_over(monkeypatch, tmp_index) -> None:
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_index))
    monkeypatch.setenv("AGENT_MAX_CONTEXT_TOKENS", "40")
    from app.core.config import get_settings

    get_settings.cache_clear()

    client = MagicMock()
    # 超预算且压缩后仍超 → 不应调 LLM
    history = [
        {"role": "user", "content": "旧问题 " + ("详" * 200)},
        {"role": "assistant", "content": "旧回答 " + ("答" * 200)},
        {"role": "tool", "content": "工具结果 " + ("段" * 400)},
    ]
    result = run_agent_loop(
        "新问题还是很长 " + ("问" * 80),
        client=client,
        index_dir=tmp_index,
        history_messages=history,
    )
    assert result.stop_reason == STOP_CLARIFY
    assert result.budget_compressed is True
    assert client.chat_turn.call_count == 0
    assert any(s.get("action") == "clarify" for s in result.agent_trace)


def test_cache_counters_hit_miss(tmp_index) -> None:
    clear_index_cache()
    reset_cache_counters()
    get_index(tmp_index)
    c1 = get_cache_counters()
    assert c1["cache_miss"] >= 1
    get_index(tmp_index)
    c2 = get_cache_counters()
    assert c2["cache_hit"] >= 1


def test_build_ask_metric_and_traces_jsonl(tmp_path: Path) -> None:
    record = build_ask_metric(
        request_id="rid-1",
        path="/ask",
        ok=True,
        status_code=200,
        latency_ms_total=12,
        mode="agent",
        agent_trace=[{"step_idx": 1, "action": "plan"}],
        max_context_tokens=6000,
        context_tokens_used=120,
        max_output_tokens=2048,
        budget_compressed=False,
        cache_hit=1,
        cache_miss=0,
    )
    assert record["agent_trace"][0]["action"] == "plan"
    assert record["cache_hit"] == 1

    out = tmp_path / "traces.jsonl"
    append_trace_metric(
        {
            "request_id": "rid-1",
            "agent_trace": record["agent_trace"],
            "context_tokens_used": 120,
        },
        path=out,
    )
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["request_id"] == "rid-1"
    assert row["agent_trace"][0]["step_idx"] == 1
