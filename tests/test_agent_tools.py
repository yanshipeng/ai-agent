"""Agent 工具与 Tool Runner 单测（mock LLM，不打真实网络）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.agent.runner import (
    STOP_CLARIFY,
    STOP_DEGRADED_TO_RAG,
    STOP_FINAL_ANSWER,
    STOP_MAX_STEPS,
    STOP_TIMEOUT,
    STOP_UPSTREAM_ERROR,
    run_agent_loop,
)
from app.agent.tools import (
    TOOL_INVALID_ARGS,
    TOOL_NOT_FOUND,
    TOOL_TIMEOUT,
    execute_tool,
    tool_kb_search,
)
from app.kb.index_store import build_index, save_index
from app.kb.retriever import clear_index_cache
from app.main import create_app
from app.services.llm_client import LLMResult, LLMTurnResult, ToolCall
from app.services.metrics_store import build_ask_metric


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
    index = build_index(_sample_chunks(), dim=64)
    save_index(index, index_dir)
    yield index_dir
    clear_index_cache()


def test_execute_tool_unknown_name():
    out = execute_tool("no_such_tool", {"query": "x"})
    assert out["ok"] is False
    assert out["error_code"] == TOOL_NOT_FOUND


def test_execute_tool_invalid_json_args():
    out = execute_tool("kb_search", "{not-json")
    assert out["ok"] is False
    assert out["error_code"] == TOOL_INVALID_ARGS


def test_kb_search_and_get_chunk(tmp_index: Path):
    search = execute_tool(
        "kb_search",
        {"query": "ANR traces", "top_k": 3},
        index_dir=tmp_index,
    )
    assert search["ok"] is True
    assert search["hit_count"] >= 1
    chunk_id = search["results"][0]["chunk_id"]

    got = execute_tool(
        "kb_get_chunk",
        {"chunk_id": chunk_id},
        index_dir=tmp_index,
    )
    assert got["ok"] is True
    assert "traces.txt" in got["chunk"]["text"]


def test_kb_search_missing_query(tmp_index: Path):
    out = tool_kb_search({}, index_dir=tmp_index)
    assert out["ok"] is False
    assert out["error_code"] == TOOL_INVALID_ARGS


def test_execute_tool_timeout(monkeypatch, tmp_index: Path):
    def slow_search(args, *, index_dir=None):
        time.sleep(0.5)
        return {"ok": True}

    monkeypatch.setitem(
        __import__("app.agent.tools", fromlist=["_HANDLERS"])._HANDLERS,
        "kb_search",
        slow_search,
    )
    out = execute_tool(
        "kb_search",
        {"query": "ANR"},
        index_dir=tmp_index,
        timeout_seconds=0.05,
    )
    assert out["ok"] is False
    assert out["error_code"] == TOOL_TIMEOUT


def test_run_agent_loop_with_real_tool_calls(tmp_index, monkeypatch):
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_index))
    from app.core.config import get_settings

    get_settings.cache_clear()

    client = MagicMock()
    client.chat_turn.side_effect = [
        LLMTurnResult(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="kb_search",
                    arguments=json.dumps({"query": "ANR 怎么排查", "top_k": 3}),
                )
            ],
            model="deepseek-v4-flash",
            finish_reason="tool_calls",
            latency_ms=10,
        ),
        LLMTurnResult(
            content="根据知识库：先查 traces.txt。",
            tool_calls=[],
            model="deepseek-v4-flash",
            finish_reason="stop",
            latency_ms=12,
        ),
    ]

    result = run_agent_loop("Android ANR 怎么排查", client=client, index_dir=tmp_index)
    assert result.http_error_code is None
    assert result.tool_calls_count == 1
    assert result.tools_used == ["kb_search"]
    assert result.agent_steps == 2
    assert "traces" in result.answer or "知识库" in result.answer
    assert result.citations
    assert "plan" in result.phase_trace
    assert "act" in result.phase_trace
    assert "observe" in result.phase_trace
    assert "final" in result.phase_trace
    assert result.degraded_to is None
    assert result.stop_reason == STOP_FINAL_ANSWER
    assert result.agent_steps <= result.max_steps
    assert result.max_steps >= 1


def test_max_steps_degrade_to_rag(tmp_index, monkeypatch):
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_index))
    from app.core.config import get_settings

    get_settings.cache_clear()

    always_tool = LLMTurnResult(
        content=None,
        tool_calls=[
            ToolCall(
                id="call_loop",
                name="kb_search",
                arguments=json.dumps({"query": "ANR", "top_k": 2}),
            )
        ],
        model="deepseek-v4-flash",
        finish_reason="tool_calls",
        latency_ms=5,
    )
    client = MagicMock()
    # max_steps=2：两轮 Plan 都调工具 → 第 3 次进 Plan 时超限 → 降级 RAG
    client.chat_turn.side_effect = [always_tool, always_tool]
    client.chat.return_value = LLMResult(
        answer="降级 RAG：先看 traces.txt。",
        model="deepseek-v4-flash",
        finish_reason="stop",
        latency_ms=20,
    )

    result = run_agent_loop(
        "Android ANR 怎么排查",
        client=client,
        index_dir=tmp_index,
        max_steps=2,
        on_max_steps="rag",
        max_total_time_ms=60_000,
    )
    assert result.http_error_code is None
    assert result.degraded_to == "rag"
    assert result.fallback is True
    assert result.finish_reason == "degraded_rag"
    assert result.stop_reason == STOP_DEGRADED_TO_RAG
    assert result.agent_steps <= result.max_steps == 2
    assert "traces" in result.answer
    assert client.chat.called


def test_max_steps_clarify(tmp_index, monkeypatch):
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_index))
    from app.core.config import get_settings

    get_settings.cache_clear()

    always_tool = LLMTurnResult(
        content=None,
        tool_calls=[
            ToolCall(
                id="call_c",
                name="kb_search",
                arguments='{"query":"x","top_k":1}',
            )
        ],
        model="m",
        finish_reason="tool_calls",
        latency_ms=1,
    )
    client = MagicMock()
    client.chat_turn.side_effect = [always_tool]
    result = run_agent_loop(
        "问题",
        client=client,
        request_id="rid-clarify",
        index_dir=tmp_index,
        max_steps=1,
        on_max_steps="clarify",
        max_total_time_ms=60_000,
    )
    assert result.degraded_to == "clarify"
    assert result.fallback is True
    assert result.stop_reason == STOP_CLARIFY
    assert result.agent_steps <= result.max_steps == 1
    assert "rid-clarify" in result.answer
    assert result.http_error_code is None


def test_max_steps_hard_error_policy(tmp_index, monkeypatch):
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_index))
    from app.core.config import get_settings

    get_settings.cache_clear()

    always_tool = LLMTurnResult(
        content=None,
        tool_calls=[
            ToolCall(id="c", name="kb_search", arguments='{"query":"x","top_k":1}')
        ],
        model="m",
        finish_reason="tool_calls",
        latency_ms=1,
    )
    client = MagicMock()
    client.chat_turn.side_effect = [always_tool]
    result = run_agent_loop(
        "问题",
        client=client,
        request_id="rid-max",
        index_dir=tmp_index,
        max_steps=1,
        on_max_steps="error",
        max_total_time_ms=60_000,
    )
    assert result.stop_reason == STOP_MAX_STEPS
    assert result.agent_steps <= result.max_steps
    assert "rid-max" in result.answer


def test_total_time_timeout_fallback(tmp_index, monkeypatch):
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_index))
    from app.core.config import get_settings

    get_settings.cache_clear()

    client = MagicMock()

    def slow_turn(*args, **kwargs):
        time.sleep(0.08)
        return LLMTurnResult(
            content=None,
            tool_calls=[
                ToolCall(id="c1", name="kb_search", arguments='{"query":"ANR","top_k":1}')
            ],
            model="m",
            finish_reason="tool_calls",
            latency_ms=1,
        )

    client.chat_turn.side_effect = slow_turn
    result = run_agent_loop(
        "ANR",
        client=client,
        request_id="rid-timeout",
        index_dir=tmp_index,
        max_steps=5,
        max_total_time_ms=50,  # 首轮 Plan 后即超时
    )
    assert result.degraded_to == "timeout"
    assert result.fallback is True
    assert result.error_code == "AGENT_TIMEOUT"
    assert result.stop_reason == STOP_TIMEOUT
    assert result.agent_steps <= result.max_steps
    assert "rid-timeout" in result.answer
    assert result.http_error_code is None


def test_upstream_error_stop_reason(tmp_index, monkeypatch):
    from app.services.llm_client import LLMError, UPSTREAM_TIMEOUT

    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_index))
    from app.core.config import get_settings

    get_settings.cache_clear()
    client = MagicMock()
    client.chat_turn.side_effect = LLMError(UPSTREAM_TIMEOUT, "boom", status_code=504)
    result = run_agent_loop("ANR", client=client, index_dir=tmp_index, max_steps=3)
    assert result.stop_reason == STOP_UPSTREAM_ERROR
    assert result.http_error_code == UPSTREAM_TIMEOUT
    assert result.agent_steps <= result.max_steps


def test_ask_mode_agent_writes_metrics(tmp_path, monkeypatch, tmp_index):
    metrics_path = tmp_path / "requests.jsonl"
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(metrics_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_index))
    from app.core.config import get_settings

    get_settings.cache_clear()

    mock_client = MagicMock()
    mock_client.chat_turn.side_effect = [
        LLMTurnResult(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_a",
                    name="kb_search",
                    arguments='{"query":"OOM","top_k":2}',
                )
            ],
            model="deepseek-v4-flash",
            finish_reason="tool_calls",
            latency_ms=8,
        ),
        LLMTurnResult(
            content="OOM 可用 LeakCanary。",
            tool_calls=[],
            model="deepseek-v4-flash",
            finish_reason="stop",
            latency_ms=9,
        ),
    ]

    app = create_app()
    with TestClient(app) as client:
        client.app.state.llm_client = mock_client
        resp = client.post("/ask?mode=agent", json={"query": "OOM 怎么查"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["meta"]["mode"] == "agent"
        assert body["meta"]["tool_calls_count"] >= 1
        assert "kb_search" in body["meta"]["tools_used"]
        assert body["meta"]["agent_steps"] >= 1
        assert body["meta"]["agent_steps"] <= body["meta"]["max_steps"]
        assert body["meta"]["stop_reason"] == STOP_FINAL_ANSWER

    row = json.loads(metrics_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert row["mode"] == "agent"
    assert row["agent_steps"] >= 1
    assert row["agent_steps"] <= row["max_steps"]
    assert row["tool_calls_count"] >= 1
    assert row["tools_used"] == ["kb_search"]
    assert row["stop_reason"] == STOP_FINAL_ANSWER


def test_build_ask_metric_agent_fields():
    row = build_ask_metric(
        request_id="r-agent",
        ok=True,
        status_code=200,
        mode="agent",
        agent_steps=3,
        max_steps=5,
        tool_calls_count=2,
        tools_used=["kb_search", "kb_get_chunk"],
        stop_reason=STOP_FINAL_ANSWER,
    )
    assert row["mode"] == "agent"
    assert row["agent_steps"] == 3
    assert row["max_steps"] == 5
    assert row["tool_calls_count"] == 2
    assert row["tools_used"] == ["kb_search", "kb_get_chunk"]
    assert row["stop_reason"] == STOP_FINAL_ANSWER
