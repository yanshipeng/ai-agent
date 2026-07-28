"""多轮 session：滑窗 / 截断 / 摘要 + /ask 带入历史。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.services.conversation import (
    MEMORY_USER_PREFIX,
    apply_sliding_window,
    compact_messages,
    truncate_text,
)
from app.services.llm_client import LLMResult
from app.services.session_store import (
    clear_all_sessions,
    get_session_messages,
    set_session_messages,
)

@pytest.fixture(autouse=True)
def _clean_sessions():
    clear_all_sessions()
    get_settings.cache_clear()
    yield
    clear_all_sessions()
    get_settings.cache_clear()


def test_truncate_text_marks_truncated():
    out = truncate_text("a" * 100, 30)
    assert len(out) <= 30
    assert "truncated" in out


def test_sliding_window_keeps_last_n_turns():
    msgs = []
    for i in range(5):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    kept, dropped, turns = apply_sliding_window(msgs, max_turns=2)
    assert turns == 2
    assert [m["content"] for m in kept if m["role"] == "user"] == ["q3", "q4"]
    assert len(dropped) == 6


def test_compact_summarizes_when_over_budget(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_TURNS", "2")
    monkeypatch.setenv("SESSION_MAX_CHARS", "500")
    monkeypatch.setenv("SESSION_ENABLE_SUMMARY", "true")
    monkeypatch.setenv("SESSION_CONTENT_MAX_CHARS", "200")
    monkeypatch.setenv("SESSION_TOOL_RESULT_MAX_CHARS", "100")
    get_settings.cache_clear()

    msgs = []
    for i in range(4):
        msgs.append({"role": "user", "content": f"问题{i}：" + ("详" * 40)})
        msgs.append({"role": "assistant", "content": f"答{i}：" + ("案" * 40)})

    result = compact_messages(msgs, enable_summary=True)
    assert result.stats.summarized is True
    assert result.stats.turns_kept <= 3  # memory + window
    assert any(
        str(m.get("content", "")).startswith(MEMORY_USER_PREFIX)
        for m in result.messages
    )
    assert result.stats.output_chars <= 500 or result.stats.truncated_msgs > 0


def test_ask_llm_multi_turn_carries_history(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_TURNS", "8")
    get_settings.cache_clear()

    seen_messages: list[list] = []

    class FakeClient:
        def chat(self, messages, *, request_id=None):
            seen_messages.append(list(messages))
            n = len(seen_messages)
            return LLMResult(
                answer=f"答{n}",
                model="fake",
                latency_ms=1,
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

    app = create_app()
    app.state.llm_client = FakeClient()
    client = TestClient(app)

    r1 = client.post(
        "/ask",
        json={"query": "我叫小明", "session_id": "s-mt-1", "mode": "llm"},
    )
    assert r1.status_code == 200
    assert r1.json()["answer"] == "答1"
    assert len(seen_messages[0]) == 1
    assert seen_messages[0][0]["content"] == "我叫小明"

    r2 = client.post(
        "/ask",
        json={"query": "我叫什么？", "session_id": "s-mt-1", "mode": "llm"},
    )
    assert r2.status_code == 200
    assert r2.json()["answer"] == "答2"
    roles = [m["role"] for m in seen_messages[1]]
    assert roles == ["user", "assistant", "user"]
    assert seen_messages[1][0]["content"] == "我叫小明"
    assert seen_messages[1][1]["content"] == "答1"
    assert seen_messages[1][2]["content"] == "我叫什么？"
    assert r2.json()["meta"].get("session_id") == "s-mt-1"
    assert r2.json()["meta"].get("history_messages") == 2
    assert r2.json()["meta"].get("history_chars", 0) > 0

    stored = get_session_messages("s-mt-1")
    assert len(stored) == 4  # 两轮 user/assistant


def test_five_turns_remember_constraint(monkeypatch, tmp_path):
    """同 session 连续 5 轮：约束进入后续 messages；jsonl 含 session 字段。"""
    metrics_path = tmp_path / "requests.jsonl"
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(metrics_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    get_settings.cache_clear()
    clear_all_sessions()

    constraint = "只讨论 Android 推送"
    turns = [
        constraint,
        "FCM 和厂商通道区别？",
        "到达率怎么查？",
        "写个 React 组件",
        "约束是什么？",
    ]
    seen: list[list] = []

    class FakeClient:
        def chat(self, messages, *, request_id=None):
            seen.append(list(messages))
            return LLMResult(
                answer="收到，只讨论 Android 推送。",
                model="fake",
                latency_ms=5,
                finish_reason="stop",
            )

    app = create_app()
    with TestClient(app) as client:
        client.app.state.llm_client = FakeClient()
        for q in turns:
            resp = client.post(
                "/ask",
                json={"query": q, "session_id": "s-5turn", "mode": "llm"},
            )
            assert resp.status_code == 200

    # 第 5 轮 messages 应仍含第 1 轮约束
    last = seen[-1]
    blob = "\n".join(str(m.get("content") or "") for m in last)
    assert constraint in blob
    assert last[-1]["content"] == turns[-1]
    assert len(last) >= 9  # 4 轮历史(8) + 本轮 user

    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert [r["history_messages"] for r in rows] == [0, 2, 4, 6, 8]
    assert all(r["session_id"] == "s-5turn" for r in rows)
    assert all(isinstance(r["history_chars"], int) for r in rows)
    clear_all_sessions()


def test_ask_without_session_is_stateless(monkeypatch):
    seen: list[int] = []

    class FakeClient:
        def chat(self, messages, *, request_id=None):
            seen.append(len(messages))
            return LLMResult(
                answer="ok",
                model="fake",
                latency_ms=1,
                finish_reason="stop",
                usage=None,
            )

    app = create_app()
    app.state.llm_client = FakeClient()
    client = TestClient(app)
    client.post("/ask", json={"query": "第一轮", "mode": "llm"})
    client.post("/ask", json={"query": "第二轮", "mode": "llm"})
    assert seen == [1, 1]


def test_rag_storage_keeps_plain_query_not_context():
    """session 里只存短 query，不存 RAG Context 大段。"""
    set_session_messages(
        "s-rag",
        [
            {"role": "user", "content": "短问题"},
            {"role": "assistant", "content": "短回答"},
        ],
    )
    history = get_session_messages("s-rag")
    assert "Context" not in history[0]["content"]
    assert len(history[0]["content"]) < 20
