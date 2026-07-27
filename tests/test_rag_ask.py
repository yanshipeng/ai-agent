"""RAG v1：prompt / citations / /ask?mode=rag 契约。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.kb.index_store import build_index, save_index
from app.kb.rag import (
    RAG_SYSTEM_PROMPT,
    build_rag_messages,
    hits_to_citations,
    run_rag_retrieve,
)
from app.kb.retriever import clear_index_cache
from app.main import create_app
from app.services.llm_client import LLMResult
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
        },
        {
            "chunk_id": "C-1:0:0",
            "doc_id": "C-1",
            "category": "C",
            "category_name": "内存",
            "title": "OOM 与内存泄漏",
            "url": "https://example.com/oom",
            "source": "test",
            "tags": ["OOM"],
            "section_path": "LeakCanary",
            "is_code": False,
            "char_len": 40,
            "text": "OOM 常见于 Bitmap 未回收，可用 LeakCanary 查泄漏。",
        },
    ]


def test_hits_to_citations_ref_id_starts_at_1():
    hits = [
        {
            "chunk_id": "a",
            "url": "u1",
            "title": "t1",
            "section_path": "s1",
            "is_code": False,
        },
        {
            "chunk_id": "b",
            "url": "u2",
            "title": "t2",
            "section_path": "",
            "is_code": True,
        },
    ]
    citations = hits_to_citations(hits)
    assert citations[0]["ref_id"] == 1
    assert citations[1]["ref_id"] == 2
    assert citations[1]["is_code"] is True
    for key in ("ref_id", "chunk_id", "url", "title", "section_path", "is_code"):
        assert key in citations[0]


def test_build_rag_messages_contains_constraints_and_context():
    hits = [
        {
            "chunk_id": "A-1:0:0",
            "title": "ANR",
            "url": "https://example.com/anr",
            "section_path": "traces",
            "is_code": False,
            "text": "先看 traces.txt",
        }
    ]
    messages = build_rag_messages("ANR 怎么查", hits)
    assert messages[0]["role"] == "system"
    assert "只能基于" in RAG_SYSTEM_PROMPT or "只能基于" in messages[0]["content"]
    assert "Context" in messages[1]["content"]
    assert "[1]" in messages[1]["content"]
    assert "traces.txt" in messages[1]["content"]
    assert "ANR 怎么查" in messages[1]["content"]


def test_run_rag_retrieve_pack(tmp_path: Path):
    index_dir = tmp_path / "index"
    save_index(build_index(_sample_chunks(), dim=256), index_dir)
    clear_index_cache()
    pack = run_rag_retrieve("ANR traces", top_k=2, index_dir=index_dir)
    assert pack["context_chunks"] >= 1
    assert pack["citations"][0]["ref_id"] == 1
    assert pack["messages"][0]["role"] == "system"
    assert isinstance(pack["retrieve_ms"], int)


def test_build_ask_metric_rag_fields():
    row = build_ask_metric(
        request_id="r1",
        ok=True,
        status_code=200,
        mode="rag",
        top_k=5,
        retrieve_ms=12,
        context_chunks=5,
        citations_count=5,
    )
    assert row["mode"] == "rag"
    assert row["top_k"] == 5
    assert row["retrieve_ms"] == 12
    assert row["context_chunks"] == 5
    assert row["citations_count"] == 5


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("APP_VERSION", "0.1.0ba")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_ask_rag_mode_fills_citations_and_metrics(client: TestClient, tmp_path, monkeypatch):
    index_dir = tmp_path / "kb_index"
    save_index(build_index(_sample_chunks(), dim=256), index_dir)
    clear_index_cache()
    monkeypatch.setenv("KB_INDEX_DIR", str(index_dir))
    from app.core.config import get_settings

    get_settings.cache_clear()

    mock_client = MagicMock()
    mock_client.chat.return_value = LLMResult(
        answer="先看 traces.txt。[1]",
        model="deepseek-v4-flash",
        finish_reason="stop",
        latency_ms=100,
        usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        retry_count=0,
    )
    client.app.state.llm_client = mock_client

    resp = client.post(
        "/ask?mode=rag",
        json={"query": "Android ANR 怎么排查", "top_k": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"].startswith("先看")
    assert len(body["citations"]) >= 1
    cite = body["citations"][0]
    for key in ("ref_id", "chunk_id", "url", "title", "section_path", "is_code"):
        assert key in cite
    assert cite["ref_id"] == 1
    assert body["meta"]["mode"] == "rag"
    assert body["meta"]["top_k"] == 2
    assert body["meta"]["context_chunks"] == len(body["citations"])
    assert "retrieve_ms" in body["meta"]

    args, _kwargs = mock_client.chat.call_args
    messages = args[0]
    assert messages[0]["role"] == "system"
    assert "Context" in messages[1]["content"]

    from app.core.config import get_settings as gs

    rows = Path(gs().requests_jsonl_path).read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(rows[-1])
    assert row["mode"] == "rag"
    assert row["top_k"] == 2
    assert row["context_chunks"] == len(body["citations"])
    assert row["citations_count"] == len(body["citations"])
    assert isinstance(row["retrieve_ms"], int)


def test_ask_rag_index_missing_returns_503(client: TestClient, tmp_path, monkeypatch):
    monkeypatch.setenv("KB_INDEX_DIR", str(tmp_path / "missing_index"))
    from app.core.config import get_settings

    get_settings.cache_clear()
    client.app.state.llm_client = MagicMock()

    resp = client.post("/ask", json={"query": "ANR", "mode": "rag"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "INDEX_NOT_READY"


def test_ask_llm_mode_still_empty_citations(client: TestClient):
    mock_client = MagicMock()
    mock_client.chat.return_value = LLMResult(
        answer="ok",
        model="deepseek-v4-flash",
        finish_reason="stop",
        latency_ms=10,
        usage={},
    )
    client.app.state.llm_client = mock_client
    resp = client.post("/ask", json={"query": "1+1"})
    assert resp.status_code == 200
    assert resp.json()["citations"] == []
    assert resp.json()["meta"]["mode"] == "llm"
