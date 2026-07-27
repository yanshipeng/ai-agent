"""KB retrieve / index 单元测试（不依赖本机完整语料）。"""

from __future__ import annotations

from pathlib import Path

from app.kb.index_store import build_index, save_index
from app.kb.retriever import RESULT_FIELDS, clear_index_cache, retrieve


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
        {
            "chunk_id": "F-1:0:0",
            "doc_id": "F-1",
            "category": "F",
            "category_name": "WebView",
            "title": "WebView 白屏排查",
            "url": "https://example.com/webview",
            "source": "test",
            "tags": ["WebView"],
            "section_path": "白屏",
            "is_code": True,
            "char_len": 30,
            "text": "WebView 白屏先看 onReceivedError 与 JSBridge 回调。",
        },
    ]


def test_retrieve_schema_and_score_order(tmp_path: Path):
    index_dir = tmp_path / "index"
    index = build_index(_sample_chunks(), dim=256)
    save_index(index, index_dir)
    clear_index_cache()

    out = retrieve("Android ANR traces 怎么查", top_k=2, index_dir=index_dir)

    assert out["query"]
    assert out["top_k"] == 2
    assert isinstance(out["retrieve_ms"], int) and out["retrieve_ms"] >= 0
    assert 1 <= len(out["results"]) <= 2

    scores = [r["score"] for r in out["results"]]
    assert scores == sorted(scores, reverse=True)

    top = out["results"][0]
    for key in RESULT_FIELDS:
        assert key in top
    assert top["chunk_id"] == "A-1:0:0"
    assert "ANR" in (top["title"] or "")
    assert top["url"].startswith("https://")
    assert isinstance(top["is_code"], bool)


def test_retrieve_category_filter(tmp_path: Path):
    index_dir = tmp_path / "index"
    save_index(build_index(_sample_chunks(), dim=256), index_dir)
    clear_index_cache()

    out = retrieve("内存泄漏 OOM", top_k=5, index_dir=index_dir, category="C")
    assert out["results"]
    assert all("oom" in (r["url"] or "") or "OOM" in (r["title"] or "") for r in out["results"])


def test_retrieve_without_snippet(tmp_path: Path):
    index_dir = tmp_path / "index"
    save_index(build_index(_sample_chunks(), dim=256), index_dir)
    clear_index_cache()

    out = retrieve("WebView 白屏", top_k=1, index_dir=index_dir, include_snippet=False)
    assert "text_snippet" not in out["results"][0]
    assert out["results"][0]["is_code"] is True
