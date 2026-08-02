"""Day16：混合检索 / 去重过滤 / 可观测字段。"""

from __future__ import annotations

from pathlib import Path

from app.kb.bm25 import build_bm25_index, min_max_normalize
from app.kb.index_store import build_index, save_index
from app.kb.retriever import (
    clear_index_cache,
    dedupe_candidates,
    fuse_hybrid_scores,
    is_noise_chunk,
    retrieve,
    retrieve_stat_fields,
)


def _chunks() -> list[dict]:
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
            "char_len": 48,
            "text": "发生 ANR 时先看 /data/anr/traces.txt 主线程堆栈。",
        },
        {
            "chunk_id": "A-1:0:1",
            "doc_id": "A-1",
            "category": "A",
            "category_name": "ANR",
            "title": "Android ANR 排查指南",
            "url": "https://example.com/anr",
            "source": "test",
            "tags": ["ANR"],
            "section_path": "分析 traces",
            "is_code": False,
            "char_len": 48,
            "text": "发生 ANR 时先看 /data/anr/traces.txt 主线程堆栈。",
        },
        {
            "chunk_id": "A-1:1:0",
            "doc_id": "A-1",
            "category": "A",
            "category_name": "ANR",
            "title": "Android ANR 排查指南",
            "url": "https://example.com/anr",
            "source": "test",
            "tags": ["ANR"],
            "section_path": "Watchdog",
            "is_code": False,
            "char_len": 42,
            "text": "Watchdog 超时也会触发 ANR，需对照系统日志。",
        },
        {
            "chunk_id": "A-1:2:0",
            "doc_id": "A-1",
            "category": "A",
            "category_name": "ANR",
            "title": "Android ANR 排查指南",
            "url": "https://example.com/anr",
            "source": "test",
            "tags": ["ANR"],
            "section_path": "补充",
            "is_code": False,
            "char_len": 40,
            "text": "第三段 ANR 材料：广播接收器耗时过长也会卡死。",
        },
        {
            "chunk_id": "N-1:0:0",
            "doc_id": "N-1",
            "category": "A",
            "category_name": "ANR",
            "title": "导航噪声页",
            "url": "https://example.com/nav",
            "source": "test",
            "tags": [],
            "section_path": "nav",
            "is_code": False,
            "char_len": 20,
            "text": "首页\n登录\n注册\n下一页\n广告",
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


def test_min_max_normalize_and_fuse():
    assert min_max_normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert min_max_normalize([0.8]) == [1.0]  # 单文档不滤光
    assert min_max_normalize([0.0, 0.0]) == [0.0, 0.0]
    fused = fuse_hybrid_scores([0.0, 1.0], [1.0, 0.0], hybrid_weight=0.6)
    assert fused[0] == 0.4  # 0.6*0 + 0.4*1
    assert fused[1] == 0.6  # 0.6*1 + 0.4*0


def test_bm25_prefers_keyword_hit():
    bm25 = build_bm25_index(_chunks())
    scores = bm25.scores("LeakCanary OOM")
    best = max(range(len(scores)), key=lambda i: scores[i])
    assert _chunks()[best]["chunk_id"] == "C-1:0:0"


def test_is_noise_chunk():
    assert is_noise_chunk(_chunks()[4]) is True
    assert is_noise_chunk(_chunks()[0]) is False


def test_dedupe_drops_near_dup_and_caps_per_doc():
    meta = _chunks()
    # 分数降序：三条同 doc，其中前两条正文完全相同
    scored = [(0.9, 0), (0.8, 1), (0.7, 2), (0.6, 3)]
    kept, dropped = dedupe_candidates(scored, meta, max_per_doc=2)
    kept_ids = [meta[i]["chunk_id"] for _, i in kept]
    assert "A-1:0:0" in kept_ids
    assert "A-1:0:1" not in kept_ids  # 近重复
    assert dropped >= 1
    assert len(kept) <= 2  # max_per_doc


def test_retrieve_hybrid_observability(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAG_HYBRID_WEIGHT", "0.55")
    monkeypatch.setenv("RAG_MIN_SCORE", "0.01")
    from app.core.config import get_settings

    get_settings.cache_clear()

    index_dir = tmp_path / "index"
    save_index(build_index(_chunks(), dim=256), index_dir)
    clear_index_cache()

    out = retrieve("Android ANR traces", top_k=3, index_dir=index_dir)
    assert out["retrieve_candidates"] >= out["retrieve_kept"]
    assert out["retrieve_before_dedup"] >= out["retrieve_after_dedup"]
    assert out["retrieve_after_dedup"] >= out["retrieve_kept"]
    assert out["retrieve_kept"] == len(out["results"])
    assert out["hybrid_weight"] == 0.55
    assert isinstance(out["dedup_dropped"], int)
    assert out["retrieve_kept"] >= 1
    # 噪声页不应进入结果
    assert all(r["chunk_id"] != "N-1:0:0" for r in out["results"])
    # ANR 相关应靠前
    assert "ANR" in (out["results"][0]["title"] or "")

    stats = retrieve_stat_fields(out)
    for key in (
        "retrieve_candidates",
        "retrieve_before_dedup",
        "retrieve_after_dedup",
        "retrieve_kept",
        "hybrid_weight",
        "dedup_dropped",
    ):
        assert key in stats

    get_settings.cache_clear()
