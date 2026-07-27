"""检索：query → Embedding → TopK chunk。

==========================================================================
做什么
==========================================================================
把用户问题变成向量，与本地索引比相似度，返回 TopK 命中（带 score 与来源字段），
并记录 retrieve_ms。

统一命中字段：
  chunk_id, score, title, url, section_path, is_code
  可选：text_snippet（给人看）、text（给 RAG 拼 Context）

==========================================================================
为什么这样设计
==========================================================================
1) 检索与 HTTP / LLM 解耦：scripts/retrieve_kb.py 与 /ask?mode=rag 共用同一函数。
2) 进程内缓存索引：避免每个请求都读几百 MB JSONL；重建索引后要 clear_index_cache()。
3) include_text 默认 False：普通冒烟不必拖全文；RAG 再显式打开。
4) category 过滤：评测或「只在 ANR 类里搜」时有用，默认不过滤。
==========================================================================
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.kb.embedder import cosine_scores, embed_query
from app.kb.index_store import DEFAULT_INDEX_DIR, load_index

DEFAULT_TOP_K = 5
DEFAULT_SNIPPET_CHARS = 180

# 单条命中统一字段（text_snippet / text 可选）
RESULT_FIELDS = (
    "chunk_id",
    "score",
    "title",
    "url",
    "section_path",
    "is_code",
    "text_snippet",
)

# 进程内缓存：key=绝对路径。多 worker 时各自一份（uvicorn 多进程不共享内存）。
_INDEX_CACHE: dict[str, dict[str, Any]] = {}


def get_index(index_dir: Path | str = DEFAULT_INDEX_DIR) -> dict[str, Any]:
    """加载（或复用缓存）索引。"""
    key = str(Path(index_dir).resolve())
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = load_index(index_dir)
    return _INDEX_CACHE[key]


def clear_index_cache() -> None:
    """测试或重建索引后必须清空，否则会继续用旧向量。"""
    _INDEX_CACHE.clear()


def _snippet(text: str, *, limit: int = DEFAULT_SNIPPET_CHARS) -> str:
    """截断预览：方便人眼验收命中对不对；喂模型请用全文 text。"""
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def format_hit(
    meta: dict[str, Any],
    score: float,
    *,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    include_snippet: bool = True,
    include_text: bool = False,
) -> dict[str, Any]:
    """统一命中结果 schema，避免各处手写字段不一致。"""
    row: dict[str, Any] = {
        "chunk_id": meta.get("chunk_id"),
        "score": round(float(score), 6),
        "title": meta.get("title"),
        "url": meta.get("url"),
        "section_path": meta.get("section_path") or "",
        "is_code": bool(meta.get("is_code")),
    }
    text = str(meta.get("text") or "")
    if include_snippet:
        row["text_snippet"] = _snippet(text, limit=snippet_chars)
    if include_text:
        row["text"] = text
    return row


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    category: str | None = None,
    include_snippet: bool = True,
    include_text: bool = False,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> dict[str, Any]:
    """检索入口。

    返回：
      {
        "query": str,
        "top_k": int,
        "retrieve_ms": int,
        "results": [ {chunk_id, score, title, url, ...}, ... ]
      }
    """
    started = time.perf_counter()
    index = get_index(index_dir)
    dim = int(index["dim"])
    qvec = embed_query(query, dim=dim)
    scores = cosine_scores(qvec, index["vectors"])

    ranked: list[tuple[float, int]] = []
    for i, score in enumerate(scores):
        meta = index["meta"][i]
        if category and str(meta.get("category") or "") != category:
            continue
        ranked.append((float(score), i))
    # 分数高者优先；同分时保持稳定排序依赖 Python sort 稳定性
    ranked.sort(key=lambda x: x[0], reverse=True)

    results = [
        format_hit(
            index["meta"][i],
            score,
            snippet_chars=snippet_chars,
            include_snippet=include_snippet,
            include_text=include_text,
        )
        for score, i in ranked[: max(top_k, 0)]
    ]
    retrieve_ms = int((time.perf_counter() - started) * 1000)
    return {
        "query": query,
        "top_k": top_k,
        "retrieve_ms": retrieve_ms,
        "results": results,
    }
