"""检索 v2：混合检索（向量 + BM25）→ 过滤/去重 → TopK。

==========================================================================
做什么
==========================================================================
1) 向量检索（余弦）+ 关键词检索（BM25）
2) 按 hybrid_weight 加权融合，取候选池
3) 过滤：低分 / 过短 / 噪声 chunk
4) 去重：同 doc_id / 同 url / 高相似正文
5) 落盘可观测：retrieve_candidates / retrieve_kept / hybrid_weight / dedup_dropped
6) Day18：进程内索引/BM25 缓存命中计数（reset_cache_counters / get_cache_counters）

==========================================================================
为什么这样设计
==========================================================================
1) 纯向量偏词面哈希，专名/错误码类问题 BM25 更稳；融合后引用覆盖不掉。
2) 先放大候选再过滤/去重，避免 TopK 被同一文档占满。
3) 检索与 HTTP / LLM 解耦：CLI、RAG、Agent 工具共用 retrieve()。
4) cache 计数按「请求」统计：api 入口 reset，结束读 hit/miss 写入 jsonl。
==========================================================================
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.kb.bm25 import BM25Index, build_bm25_index, min_max_normalize
from app.kb.embedder import cosine_scores, embed_query, tokenize
from app.kb.index_store import DEFAULT_INDEX_DIR, load_index

DEFAULT_TOP_K = 5
DEFAULT_SNIPPET_CHARS = 180
DEFAULT_HYBRID_WEIGHT = 0.6
DEFAULT_MIN_SCORE = 0.05
DEFAULT_MIN_CHUNK_CHARS = 40
DEFAULT_CANDIDATE_MULTIPLIER = 4
DEFAULT_MAX_PER_DOC = 2
NEAR_DUP_JACCARD = 0.85

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

# 检索可观测字段（日志 / meta / requests.jsonl）
RETRIEVE_STAT_KEYS = (
    "retrieve_candidates",
    "retrieve_before_dedup",
    "retrieve_after_dedup",
    "retrieve_kept",
    "hybrid_weight",
    "dedup_dropped",
)

# 噪声：导航/广告残留（清洗后仍可能漏网）
_NOISE_LINE_RE = re.compile(
    r"^(首页|登录|注册|下一页|上一页|返回顶部|点击这里|广告|"
    r"相关推荐|分享到|微信扫一扫|版权所有|copyright|home\s*page|"
    r"sign\s*in|log\s*in|next\s*page|prev(?:ious)?\s*page)\b",
    re.IGNORECASE,
)

# 进程内缓存：key=绝对路径。多 worker 时各自一份。
_INDEX_CACHE: dict[str, dict[str, Any]] = {}
_BM25_CACHE: dict[str, BM25Index] = {}

# Day18：请求范围内的缓存命中计数（ask 开始 reset，结束读取）
_CACHE_COUNTERS: dict[str, int] = {"cache_hit": 0, "cache_miss": 0}


def reset_cache_counters() -> None:
    """每个 /ask 请求开始时清零。"""
    _CACHE_COUNTERS["cache_hit"] = 0
    _CACHE_COUNTERS["cache_miss"] = 0


def get_cache_counters() -> dict[str, int]:
    """返回当前请求累计的 cache_hit / cache_miss（拷贝）。"""
    return {
        "cache_hit": int(_CACHE_COUNTERS.get("cache_hit") or 0),
        "cache_miss": int(_CACHE_COUNTERS.get("cache_miss") or 0),
    }


def get_index(index_dir: Path | str = DEFAULT_INDEX_DIR) -> dict[str, Any]:
    """加载（或复用缓存）索引。"""
    key = str(Path(index_dir).resolve())
    if key in _INDEX_CACHE:
        _CACHE_COUNTERS["cache_hit"] += 1
        return _INDEX_CACHE[key]
    _CACHE_COUNTERS["cache_miss"] += 1
    _INDEX_CACHE[key] = load_index(index_dir)
    return _INDEX_CACHE[key]


def get_bm25(index_dir: Path | str = DEFAULT_INDEX_DIR) -> BM25Index:
    """加载（或复用缓存）BM25 统计，与向量索引同 key。"""
    key = str(Path(index_dir).resolve())
    if key in _BM25_CACHE:
        _CACHE_COUNTERS["cache_hit"] += 1
        return _BM25_CACHE[key]
    _CACHE_COUNTERS["cache_miss"] += 1
    index = get_index(index_dir)
    _BM25_CACHE[key] = build_bm25_index(index["meta"])
    return _BM25_CACHE[key]


def clear_index_cache() -> None:
    """测试或重建索引后必须清空，否则会继续用旧向量/BM25。"""
    _INDEX_CACHE.clear()
    _BM25_CACHE.clear()
    reset_cache_counters()


def retrieve_stat_fields(source: dict[str, Any] | None) -> dict[str, Any]:
    """从 retrieve / rag_pack 抽取可观测字段（跳过 None）。"""
    if not source:
        return {}
    out: dict[str, Any] = {}
    for key in RETRIEVE_STAT_KEYS:
        value = source.get(key)
        if value is not None:
            out[key] = value
    return out


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
        "doc_id": meta.get("doc_id"),
    }
    text = str(meta.get("text") or "")
    if include_snippet:
        row["text_snippet"] = _snippet(text, limit=snippet_chars)
    if include_text:
        row["text"] = text
    return row


def get_chunk(
    chunk_id: str,
    *,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
) -> dict[str, Any] | None:
    """按 chunk_id 取索引中的全文与元数据；不存在返回 None。"""
    cid = (chunk_id or "").strip()
    if not cid:
        return None
    index = get_index(index_dir)
    for meta in index["meta"]:
        if str(meta.get("chunk_id") or "") == cid:
            return {
                "chunk_id": meta.get("chunk_id"),
                "title": meta.get("title"),
                "url": meta.get("url"),
                "section_path": meta.get("section_path") or "",
                "is_code": bool(meta.get("is_code")),
                "category": meta.get("category"),
                "text": str(meta.get("text") or ""),
            }
    return None


def _resolve_hybrid_weight(explicit: float | None) -> float:
    if explicit is not None:
        return max(0.0, min(1.0, float(explicit)))
    return max(0.0, min(1.0, float(get_settings().rag_hybrid_weight)))


def _resolve_min_score(explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    return float(get_settings().rag_min_score)


def _token_set(text: str) -> set[str]:
    return set(tokenize(text or ""))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def is_noise_chunk(meta: dict[str, Any]) -> bool:
    """判断导航/广告类噪声片段。"""
    if bool(meta.get("is_code")):
        return False
    text = str(meta.get("text") or "").strip()
    if not text:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    noise_hits = sum(1 for ln in lines if _NOISE_LINE_RE.match(ln))
    if noise_hits >= 2 and noise_hits / len(lines) >= 0.5:
        return True
    # 整段极短且像菜单
    if len(text) < DEFAULT_MIN_CHUNK_CHARS and noise_hits >= 1:
        return True
    return False


def is_too_short_chunk(meta: dict[str, Any], *, min_chars: int = DEFAULT_MIN_CHUNK_CHARS) -> bool:
    """过短非代码 chunk：信息量不足，过滤掉。"""
    if bool(meta.get("is_code")):
        return False
    text = str(meta.get("text") or "").strip()
    char_len = int(meta.get("char_len") or len(text))
    return char_len < min_chars


def fuse_hybrid_scores(
    vector_scores: list[float],
    keyword_scores: list[float],
    *,
    hybrid_weight: float,
) -> list[float]:
    """加权融合：score = w * vec_norm + (1-w) * bm25_norm。"""
    w = max(0.0, min(1.0, hybrid_weight))
    vec_n = min_max_normalize(vector_scores)
    kw_n = min_max_normalize(keyword_scores)
    n = max(len(vec_n), len(kw_n))
    fused: list[float] = []
    for i in range(n):
        v = vec_n[i] if i < len(vec_n) else 0.0
        k = kw_n[i] if i < len(kw_n) else 0.0
        fused.append(w * v + (1.0 - w) * k)
    return fused


def _filter_candidates(
    scored: list[tuple[float, int]],
    meta_rows: list[dict[str, Any]],
    *,
    min_score: float,
) -> list[tuple[float, int]]:
    """低分 / 过短 / 噪声过滤。"""
    kept: list[tuple[float, int]] = []
    for score, idx in scored:
        if score < min_score:
            continue
        meta = meta_rows[idx]
        if is_too_short_chunk(meta):
            continue
        if is_noise_chunk(meta):
            continue
        kept.append((score, idx))
    return kept


def dedupe_candidates(
    scored: list[tuple[float, int]],
    meta_rows: list[dict[str, Any]],
    *,
    max_per_doc: int = DEFAULT_MAX_PER_DOC,
    near_dup_jaccard: float = NEAR_DUP_JACCARD,
) -> tuple[list[tuple[float, int]], int]:
    """按 doc_id / url / 高相似正文去重；输入须已按分数降序。

    返回 (保留列表, 丢弃数)。
    """
    kept: list[tuple[float, int]] = []
    kept_meta: list[dict[str, Any]] = []
    kept_tokens: list[set[str]] = []
    per_doc: dict[str, int] = {}
    dropped = 0

    for score, idx in scored:
        meta = meta_rows[idx]
        doc_id = str(meta.get("doc_id") or "")
        url = str(meta.get("url") or "").strip()
        text = str(meta.get("text") or "")
        tokens = _token_set(text)

        if doc_id and per_doc.get(doc_id, 0) >= max_per_doc:
            dropped += 1
            continue

        # 高相似正文合并（同 doc/url 近重复也落在此阈值内）
        if any(_jaccard(tokens, prev_tok) >= near_dup_jaccard for prev_tok in kept_tokens):
            dropped += 1
            continue
        # 同 url 且已保留过：即使文本略有差异也合并（避免同页多段刷屏）
        if url and any(url == str(m.get("url") or "").strip() for m in kept_meta):
            # 同 url 允许多 section，但若 section_path 也相同则丢
            section = str(meta.get("section_path") or "")
            if any(
                url == str(m.get("url") or "").strip()
                and section == str(m.get("section_path") or "")
                for m in kept_meta
            ):
                dropped += 1
                continue

        kept.append((score, idx))
        kept_meta.append(meta)
        kept_tokens.append(tokens)
        if doc_id:
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1

    return kept, dropped


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    category: str | None = None,
    include_snippet: bool = True,
    include_text: bool = False,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    hybrid_weight: float | None = None,
    min_score: float | None = None,
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
) -> dict[str, Any]:
    """混合检索入口。

    返回：
      {
        "query", "top_k", "retrieve_ms", "results",
        "retrieve_candidates",          # 候选池（过滤前）
        "retrieve_before_dedup",        # 过滤后、去重前
        "retrieve_after_dedup",         # 去重后（截 TopK 前）
        "retrieve_kept",                # 最终保留
        "hybrid_weight", "dedup_dropped",
      }
    """
    started = time.perf_counter()
    weight = _resolve_hybrid_weight(hybrid_weight)
    score_floor = _resolve_min_score(min_score)
    k = max(int(top_k), 0)

    index = get_index(index_dir)
    meta_rows: list[dict[str, Any]] = index["meta"]
    dim = int(index["dim"])

    qvec = embed_query(query, dim=dim)
    vector_scores = cosine_scores(qvec, index["vectors"])
    keyword_scores = get_bm25(index_dir).scores(query)
    fused = fuse_hybrid_scores(vector_scores, keyword_scores, hybrid_weight=weight)

    ranked: list[tuple[float, int]] = []
    for i, score in enumerate(fused):
        meta = meta_rows[i]
        if category and str(meta.get("category") or "") != category:
            continue
        ranked.append((float(score), i))
    ranked.sort(key=lambda x: x[0], reverse=True)

    pool_n = max(k * max(int(candidate_multiplier), 1), k)
    pool = ranked[:pool_n] if pool_n else []
    retrieve_candidates = len(pool)

    filtered = _filter_candidates(pool, meta_rows, min_score=score_floor)
    retrieve_before_dedup = len(filtered)
    deduped, dedup_dropped = dedupe_candidates(filtered, meta_rows)
    retrieve_after_dedup = len(deduped)
    selected = deduped[:k]

    results = [
        format_hit(
            meta_rows[i],
            score,
            snippet_chars=snippet_chars,
            include_snippet=include_snippet,
            include_text=include_text,
        )
        for score, i in selected
    ]
    retrieve_ms = int((time.perf_counter() - started) * 1000)
    return {
        "query": query,
        "top_k": top_k,
        "retrieve_ms": retrieve_ms,
        "results": results,
        "retrieve_candidates": retrieve_candidates,
        "retrieve_before_dedup": retrieve_before_dedup,
        "retrieve_after_dedup": retrieve_after_dedup,
        "retrieve_kept": len(results),
        "hybrid_weight": round(weight, 4),
        "dedup_dropped": dedup_dropped,
    }
