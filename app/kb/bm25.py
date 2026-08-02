"""简易 BM25（Okapi）关键词检索。

==========================================================================
做什么
==========================================================================
对索引里每条 chunk 的正文建倒排统计，查询时给出 BM25 分数。
与向量检索并列，供 hybrid 融合使用。

==========================================================================
为什么自己实现、不用第三方
==========================================================================
1) 教学项目少依赖，结果可复现。
2) 分词复用 embedder.tokenize（字/词 + bigram），与向量侧词面一致。
3) 索引不大（几百～几千 chunk），内存倒排足够快。
==========================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.kb.embedder import tokenize

# Okapi BM25 经典默认；对本库短中文 chunk 足够稳
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(frozen=True)
class BM25Index:
    """一份索引对应的 BM25 统计（与 meta 行一一对应）。"""

    doc_tfs: list[dict[str, int]]
    doc_lens: list[int]
    avgdl: float
    df: dict[str, int]
    n_docs: int
    k1: float = BM25_K1
    b: float = BM25_B

    def idf(self, term: str) -> float:
        """标准 BM25 IDF，避免负值。"""
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        """查询 → 每条文档的 BM25 分（未归一化）。"""
        q_terms = tokenize(query)
        if not q_terms or self.n_docs == 0:
            return [0.0] * self.n_docs

        # 查询侧去重：同一 term 只算一次（教学简化；可改为 qtf）
        unique_terms = list(dict.fromkeys(q_terms))
        out = [0.0] * self.n_docs
        avgdl = self.avgdl if self.avgdl > 0 else 1.0
        k1 = self.k1
        b = self.b

        for term in unique_terms:
            idf = self.idf(term)
            if idf <= 0:
                continue
            for i, tf_map in enumerate(self.doc_tfs):
                tf = tf_map.get(term)
                if not tf:
                    continue
                dl = self.doc_lens[i] or 1
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                out[i] += idf * (tf * (k1 + 1.0) / denom)
        return out


def _doc_text_for_bm25(meta: dict[str, Any]) -> str:
    """拼检索文本：标题/路径/标签/正文，提高关键词命中。"""
    parts = [
        str(meta.get("title") or ""),
        str(meta.get("section_path") or ""),
        str(meta.get("category_name") or ""),
        " ".join(str(t) for t in (meta.get("tags") or [])),
        str(meta.get("text") or ""),
    ]
    return "\n".join(p for p in parts if p)


def build_bm25_index(meta_rows: list[dict[str, Any]]) -> BM25Index:
    """从 index meta 行构建 BM25 统计。"""
    doc_tfs: list[dict[str, int]] = []
    doc_lens: list[int] = []
    df: dict[str, int] = {}

    for meta in meta_rows:
        tokens = tokenize(_doc_text_for_bm25(meta))
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        doc_tfs.append(tf)
        doc_lens.append(len(tokens))
        for term in tf:
            df[term] = df.get(term, 0) + 1

    n_docs = len(doc_tfs)
    avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0
    return BM25Index(
        doc_tfs=doc_tfs,
        doc_lens=doc_lens,
        avgdl=avgdl,
        df=df,
        n_docs=n_docs,
    )


def min_max_normalize(scores: list[float]) -> list[float]:
    """线性归一到 [0, 1]，便于与余弦分加权融合。

    全相等时：若分数 > 0 则记为 1.0（避免单文档/同分被 min_score 滤光）；
    否则记为 0.0。
    """
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo <= 1e-12:
        fill = 1.0 if hi > 0.0 else 0.0
        return [fill] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]
