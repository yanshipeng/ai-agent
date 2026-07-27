"""本地文本向量化（Embedding v0）。

==========================================================================
做什么
==========================================================================
把一段文字变成固定长度的数字向量。意思相近的文字，向量夹角更小。

==========================================================================
为什么这么做（设计理由）
==========================================================================
1) 第二周目标是「先把检索链路跑通」，不是立刻上最强语义模型。
2) 不依赖外部 Embedding API / 不下载几 GB 模型：本地可复现、无 Key、可离线。
3) 用 Hashing Trick：不必维护巨大词表；中文用字 unigram + bigram，对短词更稳。
4) L2 归一化后，点积 = 余弦相似度，检索实现简单。
5) 对外只暴露 embed_texts / embed_query / cosine_scores：
   以后换成 BGE / OpenAI Embedding，检索层几乎不用改。

代价（要心里有数）：
  偏「词面重合」，语义弱于专用模型。「卡死」和「ANR」若正文都出现才更稳。
==========================================================================
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

# 向量维度（哈希桶数）：越大越细、碰撞越少，但内存与算力也更大。
# 781 条 chunk × 1024 维对本教学项目足够；上万条可再评估。
DEFAULT_DIM = 1024
# 字符 n-gram：中文单字（1）+ 相邻两字（2）对检索帮助大；更大 n 噪声也更大。
NGRAM_MIN = 1
NGRAM_MAX = 2

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """极简分词：英文单词 + 中文单字，再组成 bigram。

    为什么不用结巴等分词器？
      - 少依赖、结果稳定、便于教学复现；
      - bigram 能部分弥补「不切词」的损失（如「内」「存」→「内存」）。
    """
    raw = _TOKEN_RE.findall((text or "").lower())
    if not raw:
        return []
    grams: list[str] = []
    if NGRAM_MIN <= 1 <= NGRAM_MAX:
        grams.extend(raw)
    # 相邻 bigram：对中文短语匹配特别有用
    if NGRAM_MAX >= 2 and len(raw) >= 2:
        for i in range(len(raw) - 1):
            grams.append(raw[i] + raw[i + 1])
    return grams


def _stable_hash(token: str, dim: int) -> int:
    """把 token 映射到 [0, dim)。

    为什么不用内置 hash()？
      Python 的 hash() 受 PYTHONHASHSEED 影响，进程间不可复现，索引会「漂移」。
      这里用 FNV-1a 风格，纯函数、跨进程稳定。
    """
    h = 2166136261
    for ch in token.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h % dim


def hashing_tf_vector(text: str, *, dim: int = DEFAULT_DIM) -> list[float]:
    """Hashing Trick：词频进固定桶（未归一化）。

    有符号哈希（±）：同一桶内正负抵消一部分碰撞噪声，是常见工程技巧。
    """
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    counts = Counter(tokens)
    for tok, tf in counts.items():
        idx = _stable_hash(tok, dim)
        sign = 1.0 if (_stable_hash(tok + "#", dim) % 2 == 0) else -1.0
        vec[idx] += sign * float(tf)
    return vec


def l2_normalize(vec: list[float]) -> list[float]:
    """L2 归一化：||v||=1 时，点积即余弦相似度，检索时不必再除范数。"""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return vec
    return [v / norm for v in vec]


def embed_texts(texts: Iterable[str], *, dim: int = DEFAULT_DIM) -> list[list[float]]:
    """批量文本 → 向量。建索引时用；查询与文档必须同一套算法与 dim。"""
    return [l2_normalize(hashing_tf_vector(t, dim=dim)) for t in texts]


def embed_query(query: str, *, dim: int = DEFAULT_DIM) -> list[float]:
    """单条查询向量。与 embed_texts 同算法，否则分数不可比。"""
    return l2_normalize(hashing_tf_vector(query, dim=dim))


def cosine_scores(query_vec: list[float], doc_matrix: list[list[float]]) -> list[float]:
    """查询向量与文档矩阵的相似度（已 L2 归一 ⇒ 点积）。

    为什么暴力扫全库？
      当前几百～几千条足够快；上十万再换 FAISS / 向量库，接口可保持不变。
    """
    scores: list[float] = []
    for row in doc_matrix:
        scores.append(sum(a * b for a, b in zip(query_vec, row, strict=False)))
    return scores
