"""知识库向量索引：从 chunks.jsonl 建索引并落盘。

==========================================================================
做什么
==========================================================================
离线把每个 chunk 变成向量，连同 title/url/text 等元数据写到磁盘。
查询时只需对「用户问题」算一次向量，再和已存向量比相似度。

目录默认：data/stability_kb/index/
  - meta.jsonl     与向量行一一对应的 chunk 元数据（含全文 text）
  - vectors.jsonl  每行一个向量（纯 JSON list，避免强依赖 numpy）
  - manifest.json  dim / size / 生成时间等

==========================================================================
为什么要离线建索引（而不是每次提问现算全部 chunk）
==========================================================================
1) Embedding 相对贵：N 条 chunk 建一次，查询只算 1 次问题向量。
2) meta 与 vectors 分开：换展示字段不必重算向量；排查时也好打开看。
3) 用 JSONL 而不是专用二进制：教学项目可读、可 diff、无额外依赖。
==========================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.kb.embedder import DEFAULT_DIM, embed_texts
from app.kb.jsonl_io import load_jsonl, utc_now_iso, write_jsonl

DEFAULT_INDEX_DIR = Path("data/stability_kb/index")
DEFAULT_CHUNKS_PATH = Path("data/stability_kb/chunks.jsonl")


def _chunk_embed_text(chunk: dict[str, Any]) -> str:
    """拼进向量的文本：标题 + 章节 + 分类 + 标签 + 正文。

    为什么不只 embed 正文？
      很多排障问题的关键词出现在标题（如「ANR 分析套路」），
      正文可能是一长段堆栈。标题/标签进向量，能提高「问得到」的概率。
    """
    parts = [
        str(chunk.get("title") or ""),
        str(chunk.get("section_path") or ""),
        str(chunk.get("category_name") or ""),
        " ".join(str(t) for t in (chunk.get("tags") or [])),
        str(chunk.get("text") or ""),
    ]
    return "\n".join(p for p in parts if p)


def build_index(
    chunks: list[dict[str, Any]],
    *,
    dim: int = DEFAULT_DIM,
    progress: bool = False,
) -> dict[str, Any]:
    """内存中构建索引结构（尚未写盘）。

    返回：
      {
        "dim": int,
        "size": int,
        "meta": [chunk元数据...],   # 与 vectors 下标严格一一对应
        "vectors": [[float...], ...],
      }
    """
    texts = [_chunk_embed_text(c) for c in chunks]
    if progress:
        print(f"embedding {len(texts)} chunks, dim={dim} ...")
    vectors = embed_texts(texts, dim=dim)
    meta: list[dict[str, Any]] = []
    for c in chunks:
        meta.append(
            {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "category": c.get("category"),
                "category_name": c.get("category_name"),
                "title": c.get("title"),
                "url": c.get("url"),
                "source": c.get("source"),
                "tags": list(c.get("tags") or []),
                "section_path": c.get("section_path") or "",
                "is_code": bool(c.get("is_code")),
                "char_len": c.get("char_len"),
                # 保留全文：retrieve(include_text=True) / RAG 拼 Context 时用
                "text": c.get("text") or "",
            }
        )
    return {"dim": dim, "size": len(meta), "meta": meta, "vectors": vectors}


def save_index(index: dict[str, Any], index_dir: Path | str) -> Path:
    """把索引写到目录。meta 与 vectors 分行存储，便于人工抽查。"""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(index_dir / "meta.jsonl", index["meta"])
    with (index_dir / "vectors.jsonl").open("w", encoding="utf-8") as fp:
        for vec in index["vectors"]:
            fp.write(json.dumps(vec, ensure_ascii=False) + "\n")
    manifest = {
        "generated_at": utc_now_iso(),
        "dim": index["dim"],
        "size": index["size"],
        "embedder": "hashing_tf_char_ngram_v0",
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index_dir


def load_index(index_dir: Path | str) -> dict[str, Any]:
    """从目录加载索引。meta 行数必须等于 vectors 行数，否则视为损坏。"""
    index_dir = Path(index_dir)
    meta = load_jsonl(index_dir / "meta.jsonl")
    vectors: list[list[float]] = []
    with (index_dir / "vectors.jsonl").open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                vectors.append(json.loads(line))
    if len(meta) != len(vectors):
        raise ValueError(
            f"index corrupt: meta={len(meta)} vectors={len(vectors)} in {index_dir}"
        )
    manifest_path = index_dir / "manifest.json"
    dim = len(vectors[0]) if vectors else DEFAULT_DIM
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dim = int(manifest.get("dim") or dim)
    return {"dim": dim, "size": len(meta), "meta": meta, "vectors": vectors}


def build_index_from_chunks_file(
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    *,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    dim: int = DEFAULT_DIM,
    progress: bool = False,
) -> dict[str, Any]:
    """读 chunks.jsonl → 建索引 → 落盘。CLI / 一键脚本的主入口。"""
    chunks = load_jsonl(chunks_path)
    index = build_index(chunks, dim=dim, progress=progress)
    save_index(index, index_dir)
    return {
        "chunks_path": str(chunks_path),
        "index_dir": str(index_dir),
        "dim": index["dim"],
        "size": index["size"],
        "generated_at": utc_now_iso(),
    }
