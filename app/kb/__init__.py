"""知识库语料处理核心包（前期数据处理层）。

==========================================================================
大白话：后厨 vs 点餐员
==========================================================================
真正干活的逻辑在本包（后厨）；scripts/*.py 只是薄 CLI（点餐员）：
  读参数 → 调用本包 → 写文件 / 打印进度。

为什么要下沉到 app/kb，而不是只写在 scripts 里？
  FastAPI /ask、评测、单测都要 import 同一套 cleaner/chunker/retrieve/rag，
  避免「脚本一份、服务一份」双份逻辑漂移。

流水线：
  articles → cleaner → docs → chunker → chunks
           → embedder + index_store → index/
           → retriever / rag → /ask?mode=rag
==========================================================================
"""

from app.kb.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    build_chunk_row,
    chunk_doc,
    chunk_docs,
)
from app.kb.cleaner import (
    build_doc_from_article,
    build_docs_from_articles,
    clean_markdown,
    dedupe_key,
    html_to_markdown,
    is_bad_content,
)
from app.kb.embedder import DEFAULT_DIM, cosine_scores, embed_query, embed_texts
from app.kb.index_store import (
    build_index,
    build_index_from_chunks_file,
    load_index,
    save_index,
)
from app.kb.jsonl_io import load_jsonl, sha256_8, utc_now_iso, write_jsonl
from app.kb.rag import (
    ASK_MODE_LLM,
    ASK_MODE_RAG,
    build_rag_messages,
    hits_to_citations,
    run_rag_retrieve,
)
from app.kb.retriever import RESULT_FIELDS, clear_index_cache, get_index, retrieve

__all__ = [
    "ASK_MODE_LLM",
    "ASK_MODE_RAG",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_DIM",
    "DEFAULT_OVERLAP",
    "RESULT_FIELDS",
    "build_chunk_row",
    "build_doc_from_article",
    "build_docs_from_articles",
    "build_index",
    "build_index_from_chunks_file",
    "build_rag_messages",
    "chunk_doc",
    "chunk_docs",
    "clean_markdown",
    "clear_index_cache",
    "cosine_scores",
    "dedupe_key",
    "embed_query",
    "embed_texts",
    "get_index",
    "hits_to_citations",
    "html_to_markdown",
    "is_bad_content",
    "load_index",
    "load_jsonl",
    "retrieve",
    "run_rag_retrieve",
    "save_index",
    "sha256_8",
    "utc_now_iso",
    "write_jsonl",
]
