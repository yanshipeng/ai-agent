"""Chunking v1 核心：把干净长文（docs）切成检索小段（chunks）。

==========================================================================
大白话
==========================================================================
一整篇文章太长：搜不准，也塞不进模型。切成「卡片」，每张带链接与章节路径。

==========================================================================
为什么按「标题 → 长度 → 代码块」切（v1 写死规则）
==========================================================================
1) 先按 # / ## / ###：保留 section_path，引用时知道在哪一节。
2) 段内再按 ~1000 字切、重叠 ~120 字：
   - 1000：兼顾上下文与 TopK 条数下的 prompt 体积；
   - 重叠：避免答案卡在切割缝上（上一段尾巴 + 下一段开头都有）。
3) ``` 代码块单独成块（is_code=true）：方便「给我命令/示例」类问题命中。
4) 过短碎块丢掉：减少噪声，避免检索被目录行带偏。

谁调用：scripts/chunk_stability_docs.py；或 from app.kb import chunk_docs
A/B：Day10 可对比 chunk_size 800 vs 1200（见 run_rag_eval --prepare-chunk-ab）
==========================================================================
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.kb.jsonl_io import utc_now_iso

DEFAULT_CHUNK_SIZE = 1000  # 目标块长：太大则 Context 臃肿，太小则语义破碎
DEFAULT_OVERLAP = 120  # 相邻块重叠：防止答案刚好落在切割边界
DEFAULT_MIN_TEXT_CHARS = 40  # 短于此时常是噪声行/目录，直接丢弃

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.M)
FENCE_RE = re.compile(r"(```[\s\S]*?```)")


def split_by_fences(text: str) -> list[tuple[str, bool]]:
    """按代码围栏切分，返回 [(piece, is_code), ...]。"""
    parts = FENCE_RE.split(text)
    out: list[tuple[str, bool]] = []
    for part in parts:
        if not part or not part.strip():
            continue
        is_code = part.startswith("```") and part.endswith("```")
        out.append((part.strip("\n") + ("\n" if is_code else ""), is_code))
    return out


def parse_heading_sections(md: str) -> list[dict[str, Any]]:
    """按 #/##/### 切成 section；无标题则整篇一个 section。"""
    matches = list(HEADING_RE.finditer(md))
    if not matches:
        return [{"section_path": "", "section_index": 0, "text": md.strip()}]

    sections: list[dict[str, Any]] = []
    preface = md[: matches[0].start()].strip()
    if preface:
        sections.append(
            {"section_path": "前言", "section_index": 0, "text": preface}
        )

    stack: list[tuple[int, str]] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        section_path = "/".join(t for _, t in stack)
        sections.append(
            {
                "section_path": section_path,
                "section_index": len(sections),
                "text": f"{'#' * level} {title}\n\n{body}".strip(),
            }
        )
    return sections


def split_text_with_overlap(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind("。"))
            if cut >= int(chunk_size * 0.4):
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
        if chunks and start >= end:
            start = end
    return chunks


def chunk_section_text(
    section_text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, bool]]:
    """返回 [(text, is_code), ...]。代码块独立；普通文本再按长度切。"""
    results: list[tuple[str, bool]] = []
    for piece, is_code in split_by_fences(section_text):
        if is_code:
            results.append((piece.strip() + "\n", True))
            continue
        for part in split_text_with_overlap(
            piece, chunk_size=chunk_size, overlap=overlap
        ):
            results.append((part, False))
    return results


def build_chunk_row(
    doc: dict[str, Any],
    *,
    section_index: int,
    section_path: str,
    chunk_index: int,
    text: str,
    is_code: bool,
) -> dict[str, Any]:
    doc_id = doc["doc_id"]
    return {
        "chunk_id": f"{doc_id}:{section_index}:{chunk_index}",
        "doc_id": doc_id,
        "category": doc.get("category"),
        "category_name": doc.get("category_name"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "source": doc.get("source"),
        "tags": list(doc.get("tags") or []),
        "section_path": section_path,
        "text": text,
        "char_len": len(text),
        "is_code": is_code,
    }


def chunk_doc(
    doc: dict[str, Any],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
) -> list[dict[str, Any]]:
    """把一篇干净文档切成多个 chunk。

    顺序：标题分段 → 代码块独立 → 长段按 chunk_size/overlap 再切 → 过滤过短文字。
    """
    content = str(doc.get("content") or "")
    sections = parse_heading_sections(content)
    rows: list[dict[str, Any]] = []
    for section in sections:
        pieces = chunk_section_text(
            section["text"], chunk_size=chunk_size, overlap=overlap
        )
        for chunk_index, (text, is_code) in enumerate(pieces):
            if not text.strip():
                continue
            if not is_code and len(text) < min_text_chars:
                continue
            rows.append(
                build_chunk_row(
                    doc,
                    section_index=int(section["section_index"]),
                    section_path=str(section["section_path"]),
                    chunk_index=chunk_index,
                    text=text,
                    is_code=is_code,
                )
            )

    if not rows and content.strip():
        rows.append(
            build_chunk_row(
                doc,
                section_index=0,
                section_path="",
                chunk_index=0,
                text=content.strip(),
                is_code=False,
            )
        )

    # 按 section 重编号 chunk_index，保证 id 连续
    from collections import defaultdict

    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sec = int(str(row["chunk_id"]).split(":")[1])
        buckets[sec].append(row)
    renum: list[dict[str, Any]] = []
    for sec, items in buckets.items():
        for i, row in enumerate(items):
            item = dict(row)
            item["chunk_id"] = f"{item['doc_id']}:{sec}:{i}"
            renum.append(item)
    return renum


def chunk_docs(
    docs: list[dict[str, Any]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """批量切块。

    返回：(chunks列表, report汇总字典)
    report 里有 chunks/code_chunks/avg_char_len/by_category 等。
    """
    all_chunks: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        chunks = chunk_doc(
            doc,
            chunk_size=chunk_size,
            overlap=overlap,
            min_text_chars=min_text_chars,
        )
        all_chunks.extend(chunks)
        if progress:
            code_n = sum(1 for c in chunks if c["is_code"])
            print(
                f"[{index}/{len(docs)}] {doc.get('doc_id')} "
                f"chunks={len(chunks)} code={code_n}"
            )

    report = {
        "generated_at": utc_now_iso(),
        "docs": len(docs),
        "chunks": len(all_chunks),
        "code_chunks": sum(1 for c in all_chunks if c["is_code"]),
        "avg_char_len": int(
            sum(c["char_len"] for c in all_chunks) / len(all_chunks)
        )
        if all_chunks
        else 0,
        "by_category": dict(Counter(c.get("category") for c in all_chunks)),
        "chunk_size_chars": chunk_size,
        "overlap_chars": overlap,
        "min_text_chars": min_text_chars,
    }
    return all_chunks, report
