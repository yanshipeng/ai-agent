#!/usr/bin/env python3
"""薄 CLI（命令行小入口）：把「干净长文」切成「可检索小段」。

【这是什么】
  本文件是「点餐员」，不负责真正切块。
  真正的切块逻辑在：app/kb/chunker.py（后厨）。

【在整条链路里的位置】
  docs.jsonl（一整篇干净文章）
       ↓
  【本脚本】→ 调用 app.kb.chunker
       ↓
  产出 chunks.jsonl（一小段一小段，后面问答检索用这一层）

【为什么要切块】
  大模型一次看不了太长文字。
  用户问「ANR 怎么查」，系统应找出相关「段落」，而不是整篇 8000 字。
  所以把文章按小标题/长度切开；代码块单独成一块，方便「问代码怎么写」。

【本脚本只做 3 件事】
  1. 解析参数（chunk 多长、重叠多少）
  2. 调用 chunk_docs(...)
  3. 写入 chunks.jsonl 与 chunks_report.json

【常用命令】
  python scripts/chunk_stability_docs.py
  python scripts/chunk_stability_docs.py --chunk-size 1000 --overlap 120
  python scripts/chunk_stability_docs.py --limit 3   # 调试用
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_docs
from app.kb.cli_log import ok, step
from app.kb.jsonl_io import load_jsonl, write_jsonl

DEFAULT_DOCS = ROOT / "data" / "stability_kb" / "docs.jsonl"
DEFAULT_OUT = ROOT / "data" / "stability_kb" / "chunks.jsonl"
DEFAULT_REPORT = ROOT / "data" / "stability_kb" / "chunks_report.json"


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    p = argparse.ArgumentParser(
        description=(
            "【薄 CLI】切块 docs → chunks。"
            "核心逻辑在 app.kb.chunker，本脚本只负责传参与写文件。"
        ),
    )
    p.add_argument(
        "--docs",
        default=str(DEFAULT_DOCS),
        help="干净文档 JSONL（默认 data/stability_kb/docs.jsonl）",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="切块结果输出（默认 data/stability_kb/chunks.jsonl）",
    )
    p.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="切块汇总报告（默认 data/stability_kb/chunks_report.json）",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="普通文本块目标长度（字符数，默认 1000；建议 800–1200）",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="相邻块重叠字符数（默认 120；避免答案卡在切割缝上）",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 篇文档（调试用；正式跑不要加）",
    )
    return p


def main() -> int:
    """入口：读 docs → 调核心切块 → 写 chunks。"""
    args = build_parser().parse_args()

    step("切块开始", "docs → chunks（核心：app.kb.chunker）")
    docs = load_jsonl(args.docs)
    if args.limit is not None:
        docs = docs[: args.limit]

    print(f"docs : {len(docs)} from {args.docs}")
    print(f"core : app.kb.chunker  size={args.chunk_size} overlap={args.overlap}")

    chunks, report = chunk_docs(
        docs,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        progress=True,
    )
    write_jsonl(args.out, chunks)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"\nchunks={report['chunks']} code={report['code_chunks']} "
        f"avg_len={report['avg_char_len']}"
    )
    print(f"out    => {args.out}")
    print(f"report => {args.report}")
    ok("切块完成：下一步可 python scripts/build_kb_index.py 建索引")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
