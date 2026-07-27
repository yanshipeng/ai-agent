#!/usr/bin/env python3
"""薄 CLI：从 chunks.jsonl 建向量索引。

【在整条链路里的位置】
  chunks.jsonl
       ↓
  【本脚本】→ 调用 app.kb.index_store / embedder
       ↓
  data/stability_kb/index/（meta.jsonl + vectors.jsonl + manifest.json）

【常用命令】
  python scripts/build_kb_index.py
  python scripts/build_kb_index.py --dim 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb.embedder import DEFAULT_DIM
from app.kb.cli_log import fail, ok, step
from app.kb.index_store import build_index_from_chunks_file
from app.kb.retriever import clear_index_cache

DEFAULT_CHUNKS = ROOT / "data" / "stability_kb" / "chunks.jsonl"
DEFAULT_INDEX_DIR = ROOT / "data" / "stability_kb" / "index"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="【薄 CLI】chunks → 向量索引。核心在 app.kb.index_store / embedder。",
    )
    p.add_argument(
        "--chunks",
        default=str(DEFAULT_CHUNKS),
        help="切块 JSONL（默认 data/stability_kb/chunks.jsonl）",
    )
    p.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="索引输出目录（默认 data/stability_kb/index）",
    )
    p.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help=f"向量维度（默认 {DEFAULT_DIM}）",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        fail(f"chunks not found: {chunks_path}")
        return 1

    step("建索引开始", "chunks → 向量 index（核心：embedder + index_store）")
    print(f"chunks : {chunks_path}")
    print(f"index  : {args.index_dir}")
    print(f"dim    : {args.dim}")
    print(f"core   : app.kb.index_store / embedder")

    report = build_index_from_chunks_file(
        chunks_path,
        index_dir=args.index_dir,
        dim=args.dim,
        progress=True,
    )
    clear_index_cache()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok(f"建索引完成 size={report['size']} dim={report['dim']}；可 retrieve 或 /ask?mode=rag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
