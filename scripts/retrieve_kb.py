#!/usr/bin/env python3
"""薄 CLI：对知识库做一次检索冒烟。

【常用命令】
  python scripts/retrieve_kb.py "Android ANR 怎么排查"
  python scripts/retrieve_kb.py "OOM 内存泄漏" --top-k 5
  python scripts/retrieve_kb.py "WebView 白屏" --category F
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb.retriever import DEFAULT_TOP_K, retrieve

DEFAULT_INDEX_DIR = ROOT / "data" / "stability_kb" / "index"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="【薄 CLI】query → TopK chunks。核心在 app.kb.retriever。",
    )
    p.add_argument("query", help="用户问题")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="返回条数")
    p.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="索引目录（默认 data/stability_kb/index）",
    )
    p.add_argument(
        "--category",
        default=None,
        help="可选：只搜某一类 A–G",
    )
    p.add_argument(
        "--no-snippet",
        action="store_true",
        help="不返回 text_snippet",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    index_dir = Path(args.index_dir)
    if not (index_dir / "manifest.json").exists():
        print(
            f"ERROR: index not found at {index_dir}. "
            "Run: python scripts/build_kb_index.py",
            file=sys.stderr,
        )
        return 1

    out = retrieve(
        args.query,
        top_k=args.top_k,
        index_dir=index_dir,
        category=args.category,
        include_snippet=not args.no_snippet,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(
        f"\nretrieve_ms={out['retrieve_ms']} "
        f"candidates={out.get('retrieve_candidates')} "
        f"before_dedup={out.get('retrieve_before_dedup')} → "
        f"after_dedup={out.get('retrieve_after_dedup')} "
        f"kept={out.get('retrieve_kept', len(out['results']))} "
        f"dedup_dropped={out.get('dedup_dropped')} "
        f"hybrid_weight={out.get('hybrid_weight')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
