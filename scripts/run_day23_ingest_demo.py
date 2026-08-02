#!/usr/bin/env python3
"""Day23 Demo：同一批 docs 入库两次，展示增量；再回滚到旧版本。

【常用】
  python scripts/run_day23_ingest_demo.py
  python scripts/run_day23_ingest_demo.py --docs data/stability_kb/docs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.kb.ingest_pipeline import rollback_dataset, run_ingest_from_docs
from app.kb.dataset_registry import active_dataset_info


def main() -> int:
    parser = argparse.ArgumentParser(description="Day23 incremental ingest demo")
    parser.add_argument("--docs", default=None)
    parser.add_argument("--dim", type=int, default=1024)
    args = parser.parse_args()

    settings = get_settings()
    docs = Path(args.docs) if args.docs else Path(settings.kb_docs_path)
    if not docs.is_absolute():
        docs = ROOT / docs
    if not docs.exists():
        print(f"docs not found: {docs}")
        print("请先跑：python scripts/build_stability_docs.py")
        return 1

    print(f"docs={docs}")
    print("--- 第一次入库（全量/基线）---")
    r1 = run_ingest_from_docs(docs, incremental=True, dim=args.dim)
    print(json.dumps({k: r1[k] for k in (
        "dataset_version", "docs_total", "docs_added", "docs_changed",
        "docs_unchanged", "chunks_rebuilt", "chunks_reused", "vectors_embedded",
    ) if k in r1}, ensure_ascii=False, indent=2))

    print("--- 第二次入库（相同数据，期望 rebuilt≈0）---")
    r2 = run_ingest_from_docs(docs, incremental=True, dim=args.dim)
    print(json.dumps({k: r2[k] for k in (
        "dataset_version", "docs_total", "docs_added", "docs_changed",
        "docs_unchanged", "chunks_rebuilt", "chunks_reused", "vectors_embedded",
        "parent_version",
    ) if k in r2}, ensure_ascii=False, indent=2))

    saved = r2["vectors_embedded"] < max(r1["vectors_embedded"], 1)
    print(f"incremental_ok={saved}  (2nd embedded {r2['vectors_embedded']} << 1st {r1['vectors_embedded']})")

    print("--- 回滚到第一次版本 ---")
    rolled = rollback_dataset(r1["dataset_version"])
    print(json.dumps({
        "dataset_version": rolled.get("dataset_version"),
        "index_dir": rolled.get("index_dir"),
        "current": rolled.get("current"),
    }, ensure_ascii=False, indent=2))
    print("active:", json.dumps(active_dataset_info(), ensure_ascii=False, indent=2))
    return 0 if saved else 2


if __name__ == "__main__":
    raise SystemExit(main())
