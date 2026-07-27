#!/usr/bin/env python3
"""薄 CLI（命令行小入口）：把「原始采集」洗成「干净文档」。

【这是什么】
  本文件是「点餐员」，不负责真正炒菜。
  真正的清洗/去重逻辑在：app/kb/cleaner.py（后厨）。

【在整条链路里的位置】
  采集 articles.jsonl
       ↓
  【本脚本】→ 调用 app.kb.cleaner
       ↓
  产出 docs.jsonl（干净文档，统一字段）
       ↓
  再由 chunk 脚本切成 chunks.jsonl（检索小段）

【本脚本只做 3 件事】
  1. 解析命令行参数（读哪个文件、输出到哪、是否联网重抓）
  2. 调用 build_docs_from_articles(...)
  3. 把结果写入 docs.jsonl，并把汇总写入 docs_report.json

【常用命令】
  # 只用本地已有正文清洗（不联网，推荐日常）
  python scripts/build_stability_docs.py --no-refetch

  # 重新打开网页抓正文再清洗（更慢，需要网络）
  python scripts/build_stability_docs.py

  # 调试：只处理前 5 条
  python scripts/build_stability_docs.py --no-refetch --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 保证从任意目录运行脚本时，都能 import 到 app 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb.cleaner import DEFAULT_DELAY_SEC, HAS_BS4, build_docs_from_articles
from app.kb.cli_log import ok, step
from app.kb.jsonl_io import load_jsonl, write_jsonl

# 默认输入/输出路径（都在 data/stability_kb/ 下）
DEFAULT_ARTICLES = ROOT / "data" / "stability_kb" / "articles.jsonl"
DEFAULT_OUT = ROOT / "data" / "stability_kb" / "docs.jsonl"
DEFAULT_REPORT = ROOT / "data" / "stability_kb" / "docs_report.json"


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数：告诉用户「这个开关是干什么的」。"""
    p = argparse.ArgumentParser(
        description=(
            "【薄 CLI】清洗 articles → docs。"
            "核心逻辑在 app.kb.cleaner，本脚本只负责传参与写文件。"
        ),
    )
    p.add_argument(
        "--from-articles",
        default=str(DEFAULT_ARTICLES),
        help="原始采集 JSONL（默认 data/stability_kb/articles.jsonl）",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="干净文档输出路径（默认 data/stability_kb/docs.jsonl）",
    )
    p.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="本次清洗汇总报告（默认 data/stability_kb/docs_report.json）",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 条（调试用；正式跑不要加，以免覆盖成不完整结果）",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SEC,
        help="联网重抓时每条之间的间隔秒数，降低被网站限流概率",
    )
    p.add_argument(
        "--no-refetch",
        dest="refetch",
        action="store_false",
        help="不重新打开网页；只用 articles 里已有的 text_excerpt 清洗",
    )
    # 默认会 refetch=True；加上 --no-refetch 后变成 False
    p.set_defaults(refetch=True)
    return p


def main() -> int:
    """入口：读参数 → 调核心 → 写文件。"""
    args = build_parser().parse_args()

    step("清洗开始", "articles → docs（核心：app.kb.cleaner）")
    # 1) 读入原始采集
    articles = load_jsonl(args.from_articles)
    if args.limit is not None:
        articles = articles[: args.limit]

    print(f"articles : {len(articles)} from {args.from_articles}")
    print(f"refetch  : {args.refetch}  bs4={HAS_BS4}  core=app.kb.cleaner")
    if not HAS_BS4:
        print("tip: pip install beautifulsoup4  # 联网转 Markdown 时正文质量更好")

    # 2) 调用核心清洗（真正干活的地方）
    docs, report = build_docs_from_articles(
        articles,
        refetch=args.refetch,
        delay_sec=args.delay,
        progress=True,
    )

    # 3) 落盘：干净文档 + 汇总报告
    write_jsonl(args.out, docs)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"\ndocs={report['docs']} dup={report['duplicates']} "
        f"skipped={report['skipped']}"
    )
    print(f"out    => {args.out}")
    print(f"report => {args.report}")
    ok("清洗完成：下一步可 python scripts/chunk_stability_docs.py 切块")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
