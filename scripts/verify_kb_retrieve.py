#!/usr/bin/env python3
"""Day 8 检索验收：验证索引完整性 + retrieve() 契约 + 冒烟相关性。

【用途】
  在本地建好 index 之后跑一遍，确认「能查、格式对、基本相关」。

【常用命令】
  python scripts/verify_kb_retrieve.py
  python scripts/verify_kb_retrieve.py --index-dir data/stability_kb/index
  python scripts/verify_kb_retrieve.py --json   # 机器可读报告

【判定】
  - 硬失败（exit 1）：索引缺失/条数不一致、schema 不对、score 未降序、命中数为 0
  - 软失败（默认也算 fail）：10 条 query 中 TopK 不含期望关键词
  - 可用 --soft-warn 把主题冒烟降级为警告，仅硬检查决定退出码
  - 某 category 在索引中为 0 条时，自动跳过该类过滤检查（记 WARN）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb.retriever import RESULT_FIELDS, clear_index_cache, retrieve

DEFAULT_INDEX_DIR = ROOT / "data" / "stability_kb" / "index"
DEFAULT_CHUNKS = ROOT / "data" / "stability_kb" / "chunks.jsonl"
DEFAULT_TOP_K = 5
MAX_RETRIEVE_MS = 5_000

# 10 条冒烟 query（覆盖 A–F；G 语料若缺失则跳过 category 过滤）
# expect_any：TopK 的 title/snippet/url 中至少命中一个关键词即算相关
SMOKE_CASES: list[dict[str, Any]] = [
    {
        "id": "Q01",
        "query": "Android ANR 怎么排查",
        "expect_any": ["ANR", "anr", "traces", "卡顿", "无响应"],
        "category": "A",
    },
    {
        "id": "Q02",
        "query": "主线程卡顿怎么分析",
        "expect_any": ["卡顿", "主线程", "ANR", "anr", "掉帧", "Jank"],
        "category": "A",
    },
    {
        "id": "Q03",
        "query": "Crash 堆栈怎么看",
        "expect_any": ["Crash", "crash", "崩溃", "堆栈", "闪退"],
        "category": "B",
    },
    {
        "id": "Q04",
        "query": "App 启动白屏原因",
        "expect_any": ["白屏", "启动", "Splash", "splash", "冷启动"],
        "category": "B",
    },
    {
        "id": "Q05",
        "query": "OOM 内存泄漏怎么查",
        "expect_any": ["OOM", "内存", "泄漏", "leak", "Heap"],
        "category": "C",
    },
    {
        "id": "Q06",
        "query": "Bitmap 内存优化",
        "expect_any": ["Bitmap", "bitmap", "内存", "OOM", "图片"],
        "category": "C",
    },
    {
        "id": "Q07",
        "query": "网络请求超时重试",
        "expect_any": ["网络", "超时", "重试", "请求", "HTTP", "axios", "OkHttp"],
        "category": "D",
    },
    {
        "id": "Q08",
        "query": "定位不准或漂移怎么办",
        "expect_any": ["定位", "GPS", "经纬度", "导航", "地图", "漂移"],
        "category": "E",
    },
    {
        "id": "Q09",
        "query": "WebView 白屏怎么办",
        "expect_any": ["WebView", "webview", "白屏", "Hybrid", "JSBridge"],
        "category": "F",
    },
    {
        "id": "Q10",
        "query": "推送收不到消息怎么查",
        "expect_any": ["推送", "消息", "IM", "Push", "通知", "RocketMQ", "MQTT"],
        "category": "G",  # 本机可能无 G 类语料，category 过滤会自动跳过
    },
]

# retrieve 单条结果必须字段（与 app.kb.retriever 对齐；脚本侧再写一份防漂移）
HIT_REQUIRED = (
    "chunk_id",
    "score",
    "title",
    "url",
    "section_path",
    "text_snippet",
    "is_code",
)
OUTER_REQUIRED = ("query", "top_k", "retrieve_ms", "results")


def _ok(name: str, detail: str = "") -> dict[str, Any]:
    return {"name": name, "status": "pass", "detail": detail}


def _fail(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "fail", "detail": detail}


def _warn(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "warn", "detail": detail}


def check_index_files(index_dir: Path, chunks_path: Path | None) -> list[dict[str, Any]]:
    """检查索引文件是否存在、条数是否一致。"""
    checks: list[dict[str, Any]] = []
    manifest = index_dir / "manifest.json"
    meta = index_dir / "meta.jsonl"
    vectors = index_dir / "vectors.jsonl"

    if not manifest.exists():
        checks.append(_fail("index.manifest", f"缺失: {manifest}"))
        return checks
    checks.append(_ok("index.manifest", str(manifest)))

    for path, label in ((meta, "index.meta"), (vectors, "index.vectors")):
        if not path.exists():
            checks.append(_fail(label, f"缺失: {path}"))
        else:
            checks.append(_ok(label, str(path)))

    if not all(p.exists() for p in (meta, vectors)):
        return checks

    info = json.loads(manifest.read_text(encoding="utf-8"))
    meta_n = sum(1 for line in meta.open(encoding="utf-8") if line.strip())
    vec_n = sum(1 for line in vectors.open(encoding="utf-8") if line.strip())
    size = int(info.get("size") or -1)

    if meta_n != vec_n:
        checks.append(_fail("index.align", f"meta={meta_n} vectors={vec_n}"))
    else:
        checks.append(_ok("index.align", f"meta=vectors={meta_n}"))

    if size != meta_n:
        checks.append(_fail("index.size", f"manifest.size={size} meta={meta_n}"))
    else:
        checks.append(_ok("index.size", f"size={size}"))

    if chunks_path and chunks_path.exists():
        chunk_n = sum(1 for line in chunks_path.open(encoding="utf-8") if line.strip())
        if chunk_n != meta_n:
            checks.append(
                _fail("index.vs_chunks", f"chunks={chunk_n} index={meta_n}")
            )
        else:
            checks.append(_ok("index.vs_chunks", f"chunks=index={chunk_n}"))
    elif chunks_path:
        checks.append(_warn("index.vs_chunks", f"chunks 不存在，跳过: {chunks_path}"))

    return checks


def check_retrieve_schema(out: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    """检查 retrieve 外层与命中字段。"""
    checks: list[dict[str, Any]] = []
    missing_outer = [k for k in OUTER_REQUIRED if k not in out]
    if missing_outer:
        checks.append(_fail("retrieve.outer_fields", f"缺字段: {missing_outer}"))
        return checks
    checks.append(_ok("retrieve.outer_fields", ",".join(OUTER_REQUIRED)))

    ms = out["retrieve_ms"]
    if not isinstance(ms, int) or ms < 0 or ms > MAX_RETRIEVE_MS:
        checks.append(
            _fail(
                "retrieve.retrieve_ms",
                f"非法 retrieve_ms={ms!r}（期望 0..{MAX_RETRIEVE_MS} 的 int）",
            )
        )
    else:
        checks.append(_ok("retrieve.retrieve_ms", f"{ms}ms"))

    results = out["results"]
    if not isinstance(results, list) or len(results) == 0:
        checks.append(_fail("retrieve.non_empty", f"results={results!r}"))
        return checks
    if len(results) > top_k:
        checks.append(_fail("retrieve.top_k", f"len={len(results)} > top_k={top_k}"))
    else:
        checks.append(_ok("retrieve.top_k", f"hits={len(results)} top_k={top_k}"))

    bad_rows: list[str] = []
    for i, row in enumerate(results):
        miss = [k for k in HIT_REQUIRED if k not in row]
        if miss:
            bad_rows.append(f"[{i}]缺{miss}")
            continue
        if not isinstance(row["score"], (int, float)):
            bad_rows.append(f"[{i}]score类型={type(row['score'])}")
        if not isinstance(row["is_code"], bool):
            bad_rows.append(f"[{i}]is_code类型={type(row['is_code'])}")
    if bad_rows:
        checks.append(_fail("retrieve.hit_fields", "; ".join(bad_rows[:5])))
    else:
        checks.append(_ok("retrieve.hit_fields", ",".join(HIT_REQUIRED)))

    scores = [float(r["score"]) for r in results]
    if scores != sorted(scores, reverse=True):
        checks.append(_fail("retrieve.score_desc", f"scores={scores}"))
    else:
        checks.append(_ok("retrieve.score_desc", f"top={scores[0]:.4f}"))

    return checks


def _haystack(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(k) or "")
        for k in ("title", "text_snippet", "section_path", "url")
    )


def _category_counts(index_dir: Path) -> dict[str, int]:
    """统计索引里各类别条数（用于空类跳过 category 过滤）。"""
    from app.kb.retriever import get_index

    index = get_index(index_dir)
    counts: dict[str, int] = {}
    for meta in index["meta"]:
        cat = str(meta.get("category") or "")
        if cat:
            counts[cat] = counts.get(cat, 0) + 1
    return counts


def check_smoke_relevance(
    index_dir: Path,
    *,
    top_k: int,
    soft_warn: bool,
) -> list[dict[str, Any]]:
    """用 10 条 query 做主题冒烟：TopK 里至少一条命中期望关键词。"""
    checks: list[dict[str, Any]] = []
    cat_counts = _category_counts(index_dir)
    query_pass = 0

    for case in SMOKE_CASES:
        qid = case.get("id") or "?"
        query = case["query"]
        expect = list(case["expect_any"])
        out = retrieve(query, top_k=top_k, index_dir=index_dir)
        blob = " | ".join(_haystack(r) for r in out["results"])
        hit = any(tok in blob for tok in expect)
        name = f"smoke.{qid}"
        detail = (
            f"q={query!r} top1={out['results'][0].get('title', '')[:40]!r} "
            f"score={out['results'][0].get('score')} ms={out['retrieve_ms']}"
        )
        if hit:
            query_pass += 1
            checks.append(_ok(name, detail))
        elif soft_warn:
            checks.append(_warn(name, f"未命中关键词{expect}; {detail}"))
        else:
            checks.append(_fail(name, f"未命中关键词{expect}; {detail}"))

        cat = case.get("category")
        if not cat:
            continue
        cname = f"smoke.{qid}.category_filter"
        if cat_counts.get(cat, 0) <= 0:
            checks.append(
                _warn(cname, f"索引无 category={cat} 语料，跳过过滤检查")
            )
            continue
        out_c = retrieve(query, top_k=top_k, index_dir=index_dir, category=cat)
        if out_c["results"]:
            checks.append(
                _ok(cname, f"cat={cat} hits={len(out_c['results'])} ms={out_c['retrieve_ms']}")
            )
        else:
            msg = f"category={cat} 过滤后无命中"
            checks.append(_warn(cname, msg) if soft_warn else _fail(cname, msg))

    total_q = len(SMOKE_CASES)
    summary_name = "smoke.queries_summary"
    summary_detail = f"{query_pass}/{total_q} queries 关键词命中"
    if query_pass == total_q:
        checks.append(_ok(summary_name, summary_detail))
    elif soft_warn:
        checks.append(_warn(summary_name, summary_detail))
    else:
        checks.append(_fail(summary_name, summary_detail))
    return checks


def run_verify(
    *,
    index_dir: Path,
    chunks_path: Path | None,
    top_k: int,
    soft_warn: bool,
) -> dict[str, Any]:
    clear_index_cache()
    checks: list[dict[str, Any]] = []
    checks.extend(check_index_files(index_dir, chunks_path))

    hard_failed = any(c["status"] == "fail" for c in checks)
    if hard_failed:
        return _summarize(checks)

    # 用第一条冒烟问句做 schema 检查
    sample_q = SMOKE_CASES[0]["query"]
    out = retrieve(sample_q, top_k=top_k, index_dir=index_dir)
    checks.extend(check_retrieve_schema(out, top_k=top_k))
    checks.extend(check_smoke_relevance(index_dir, top_k=top_k, soft_warn=soft_warn))
    return _summarize(checks)


def _summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for c in checks if c["status"] == "pass")
    failed = sum(1 for c in checks if c["status"] == "fail")
    warned = sum(1 for c in checks if c["status"] == "warn")
    return {
        "ok": failed == 0,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="验证 KB 索引与 retrieve()")
    p.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument(
        "--soft-warn",
        action="store_true",
        help="主题冒烟失败时只警告，不导致 exit 1",
    )
    p.add_argument("--json", action="store_true", help="输出 JSON 报告")
    p.add_argument(
        "--no-chunks-check",
        action="store_true",
        help="不对比 chunks.jsonl 行数",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    index_dir = Path(args.index_dir)
    chunks_path = None if args.no_chunks_check else Path(args.chunks)

    # 与 retriever.RESULT_FIELDS 对齐，防止脚本与核心漂移
    missing = [k for k in HIT_REQUIRED if k not in RESULT_FIELDS]
    if missing:
        print(f"ERROR: HIT_REQUIRED 与 RESULT_FIELDS 不一致: {missing}", file=sys.stderr)
        return 1

    report = run_verify(
        index_dir=index_dir,
        chunks_path=chunks_path,
        top_k=args.top_k,
        soft_warn=args.soft_warn,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for c in report["checks"]:
            mark = {"pass": "PASS", "fail": "FAIL", "warn": "WARN"}[c["status"]]
            print(f"[{mark}] {c['name']}: {c['detail']}")
        print(
            f"\nsummary: ok={report['ok']} "
            f"pass={report['passed']} fail={report['failed']} warn={report['warned']}"
        )
        if report["ok"]:
            print("验收通过：索引完整，retrieve 契约与主题冒烟 OK。")
        else:
            print("验收未通过：请根据 FAIL 项排查（先 build_kb_index，再看命中关键词）。")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
