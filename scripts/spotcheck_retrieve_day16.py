#!/usr/bin/env python3
"""Day16 验收：随机抽检 20 条 Top3 相关性（主观记录）+ 可观测字段落盘。

做什么
  1) 从 eval_samples_rag.jsonl 的 normal 题里随机抽 N 条（默认 20）
  2) 本地 hybrid retrieve Top3，生成抽检表（含 reason 空字段供填写）
  3) 可选：走 /ask?mode=rag，把 retrieve_ms / 去重前后数量写入 requests.jsonl

【常用命令】
  # 只做本地 Top3 抽检表（不打 LLM，最快）
  python scripts/spotcheck_retrieve_day16.py

  # 抽检 + 打 /ask?mode=rag，验收 requests.jsonl 字段（服务须已启动）
  python scripts/spotcheck_retrieve_day16.py --via-ask

  # 填完理由后看汇总
  python scripts/spotcheck_retrieve_day16.py --summarize reports/day16_spotcheck.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb.retriever import clear_index_cache, retrieve  # noqa: E402

DEFAULT_SAMPLES = ROOT / "eval" / "eval_samples_rag.jsonl"
DEFAULT_INDEX = ROOT / "data" / "stability_kb" / "index"
DEFAULT_OUT = ROOT / "reports" / "day16_spotcheck.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def load_normal_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if str(row.get("tag") or "") == "normal":
                rows.append(row)
    return rows


def pick_samples(
    rows: list[dict[str, Any]],
    *,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, n)


def hit_card(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": hit.get("chunk_id"),
        "score": hit.get("score"),
        "title": hit.get("title"),
        "url": hit.get("url"),
        "section_path": hit.get("section_path") or "",
        "is_code": bool(hit.get("is_code")),
        "text_snippet": hit.get("text_snippet") or "",
    }


def local_top3(
    query: str,
    *,
    index_dir: Path,
    top_k: int = 3,
) -> dict[str, Any]:
    out = retrieve(
        query,
        top_k=top_k,
        index_dir=index_dir,
        include_snippet=True,
        include_text=False,
    )
    hits = [hit_card(h, i) for i, h in enumerate(out.get("results") or [], start=1)]
    return {
        "top3": hits,
        "retrieve_ms": out.get("retrieve_ms"),
        "retrieve_candidates": out.get("retrieve_candidates"),
        "retrieve_before_dedup": out.get("retrieve_before_dedup"),
        "retrieve_after_dedup": out.get("retrieve_after_dedup"),
        "retrieve_kept": out.get("retrieve_kept"),
        "hybrid_weight": out.get("hybrid_weight"),
        "dedup_dropped": out.get("dedup_dropped"),
    }


def ask_rag(
    query: str,
    *,
    base_url: str,
    top_k: int,
    timeout: float,
) -> dict[str, Any]:
    import httpx

    url = f"{base_url.rstrip('/')}/ask"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            url,
            params={"mode": "rag"},
            json={"query": query, "top_k": top_k},
        )
    body: dict[str, Any]
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    meta = body.get("meta") if isinstance(body, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    return {
        "http_status": resp.status_code,
        "request_id": body.get("request_id") if isinstance(body, dict) else None,
        "retrieve_ms": meta.get("retrieve_ms"),
        "retrieve_candidates": meta.get("retrieve_candidates"),
        "retrieve_before_dedup": meta.get("retrieve_before_dedup"),
        "retrieve_after_dedup": meta.get("retrieve_after_dedup"),
        "retrieve_kept": meta.get("retrieve_kept"),
        "hybrid_weight": meta.get("hybrid_weight"),
        "dedup_dropped": meta.get("dedup_dropped"),
        "citations_count": meta.get("citations_count"),
    }


def build_item(
    sample: dict[str, Any],
    pack: dict[str, Any],
    *,
    ask_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成一条抽检记录；主观字段留空待填。"""
    return {
        "id": sample.get("id"),
        "query": sample.get("query"),
        "tag": sample.get("tag"),
        "category": sample.get("category"),
        "top3": pack.get("top3") or [],
        "retrieve_ms": pack.get("retrieve_ms"),
        "retrieve_candidates": pack.get("retrieve_candidates"),
        "retrieve_before_dedup": pack.get("retrieve_before_dedup"),
        "retrieve_after_dedup": pack.get("retrieve_after_dedup"),
        "retrieve_kept": pack.get("retrieve_kept"),
        "hybrid_weight": pack.get("hybrid_weight"),
        "dedup_dropped": pack.get("dedup_dropped"),
        "ask": ask_meta,
        # ---- 主观抽检（人工填写）----
        "top3_more_relevant": None,  # true / false
        "reason": "",  # 为何认为 Top3 更相关 / 不相关
        "notes": "",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# Day16 检索抽检表（Top3）",
        "",
        f"- 生成时间：{report.get('ts')}",
        f"- seed={report.get('seed')}  n={report.get('n')}  index={report.get('index_dir')}",
        f"- 填写说明：把每条的 `top3_more_relevant` 改成 true/false，并写 `reason`",
        "",
    ]
    for i, item in enumerate(report.get("items") or [], start=1):
        lines.append(f"## {i}. {item.get('id')} · category={item.get('category')}")
        lines.append("")
        lines.append(f"**Query**: {item.get('query')}")
        lines.append("")
        lines.append(
            f"可观测：retrieve_ms={item.get('retrieve_ms')}ms，"
            f"candidates={item.get('retrieve_candidates')}，"
            f"before_dedup={item.get('retrieve_before_dedup')} → "
            f"after_dedup={item.get('retrieve_after_dedup')} "
            f"(dropped={item.get('dedup_dropped')})，"
            f"kept={item.get('retrieve_kept')}"
        )
        lines.append("")
        for hit in item.get("top3") or []:
            lines.append(
                f"- **[{hit.get('rank')}]** score={hit.get('score')} "
                f"`{hit.get('chunk_id')}` | {hit.get('title')} | "
                f"{hit.get('section_path')}"
            )
            snip = (hit.get("text_snippet") or "").replace("\n", " ")
            if snip:
                lines.append(f"  - snippet: {snip[:160]}")
        lines.append("")
        lines.append(f"- top3_more_relevant: `{item.get('top3_more_relevant')}`")
        lines.append(f"- reason: {item.get('reason') or '（待填写）'}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_filled(path: Path) -> int:
    report = json.loads(path.read_text(encoding="utf-8"))
    items = list(report.get("items") or [])
    filled = [x for x in items if x.get("top3_more_relevant") is not None]
    yes = sum(1 for x in filled if x.get("top3_more_relevant") is True)
    no = sum(1 for x in filled if x.get("top3_more_relevant") is False)
    print(f"file          : {path}")
    print(f"total items   : {len(items)}")
    print(f"filled        : {len(filled)}")
    print(f"more_relevant : {yes}")
    print(f"not_relevant  : {no}")
    if filled:
        print(f"pass_rate     : {yes / len(filled):.1%}")
    empty_reason = sum(1 for x in filled if not str(x.get("reason") or "").strip())
    if empty_reason:
        print(f"WARN: {empty_reason} 条已判相关但未写 reason", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Day16 Top3 随机抽检")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--n", type=int, default=20, help="抽检条数，默认 20")
    parser.add_argument("--seed", type=int, default=16, help="随机种子，可复现")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--via-ask",
        action="store_true",
        help="额外 POST /ask?mode=rag，把检索指标写入 requests.jsonl",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--summarize",
        default=None,
        help="汇总已填写的抽检 JSON（跳过检索）",
    )
    args = parser.parse_args()

    if args.summarize:
        return summarize_filled(Path(args.summarize))

    samples_path = Path(args.samples)
    index_dir = Path(args.index_dir)
    if not samples_path.exists():
        print(f"ERROR: samples not found: {samples_path}", file=sys.stderr)
        return 2
    if not (index_dir / "manifest.json").exists():
        print(
            f"ERROR: index not found at {index_dir}. "
            "Run: python scripts/build_kb_index.py",
            file=sys.stderr,
        )
        return 2

    normals = load_normal_samples(samples_path)
    if not normals:
        print("ERROR: no normal samples", file=sys.stderr)
        return 2

    picked = pick_samples(normals, n=max(args.n, 1), seed=args.seed)
    clear_index_cache()

    items: list[dict[str, Any]] = []
    print(f"[spotcheck] normals={len(normals)} picked={len(picked)} seed={args.seed}")
    for idx, sample in enumerate(picked, start=1):
        q = str(sample.get("query") or "")
        print(f"[{idx}/{len(picked)}] {sample.get('id')} …", flush=True)
        pack = local_top3(q, index_dir=index_dir, top_k=args.top_k)
        ask_meta = None
        if args.via_ask:
            try:
                ask_meta = ask_rag(
                    q,
                    base_url=args.base_url,
                    top_k=max(args.top_k, 5),
                    timeout=args.timeout,
                )
                # 以服务端可观测字段为准（jsonl 同源）
                for key in (
                    "retrieve_ms",
                    "retrieve_candidates",
                    "retrieve_before_dedup",
                    "retrieve_after_dedup",
                    "retrieve_kept",
                    "hybrid_weight",
                    "dedup_dropped",
                ):
                    if ask_meta.get(key) is not None:
                        pack[key] = ask_meta[key]
            except Exception as exc:
                ask_meta = {"error": str(exc)}
                print(f"  WARN via-ask failed: {exc}", file=sys.stderr)
            time.sleep(0.05)
        items.append(build_item(sample, pack, ask_meta=ask_meta))

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n": len(items),
        "top_k": args.top_k,
        "index_dir": str(index_dir),
        "via_ask": bool(args.via_ask),
        "instruction": (
            "主观验收：逐条把 top3_more_relevant 填 true/false，"
            "并在 reason 写清为何 Top3 更相关（或不够相关）。"
        ),
        "items": items,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path = out_path.with_suffix(".md")
    write_markdown(report, md_path)

    # 可观测汇总（本地/via-ask）
    ms_vals = [i["retrieve_ms"] for i in items if isinstance(i.get("retrieve_ms"), int)]
    before = [i["retrieve_before_dedup"] for i in items if isinstance(i.get("retrieve_before_dedup"), int)]
    after = [i["retrieve_after_dedup"] for i in items if isinstance(i.get("retrieve_after_dedup"), int)]
    dropped = [i["dedup_dropped"] for i in items if isinstance(i.get("dedup_dropped"), int)]

    print(f"\n[ok] wrote {out_path}")
    print(f"[ok] wrote {md_path}")
    if ms_vals:
        print(
            f"retrieve_ms: min={min(ms_vals)} p50≈{sorted(ms_vals)[len(ms_vals)//2]} "
            f"max={max(ms_vals)}"
        )
    if before and after:
        print(
            f"dedup: before_avg={sum(before)/len(before):.1f} → "
            f"after_avg={sum(after)/len(after):.1f} "
            f"(dropped_sum={sum(dropped) if dropped else 0})"
        )
    print(
        "\n下一步：打开 JSON/MD，填写每条 top3_more_relevant + reason；"
        "若用了 --via-ask，再跑：\n"
        "  python scripts/stats_requests.py --path ./data/runtime/requests.jsonl --mode rag"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
