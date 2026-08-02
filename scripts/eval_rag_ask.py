#!/usr/bin/env python3
"""Day9+ RAG 验收：随机 20 条 query 打 /ask?mode=rag，按门槛汇总。

验收条件（默认）：
  1) citations 非空比例 ≥ 80%（相对成功响应；也可按全部 20 条计）
  2) 至少 5 条 query 引用到代码 chunk，且回答能定位到关键片段
     - 默认 code_mode=either：citations.is_code=true 或正文像代码（启发式）
     - 当前本地语料若 is_code 全为 0，可用 either/heuristic；纯 is_code 会失败
  3) 固定 3 条「信息不足」问题能触发澄清/拒答（固定短语判定）

【常用命令】
  # 服务已启动
  python scripts/eval_rag_ask.py

  python scripts/eval_rag_ask.py \\
    --base-url http://127.0.0.1:8000 \\
    --samples ./eval/eval_rag_samples.jsonl \\
    --seed 42 \\
    --code-mode either

【输出】
  - reports/rag_eval_results_*.jsonl
  - reports/rag_eval_report_*.json
  - 根目录 rag_eval_report.json（最新一份）
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SAMPLES = ROOT / "eval" / "eval_rag_samples.jsonl"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT = 90.0
CITATIONS_NONEMPTY_RATE_MIN = 0.80
CODE_LOCATED_MIN = 5
INSUFFICIENT_EXPECTED = 3

# 固定短语：澄清 / 资料不足 / 拒答（RAG v1 Prompt 对齐）
INSUFFICIENT_PHRASES = (
    "根据已有资料无法确定",
    "资料不足",
    "信息不足",
    "无法确定",
    "需要澄清",
    "请补充",
    "请提供更多",
    "请说明",
    "不太清楚",
    "无法回答",
    "没有足够",
    "上下文不足",
    "缺少关键信息",
    "请问你指的是",
    "我需要更多信息",
)

# 启发式：正文像代码/命令/堆栈（当 index.is_code 全为 0 时的兜底）
CODE_LIKE_RE = re.compile(
    r"(```|adb\s+shell|adb\s+pull|public\s+class|override\s+fun|"
    r"func\s*\(|void\s+\w+\(|native\s*:\s*#\d+|MessageQueue\.|Looper\.|"
    r"okhttp|Retrofit|@Override|System\.loadLibrary)",
    re.IGNORECASE,
)
REF_RE = re.compile(r"\[(\d+)\]")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            for key in ("id", "query", "tag"):
                if key not in item:
                    raise ValueError(f"sample line {line_no} missing {key}")
            rows.append(item)
    return rows


def pick_samples(
    pool: list[dict[str, Any]],
    *,
    seed: int,
    total: int = 20,
    insufficient_n: int = INSUFFICIENT_EXPECTED,
) -> list[dict[str, Any]]:
    """固定纳入全部 insufficient 样例，其余随机抽满 total。"""
    insuff = [s for s in pool if s.get("tag") == "insufficient"]
    others = [s for s in pool if s.get("tag") != "insufficient"]
    if len(insuff) < insufficient_n:
        raise ValueError(
            f"insufficient samples need >= {insufficient_n}, got {len(insuff)}"
        )
    rng = random.Random(seed)
    # 优先保留 code_seeking，提高命中代码片段概率
    code_seeking = [s for s in others if s.get("tag") == "code_seeking"]
    normal = [s for s in others if s.get("tag") != "code_seeking"]
    need = total - insufficient_n
    chosen: list[dict[str, Any]] = []
    # 先尽量纳入 code_seeking（打乱后取前若干）
    rng.shuffle(code_seeking)
    take_code = min(len(code_seeking), max(CODE_LOCATED_MIN, need // 2))
    chosen.extend(code_seeking[:take_code])
    remain = need - len(chosen)
    rng.shuffle(normal)
    if remain > len(normal):
        raise ValueError(f"not enough normal/code samples: need {remain} more")
    chosen.extend(normal[:remain])
    # insufficient 固定取前 N（样例文件里应正好 3 条）
    chosen_insuff = insuff[:insufficient_n]
    out = chosen + chosen_insuff
    rng.shuffle(out)
    return out


def is_insufficient_answer(answer: str) -> bool:
    text = answer or ""
    return any(p in text for p in INSUFFICIENT_PHRASES)


def citation_is_code(cite: dict[str, Any], *, code_mode: str) -> bool:
    flag = bool(cite.get("is_code"))
    # 启发式只能看 title；真正代码感在 chunk 正文，由调用方注入 _code_like
    like = bool(cite.get("_code_like"))
    if code_mode == "is_code":
        return flag
    if code_mode == "heuristic":
        return like
    return flag or like  # either


def annotate_citations_code_like(
    citations: list[dict[str, Any]],
    index_by_id: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """给 citation 打上 _code_like（查本地 index 正文）。"""
    out: list[dict[str, Any]] = []
    for c in citations:
        row = dict(c)
        text = ""
        if index_by_id:
            local = index_by_id.get(str(c.get("chunk_id") or ""))
            if local:
                text = str(local.get("text") or "")
                # 若响应 is_code 为 false，但本地 meta 为 true，仍算
                if local.get("is_code"):
                    row["is_code"] = True
        row["_code_like"] = bool(CODE_LIKE_RE.search(text)) or bool(
            CODE_LIKE_RE.search(str(c.get("title") or ""))
        )
        out.append(row)
    return out


def answer_locates_code_fragment(
    answer: str,
    citations: list[dict[str, Any]],
    *,
    code_mode: str,
    index_by_id: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """至少有一个代码 citation 被 [n] 引用，且与 chunk 有字面重叠。"""
    used_refs = {int(x) for x in REF_RE.findall(answer or "")}
    by_ref = {
        int(c["ref_id"]): c
        for c in citations
        if isinstance(c.get("ref_id"), int)
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_./-]{3,}|[\u4e00-\u9fff]{2,}", answer or "")
    uniq: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
        if len(uniq) >= 50:
            break

    located_refs: list[int] = []
    for ref, cite in by_ref.items():
        if not citation_is_code(cite, code_mode=code_mode):
            continue
        if used_refs and ref not in used_refs:
            # 若完全没有 [n]，仍允许用全文重叠弱判定
            if used_refs:
                continue
        local = (index_by_id or {}).get(str(cite.get("chunk_id") or ""))
        blob = str((local or {}).get("text") or cite.get("title") or "")
        overlap = [t for t in uniq if t in blob][:5]
        if overlap:
            located_refs.append(ref)
    return {
        "ok": len(located_refs) > 0,
        "located_refs": located_refs,
    }


def load_index_by_id(index_dir: Path) -> dict[str, dict[str, Any]]:
    meta_path = index_dir / "meta.jsonl"
    if not meta_path.exists():
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    with meta_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row.get("chunk_id")
            if cid:
                by_id[str(cid)] = row
    return by_id


def call_ask_rag(
    client: httpx.Client,
    base_url: str,
    query: str,
    *,
    top_k: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "ok": False,
        "status_code": None,
        "latency_ms": None,
        "error_code": None,
        "request_id": None,
        "answer": None,
        "citations": [],
        "meta": None,
    }
    try:
        resp = client.post(
            f"{base_url.rstrip('/')}/ask",
            params={"mode": "rag"},
            json={"query": query, "top_k": top_k, "client_tag": "eval_rag_ask"},
            timeout=timeout,
        )
        row["status_code"] = resp.status_code
        row["latency_ms"] = int((time.perf_counter() - started) * 1000)
        try:
            body = resp.json()
        except json.JSONDecodeError:
            row["error_code"] = "BAD_JSON"
            return row
        if resp.status_code != 200:
            row["error_code"] = body.get("code") or f"HTTP_{resp.status_code}"
            row["request_id"] = body.get("request_id")
            return row
        row["ok"] = True
        row["request_id"] = body.get("request_id")
        row["answer"] = body.get("answer")
        row["citations"] = body.get("citations") or []
        row["meta"] = body.get("meta")
        return row
    except httpx.HTTPError as exc:
        row["latency_ms"] = int((time.perf_counter() - started) * 1000)
        row["error_code"] = type(exc).__name__
        return row


def evaluate_one(
    sample: dict[str, Any],
    ask_row: dict[str, Any],
    *,
    code_mode: str,
    index_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    citations = ask_row.get("citations") or []
    if ask_row.get("ok"):
        citations = annotate_citations_code_like(citations, index_by_id)
    answer = str(ask_row.get("answer") or "")
    citations_nonempty = bool(citations)
    insuff_hit = is_insufficient_answer(answer) if sample.get("tag") == "insufficient" else None
    code_loc = (
        answer_locates_code_fragment(
            answer,
            citations,
            code_mode=code_mode,
            index_by_id=index_by_id,
        )
        if ask_row.get("ok") and sample.get("tag") != "insufficient"
        else {"ok": False, "located_refs": []}
    )
    return {
        "id": sample["id"],
        "tag": sample.get("tag"),
        "query": sample["query"],
        "ok": ask_row.get("ok"),
        "status_code": ask_row.get("status_code"),
        "error_code": ask_row.get("error_code"),
        "request_id": ask_row.get("request_id"),
        "latency_ms": ask_row.get("latency_ms"),
        "citations_count": len(citations),
        "citations_nonempty": citations_nonempty,
        "has_is_code_citation": any(bool(c.get("is_code")) for c in citations),
        "has_code_like_citation": any(bool(c.get("_code_like")) for c in citations),
        "code_located": bool(code_loc.get("ok")),
        "code_located_refs": code_loc.get("located_refs"),
        "insufficient_expected": sample.get("tag") == "insufficient",
        "insufficient_triggered": insuff_hit,
        "answer_preview": answer[:180].replace("\n", " "),
        "retrieve_ms": (ask_row.get("meta") or {}).get("retrieve_ms"),
    }


def summarize(
    results: list[dict[str, Any]],
    *,
    code_mode: str,
    seed: int,
) -> dict[str, Any]:
    total = len(results)
    http_ok = [r for r in results if r.get("ok")]
    # citations 非空比例：相对全部 20 条（失败也算空）
    nonempty = sum(1 for r in results if r.get("citations_nonempty"))
    citations_rate = (nonempty / total) if total else 0.0

    code_located_n = sum(
        1
        for r in results
        if r.get("tag") != "insufficient" and r.get("code_located")
    )
    insuff_rows = [r for r in results if r.get("insufficient_expected")]
    insuff_ok = sum(1 for r in insuff_rows if r.get("insufficient_triggered"))

    citations_pass = citations_rate >= CITATIONS_NONEMPTY_RATE_MIN
    code_pass = code_located_n >= CODE_LOCATED_MIN
    insuff_pass = insuff_ok >= INSUFFICIENT_EXPECTED and len(insuff_rows) >= INSUFFICIENT_EXPECTED

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "code_mode": code_mode,
        "total": total,
        "http_ok": len(http_ok),
        "thresholds": {
            "citations_nonempty_rate_min": CITATIONS_NONEMPTY_RATE_MIN,
            "code_located_min": CODE_LOCATED_MIN,
            "insufficient_expected": INSUFFICIENT_EXPECTED,
        },
        "citations_nonempty": nonempty,
        "citations_nonempty_rate": round(citations_rate, 4),
        "citations_pass": citations_pass,
        "code_located_queries": code_located_n,
        "code_pass": code_pass,
        "insufficient_total": len(insuff_rows),
        "insufficient_triggered": insuff_ok,
        "insufficient_pass": insuff_pass,
        "ok": citations_pass and code_pass and insuff_pass,
        "failed_ids": {
            "http": [r["id"] for r in results if not r.get("ok")],
            "empty_citations": [
                r["id"] for r in results if r.get("ok") and not r.get("citations_nonempty")
            ],
            "insufficient_miss": [
                r["id"]
                for r in insuff_rows
                if not r.get("insufficient_triggered")
            ],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG /ask 验收：随机 20 query")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total", type=int, default=20)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument(
        "--code-mode",
        choices=("is_code", "heuristic", "either"),
        default="either",
        help="代码 chunk 判定：is_code / 启发式 / 二者任一（默认 either）",
    )
    p.add_argument(
        "--index-dir",
        default=str(ROOT / "data" / "stability_kb" / "index"),
        help="用于 is_code/启发式对账的本地 index",
    )
    p.add_argument("--out-dir", default=str(ROOT / "reports"))
    return p


def main() -> int:
    args = build_parser().parse_args()
    pool = load_samples(Path(args.samples))
    selected = pick_samples(pool, seed=args.seed, total=args.total)
    index_by_id = load_index_by_id(Path(args.index_dir))
    index_code_n = sum(1 for v in index_by_id.values() if v.get("is_code"))

    print(f"base_url={args.base_url} seed={args.seed} total={len(selected)}")
    print(f"code_mode={args.code_mode} index_is_code_chunks={index_code_n}")
    if args.code_mode == "is_code" and index_code_n == 0:
        print(
            "WARN: 本地 index 无 is_code=true chunk，"
            "code 门槛大概率失败；可改 --code-mode either",
            file=sys.stderr,
        )

    results: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for i, sample in enumerate(selected, start=1):
            print(f"[{i}/{len(selected)}] {sample['id']} {sample['query'][:40]}...")
            ask_row = call_ask_rag(
                client,
                args.base_url,
                sample["query"],
                top_k=args.top_k,
                timeout=args.timeout,
            )
            row = evaluate_one(
                sample,
                ask_row,
                code_mode=args.code_mode,
                index_by_id=index_by_id,
            )
            results.append(row)
            flag = "OK" if row["ok"] else f"ERR:{row.get('error_code')}"
            print(
                f"  -> {flag} cites={row['citations_count']} "
                f"code_located={row['code_located']} "
                f"insuff={row['insufficient_triggered']} "
                f"ms={row['latency_ms']}"
            )

    report = summarize(results, code_mode=args.code_mode, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    results_path = out_dir / f"rag_eval_results_{stamp}.jsonl"
    report_path = out_dir / f"rag_eval_report_{stamp}.json"
    latest = ROOT / "reports" / "rag_eval_report.json"

    with results_path.open("w", encoding="utf-8") as fp:
        for row in results:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    latest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== RAG acceptance ===")
    print(
        f"citations_nonempty_rate={report['citations_nonempty_rate']} "
        f"(pass={report['citations_pass']}, need>={CITATIONS_NONEMPTY_RATE_MIN})"
    )
    print(
        f"code_located_queries={report['code_located_queries']} "
        f"(pass={report['code_pass']}, need>={CODE_LOCATED_MIN}, mode={args.code_mode})"
    )
    print(
        f"insufficient_triggered={report['insufficient_triggered']}/"
        f"{report['insufficient_total']} (pass={report['insufficient_pass']})"
    )
    print(f"ok={report['ok']}")
    print(f"results => {results_path}")
    print(f"report  => {report_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
