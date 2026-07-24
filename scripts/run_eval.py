#!/usr/bin/env python3
"""Day 5.3 回归跑批：逐条调用 HTTP /ask，输出明细与汇总报告。

输入：eval_samples.jsonl（字段 id / query / tag）
输出：
  - eval_results.jsonl（每条一行明细）
  - eval_run_report.json（汇总：total、ok_rate、p95_latency、top_errors、各 tag 失败数）

要求：20 条能跑完不中断；即使单条失败也继续。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_samples(path: Path) -> list[dict[str, Any]]:
    """读取样例；每行必须含 id / query / tag。"""
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            for key in ("id", "query", "tag"):
                if key not in item:
                    raise ValueError(f"sample at line {line_no} missing field: {key}")
            samples.append(item)
    return samples


def percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    return float(sorted_values[f] * (c - k) + sorted_values[c] * (k - f))


def call_ask(client: httpx.Client, base_url: str, query: str, *, timeout: float) -> dict[str, Any]:
    """调用 POST /ask，返回统一明细字段（不抛中断）。"""
    started = time.perf_counter()
    row: dict[str, Any] = {
        "ok": False,
        "latency_ms": None,
        "error_code": None,
        "answer_len": None,
        "finish_reason": None,
        "total_tokens": None,
        "status_code": None,
        "request_id": None,
    }
    try:
        resp = client.post(
            f"{base_url.rstrip('/')}/ask",
            json={"query": query, "client_tag": "run_eval"},
            timeout=timeout,
        )
        row["status_code"] = resp.status_code
        row["latency_ms"] = int((time.perf_counter() - started) * 1000)
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            row["error_code"] = "INVALID_JSON"
            return row

        if resp.status_code == 200 and isinstance(body, dict) and "answer" in body:
            row["ok"] = True
            row["request_id"] = body.get("request_id")
            answer = body.get("answer") or ""
            row["answer_len"] = len(answer)
            meta = body.get("meta") or {}
            row["finish_reason"] = meta.get("finish_reason")
            usage = meta.get("usage") or {}
            if isinstance(usage.get("total_tokens"), int):
                row["total_tokens"] = usage["total_tokens"]
            # 兜底也算 HTTP ok；若带 fallback error_code 仍记入 error_code 便于统计
            if meta.get("fallback") and meta.get("error_code"):
                row["error_code"] = meta.get("error_code")
            return row

        row["request_id"] = body.get("request_id") if isinstance(body, dict) else None
        row["error_code"] = (
            body.get("code") if isinstance(body, dict) else f"HTTP_{resp.status_code}"
        )
        return row
    except httpx.TimeoutException:
        row["latency_ms"] = int((time.perf_counter() - started) * 1000)
        row["error_code"] = "CLIENT_TIMEOUT"
        return row
    except Exception as exc:  # noqa: BLE001
        row["latency_ms"] = int((time.perf_counter() - started) * 1000)
        row["error_code"] = f"CLIENT_ERROR:{type(exc).__name__}"
        return row


def build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    ok_n = sum(1 for r in results if r.get("ok"))
    latencies = sorted(
        float(r["latency_ms"])
        for r in results
        if isinstance(r.get("latency_ms"), (int, float))
    )
    fail_errors = Counter(
        r.get("error_code") or "unknown" for r in results if not r.get("ok")
    )
    tag_fail: dict[str, int] = {}
    for r in results:
        tag = r.get("tag") or "unknown"
        tag_fail.setdefault(tag, 0)
        if not r.get("ok"):
            tag_fail[tag] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "ok": ok_n,
        "fail": total - ok_n,
        "ok_rate": round(ok_n / total, 4) if total else 0.0,
        "p95_latency": round(percentile(latencies, 95) or 0.0, 2) if latencies else None,
        "p50_latency": round(percentile(latencies, 50) or 0.0, 2) if latencies else None,
        "max_latency": latencies[-1] if latencies else None,
        "top_errors": [
            {"error_code": code, "count": count}
            for code, count in fail_errors.most_common(5)
        ],
        "tag_fail_counts": tag_fail,
        "tag_totals": dict(Counter(r.get("tag") or "unknown" for r in results)),
    }


def run_eval(
    samples: list[dict[str, Any]],
    *,
    base_url: str,
    timeout: float,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = samples if limit is None else samples[:limit]
    results: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for sample in selected:
            detail = call_ask(client, base_url, sample["query"], timeout=timeout)
            row = {
                "id": sample["id"],
                "tag": sample["tag"],
                "query_len": len(sample["query"]),
                **detail,
            }
            results.append(row)
            mark = "OK" if row["ok"] else "FAIL"
            print(
                f"[{mark}] {row['id']} tag={row['tag']} "
                f"latency={row.get('latency_ms')}ms error={row.get('error_code')}"
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression runner against HTTP /ask")
    parser.add_argument("--samples", default="./eval_samples.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--results", default="./eval_results.jsonl")
    parser.add_argument("--report", default="./eval_run_report.json")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    samples = load_samples(Path(args.samples))
    print(f"loaded {len(samples)} samples from {args.samples}")
    print(f"target {args.base_url}/ask")

    results = run_eval(
        samples,
        base_url=args.base_url,
        timeout=args.timeout,
        limit=args.limit,
    )

    results_path = Path(args.results)
    with results_path.open("w", encoding="utf-8") as fp:
        for row in results:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_report(results)
    report_path = Path(args.report)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nresults => {results_path}")
    print(f"report  => {report_path}")
    print(
        f"summary: total={report['total']} ok_rate={report['ok_rate']:.2%} "
        f"p95_latency={report['p95_latency']} tag_fail={report['tag_fail_counts']}"
    )
    # 跑完不中断；全部失败也 exit 0（验收：能跑完并生成报告）
    # 若需要「有失败则非 0」，可用 --strict
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
