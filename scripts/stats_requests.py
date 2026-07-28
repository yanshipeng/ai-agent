#!/usr/bin/env python3
"""读取 requests.jsonl 并输出汇总指标。

输出：
  total / ok / fail / ok_rate
  p50_latency_ms / p95_latency_ms / max_latency_ms
  （有 session 时）with_session / without_session 的 p95 对比
  history_messages / history_chars 分位（若有字段）
  retry_rate
  top_error_codes（前 5）
  若有 token：p50/p95/max/avg_total_tokens
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                print(f"skip invalid json at line {line_no}: {exc}", file=sys.stderr)
    return records


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


def _is_ok(row: dict[str, Any]) -> bool:
    if "ok" in row:
        return row.get("ok") is True
    # 兼容旧格式
    return row.get("status") == "ok"


def _latency_ms(row: dict[str, Any]) -> float | None:
    for key in ("latency_ms_total", "latency_ms"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _retry_count(row: dict[str, Any]) -> int:
    value = row.get("retry_count")
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _total_tokens(row: dict[str, Any]) -> float | None:
    value = row.get("total_tokens")
    if isinstance(value, (int, float)):
        return float(value)
    usage = row.get("usage")
    if isinstance(usage, dict):
        nested = usage.get("total_tokens")
        if isinstance(nested, (int, float)):
            return float(nested)
    return None


def _error_code(row: dict[str, Any]) -> str | None:
    code = row.get("error_code")
    if isinstance(code, str) and code:
        return code
    return None


def _has_session(row: dict[str, Any]) -> bool:
    sid = row.get("session_id")
    return isinstance(sid, str) and bool(sid.strip())


def _int_field(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(v for v in (_latency_ms(r) for r in rows) if v is not None)
    return {
        "count": len(rows),
        "p50_latency_ms": _round_or_none(percentile(latencies, 50)),
        "p95_latency_ms": _round_or_none(percentile(latencies, 95)),
        "max_latency_ms": _round_or_none(latencies[-1] if latencies else None),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    ok_n = sum(1 for r in records if _is_ok(r))
    fail_n = total - ok_n
    ok_rate = round(ok_n / total, 4) if total else 0.0

    latencies = sorted(v for v in (_latency_ms(r) for r in records) if v is not None)
    retried_n = sum(1 for r in records if _retry_count(r) > 0)
    retry_rate = round(retried_n / total, 4) if total else 0.0

    fail_rows = [r for r in records if not _is_ok(r)]
    # 失败行 + 成功但带 error_code（如 fallback）也计入 top errors
    error_counter = Counter()
    for row in records:
        code = _error_code(row)
        if code:
            error_counter[code] += 1
    # 若失败行没有 error_code，记为 unknown
    for row in fail_rows:
        if not _error_code(row):
            error_counter["unknown"] += 1

    top_error_codes = [
        {"error_code": code, "count": count}
        for code, count in error_counter.most_common(5)
    ]

    summary: dict[str, Any] = {
        "total": total,
        "ok": ok_n,
        "fail": fail_n,
        "ok_rate": ok_rate,
        "p50_latency_ms": _round_or_none(percentile(latencies, 50)),
        "p95_latency_ms": _round_or_none(percentile(latencies, 95)),
        "max_latency_ms": _round_or_none(latencies[-1] if latencies else None),
        "retry_rate": retry_rate,
        "top_error_codes": top_error_codes,
    }

    with_session = [r for r in records if _has_session(r)]
    without_session = [r for r in records if not _has_session(r)]
    if with_session or without_session:
        summary["with_session"] = _latency_summary(with_session)
        summary["without_session"] = _latency_summary(without_session)

    hist_msgs = sorted(
        v for v in (_int_field(r, "history_messages") for r in with_session) if v is not None
    )
    hist_chars = sorted(
        v for v in (_int_field(r, "history_chars") for r in with_session) if v is not None
    )
    if hist_msgs:
        summary["history_messages"] = {
            "p50": _round_or_none(percentile(hist_msgs, 50)),
            "p95": _round_or_none(percentile(hist_msgs, 95)),
            "max": _round_or_none(hist_msgs[-1]),
        }
    if hist_chars:
        summary["history_chars"] = {
            "p50": _round_or_none(percentile(hist_chars, 50)),
            "p95": _round_or_none(percentile(hist_chars, 95)),
            "max": _round_or_none(hist_chars[-1]),
        }

    tokens = sorted(v for v in (_total_tokens(r) for r in records) if v is not None)
    if tokens:
        summary.update(
            {
                "p50_total_tokens": _round_or_none(percentile(tokens, 50)),
                "p95_total_tokens": _round_or_none(percentile(tokens, 95)),
                "max_total_tokens": _round_or_none(tokens[-1]),
                "avg_total_tokens": round(sum(tokens) / len(tokens), 2),
            }
        )

    return summary


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 2)


def print_human(summary: dict[str, Any], *, path: str) -> None:
    print(f"file              : {path}")
    print(f"total             : {summary['total']}")
    print(f"ok                : {summary['ok']}")
    print(f"fail              : {summary['fail']}")
    print(f"ok_rate           : {summary['ok_rate']:.2%}")
    print(f"p50_latency_ms    : {summary['p50_latency_ms']}")
    print(f"p95_latency_ms    : {summary['p95_latency_ms']}")
    print(f"max_latency_ms    : {summary['max_latency_ms']}")
    print(f"retry_rate        : {summary['retry_rate']:.2%}")
    print(f"top_error_codes   : {summary['top_error_codes']}")
    if "with_session" in summary:
        ws = summary["with_session"]
        ns = summary["without_session"]
        print(
            f"with_session      : count={ws['count']} "
            f"p50={ws['p50_latency_ms']} p95={ws['p95_latency_ms']} max={ws['max_latency_ms']}"
        )
        print(
            f"without_session   : count={ns['count']} "
            f"p50={ns['p50_latency_ms']} p95={ns['p95_latency_ms']} max={ns['max_latency_ms']}"
        )
    if "history_messages" in summary:
        hm = summary["history_messages"]
        print(
            f"history_messages  : p50={hm['p50']} p95={hm['p95']} max={hm['max']}"
        )
    if "history_chars" in summary:
        hc = summary["history_chars"]
        print(
            f"history_chars     : p50={hc['p50']} p95={hc['p95']} max={hc['max']}"
        )
    if "avg_total_tokens" in summary:
        print(f"p50_total_tokens  : {summary['p50_total_tokens']}")
        print(f"p95_total_tokens  : {summary['p95_total_tokens']}")
        print(f"max_total_tokens  : {summary['max_total_tokens']}")
        print(f"avg_total_tokens  : {summary['avg_total_tokens']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize requests.jsonl metrics")
    parser.add_argument(
        "--path",
        default="./requests.jsonl",
        help="Path to requests.jsonl (default: ./requests.jsonl)",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON summary")
    parser.add_argument(
        "--session-id",
        default=None,
        help="只统计指定 session_id 的行（验收多轮延迟）",
    )
    args = parser.parse_args()

    try:
        records = load_records(Path(args.path))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.session_id:
        sid = args.session_id.strip()
        records = [r for r in records if r.get("session_id") == sid]
        if not records:
            print(f"no records for session_id={sid}", file=sys.stderr)
            return 2

    summary = summarize(records)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_human(summary, path=args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
