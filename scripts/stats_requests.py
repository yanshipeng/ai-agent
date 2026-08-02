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
  Day16：retrieve_ms / candidates / before_dedup→after_dedup / kept / dedup_dropped
  Day18：obs_v2（trace_steps / context_tokens / budget_compressed / cache hit·miss）

用法：
  python scripts/stats_requests.py --mode rag
  python scripts/stats_requests.py --mode agent
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


def _tenant_id(row: dict[str, Any]) -> str | None:
    tid = row.get("tenant_id")
    if isinstance(tid, str) and tid.strip():
        return tid.strip()
    return None


def _tenant_bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """单租户：调用量 / 错误量 / 延迟 / token（成本代理）。"""
    total = len(rows)
    ok_n = sum(1 for r in rows if _is_ok(r))
    fail_n = total - ok_n
    latencies = sorted(v for v in (_latency_ms(r) for r in rows) if v is not None)
    tokens = sorted(v for v in (_total_tokens(r) for r in rows) if v is not None)
    err = Counter()
    for row in rows:
        code = _error_code(row)
        if code:
            err[code] += 1
        elif not _is_ok(row):
            err["unknown"] += 1
    out: dict[str, Any] = {
        "total": total,
        "ok": ok_n,
        "fail": fail_n,
        "ok_rate": round(ok_n / total, 4) if total else 0.0,
        "p50_latency_ms": _round_or_none(percentile(latencies, 50)),
        "p95_latency_ms": _round_or_none(percentile(latencies, 95)),
        "max_latency_ms": _round_or_none(latencies[-1] if latencies else None),
        "error_count": fail_n,
        "top_error_codes": [
            {"error_code": c, "count": n} for c, n in err.most_common(5)
        ],
    }
    if tokens:
        out["sum_total_tokens"] = int(sum(tokens))
        out["avg_total_tokens"] = round(sum(tokens) / len(tokens), 2)
        out["p95_total_tokens"] = _round_or_none(percentile(tokens, 95))
    return out


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

    # Day22：按 tenant 汇总调用量 / 错误 / 延迟 / token
    by_tenant: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = _tenant_id(row) or "_none"
        by_tenant.setdefault(key, []).append(row)
    if any(k != "_none" for k in by_tenant):
        summary["by_tenant"] = {
            tid: _tenant_bucket_summary(rows) for tid, rows in sorted(by_tenant.items())
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

    retrieve = _retrieve_summary(records)
    if retrieve:
        summary["retrieve"] = retrieve

    day18 = _day18_summary(records)
    if day18:
        summary["observability_v2"] = day18

    return summary


def _day18_summary(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Day18：agent_trace 长度、context token、cache hit/miss。"""
    rows = [
        r
        for r in records
        if r.get("agent_trace") is not None
        or isinstance(r.get("context_tokens_used"), (int, float))
        or isinstance(r.get("cache_hit"), (int, float))
        or isinstance(r.get("cache_miss"), (int, float))
    ]
    if not rows:
        return None

    trace_lens = sorted(
        len(r["agent_trace"])
        for r in rows
        if isinstance(r.get("agent_trace"), list)
    )
    ctx = sorted(
        v for v in (_int_field(r, "context_tokens_used") for r in rows) if v is not None
    )
    hits = [_int_field(r, "cache_hit") for r in rows]
    misses = [_int_field(r, "cache_miss") for r in rows]
    hit_sum = int(sum(v for v in hits if v is not None))
    miss_sum = int(sum(v for v in misses if v is not None))
    compressed_n = sum(1 for r in rows if r.get("budget_compressed") is True)

    out: dict[str, Any] = {"count": len(rows)}
    if trace_lens:
        out["agent_trace_steps"] = {
            "p50": _round_or_none(percentile(trace_lens, 50)),
            "p95": _round_or_none(percentile(trace_lens, 95)),
            "max": _round_or_none(trace_lens[-1]),
        }
    if ctx:
        out["context_tokens_used"] = {
            "p50": _round_or_none(percentile(ctx, 50)),
            "p95": _round_or_none(percentile(ctx, 95)),
            "max": _round_or_none(ctx[-1]),
        }
    out["budget_compressed_n"] = compressed_n
    out["cache_hit_sum"] = hit_sum
    out["cache_miss_sum"] = miss_sum
    return out


def _retrieve_summary(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Day16：检索耗时与去重前后数量变化。"""
    retrieve_rows = [
        r
        for r in records
        if isinstance(r.get("retrieve_ms"), (int, float))
        or isinstance(r.get("retrieve_candidates"), (int, float))
    ]
    if not retrieve_rows:
        return None

    def _vals(key: str) -> list[float]:
        return sorted(
            v for v in (_int_field(r, key) for r in retrieve_rows) if v is not None
        )

    ms = _vals("retrieve_ms")
    candidates = _vals("retrieve_candidates")
    before = _vals("retrieve_before_dedup")
    after = _vals("retrieve_after_dedup")
    kept = _vals("retrieve_kept")
    dropped = _vals("dedup_dropped")

    out: dict[str, Any] = {"count": len(retrieve_rows)}
    if ms:
        out["retrieve_ms"] = {
            "p50": _round_or_none(percentile(ms, 50)),
            "p95": _round_or_none(percentile(ms, 95)),
            "max": _round_or_none(ms[-1]),
        }
    if candidates:
        out["retrieve_candidates_avg"] = round(sum(candidates) / len(candidates), 2)
    if before:
        out["retrieve_before_dedup_avg"] = round(sum(before) / len(before), 2)
    if after:
        out["retrieve_after_dedup_avg"] = round(sum(after) / len(after), 2)
    if kept:
        out["retrieve_kept_avg"] = round(sum(kept) / len(kept), 2)
    if dropped:
        out["dedup_dropped_sum"] = int(sum(dropped))
        out["dedup_dropped_avg"] = round(sum(dropped) / len(dropped), 2)
    return out


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
    if "retrieve" in summary:
        rv = summary["retrieve"]
        print(f"retrieve.count    : {rv.get('count')}")
        if "retrieve_ms" in rv:
            rms = rv["retrieve_ms"]
            print(
                f"retrieve_ms       : p50={rms.get('p50')} "
                f"p95={rms.get('p95')} max={rms.get('max')}"
            )
        if "retrieve_before_dedup_avg" in rv:
            print(
                f"dedup_flow        : candidates_avg={rv.get('retrieve_candidates_avg')} "
                f"before={rv.get('retrieve_before_dedup_avg')} → "
                f"after={rv.get('retrieve_after_dedup_avg')} "
                f"kept_avg={rv.get('retrieve_kept_avg')} "
                f"dropped_sum={rv.get('dedup_dropped_sum')}"
            )
    if "observability_v2" in summary:
        ov = summary["observability_v2"]
        print(f"obs_v2.count      : {ov.get('count')}")
        if "agent_trace_steps" in ov:
            ts = ov["agent_trace_steps"]
            print(
                f"trace_steps       : p50={ts.get('p50')} "
                f"p95={ts.get('p95')} max={ts.get('max')}"
            )
        if "context_tokens_used" in ov:
            ct = ov["context_tokens_used"]
            print(
                f"context_tokens    : p50={ct.get('p50')} "
                f"p95={ct.get('p95')} max={ct.get('max')}"
            )
        print(
            f"budget_compressed : {ov.get('budget_compressed_n')}  "
            f"cache hit/miss={ov.get('cache_hit_sum')}/{ov.get('cache_miss_sum')}"
        )
    if "by_tenant" in summary:
        print("by_tenant:")
        for tid, bucket in summary["by_tenant"].items():
            print(
                f"  {tid}: total={bucket['total']} fail={bucket['fail']} "
                f"p95_lat={bucket.get('p95_latency_ms')} "
                f"sum_tokens={bucket.get('sum_total_tokens')}"
            )


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
    parser.add_argument(
        "--mode",
        default=None,
        help="只统计指定 mode（如 rag / llm / agent）",
    )
    parser.add_argument(
        "--tenant-id",
        default=None,
        help="Day22：只统计指定 tenant_id",
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

    if args.mode:
        mode = args.mode.strip()
        records = [r for r in records if r.get("mode") == mode]
        if not records:
            print(f"no records for mode={mode}", file=sys.stderr)
            return 2

    if args.tenant_id:
        tid = args.tenant_id.strip()
        records = [r for r in records if r.get("tenant_id") == tid]
        if not records:
            print(f"no records for tenant_id={tid}", file=sys.stderr)
            return 2

    summary = summarize(records)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_human(summary, path=args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
