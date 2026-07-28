#!/usr/bin/env python3
"""Week3 Agent 评测：mode=agent 打样例，输出 agent_eval_report.json。

样例：agent_eval_samples.jsonl（≥30，默认围绕稳定性知识库 A=ANR）
  - tag=tool：应触发 tool_calls（可检索问题）
  - tag=clarify：信息不足，应澄清

最小指标：
  - tool_call_rate
  - citation_coverage
  - clarify_rate（仅 expect_clarify / tag=clarify 子集）
  - p50/p95 latency
  - tool_fail_rate + top_errors
  - avg_steps / p95_steps

常用：
  ./scripts/start_server.sh 2>&1 | tee /tmp/app.log
  python scripts/run_agent_eval.py
  python scripts/run_agent_eval.py --limit 5   # 冒烟
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

DEFAULT_SAMPLES = ROOT / "agent_eval_samples.jsonl"
DEFAULT_REPORT = ROOT / "agent_eval_report.json"
DEFAULT_DETAILS = ROOT / "reports" / "agent_eval_details.jsonl"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 120.0

TAG_TOOL = "tool"
TAG_CLARIFY = "clarify"

CLARIFY_PHRASES = (
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
    "需要更多上下文",
    "请告知",
    "证据仍不足",
    "无法给出可靠",
    "更具体的关键词",
    "复现路径",
    "机型/系统",
    "能否提供",
    "方便补充",
)

TOOL_FAIL_PREFIXES = ("TOOL_", "AGENT_")
TOOL_FAIL_CODES = frozenset(
    {
        "TOOL_INVALID_ARGS",
        "TOOL_TIMEOUT",
        "TOOL_NOT_FOUND",
        "TOOL_INDEX_NOT_READY",
        "AGENT_TIMEOUT",
        "AGENT_MAX_STEPS",
        "AGENT_NO_ANSWER",
    }
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def hit_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(p in (text or "") for p in phrases)


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
            tag = str(item["tag"]).strip().lower()
            item["tag"] = tag
            if "expect_clarify" not in item:
                item["expect_clarify"] = tag == TAG_CLARIFY
            if "expect_tool_calls" not in item:
                item["expect_tool_calls"] = tag == TAG_TOOL
            rows.append(item)
    return rows


def validate_sample_distribution(samples: list[dict[str, Any]]) -> dict[str, int]:
    if len(samples) < 30:
        raise ValueError(f"need >= 30 samples, got {len(samples)}")
    counts = Counter(s.get("tag") or "unknown" for s in samples)
    if counts.get(TAG_TOOL, 0) < 20:
        raise ValueError(f"tag=tool need >= 20, got {counts.get(TAG_TOOL, 0)}")
    if counts.get(TAG_CLARIFY, 0) < 8:
        raise ValueError(f"tag=clarify need >= 8, got {counts.get(TAG_CLARIFY, 0)}")
    return dict(counts)


def is_tool_fail_error(error_code: str | None) -> bool:
    if not error_code:
        return False
    if error_code in TOOL_FAIL_CODES:
        return True
    return any(error_code.startswith(p) for p in TOOL_FAIL_PREFIXES)


def is_clarified(*, answer: str, stop_reason: str | None, finish_reason: str | None) -> bool:
    if stop_reason == "clarify" or finish_reason == "clarify":
        return True
    return hit_any_phrase(answer, CLARIFY_PHRASES)


def call_ask_agent(
    client: httpx.Client,
    base_url: str,
    query: str,
    *,
    top_k: int | None,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "ok": False,
        "status_code": None,
        "latency_ms_total": None,
        "error_code": None,
        "request_id": None,
        "answer": None,
        "citations": [],
        "meta": {},
        "tool_calls_count": 0,
        "agent_steps": None,
        "stop_reason": None,
        "tools_used": [],
    }
    payload: dict[str, Any] = {
        "query": query,
        "client_tag": "run_agent_eval",
        "mode": "agent",
    }
    if top_k is not None:
        payload["top_k"] = top_k
    try:
        resp = client.post(
            f"{base_url.rstrip('/')}/ask",
            params={"mode": "agent"},
            json=payload,
            timeout=timeout,
        )
        row["status_code"] = resp.status_code
        row["latency_ms_total"] = int((time.perf_counter() - started) * 1000)
        try:
            body = resp.json()
        except json.JSONDecodeError:
            row["error_code"] = "BAD_JSON"
            return row
        if resp.status_code != 200:
            row["error_code"] = body.get("code") or f"HTTP_{resp.status_code}"
            row["request_id"] = body.get("request_id")
            return row
        meta = body.get("meta") or {}
        row["ok"] = True
        row["request_id"] = body.get("request_id")
        row["answer"] = body.get("answer")
        row["citations"] = body.get("citations") or []
        row["meta"] = meta
        row["tool_calls_count"] = int(meta.get("tool_calls_count") or 0)
        row["agent_steps"] = meta.get("agent_steps")
        row["stop_reason"] = meta.get("stop_reason")
        row["tools_used"] = list(meta.get("tools_used") or [])
        if meta.get("fallback") and meta.get("error_code"):
            row["error_code"] = meta.get("error_code")
        return row
    except httpx.TimeoutException:
        row["latency_ms_total"] = int((time.perf_counter() - started) * 1000)
        row["error_code"] = "CLIENT_TIMEOUT"
        return row
    except httpx.HTTPError as exc:
        row["latency_ms_total"] = int((time.perf_counter() - started) * 1000)
        row["error_code"] = f"CLIENT_ERROR:{type(exc).__name__}"
        return row


def evaluate_row(sample: dict[str, Any], ask: dict[str, Any]) -> dict[str, Any]:
    answer = str(ask.get("answer") or "")
    citations = ask.get("citations") or []
    stop_reason = ask.get("stop_reason")
    finish_reason = (ask.get("meta") or {}).get("finish_reason")
    tool_calls = int(ask.get("tool_calls_count") or 0)
    expect_clarify = bool(sample.get("expect_clarify"))
    clarified = is_clarified(
        answer=answer,
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
    )
    error_code = ask.get("error_code")
    return {
        "id": sample["id"],
        "tag": sample.get("tag"),
        "category": sample.get("category"),
        "query": sample["query"],
        "expect_tool_calls": bool(sample.get("expect_tool_calls")),
        "expect_clarify": expect_clarify,
        "ok": ask.get("ok"),
        "status_code": ask.get("status_code"),
        "error_code": error_code,
        "request_id": ask.get("request_id"),
        "latency_ms_total": ask.get("latency_ms_total"),
        "tool_calls_count": tool_calls,
        "had_tool_calls": tool_calls >= 1,
        "tools_used": ask.get("tools_used") or [],
        "citations_count": len(citations),
        "citations_nonempty": bool(citations),
        "agent_steps": ask.get("agent_steps"),
        "stop_reason": stop_reason,
        "clarified": clarified if expect_clarify else None,
        "tool_failed": is_tool_fail_error(
            error_code if isinstance(error_code, str) else None
        ),
        "answer_preview": answer[:160].replace("\n", " "),
    }


def build_report(
    results: list[dict[str, Any]],
    *,
    sample_counts: dict[str, int] | None = None,
    label: str = "agent",
) -> dict[str, Any]:
    total = len(results)
    http_ok = sum(1 for r in results if r.get("ok"))
    tool_n = sum(1 for r in results if r.get("had_tool_calls"))
    citation_n = sum(1 for r in results if r.get("citations_nonempty"))
    tool_fail_n = sum(1 for r in results if r.get("tool_failed"))

    clarify_rows = [r for r in results if r.get("expect_clarify")]
    clarify_ok = sum(1 for r in clarify_rows if r.get("clarified"))
    clarify_rate = (
        round(clarify_ok / len(clarify_rows), 4) if clarify_rows else None
    )

    latencies = sorted(
        float(r["latency_ms_total"])
        for r in results
        if isinstance(r.get("latency_ms_total"), (int, float))
    )
    steps = sorted(
        float(r["agent_steps"])
        for r in results
        if isinstance(r.get("agent_steps"), (int, float))
    )

    error_counter = Counter()
    for r in results:
        code = r.get("error_code")
        if code:
            error_counter[str(code)] += 1
        elif not r.get("ok"):
            error_counter["unknown"] += 1

    # 期望调工具却完全没调：也记入 top 观察（非硬错误）
    miss_tool = sum(
        1
        for r in results
        if r.get("expect_tool_calls") and r.get("ok") and not r.get("had_tool_calls")
    )
    if miss_tool:
        error_counter["EXPECT_TOOL_BUT_NONE"] = miss_tool

    avg_steps = round(sum(steps) / len(steps), 4) if steps else None

    return {
        "generated_at": utc_now_iso(),
        "label": label,
        "mode": "agent",
        "total": total,
        "http_ok": http_ok,
        "http_ok_rate": round(http_ok / total, 4) if total else 0.0,
        "sample_tag_counts": sample_counts
        or dict(Counter(r.get("tag") for r in results)),
        "tool_call_rate": round(tool_n / total, 4) if total else 0.0,
        "tool_calls_triggered": tool_n,
        "citation_coverage": round(citation_n / total, 4) if total else 0.0,
        "citations_nonempty": citation_n,
        "clarify_rate": clarify_rate,
        "clarify_handled": clarify_ok,
        "clarify_total": len(clarify_rows),
        "latency_ms_total": {
            "p50": round(percentile(latencies, 50) or 0.0, 2) if latencies else None,
            "p95": round(percentile(latencies, 95) or 0.0, 2) if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
        "tool_fail_rate": round(tool_fail_n / total, 4) if total else 0.0,
        "tool_fail_count": tool_fail_n,
        "top_errors": [
            {"error_code": code, "count": count}
            for code, count in error_counter.most_common(8)
        ],
        "avg_steps": avg_steps,
        "p95_steps": round(percentile(steps, 95) or 0.0, 2) if steps else None,
        "max_steps_observed": steps[-1] if steps else None,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(report: dict[str, Any], *, report_path: Path) -> None:
    lat = report.get("latency_ms_total") or {}
    print(f"report            : {report_path}")
    print(f"total             : {report.get('total')}")
    print(f"http_ok_rate      : {report.get('http_ok_rate')}")
    print(f"tool_call_rate    : {report.get('tool_call_rate')}")
    print(f"citation_coverage : {report.get('citation_coverage')}")
    print(f"clarify_rate      : {report.get('clarify_rate')}")
    print(f"p50_latency_ms    : {lat.get('p50')}")
    print(f"p95_latency_ms    : {lat.get('p95')}")
    print(f"tool_fail_rate    : {report.get('tool_fail_rate')}")
    print(f"top_errors        : {report.get('top_errors')}")
    print(f"avg_steps         : {report.get('avg_steps')}")
    print(f"p95_steps         : {report.get('p95_steps')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent mode 评测 → agent_eval_report.json")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--details", default=str(DEFAULT_DETAILS))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（冒烟）")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--label", default="agent")
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="跳过 ≥30 / tag 分布校验（配合 --limit）",
    )
    args = parser.parse_args()

    samples_path = Path(args.samples)
    report_path = Path(args.report)
    details_path = Path(args.details)

    try:
        samples = load_samples(samples_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL load samples: {exc}", file=sys.stderr)
        return 2

    if args.limit is not None:
        samples = samples[: max(0, args.limit)]

    sample_counts: dict[str, int] | None = None
    if not args.skip_validate and args.limit is None:
        try:
            sample_counts = validate_sample_distribution(samples)
        except ValueError as exc:
            print(f"FAIL sample distribution: {exc}", file=sys.stderr)
            return 2
    else:
        sample_counts = dict(Counter(s.get("tag") for s in samples))

    print(f"samples={len(samples)} tags={sample_counts} base_url={args.base_url}")

    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{args.base_url.rstrip('/')}/health")
        if health.status_code != 200:
            print(f"FAIL health status={health.status_code}", file=sys.stderr)
            return 1

        for i, sample in enumerate(samples, start=1):
            ask = call_ask_agent(
                client,
                args.base_url,
                sample["query"],
                top_k=args.top_k,
                timeout=args.timeout,
            )
            row = evaluate_row(sample, ask)
            results.append(row)
            print(
                f"[{i}/{len(samples)}] {sample['id']} ok={row.get('ok')} "
                f"tools={row.get('tool_calls_count')} steps={row.get('agent_steps')} "
                f"cite={row.get('citations_count')} stop={row.get('stop_reason')} "
                f"latency={row.get('latency_ms_total')}"
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

    report = build_report(results, sample_counts=sample_counts, label=args.label)
    write_json(report_path, report)
    write_jsonl(details_path, results)
    print_summary(report, report_path=report_path)
    print(f"details           : {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
