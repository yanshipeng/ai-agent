#!/usr/bin/env python3
"""多轮 session 验收：同 session 连续 5 轮，检查约束记忆 + 指标字段 + P95。

验收：
  1) 同 session_id 连续 5 轮；约束如「只讨论 Android 推送」能被记住
  2) requests.jsonl 含 session_id / history_messages / history_chars
  3) 本批 p95 相对第 1 轮不明显恶化（默认允许 ≤ 3x 第 1 轮延迟）

常用：
  ./scripts/start_server.sh 2>&1 | tee /tmp/app.log
  python scripts/smoke_session_memory.py
  python scripts/stats_requests.py --path ./requests.jsonl --session-id smoke-session-push
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONSTRAINT = "从现在起只讨论 Android 推送相关问题，其它话题请明确拒绝并提醒约束。"

DEFAULT_TURNS = [
    CONSTRAINT,
    "FCM 和厂商推送通道有什么区别？",
    "推送到达率突然下降怎么排查？",
    "帮我写一个 React 登录页组件。",  # 偏题：应拒绝 / 提醒约束
    "请复述本会话必须遵守的约束是什么？",
]

# 第 4 轮偏题、第 5 轮复述：答案里应出现这些线索之一
CONSTRAINT_HINTS = ("推送", "Android", "约束", "拒绝", "只讨论", "FCM", "通道")
OFFTOPIC_HINTS = ("拒绝", "推送", "约束", "不讨论", "范围", "Android")


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    return float(sorted_values[f] * (c - k) + sorted_values[c] * (k - f))


def answer_has_any(text: str, hints: tuple[str, ...]) -> bool:
    lower = text or ""
    return any(h in lower for h in hints)


def load_metric_rows(path: Path, request_ids: list[str]) -> list[dict]:
    if not path.exists():
        return []
    wanted = set(request_ids)
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("request_id") in wanted:
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="同 session 5 轮记忆 + 指标验收")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics", default="./requests.jsonl")
    parser.add_argument("--mode", default="llm", choices=["llm", "rag", "agent"])
    parser.add_argument(
        "--session-id",
        default=None,
        help="默认每次随机生成，避免旧历史污染",
    )
    parser.add_argument(
        "--max-p95-ratio",
        type=float,
        default=3.0,
        help="本批 p95 相对第 1 轮 latency 的上限倍数（默认 3）",
    )
    args = parser.parse_args()

    session_id = (args.session_id or f"smoke-session-push-{uuid.uuid4().hex[:8]}").strip()
    base = args.base_url.rstrip("/")
    metrics_path = Path(args.metrics)

    request_ids: list[str] = []
    latencies: list[float] = []
    answers: list[str] = []

    print(f"session_id={session_id} mode={args.mode}")

    with httpx.Client(timeout=120.0) as client:
        health = client.get(f"{base}/health")
        if health.status_code != 200:
            print(f"FAIL health status={health.status_code}")
            return 1

        for i, query in enumerate(DEFAULT_TURNS, start=1):
            resp = client.post(
                f"{base}/ask",
                params={"mode": args.mode} if args.mode != "llm" else None,
                json={
                    "query": query,
                    "session_id": session_id,
                    "client_tag": "smoke_session",
                    "mode": args.mode,
                },
            )
            body = (
                resp.json()
                if resp.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            rid = body.get("request_id")
            answer = body.get("answer") or ""
            meta = body.get("meta") or {}
            latency = body.get("latency_ms")
            if rid:
                request_ids.append(rid)
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
            answers.append(answer)
            print(
                f"[{i}/5] HTTP {resp.status_code} request_id={rid} "
                f"latency_ms={latency} history_messages={meta.get('history_messages')} "
                f"history_chars={meta.get('history_chars')}"
            )
            print(f"       Q: {query[:60]}")
            print(f"       A: {answer[:160].replace(chr(10), ' ')}")
            if resp.status_code != 200:
                print("FAIL non-200 response")
                return 1
            time.sleep(0.2)

    # --- 记忆验收：第 4 轮偏题应守约束；第 5 轮能复述 ---
    if not answer_has_any(answers[3], OFFTOPIC_HINTS):
        print(
            "FAIL turn4 off-topic：回答未体现「只讨论 Android 推送」约束 "
            f"(answer={answers[3][:200]!r})"
        )
        return 1
    if not answer_has_any(answers[4], CONSTRAINT_HINTS):
        print(
            "FAIL turn5 recall：回答未复述推送相关约束 "
            f"(answer={answers[4][:200]!r})"
        )
        return 1
    print("PASS memory: constraint held across 5 turns")

    # --- requests.jsonl 字段 ---
    time.sleep(0.3)
    rows = load_metric_rows(metrics_path, request_ids)
    if len(rows) < 5:
        print(f"FAIL metrics rows for this run: {len(rows)} (need 5)")
        return 1

    rows_by_id = {r.get("request_id"): r for r in rows}
    expected_history = [0, 2, 4, 6, 8]  # 每轮结束后 +2 条；下一轮读入
    for i, rid in enumerate(request_ids):
        row = rows_by_id.get(rid)
        if not row:
            print(f"FAIL missing metrics for {rid}")
            return 1
        for key in ("session_id", "history_messages", "history_chars"):
            if key not in row:
                print(f"FAIL metrics missing {key}: {rid}")
                return 1
        if row.get("session_id") != session_id:
            print(f"FAIL session_id mismatch: {row.get('session_id')} != {session_id}")
            return 1
        hm = row.get("history_messages")
        if hm != expected_history[i]:
            print(
                f"FAIL history_messages turn{i+1}: got {hm}, expected {expected_history[i]}"
            )
            return 1
        if not isinstance(row.get("history_chars"), int) or row["history_chars"] < 0:
            print(f"FAIL history_chars invalid: {row.get('history_chars')}")
            return 1
        if i > 0 and row["history_chars"] <= 0:
            print(f"FAIL history_chars should grow after turn1: {row}")
            return 1

    print("PASS metrics: session_id / history_messages / history_chars")

    # --- P95：相对第 1 轮不明显恶化 ---
    p95 = percentile(latencies, 95)
    first = latencies[0] if latencies else None
    print(f"latencies_ms={latencies}")
    print(f"p95_latency_ms={None if p95 is None else round(p95, 2)} first={first}")
    if first and p95 and first > 0:
        ratio = p95 / first
        print(f"p95/first_ratio={ratio:.2f} (limit={args.max_p95_ratio})")
        if ratio > args.max_p95_ratio:
            print(
                f"FAIL P95 worsened too much: p95={p95:.0f}ms vs first={first:.0f}ms "
                f"(ratio={ratio:.2f} > {args.max_p95_ratio})"
            )
            return 1
    print("PASS latency: p95 within budget vs turn1")

    print(
        "\nTIP: python scripts/stats_requests.py "
        f"--path {metrics_path} --session-id {session_id}"
    )
    print("PASS session memory smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
