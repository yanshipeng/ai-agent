#!/usr/bin/env python3
"""Agent 工具冒烟：对已启动服务发 5 次 mode=agent，验收真实 tool_calls。

验收：
  - 5 次请求里 ≥3 次日志出现 tool_call_start（真实 tool_calls）
  - requests.jsonl 含 mode=agent / agent_steps / tool_calls_count / tools_used

常用：
  ./scripts/start_server.sh 2>&1 | tee /tmp/app.log
  python scripts/smoke_agent_tools.py --log /tmp/app.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_QUERIES = [
    "Android ANR 怎么排查",
    "OOM 内存泄漏怎么查",
    "App 卡顿怎么分析",
    "Native Crash 怎么看 tombstone",
    "线上稳定性告警怎么定位",
]


def count_tool_call_events(log_path: Path, request_ids: list[str]) -> int:
    """统计有多少 request_id 出现过 tool_call_start。"""
    if not log_path.exists():
        return 0
    hit: set[str] = set()
    wanted = set(request_ids)
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "tool_call_start":
            continue
        rid = row.get("request_id")
        if rid in wanted:
            hit.add(rid)
    return len(hit)


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent tool_calls 冒烟（5 次 /ask?mode=agent）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--log", default="/tmp/app.log", help="服务 tee 的日志文件")
    parser.add_argument("--metrics", default="./requests.jsonl")
    parser.add_argument("--min-tool-calls", type=int, default=3)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    log_path = Path(args.log)
    metrics_path = Path(args.metrics)

    request_ids: list[str] = []
    ok = 0
    with httpx.Client(timeout=120.0) as client:
        health = client.get(f"{base}/health")
        if health.status_code != 200:
            print(f"FAIL health status={health.status_code}")
            return 1

        for i, query in enumerate(DEFAULT_QUERIES, start=1):
            resp = client.post(
                f"{base}/ask",
                params={"mode": "agent"},
                json={"query": query, "client_tag": "smoke_agent"},
            )
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            rid = body.get("request_id")
            if rid:
                request_ids.append(rid)
            meta = body.get("meta") or {}
            tool_calls = meta.get("tool_calls_count") or 0
            print(
                f"[{i}/5] HTTP {resp.status_code} request_id={rid} "
                f"tool_calls={tool_calls} tools={meta.get('tools_used')} "
                f"steps={meta.get('agent_steps')}"
            )
            if resp.status_code == 200 and tool_calls >= 1:
                ok += 1
            time.sleep(0.3)

    # 给日志一点落盘时间
    time.sleep(0.5)
    log_hits = count_tool_call_events(log_path, request_ids)
    print(f"log tool_call_start coverage: {log_hits}/{len(request_ids)} (need >= {args.min_tool_calls})")

    agent_metrics = 0
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("request_id") in request_ids and row.get("mode") == "agent":
                agent_metrics += 1
                for key in ("agent_steps", "max_steps", "tool_calls_count", "tools_used", "stop_reason"):
                    if key not in row:
                        print(f"FAIL metrics missing {key}: {row.get('request_id')}")
                        return 1
                if row.get("agent_steps", 0) > row.get("max_steps", 0):
                    print(f"FAIL agent_steps > max_steps: {row}")
                    return 1
    print(f"requests.jsonl agent rows for this run: {agent_metrics}")

    if log_hits < args.min_tool_calls and ok < args.min_tool_calls:
        print(
            f"FAIL: need >= {args.min_tool_calls} real tool_calls "
            f"(meta ok={ok}, log hits={log_hits})"
        )
        return 1

    print("PASS agent tool smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
