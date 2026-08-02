#!/usr/bin/env python3
"""按 request_id 抽取事件链，并用中文 hint 帮助新手读懂。

成功链路（llm）：
  request_start → llm_call_start → llm_call_end → request_success

成功链路（rag）：
  request_start → retrieve_start → retrieve_end → llm_call_start
                → llm_call_end → request_success

用法：
  ./scripts/start_server.sh 2>&1 | tee /tmp/app.log
  # 另开终端请求后：
  python scripts/trace_request.py <request_id> --log /tmp/app.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SUCCESS_LLM = ("request_start", "llm_call_start", "llm_call_end", "request_success")
SUCCESS_RAG = (
    "request_start",
    "retrieve_start",
    "retrieve_end",
    "llm_call_start",
    "llm_call_end",
    "request_success",
)
ERROR_AFTER_LLM = ("request_start", "llm_call_start", "llm_call_end", "request_error")
ERROR_AFTER_RETRIEVE = (
    "request_start",
    "retrieve_start",
    "retrieve_end",
    "llm_call_start",
    "llm_call_end",
    "request_error",
)
ERROR_INDEX = ("request_start", "retrieve_start", "request_error")
VALIDATE_CHAIN = ("validate_failed", "request_error")

EVENT_TITLE = {
    "request_start": "① 收到请求",
    "validate_failed": "① 参数校验失败",
    "retrieve_start": "② 开始本地检索",
    "retrieve_end": "③ 检索结束并拼 Context",
    "llm_call_start": "④ 调用大模型",
    "llm_call_end": "⑤ 大模型返回",
    "request_success": "⑥ 整单成功",
    "request_error": "⑥ 整单失败",
    "app_startup": "服务启动",
    "app_shutdown": "服务关闭",
}

SENSITIVE_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}|\"query\"\s*:")


def iter_lines(path: str | None):
    if path:
        with Path(path).open(encoding="utf-8") as fp:
            yield from fp
    else:
        yield from sys.stdin


def covers(events: list[str], chain: tuple[str, ...]) -> bool:
    idx = 0
    for ev in events:
        if idx < len(chain) and ev == chain[idx]:
            idx += 1
    return idx == len(chain)


def main() -> int:
    parser = argparse.ArgumentParser(description="按 request_id 回放日志事件链（新手友好）")
    parser.add_argument("request_id", help="UUID from /ask response")
    parser.add_argument("--log", default=None, help="Log file path (default: stdin)")
    args = parser.parse_args()

    rows: list[dict] = []
    for line in iter_lines(args.log):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if row.get("request_id") != args.request_id:
            continue
        rows.append(row)
        if SENSITIVE_RE.search(text):
            print(f"SENSITIVE LEAK in log line: {text}", file=sys.stderr)
            return 2

    events = [str(r.get("event") or r.get("msg") or "") for r in rows]
    print(f"request_id = {args.request_id}")
    print(f"events     = {' → '.join(events) if events else '(none)'}")
    print("-" * 60)
    for row in rows:
        ev = str(row.get("event") or row.get("msg") or "")
        title = EVENT_TITLE.get(ev, ev)
        hint = row.get("hint") or ""
        print(f"[{title}] event={ev}")
        if hint:
            print(f"  说明: {hint}")
        # 关键业务字段摘要
        interesting = {
            k: row[k]
            for k in (
                "mode",
                "top_k",
                "retrieve_ms",
                "context_chunks",
                "citations_count",
                "latency_ms",
                "latency_ms_total",
                "ok",
                "error_code",
                "status_code",
                "llm_model",
                "finish_reason",
                "retry_count",
                "top_chunk_ids",
            )
            if k in row
        }
        if interesting:
            print(f"  字段: {json.dumps(interesting, ensure_ascii=False)}")
    print("-" * 60)

    if covers(events, SUCCESS_RAG):
        print("判定: 完整 RAG 成功链路 ✓")
        return 0
    if covers(events, SUCCESS_LLM):
        print("判定: 完整 LLM 成功链路 ✓（未走检索，或检索日志缺失）")
        return 0
    if covers(events, ERROR_AFTER_RETRIEVE) or covers(events, ERROR_AFTER_LLM):
        print("判定: 调用大模型后失败链路")
        return 0
    if covers(events, ERROR_INDEX):
        print("判定: 索引未就绪或检索阶段失败")
        return 0
    if covers(events, VALIDATE_CHAIN):
        print("判定: 参数校验失败链路")
        return 0

    print("判定: 事件链不完整（可能日志未 tee 全，或中途进程重启）", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
