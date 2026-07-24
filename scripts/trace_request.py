#!/usr/bin/env python3
"""按 request_id 抽取固定事件链并做敏感信息抽查。

事件：
  request_start → llm_call_start → llm_call_end → request_success
  或 request_start → ... → request_error
  或 validate_failed → request_error

用法：
  python scripts/trace_request.py <request_id> --log /tmp/app.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SUCCESS_CHAIN = ("request_start", "llm_call_start", "llm_call_end", "request_success")
ERROR_AFTER_LLM = ("request_start", "llm_call_start", "llm_call_end", "request_error")
VALIDATE_CHAIN = ("validate_failed", "request_error")

SENSITIVE_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}|\"query\"\s*:")


def iter_lines(path: str | None):
    if path:
        with Path(path).open(encoding="utf-8") as fp:
            yield from fp
    else:
        yield from sys.stdin


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace request event chain by request_id")
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

    events = [r.get("event") or r.get("msg") for r in rows]
    print(f"request_id={args.request_id}")
    for row in rows:
        ev = row.get("event") or row.get("msg")
        print(f"  [{ev}] {json.dumps(row, ensure_ascii=False)}")

    def covers(chain: tuple[str, ...]) -> bool:
        idx = 0
        for ev in events:
            if idx < len(chain) and ev == chain[idx]:
                idx += 1
        return idx == len(chain)

    if covers(SUCCESS_CHAIN):
        print("\ncomplete: success chain")
        return 0
    if covers(ERROR_AFTER_LLM):
        print("\ncomplete: error-after-llm chain")
        return 0
    if covers(VALIDATE_CHAIN):
        print("\ncomplete: validate chain")
        return 0

    print("\nincomplete event chain", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
