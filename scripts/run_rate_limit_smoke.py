#!/usr/bin/env python3
"""Day24：小规模压测限流——超限应返回 RATE_LIMITED。

【常用】
  # 服务已启动；.env 可临时 RATE_LIMIT_RPM=5
  python scripts/run_rate_limit_smoke.py --n 20 --rpm-hint 5
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

import httpx

DEFAULT_BASE = "http://127.0.0.1:8000"


def main() -> int:
    parser = argparse.ArgumentParser(description="Day24 rate limit smoke")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--tenant", default="burst-tenant")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--ask-path", default="/v1/ask")
    args = parser.parse_args()

    headers = {"X-Tenant-Id": args.tenant, "Content-Type": "application/json"}
    if args.api_key:
        headers["X-Api-Key"] = args.api_key

    codes: Counter[str] = Counter()
    status: Counter[int] = Counter()
    url = f"{args.base_url.rstrip('/')}{args.ask_path}"
    print(f"burst n={args.n} tenant={args.tenant} url={url}")
    with httpx.Client(timeout=30.0) as client:
        for i in range(args.n):
            try:
                resp = client.post(
                    url,
                    params={"mode": "llm"},
                    headers=headers,
                    json={"query": f"ping {i}"},
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{i}] CLIENT_ERROR {type(exc).__name__}: {exc}")
                codes["CLIENT_ERROR"] += 1
                continue
            status[resp.status_code] += 1
            data = resp.json() if resp.content else {}
            code = str(data.get("code") or ("OK" if resp.status_code == 200 else resp.status_code))
            codes[code] += 1
            if resp.status_code == 429:
                print(f"[{i}] RATE_LIMITED retry_after={resp.headers.get('Retry-After')}")
            time.sleep(0.01)

    print("---")
    print("status:", dict(status))
    print("codes:", dict(codes))
    if codes.get("RATE_LIMITED", 0) > 0:
        print("PASS: saw RATE_LIMITED")
        return 0
    print("WARN: no RATE_LIMITED — 调低 RATE_LIMIT_RPM 后重启服务再试")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
