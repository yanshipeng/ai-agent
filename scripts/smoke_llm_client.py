#!/usr/bin/env python3
"""不启动 FastAPI，直接连续调用 LLMClient.chat 做冒烟验证。

用途：验证密钥、网络、模型可用性；默认 10 次，成功率需 ≥ 9/10。

判定：
- 成功率 ≥ min-success（默认 9/10）
- 每次成功的 latency_ms 有值且合理（0 < latency_ms <= 30000）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.services.llm_client import LLMClient, LLMError  # noqa: E402

DEFAULT_ROUNDS = 10
MIN_SUCCESS = 9
MAX_LATENCY_MS = 30_000
QUERY = "用一句话回答：1+1等于几？"


def run_once(client: LLMClient, index: int) -> dict:
    started = time.perf_counter()
    row: dict = {"index": index, "ok": False}
    try:
        result = client.chat([{"role": "user", "content": QUERY}])
        row.update(
            {
                "ok": True,
                "answer": result.answer,
                "model": result.model,
                "finish_reason": result.finish_reason,
                "latency_ms": result.latency_ms,
                "usage": result.usage,
            }
        )
    except LLMError as exc:
        row.update(
            {
                "ok": False,
                "error_code": exc.code,
                "error_message": exc.message,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "ok": False,
                "error_code": "unexpected",
                "error_message": str(exc),
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )
    return row


def latency_ok(latency_ms: object) -> bool:
    return isinstance(latency_ms, int) and 0 < latency_ms <= MAX_LATENCY_MS


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test LLMClient without FastAPI")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--min-success", type=int, default=MIN_SUCCESS)
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    if not settings.deepseek_api_key:
        print("DEEPSEEK_API_KEY missing", file=sys.stderr)
        return 2

    client = LLMClient(settings)
    rows: list[dict] = []
    print(f"query={QUERY!r} rounds={args.rounds}")

    for i in range(1, args.rounds + 1):
        row = run_once(client, i)
        rows.append(row)
        if row["ok"]:
            print(
                f"[{i}/{args.rounds}] OK  latency_ms={row['latency_ms']} "
                f"answer={row['answer'][:60]!r}"
            )
        else:
            print(
                f"[{i}/{args.rounds}] FAIL code={row.get('error_code')} "
                f"msg={row.get('error_message')}"
            )

    successes = [r for r in rows if r["ok"]]
    success_count = len(successes)
    bad_latency = [r for r in successes if not latency_ok(r.get("latency_ms"))]
    latencies = [r["latency_ms"] for r in successes if isinstance(r.get("latency_ms"), int)]

    summary = {
        "total": args.rounds,
        "success": success_count,
        "failed": args.rounds - success_count,
        "success_rate": f"{success_count}/{args.rounds}",
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "avg": round(sum(latencies) / len(latencies), 1) if latencies else None,
        },
        "bad_latency_count": len(bad_latency),
    }
    print("\nsummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if success_count < args.min_success:
        print(f"FAIL: success {success_count}/{args.rounds} < {args.min_success}", file=sys.stderr)
        return 1
    if bad_latency:
        print(
            f"FAIL: {len(bad_latency)} success calls have unreasonable latency_ms "
            f"(need 0 < latency_ms <= {MAX_LATENCY_MS})",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: success ≥ {args.min_success}/{args.rounds}, latency_ms ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
