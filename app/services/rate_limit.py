"""Day24：最小进程内限流（按 tenant_id 或 api_key）。

滑动窗口：window 秒内最多 max_requests 次。
超限抛 RateLimitError → HTTP 429 + RATE_LIMITED。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

from app.core.config import get_settings

CODE_RATE_LIMITED = "RATE_LIMITED"


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    key: str
    limit: int
    window_seconds: int
    remaining: int
    retry_after_seconds: float | None = None


class RateLimitError(Exception):
    def __init__(self, message: str, *, decision: RateLimitDecision) -> None:
        super().__init__(message)
        self.code = CODE_RATE_LIMITED
        self.status_code = 429
        self.decision = decision
        self.message = message


_lock = threading.Lock()
_buckets: dict[str, Deque[float]] = defaultdict(deque)


def reset_rate_limit_state() -> None:
    """测试用：清空计数。"""
    with _lock:
        _buckets.clear()


def resolve_rate_limit_key(
    *,
    tenant_id: str | None,
    api_key_id: str | None,
) -> str:
    tid = (tenant_id or "").strip()
    if tid:
        return f"tenant:{tid}"
    kid = (api_key_id or "").strip()
    if kid and kid != "auth_disabled":
        return f"key:{kid}"
    return "anon:default"


def check_rate_limit(
    *,
    tenant_id: str | None = None,
    api_key_id: str | None = None,
    now: float | None = None,
) -> RateLimitDecision:
    """检查并记录一次请求；超限抛 RateLimitError。"""
    settings = get_settings()
    if not bool(getattr(settings, "rate_limit_enabled", True)):
        key = resolve_rate_limit_key(tenant_id=tenant_id, api_key_id=api_key_id)
        return RateLimitDecision(
            allowed=True,
            key=key,
            limit=0,
            window_seconds=0,
            remaining=999,
        )

    limit = max(1, int(getattr(settings, "rate_limit_rpm", 60) or 60))
    window = max(1, int(getattr(settings, "rate_limit_window_seconds", 60) or 60))
    key = resolve_rate_limit_key(tenant_id=tenant_id, api_key_id=api_key_id)
    ts = time.time() if now is None else float(now)

    with _lock:
        q = _buckets[key]
        cutoff = ts - window
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) >= limit:
            retry_after = max(0.1, (q[0] + window) - ts)
            decision = RateLimitDecision(
                allowed=False,
                key=key,
                limit=limit,
                window_seconds=window,
                remaining=0,
                retry_after_seconds=round(retry_after, 2),
            )
            raise RateLimitError(
                f"rate limit exceeded for {key}: {limit}/{window}s",
                decision=decision,
            )
        q.append(ts)
        remaining = max(0, limit - len(q))
        return RateLimitDecision(
            allowed=True,
            key=key,
            limit=limit,
            window_seconds=window,
            remaining=remaining,
        )
