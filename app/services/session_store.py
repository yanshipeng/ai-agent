"""进程内 session 消息存储（按 session_id）。

==========================================================================
做什么
==========================================================================
为多轮对话保存「同 session 的历史 messages」：
  get_session_messages(session_id) → 读出历史（深拷贝）
  set_session_messages(session_id, messages) → 覆盖写入
  touch_and_prune(ttl) → 删过期会话

为什么用内存而不是 DB / Redis？
  教学与本地单进程足够；改动面小、零依赖。
  代价：进程重启清空；多 uvicorn worker 时各有一份（与索引缓存同理）。

设计约定（与 conversation / api 配合）：
  - 不存主 system prompt（每次请求重新注入）
  - llm/rag 只存短 user/assistant；agent 可存截断后的 tool 轨迹
  - 对外日志用 session_id_sha256_8；requests.jsonl 可写明文 session_id（验收用）
==========================================================================
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# session_id → { "messages": [...], "updated_at": float, "ttl_seconds": optional }
_SESSIONS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def clear_all_sessions() -> None:
    """测试用：清空全部会话，避免用例互相污染。"""
    with _LOCK:
        _SESSIONS.clear()


def clear_session(session_id: str) -> None:
    """删除单个会话。"""
    sid = (session_id or "").strip()
    if not sid:
        return
    with _LOCK:
        _SESSIONS.pop(sid, None)


def get_session_messages(session_id: str | None) -> list[dict[str, Any]]:
    """读取会话历史（不含当前轮）；无 session_id 返回空列表。

    返回深拷贝，避免调用方原地改坏全局缓存。
    """
    sid = (session_id or "").strip()
    if not sid:
        return []
    with _LOCK:
        row = _SESSIONS.get(sid)
        if not row:
            return []
        return deepcopy(row.get("messages") or [])


def set_session_messages(
    session_id: str | None,
    messages: list[dict[str, Any]],
    *,
    ttl_seconds: float | None = None,
) -> None:
    """覆盖写入会话历史。空 session_id 则忽略（无 session = 无状态）。"""
    sid = (session_id or "").strip()
    if not sid:
        return
    with _LOCK:
        _SESSIONS[sid] = {
            "messages": deepcopy(messages),
            "updated_at": time.time(),
            "ttl_seconds": ttl_seconds,
        }


def touch_and_prune(*, ttl_seconds: float = 3600.0) -> int:
    """删除过期会话；返回删除数量。

    每条可带自己的 ttl_seconds；缺省用参数默认值（配置 SESSION_TTL_SECONDS）。
    """
    now = time.time()
    removed = 0
    with _LOCK:
        expired = [
            sid
            for sid, row in _SESSIONS.items()
            if now - float(row.get("updated_at") or 0)
            > float(row.get("ttl_seconds") or ttl_seconds)
        ]
        for sid in expired:
            _SESSIONS.pop(sid, None)
            removed += 1
    return removed


def session_count() -> int:
    """当前内存中的会话个数（运维/测试用）。"""
    with _LOCK:
        return len(_SESSIONS)
