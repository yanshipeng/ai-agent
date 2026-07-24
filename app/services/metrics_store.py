"""请求指标落盘：每次请求结束追加 1 行 JSON 到 requests.jsonl。

字段（建议）：
  ts, request_id, path, ok, status_code,
  latency_ms_total, latency_ms_llm,
  llm_model, retry_count, finish_reason, error_code,
  query_len, query_sha256_8,
  prompt_tokens / completion_tokens / total_tokens（有则写）

不写 query / answer 原文。写入失败只记日志，不阻断主流程。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger, query_sha256_8 as hash_query_sha256_8

logger = get_logger(__name__)

# 多 worker / 多线程追加时串行化写文件
_lock = threading.Lock()


def _default_path() -> Path:
    """从配置读取 JSONL 路径（REQUESTS_JSONL_PATH）。"""
    return Path(get_settings().requests_jsonl_path)


def append_request_metric(
    record: dict[str, Any],
    *,
    path: str | Path | None = None,
) -> None:
    """将一条请求指标追加写入 JSONL（1 行 = 1 请求）。

    - 自动丢弃值为 None 的字段
    - 若调用方未提供 ts，自动补 UTC ISO 时间
    """
    payload = {k: v for k, v in record.items() if v is not None}
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())

    target = Path(path) if path is not None else _default_path()
    line = json.dumps(payload, ensure_ascii=False, default=str)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with target.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
    except OSError as exc:
        logger.error(
            "metrics_write_failed",
            extra={"event": "metrics_write_failed", "path": str(target), "error": str(exc)},
        )


def _extract_tokens(usage: dict[str, Any] | None) -> dict[str, int]:
    """从 usage 中提取可落盘的 token 计数字段。"""
    if not usage:
        return {}
    tokens: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            tokens[key] = value
    return tokens


def build_ask_metric(
    *,
    request_id: str,
    path: str = "/ask",
    ok: bool,
    status_code: int,
    latency_ms_total: int | None = None,
    latency_ms_llm: int | None = None,
    llm_model: str | None = None,
    retry_count: int | None = None,
    finish_reason: str | None = None,
    error_code: str | None = None,
    query: str | None = None,
    query_len: int | None = None,
    query_sha256_8: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 /ask 请求结束指标记录（不含 query/answer 原文）。

    若传入 query，可自动补齐 query_len / query_sha256_8（仍不会把原文写入 record）。
    """
    resolved_len = query_len
    resolved_hash = query_sha256_8
    if query is not None:
        if resolved_len is None:
            resolved_len = len(query)
        if resolved_hash is None and query:
            resolved_hash = hash_query_sha256_8(query)

    record: dict[str, Any] = {
        "request_id": request_id,
        "path": path,
        "ok": ok,
        "status_code": status_code,
        "latency_ms_total": latency_ms_total,
        "latency_ms_llm": latency_ms_llm,
        "llm_model": llm_model,
        "retry_count": retry_count,
        "finish_reason": finish_reason,
        "error_code": error_code,
        "query_len": resolved_len,
        "query_sha256_8": resolved_hash,
    }
    record.update(_extract_tokens(usage))
    return record
