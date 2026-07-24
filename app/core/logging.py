"""结构化日志：JSON 输出 + 固定业务事件 + 脱敏。

固定事件名（按 request_id 串联）：
  request_start → llm_call_start → llm_call_end → request_success
  或 request_start → … → request_error
  或 validate_failed → request_error

约定：不记录 query 原文、不记录 API key。
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

# ---------- 固定事件名 ----------
EVENT_REQUEST_START = "request_start"
EVENT_VALIDATE_FAILED = "validate_failed"
EVENT_LLM_CALL_START = "llm_call_start"
EVENT_LLM_CALL_END = "llm_call_end"
EVENT_REQUEST_SUCCESS = "request_success"
EVENT_REQUEST_ERROR = "request_error"

LLM_PROVIDER = "deepseek"

# 日志脱敏：匹配 sk-xxx / api_key=xxx 等
_SENSITIVE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]+"),
    re.compile(r"(?i)(api[_-]?key|authorization)\s*[:=]\s*\S+"),
)


class JsonFormatter(logging.Formatter):
    """将 LogRecord 格式化为单行 JSON，透传 extra 业务字段。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # LogRecord 内置字段，避免写入业务 JSON
        reserved = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in reserved and not key.startswith("_"):
                payload[key] = value

        # 业务事件：msg 与 event 对齐，方便 grep
        if "event" in payload and payload["event"]:
            payload["msg"] = payload["event"]

        import json

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根 logger：stdout + JSON；降低 httpx/openai 噪音。"""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """按模块名获取 logger。"""
    return logging.getLogger(name)


def query_sha256_8(query: str) -> str:
    """生成 query 指纹：sha256 hex 前 8 位（用于关联请求，不落原文）。"""
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return digest[:8]


def sanitize_log_text(text: str | None) -> str | None:
    """脱敏文本中可能出现的 API key 等敏感片段。"""
    if text is None:
        return None
    cleaned = text
    for pattern in _SENSITIVE_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """输出结构化业务事件。

    - ts / level 由 JsonFormatter 补齐
    - None 字段自动跳过
    - 禁止通过 fields 传入 query / api_key 等敏感键
    """
    banned = {"query", "api_key", "deepseek_api_key", "authorization"}
    dirty = banned.intersection(fields)
    if dirty:
        raise ValueError(f"refusing to log sensitive fields: {sorted(dirty)}")

    payload = {"event": event}
    for key, value in fields.items():
        if value is None:
            continue
        if key in {"error_message", "message"} and isinstance(value, str):
            payload[key] = sanitize_log_text(value)
        else:
            payload[key] = value

    logger.log(level, event, extra=payload)
