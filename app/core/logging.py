"""结构化日志：JSON 输出 + 固定业务事件 + 中文 hint + 脱敏。

==========================================================================
新手怎么读日志（服务 stdout / 启动脚本重定向的文件）
==========================================================================
每行是一条 JSON。先看这几个字段：

  event   —— 发生了什么（英文事件名，可用 grep）
  hint    —— 人话解释（给新手看的中文说明）
  request_id —— 同一次 /ask 请求的身份证，用来串起整条链路

典型成功链路（mode=llm）：
  request_start → llm_call_start → llm_call_end → request_success

典型成功链路（mode=rag）：
  request_start → retrieve_start → retrieve_end → llm_call_start
                → llm_call_end → request_success

失败链路示例：
  request_start → … → request_error
  validate_failed → request_error

追踪某次请求：
  python scripts/trace_request.py <request_id> --log /tmp/app.log

约定：不记录 query 原文、不记录 API key。
详细说明见 docs/日志阅读指南.md
==========================================================================
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

# ---------- 固定事件名 ----------
EVENT_APP_STARTUP = "app_startup"
EVENT_APP_SHUTDOWN = "app_shutdown"
EVENT_REQUEST_START = "request_start"
EVENT_VALIDATE_FAILED = "validate_failed"
EVENT_RETRIEVE_START = "retrieve_start"
EVENT_RETRIEVE_END = "retrieve_end"
EVENT_LLM_CALL_START = "llm_call_start"
EVENT_LLM_CALL_END = "llm_call_end"
EVENT_REQUEST_SUCCESS = "request_success"
EVENT_REQUEST_ERROR = "request_error"

LLM_PROVIDER = "deepseek"

# 事件 → 新手可读中文说明（log_event 会自动写入 hint，也可调用方覆盖）
EVENT_HINTS: dict[str, str] = {
    EVENT_APP_STARTUP: "服务启动完成：日志已就绪，LLMClient 已创建",
    EVENT_APP_SHUTDOWN: "服务正在关闭",
    EVENT_REQUEST_START: "收到 /ask 请求，开始处理（此时还不知道会不会检索）",
    EVENT_VALIDATE_FAILED: "请求参数校验失败（例如 query 为空或超长）",
    EVENT_RETRIEVE_START: "RAG 模式：开始在本地知识库索引中检索相关片段",
    EVENT_RETRIEVE_END: "RAG 模式：检索结束，已拼好带编号的 Context，准备交给大模型",
    EVENT_LLM_CALL_START: "开始调用 DeepSeek 大模型生成回答",
    EVENT_LLM_CALL_END: "大模型调用结束（成功 / 兜底 / 失败见 ok、error_code）",
    EVENT_REQUEST_SUCCESS: "本次 /ask 整单成功，已写 requests.jsonl 指标",
    EVENT_REQUEST_ERROR: "本次 /ask 失败，见 error_code 与 status_code",
}

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


def hint_for(event: str, *, override: str | None = None) -> str:
    """返回事件的中文说明；override 优先。"""
    if override:
        return override
    return EVENT_HINTS.get(event, f"业务事件：{event}")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    hint: str | None = None,
    **fields: Any,
) -> None:
    """输出结构化业务事件。

    - 自动带上 hint（中文，方便新手读日志）
    - ts / level 由 JsonFormatter 补齐
    - None 字段自动跳过
    - 禁止通过 fields 传入 query / api_key 等敏感键
    """
    banned = {"query", "api_key", "deepseek_api_key", "authorization"}
    dirty = banned.intersection(fields)
    if dirty:
        raise ValueError(f"refusing to log sensitive fields: {sorted(dirty)}")

    payload: dict[str, Any] = {
        "event": event,
        "hint": hint_for(event, override=hint),
    }
    for key, value in fields.items():
        if value is None:
            continue
        if key in {"error_message", "message"} and isinstance(value, str):
            payload[key] = sanitize_log_text(value)
        else:
            payload[key] = value

    logger.log(level, event, extra=payload)
