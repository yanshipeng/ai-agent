"""Day22：请求级审计字段（contextvars），供 metrics 自动合并。"""

from __future__ import annotations

import contextvars
from typing import Any

_AUDIT_FIELDS: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "audit_fields",
    default={},
)


def set_audit_fields(fields: dict[str, Any] | None) -> None:
    _AUDIT_FIELDS.set(dict(fields or {}))


def get_audit_fields() -> dict[str, Any]:
    return dict(_AUDIT_FIELDS.get() or {})


def clear_audit_fields() -> None:
    _AUDIT_FIELDS.set({})
