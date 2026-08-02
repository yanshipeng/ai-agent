"""Day22：最小 API Key 鉴权 + tenant/user/role 透传。

==========================================================================
约定
==========================================================================
- Header：Authorization: Bearer <key> 或 X-Api-Key: <key>（必填，鉴权开启时）
- 身份透传：X-Tenant-Id / X-User-Id / X-Role；也可从 body 读（header 优先）
- 角色：reader（问答）/ admin（入库+评测）；key 可绑定默认角色
  API_KEYS 格式：key1:admin,key2:reader 或 key1,key2（默认 reader）

健康检查 /docs 可匿名；其余 /ask /v1/* 一律要 key。
==========================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from starlette.requests import Request as StarletteRequest

from app.core.config import get_settings

CODE_UNAUTHORIZED = "UNAUTHORIZED"
CODE_FORBIDDEN = "FORBIDDEN"

ROLE_READER = "reader"
ROLE_ADMIN = "admin"
KNOWN_ROLES = frozenset({ROLE_READER, ROLE_ADMIN})

# 匿名可访问
PUBLIC_PATH_PREFIXES = (
    "/health",
    "/v1/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)

_HEADER_API_KEY = "x-api-key"
_HEADER_TENANT = "x-tenant-id"
_HEADER_USER = "x-user-id"
_HEADER_ROLE = "x-role"
_BEARER_RE = re.compile(r"^\s*Bearer\s+(\S+)\s*$", re.I)


@dataclass(frozen=True)
class AuthContext:
    """请求身份（不落敏感原文到日志时可再用指纹）。"""

    api_key_id: str  # key 指纹（前 4…后 2），非原文
    tenant_id: str | None
    user_id: str | None
    role: str

    def metric_fields(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "role": self.role,
            "api_key_id": self.api_key_id,
        }


class AuthError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 401) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def fingerprint_key(key: str) -> str:
    raw = (key or "").strip()
    if len(raw) <= 6:
        return "***"
    return f"{raw[:4]}…{raw[-2:]}"


def parse_api_keys(raw: str | None) -> dict[str, str]:
    """解析 API_KEYS → {key: role}。"""
    text = (raw or "").strip()
    if not text:
        return {}
    out: dict[str, str] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            key, role = item.split(":", 1)
            key = key.strip()
            role = (role.strip().lower() or ROLE_READER)
        else:
            key, role = item, ROLE_READER
        if not key:
            continue
        if role not in KNOWN_ROLES:
            role = ROLE_READER
        out[key] = role
    return out


def extract_bearer_or_api_key(request: StarletteRequest) -> str | None:
    header_key = request.headers.get(_HEADER_API_KEY)
    if header_key and header_key.strip():
        return header_key.strip()
    auth = request.headers.get("authorization") or ""
    m = _BEARER_RE.match(auth)
    if m:
        return m.group(1).strip()
    return None


def is_public_path(path: str) -> bool:
    p = (path or "").rstrip("/") or "/"
    if p == "/":
        return True
    for prefix in PUBLIC_PATH_PREFIXES:
        if p == prefix.rstrip("/") or p.startswith(prefix.rstrip("/") + "/"):
            return True
        if p == prefix:
            return True
    # /health 与带尾斜杠
    if p in {"/health", "/v1/health"}:
        return True
    return False


def resolve_identity(
    request: StarletteRequest,
    *,
    body_tenant_id: str | None = None,
    body_user_id: str | None = None,
    body_role: str | None = None,
    key_default_role: str = ROLE_READER,
) -> tuple[str | None, str | None, str]:
    """header 优先，其次 body；role 最终规范化。"""
    tenant = (request.headers.get(_HEADER_TENANT) or body_tenant_id or "").strip() or None
    user = (request.headers.get(_HEADER_USER) or body_user_id or "").strip() or None
    role_raw = (request.headers.get(_HEADER_ROLE) or body_role or key_default_role or ROLE_READER)
    role = str(role_raw).strip().lower()
    if role not in KNOWN_ROLES:
        role = key_default_role if key_default_role in KNOWN_ROLES else ROLE_READER
    return tenant, user, role


def authenticate_request(request: StarletteRequest) -> AuthContext:
    """校验 API Key；失败抛 AuthError。"""
    settings = get_settings()
    if not bool(getattr(settings, "api_auth_enabled", False)):
        # 关闭鉴权时仍接收透传身份，便于本地联调
        tenant, user, role = resolve_identity(request)
        return AuthContext(
            api_key_id="auth_disabled",
            tenant_id=tenant,
            user_id=user,
            role=role,
        )

    keys = parse_api_keys(getattr(settings, "api_keys", None))
    if not keys:
        raise AuthError(
            CODE_UNAUTHORIZED,
            "API auth enabled but API_KEYS is empty",
            status_code=401,
        )

    token = extract_bearer_or_api_key(request)
    if not token:
        raise AuthError(
            CODE_UNAUTHORIZED,
            "missing API token; use Authorization: Bearer <key> or X-Api-Key",
            status_code=401,
        )
    if token not in keys:
        raise AuthError(CODE_UNAUTHORIZED, "invalid API token", status_code=401)

    key_role = keys[token]
    tenant, user, role = resolve_identity(request, key_default_role=key_role)
    return AuthContext(
        api_key_id=fingerprint_key(token),
        tenant_id=tenant,
        user_id=user,
        role=role,
    )


def require_role(auth: AuthContext, *, min_role: str) -> None:
    """admin 可做一切；reader 仅问答。"""
    if min_role == ROLE_READER:
        return
    if min_role == ROLE_ADMIN and auth.role != ROLE_ADMIN:
        raise AuthError(
            CODE_FORBIDDEN,
            f"role={auth.role} not allowed; requires {ROLE_ADMIN}",
            status_code=403,
        )


def get_request_auth(request: Request) -> AuthContext | None:
    return getattr(request.state, "auth", None)


def set_request_auth(request: Request, auth: AuthContext) -> None:
    request.state.auth = auth
