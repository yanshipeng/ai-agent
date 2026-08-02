"""Agent 工具：kb_search / kb_get_chunk。

==========================================================================
做什么
==========================================================================
给大模型「可调用的本地知识库能力」：
  1) kb_search(query, top_k?)  —— 向量检索 TopK（只回 snippet，省 token）
  2) kb_get_chunk(chunk_id)    —— 按 id 取全文（需要细节时再调）

为什么拆成两个，而不是一次 retrieve 塞全文？
  - 模型先看候选卡片，再按需读全文，符合「先广后深」
  - 控制单轮 tool 结果体积，降低超 max_tokens / 变慢 / 变贵

==========================================================================
失败怎么处理（重要）
==========================================================================
工具层**绝不向路由抛未捕获异常**：统一返回
  { "ok": false, "error_code": "TOOL_xxx", "message": "..." }
由 Runner 写成 role=tool，让模型自己决定改查询或澄清。
这样服务不会因为坏参数 / 索引缺失 / 超时而 500。

可控错误码：
  TOOL_INVALID_ARGS   —— JSON/字段不合法
  TOOL_NOT_FOUND      —— 未知工具名
  TOOL_TIMEOUT        —— 单工具执行超时（线程池 future）
  TOOL_INDEX_NOT_READY—— 本地索引未建
  TOOL_EXEC_FAILED    —— 其它执行异常（兜底）
  TOOL_NEEDS_APPROVAL —— 高风险工具需 human-in-the-loop（Day17）

==========================================================================
和 mode=rag 的关系
==========================================================================
底层都复用 app.kb.retriever；差别只在「谁发起检索」：
  rag   = 服务端固定先 retrieve
  agent = 模型通过 tool_calls 决定是否/何时检索
==========================================================================
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable

from app.core.config import get_settings
from app.kb.index_store import DEFAULT_INDEX_DIR
from app.kb.retriever import DEFAULT_TOP_K, get_chunk, retrieve

ASK_MODE_AGENT = "agent"

TOOL_KB_SEARCH = "kb_search"
TOOL_KB_GET_CHUNK = "kb_get_chunk"

TOOL_INVALID_ARGS = "TOOL_INVALID_ARGS"
TOOL_TIMEOUT = "TOOL_TIMEOUT"
TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
TOOL_EXEC_FAILED = "TOOL_EXEC_FAILED"
TOOL_INDEX_NOT_READY = "TOOL_INDEX_NOT_READY"
TOOL_NEEDS_APPROVAL = "TOOL_NEEDS_APPROVAL"

DEFAULT_TOOL_TIMEOUT_SECONDS = 10.0
MAX_TOP_K = 20
MAX_QUERY_LEN = 500
MAX_CHUNK_ID_LEN = 200

# Day17：高风险工具（当前未注册到 _HANDLERS；预留给 DB/Text-to-SQL/Shell）
# 调用时一律返回 TOOL_NEEDS_APPROVAL，除非显式 approved=True（人工确认后）。
# 约定：未来接 SQL 时可先只返回 SQL 文本、不执行；执行必须 HITL。
HIGH_RISK_TOOLS = frozenset(
    {
        "db_query",
        "execute_sql",
        "text_to_sql_execute",
        "shell_exec",
    }
)

# ---------------------------------------------------------------------------
# OpenAI / DeepSeek function-calling schema
# 原样传给 chat.completions.create(..., tools=...)
# description 写清楚「何时该调」，能显著提高真实 tool_calls 触发率
# ---------------------------------------------------------------------------
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_KB_SEARCH,
            "description": (
                "在本地稳定性知识库中混合检索（向量+关键词）最相关的 TopK 片段。"
                "返回 chunk_id / score / title / url / text_snippet。"
                "排障类问题应优先调用本工具，再按需 kb_get_chunk 取全文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索用的查询语句（中文或英文关键词）",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"返回条数，默认 {DEFAULT_TOP_K}，最大 {MAX_TOP_K}",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_KB_GET_CHUNK,
            "description": (
                "按 chunk_id 从本地索引取该片段全文与元数据。"
                "chunk_id 通常来自 kb_search 的结果。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "知识库片段 ID，例如 A-1:0:0",
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
]


def openai_tools_schema() -> list[dict[str, Any]]:
    """返回可直接传给 chat.completions 的 tools 列表（拷贝，避免被调用方改坏常量）。"""
    return list(TOOL_SPECS)


def _index_dir() -> Path:
    """Day23：优先 dataset current.json；否则 KB_INDEX_DIR。"""
    try:
        from app.kb.dataset_registry import resolve_active_index_dir

        return resolve_active_index_dir()
    except Exception:  # noqa: BLE001
        settings = get_settings()
        return Path(settings.kb_index_dir or DEFAULT_INDEX_DIR)


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """统一失败结构：ok=False + error_code，便于 Runner / 路由映射。"""
    payload = {"ok": False, "error_code": code, "message": message}
    payload.update(extra)
    return payload


def _ok(data: dict[str, Any]) -> dict[str, Any]:
    """统一成功结构：ok=True + 业务字段。"""
    return {"ok": True, **data}


def _parse_args(raw: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """解析 tool arguments。

    DeepSeek / OpenAI 常见两种形态：
      - 已是 dict（SDK 已解析）
      - JSON 字符串（需 json.loads）
    解析失败返回 (None, error_dict)，成功返回 (args, None)。
    """
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return None, _error(TOOL_INVALID_ARGS, "arguments must be a JSON object or string")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, _error(TOOL_INVALID_ARGS, "arguments is not valid JSON")
    if not isinstance(parsed, dict):
        return None, _error(TOOL_INVALID_ARGS, "arguments JSON must be an object")
    return parsed, None


def tool_kb_search(args: dict[str, Any], *, index_dir: Path | str | None = None) -> dict[str, Any]:
    """执行 kb_search：校验 query/top_k → retrieve（snippet，不含全文）。

    include_text=False：强制只回 snippet，避免一次把 TopK 全文塞进 messages。
    """
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return _error(TOOL_INVALID_ARGS, "query is required and must be a non-empty string")
    query = query.strip()
    if len(query) > MAX_QUERY_LEN:
        return _error(TOOL_INVALID_ARGS, f"query must be at most {MAX_QUERY_LEN} characters")

    top_k = args.get("top_k", DEFAULT_TOP_K)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        return _error(TOOL_INVALID_ARGS, "top_k must be an integer")
    if top_k < 1 or top_k > MAX_TOP_K:
        return _error(TOOL_INVALID_ARGS, f"top_k must be between 1 and {MAX_TOP_K}")

    resolved_dir = Path(index_dir) if index_dir is not None else _index_dir()
    if not resolved_dir.joinpath("manifest.json").exists():
        return _error(
            TOOL_INDEX_NOT_READY,
            "knowledge index not ready; run build_kb_index.py",
            index_dir=str(resolved_dir),
        )

    try:
        out = retrieve(
            query,
            top_k=top_k,
            index_dir=resolved_dir,
            include_snippet=True,
            include_text=False,
        )
    except FileNotFoundError:
        return _error(TOOL_INDEX_NOT_READY, "knowledge index not found", index_dir=str(resolved_dir))
    except Exception as exc:  # noqa: BLE001 — 工具层吞掉，转成可控码
        return _error(TOOL_EXEC_FAILED, f"kb_search failed: {exc}")

    results = out.get("results") or []
    return _ok(
        {
            "tool": TOOL_KB_SEARCH,
            "query_len": len(query),
            "top_k": out.get("top_k", top_k),
            "retrieve_ms": out.get("retrieve_ms"),
            "hit_count": len(results),
            "results": results,
        }
    )


def tool_kb_get_chunk(args: dict[str, Any], *, index_dir: Path | str | None = None) -> dict[str, Any]:
    """执行 kb_get_chunk：校验 chunk_id → get_chunk 取全文。

    chunk_id 不存在时返回 TOOL_INVALID_ARGS（参数语义错误），不是 INDEX_NOT_READY。
    """
    chunk_id = args.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        return _error(TOOL_INVALID_ARGS, "chunk_id is required and must be a non-empty string")
    chunk_id = chunk_id.strip()
    if len(chunk_id) > MAX_CHUNK_ID_LEN:
        return _error(TOOL_INVALID_ARGS, f"chunk_id must be at most {MAX_CHUNK_ID_LEN} characters")

    resolved_dir = Path(index_dir) if index_dir is not None else _index_dir()
    if not resolved_dir.joinpath("manifest.json").exists():
        return _error(
            TOOL_INDEX_NOT_READY,
            "knowledge index not ready; run build_kb_index.py",
            index_dir=str(resolved_dir),
        )

    try:
        row = get_chunk(chunk_id, index_dir=resolved_dir)
    except FileNotFoundError:
        return _error(TOOL_INDEX_NOT_READY, "knowledge index not found", index_dir=str(resolved_dir))
    except Exception as exc:  # noqa: BLE001
        return _error(TOOL_EXEC_FAILED, f"kb_get_chunk failed: {exc}")

    if row is None:
        return _error(TOOL_INVALID_ARGS, f"chunk_id not found: {chunk_id}", chunk_id=chunk_id)

    return _ok({"tool": TOOL_KB_GET_CHUNK, "chunk": row})


_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    TOOL_KB_SEARCH: tool_kb_search,
    TOOL_KB_GET_CHUNK: tool_kb_get_chunk,
}


def execute_tool(
    name: str,
    arguments: Any,
    *,
    index_dir: Path | str | None = None,
    timeout_seconds: float | None = None,
    approved: bool = False,
) -> dict[str, Any]:
    """工具统一入口：高风险审批 → 白名单 → 参数校验 → 限时执行。

    超时用 ThreadPoolExecutor + future.result(timeout=...)：
      单工具卡住时返回 TOOL_TIMEOUT，不拖垮整单 Agent。
    永不向调用方抛异常（最后一道 except → TOOL_EXEC_FAILED）。

    approved=True：仅用于未来 HITL 确认后重放高风险工具；默认 False。
    """
    tool_name = (name or "").strip()

    # Day17：高风险工具必须人工确认（优先于 NOT_FOUND，便于模型/前端识别）
    if tool_name in HIGH_RISK_TOOLS and not approved:
        return _error(
            TOOL_NEEDS_APPROVAL,
            (
                f"high-risk tool '{tool_name}' requires human-in-the-loop approval; "
                "do not execute. For SQL: return the statement for review only."
            ),
            tool=tool_name,
            requires_human=True,
        )

    if tool_name not in _HANDLERS:
        return _error(TOOL_NOT_FOUND, f"unknown tool: {tool_name or '(empty)'}", tool=tool_name)

    args, err = _parse_args(arguments)
    if err is not None:
        err["tool"] = tool_name
        return err
    assert args is not None

    settings = get_settings()
    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else float(getattr(settings, "agent_tool_timeout_seconds", DEFAULT_TOOL_TIMEOUT_SECONDS))
    )
    timeout = max(0.1, timeout)
    handler = _HANDLERS[tool_name]

    def _run() -> dict[str, Any]:
        return handler(args, index_dir=index_dir)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            return future.result(timeout=timeout)
    except FuturesTimeout:
        return _error(
            TOOL_TIMEOUT,
            f"tool timed out after {timeout:.1f}s",
            tool=tool_name,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(TOOL_EXEC_FAILED, f"tool execution failed: {exc}", tool=tool_name)
