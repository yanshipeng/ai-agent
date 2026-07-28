"""Agent：工具定义 + Plan→Act→Observe→Final 状态机。

==========================================================================
线上路径（第三周）
==========================================================================
  POST /ask?mode=agent
    → api._ask_agent
    → run_agent_loop（本包 runner）
    → chat_turn(..., tools=) 真实 tool_calls
    → execute_tool(kb_search | kb_get_chunk)
    → 超限：降级 RAG / 澄清 / 硬终止；超时兜底含 request_id

核心逻辑在 app/agent/；scripts/ 只做冒烟与评测 CLI。
导出 stop_reason 常量与 execute_tool，方便测试与外部脚本复用。
==========================================================================
"""

from app.agent.runner import (
    STOP_CLARIFY,
    STOP_DEGRADED_TO_RAG,
    STOP_FINAL_ANSWER,
    STOP_MAX_STEPS,
    STOP_REASONS,
    STOP_TIMEOUT,
    STOP_UPSTREAM_ERROR,
    AgentPhase,
    AgentResult,
    run_agent_loop,
)
from app.agent.tools import (
    ASK_MODE_AGENT,
    TOOL_INVALID_ARGS,
    TOOL_NOT_FOUND,
    TOOL_TIMEOUT,
    TOOL_SPECS,
    execute_tool,
    openai_tools_schema,
)

__all__ = [
    "ASK_MODE_AGENT",
    "AgentPhase",
    "AgentResult",
    "STOP_CLARIFY",
    "STOP_DEGRADED_TO_RAG",
    "STOP_FINAL_ANSWER",
    "STOP_MAX_STEPS",
    "STOP_REASONS",
    "STOP_TIMEOUT",
    "STOP_UPSTREAM_ERROR",
    "TOOL_INVALID_ARGS",
    "TOOL_NOT_FOUND",
    "TOOL_TIMEOUT",
    "TOOL_SPECS",
    "execute_tool",
    "openai_tools_schema",
    "run_agent_loop",
]
