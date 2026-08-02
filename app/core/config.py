"""应用配置：从环境变量 / .env 读取运行时参数。

==========================================================================
为什么用 pydantic-settings + lru_cache
==========================================================================
- 类型校验、默认值、别名（DEEPSEEK_API_KEY）一处定义，避免满项目 os.getenv。
- get_settings() 缓存：热路径不反复读盘；因此改 .env 后必须重启进程。

注意优先级：
  pydantic-settings 默认「进程环境变量 > .env 文件」。
  若 shell 里 export 过旧 Key，会盖住 .env —— 所以 start_server.sh 会用 .env 强制覆盖。

Week4 相关字段：
  RAG_HYBRID_WEIGHT / RAG_MIN_SCORE —— Day16 混合检索
  AGENT_MAX_CONTEXT_TOKENS / TRACES_JSONL_PATH —— Day18 Trace 与预算
==========================================================================
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时配置，对应项目环境变量清单。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek / LLM
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    llm_base_url: str = Field(default="https://api.deepseek.com", alias="LLM_BASE_URL")
    llm_model: str = Field(default="deepseek-v4-flash", alias="LLM_MODEL")
    # Day20：高质量路由目标模型（默认 pro）
    llm_model_pro: str = Field(default="deepseek-v4-pro", alias="LLM_MODEL_PRO")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_thinking: str = Field(default="disabled", alias="LLM_THINKING")

    # 可选
    requests_jsonl_path: str = Field(
        default="./data/runtime/requests.jsonl",
        alias="REQUESTS_JSONL_PATH",
    )
    app_version: str = Field(default="0.1.0ba", alias="APP_VERSION")

    # RAG / 知识库
    kb_index_dir: str = Field(
        default="data/stability_kb/index",
        alias="KB_INDEX_DIR",
    )
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    # Day16：混合检索权重（向量占比）；关键词 = 1 - weight
    rag_hybrid_weight: float = Field(default=0.6, alias="RAG_HYBRID_WEIGHT")
    # Day16：融合分低于此阈值的候选丢弃
    rag_min_score: float = Field(default=0.05, alias="RAG_MIN_SCORE")
    # Day20：Top1 低于此分数 → 路由到 pro（主成本策略条件）
    rag_route_pro_min_score: float = Field(
        default=0.35,
        alias="RAG_ROUTE_PRO_MIN_SCORE",
    )

    # Agent / Tools（第三周）
    agent_max_steps: int = Field(default=5, alias="AGENT_MAX_STEPS")
    agent_max_total_time_ms: int = Field(
        default=20_000,
        alias="AGENT_MAX_TOTAL_TIME_MS",
    )
    agent_on_max_steps: str = Field(
        default="rag",
        alias="AGENT_ON_MAX_STEPS",
        description="超 max_steps：rag / clarify / error",
    )
    agent_tool_timeout_seconds: float = Field(
        default=10.0,
        alias="AGENT_TOOL_TIMEOUT_SECONDS",
    )
    # Day18：Agent context token 预算（启发式估算）
    agent_max_context_tokens: int = Field(
        default=6000,
        alias="AGENT_MAX_CONTEXT_TOKENS",
    )
    traces_jsonl_path: str = Field(
        default="./data/runtime/traces.jsonl",
        alias="TRACES_JSONL_PATH",
    )

    # 多轮 session 上下文（第三周）：滑窗 / 截断 / 可选摘要
    session_max_turns: int = Field(default=8, alias="SESSION_MAX_TURNS")
    session_max_chars: int = Field(default=20_000, alias="SESSION_MAX_CHARS")
    session_tool_result_max_chars: int = Field(
        default=4_000,
        alias="SESSION_TOOL_RESULT_MAX_CHARS",
    )
    session_content_max_chars: int = Field(
        default=8_000,
        alias="SESSION_CONTENT_MAX_CHARS",
    )
    session_enable_summary: bool = Field(default=True, alias="SESSION_ENABLE_SUMMARY")
    session_summary_use_llm: bool = Field(
        default=False,
        alias="SESSION_SUMMARY_USE_LLM",
    )
    session_ttl_seconds: float = Field(default=3600.0, alias="SESSION_TTL_SECONDS")

    # Day22：最小 API Key 鉴权（开启后除 /health|/docs 外必须带 token）
    api_auth_enabled: bool = Field(default=False, alias="API_AUTH_ENABLED")
    # 格式：key1:admin,key2:reader（无 :role 时默认 reader）
    api_keys: str = Field(default="", alias="API_KEYS")

    # Day23：版本化知识库根目录（versions/ + current.json）
    kb_versions_dir: str = Field(
        default="data/stability_kb/versions",
        alias="KB_VERSIONS_DIR",
    )
    kb_docs_path: str = Field(
        default="data/stability_kb/docs.jsonl",
        alias="KB_DOCS_PATH",
    )

    # Day24：限流（按 tenant 或 api_key）；默认关以免拖垮本地 pytest
    rate_limit_enabled: bool = Field(default=False, alias="RATE_LIMIT_ENABLED")
    rate_limit_rpm: int = Field(default=60, alias="RATE_LIMIT_RPM")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")
    # Day24：单次请求 completion token 硬顶（控 p95）
    request_token_budget: int = Field(default=1024, alias="REQUEST_TOKEN_BUDGET")
    request_budget_top_k_cap: int = Field(default=3, alias="REQUEST_BUDGET_TOP_K_CAP")

    # Day25：反馈 / badcase
    feedback_jsonl_path: str = Field(
        default="data/feedback/feedback.jsonl",
        alias="FEEDBACK_JSONL_PATH",
    )
    badcases_pending_path: str = Field(
        default="data/feedback/badcases_pending.jsonl",
        alias="BADCASES_PENDING_PATH",
    )


@lru_cache
def get_settings() -> Settings:
    """获取单例配置；测试中修改环境变量后需调用 get_settings.cache_clear()。"""
    return Settings()
