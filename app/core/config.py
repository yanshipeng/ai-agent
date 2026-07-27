"""应用配置：从环境变量 / .env 读取运行时参数。

==========================================================================
为什么用 pydantic-settings + lru_cache
==========================================================================
- 类型校验、默认值、别名（DEEPSEEK_API_KEY）一处定义，避免满项目 os.getenv。
- get_settings() 缓存：热路径不反复读盘；因此改 .env 后必须重启进程。

注意优先级：
  pydantic-settings 默认「进程环境变量 > .env 文件」。
  若 shell 里 export 过旧 Key，会盖住 .env —— 所以 start_server.sh 会用 .env 强制覆盖。
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
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_thinking: str = Field(default="disabled", alias="LLM_THINKING")

    # 可选
    requests_jsonl_path: str = Field(default="./requests.jsonl", alias="REQUESTS_JSONL_PATH")
    app_version: str = Field(default="0.1.0ba", alias="APP_VERSION")

    # RAG / 知识库
    kb_index_dir: str = Field(
        default="data/stability_kb/index",
        alias="KB_INDEX_DIR",
    )
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")


@lru_cache
def get_settings() -> Settings:
    """获取单例配置；测试中修改环境变量后需调用 get_settings.cache_clear()。"""
    return Settings()
