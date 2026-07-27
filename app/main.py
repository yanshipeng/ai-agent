"""FastAPI 应用创建与启动入口。

==========================================================================
为什么用 lifespan 注入 LLMClient
==========================================================================
进程内复用一个 client（连接池 / 配置一致），避免每个请求 new OpenAI()。
测试里可替换 app.state.llm_client = MagicMock()，不必真打网。

启动时打 app_startup（含 kb_index_dir / rag_top_k 与中文 hint），
方便新手确认「服务起来了、索引路径对不对」。
==========================================================================
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api import ask_validation_exception_handler, router
from app.core.config import get_settings
from app.core.logging import (
    EVENT_APP_SHUTDOWN,
    EVENT_APP_STARTUP,
    get_logger,
    log_event,
    setup_logging,
)
from app.services.llm_client import LLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化日志与 LLMClient，关闭时打 shutdown 日志。"""
    setup_logging()
    settings = get_settings()
    logger = get_logger(__name__)
    # 进程内复用同一个 client，避免每次请求重建连接
    app.state.llm_client = LLMClient(settings)
    log_event(
        logger,
        EVENT_APP_STARTUP,
        version=settings.app_version,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        kb_index_dir=settings.kb_index_dir,
        rag_top_k=settings.rag_top_k,
        requests_jsonl_path=settings.requests_jsonl_path,
        hint=(
            "服务已启动。健康检查 GET /health；问答 POST /ask；"
            f"RAG 索引目录={settings.kb_index_dir}；指标文件={settings.requests_jsonl_path}"
        ),
    )
    yield
    log_event(logger, EVENT_APP_SHUTDOWN)


def create_app() -> FastAPI:
    """工厂方法：创建并配置 FastAPI 实例（测试与生产共用）。"""
    settings = get_settings()
    app = FastAPI(
        title="AI Start Agent",
        version=settings.app_version,
        lifespan=lifespan,
    )
    # 统一把 RequestValidationError 转成 INVALID_ARGUMENT 错误体
    app.add_exception_handler(RequestValidationError, ask_validation_exception_handler)
    app.include_router(router)
    return app


# uvicorn 默认加载的 ASGI 对象
app = create_app()


def main() -> None:
    """本地直接 python -m app.main 时的启动方式。"""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
