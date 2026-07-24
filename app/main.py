"""FastAPI 应用创建与启动入口。

- create_app()：组装路由、校验异常处理、生命周期
- lifespan：初始化结构化日志，并向 app.state 注入共享 LLMClient
- python -m app.main / uvicorn app.main:app 均可启动
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api import ask_validation_exception_handler, router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.llm_client import LLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化日志与 LLMClient，关闭时打 shutdown 日志。"""
    setup_logging()
    settings = get_settings()
    logger = get_logger(__name__)
    # 进程内复用同一个 client，避免每次请求重建连接
    app.state.llm_client = LLMClient(settings)
    logger.info(
        "app_startup",
        extra={
            "event": "app_startup",
            "version": settings.app_version,
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "timeout_seconds": settings.llm_timeout_seconds,
            # 故意不记录 api_key（含脱敏）
        },
    )
    yield
    logger.info("app_shutdown", extra={"event": "app_shutdown"})


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
