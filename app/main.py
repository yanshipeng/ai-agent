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
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import ask_validation_exception_handler, build_error_body, router
from app.api_v1 import router as router_v1
from app.core.audit_context import clear_audit_fields, set_audit_fields
from app.core.auth import (
    AuthError,
    authenticate_request,
    is_public_path,
    set_request_auth,
)
from app.services.rate_limit import RateLimitError, check_rate_limit
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
            "服务已启动。健康检查 GET /health|/v1/health；"
            "问答 POST /ask|/v1/ask；入库 POST /v1/ingest；评测 POST /v1/eval/run；"
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

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        """Day22：API Key 鉴权；审计字段写入 contextvars。"""
        clear_audit_fields()
        path = request.url.path
        if is_public_path(path):
            return await call_next(request)
        try:
            auth = authenticate_request(request)
        except AuthError as exc:
            request_id = str(uuid.uuid4())
            return JSONResponse(
                status_code=exc.status_code,
                content=build_error_body(
                    request_id=request_id,
                    code=exc.code,
                    message=exc.message,
                ),
            )
        set_request_auth(request, auth)
        set_audit_fields(auth.metric_fields())
        # Day24：鉴权通过后按 tenant / api_key 限流
        try:
            check_rate_limit(
                tenant_id=auth.tenant_id,
                api_key_id=auth.api_key_id,
            )
        except RateLimitError as exc:
            request_id = str(uuid.uuid4())
            headers = {}
            if exc.decision.retry_after_seconds is not None:
                headers["Retry-After"] = str(int(exc.decision.retry_after_seconds) + 1)
            return JSONResponse(
                status_code=exc.status_code,
                content=build_error_body(
                    request_id=request_id,
                    code=exc.code,
                    message=exc.message,
                ),
                headers=headers,
            )
        try:
            return await call_next(request)
        finally:
            clear_audit_fields()

    app.include_router(router)
    app.include_router(router_v1)  # Day21：/v1/ask|/v1/ingest|/v1/eval/run
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
