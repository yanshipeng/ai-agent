"""Day21–25：API Contract v2 — 版本化入口 /v1/*。

核心入口：
  GET  /v1/health
  POST /v1/ask
  POST /v1/ingest
  POST /v1/eval/run
  POST /v1/feedback     （Day25：有用/无用/引用错误/幻觉）
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api import (
    CODE_INVALID_ARGUMENT,
    AskMode,
    AskRequest,
    AskResponse,
    HealthResponse,
    ask,
    build_error_body,
)
from app.core.auth import (
    ROLE_ADMIN,
    AuthError,
    get_request_auth,
    require_role,
)
from app.core.config import get_settings
from app.core.logging import get_logger, log_event
from app.kb.dataset_registry import active_dataset_info, list_versions
from app.kb.embedder import DEFAULT_DIM
from app.kb.index_store import build_index_from_chunks_file
from app.kb.ingest_pipeline import rollback_dataset, run_ingest_from_docs
from app.kb.retriever import clear_index_cache
from app.services.eval_v2_service import run_eval_batch
from app.services.feedback_store import record_feedback

router = APIRouter(prefix="/v1", tags=["v1"])
logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "stability_kb" / "chunks.jsonl"
DEFAULT_DOCS = ROOT / "data" / "stability_kb" / "docs.jsonl"
DEFAULT_SAMPLES = ROOT / "eval_samples_v2.jsonl"

EVAL_LIMIT_DEFAULT = 5
EVAL_LIMIT_MAX = 80


class IngestRequest(BaseModel):
    """入库：默认从 docs.jsonl 增量构建 dataset_version。

    action:
      ingest   — docs → chunk/embed/index（incremental 默认 true）
      rollback — 切换到指定 dataset_version
      rebuild_chunks — 兼容 Day21：直接从 chunks.jsonl 重建（不进 versions）
    """

    action: Literal["ingest", "rollback", "rebuild_chunks"] = Field(default="ingest")
    docs_path: str | None = Field(default=None, description="docs.jsonl；默认 KB_DOCS_PATH")
    chunks_path: str | None = Field(default=None, description="仅 rebuild_chunks")
    index_dir: str | None = Field(default=None, description="仅 rebuild_chunks 输出目录")
    dim: int | None = Field(default=None, ge=64, le=4096)
    incremental: bool = Field(default=True, description="仅处理新增/变更 doc")
    dataset_version: str | None = Field(
        default=None,
        description="rollback 目标版本；ingest 时可指定版本名",
    )


class IngestResponse(BaseModel):
    request_id: str
    ok: bool
    latency_ms: int
    report: dict[str, Any]
    meta: dict[str, Any] | None = None


class EvalRunRequest(BaseModel):
    offline: bool = Field(default=True)
    limit: int = Field(default=EVAL_LIMIT_DEFAULT, ge=1, le=EVAL_LIMIT_MAX)
    suite: str | None = Field(default=None)
    mode_override: Literal["llm", "rag", "agent"] | None = Field(default=None)
    samples_path: str | None = Field(default=None)


class EvalRunResponse(BaseModel):
    request_id: str
    ok: bool
    latency_ms: int
    report: dict[str, Any]
    meta: dict[str, Any] | None = None


class FeedbackRequest(BaseModel):
    """Day25：用户反馈。负面标签可沉淀为 pending badcase。"""

    label: Literal["useful", "useless", "wrong_citation", "hallucination"]
    request_id: str | None = None
    query: str | None = Field(default=None, description="原问题；badcase 沉淀需要")
    mode: Literal["llm", "rag", "agent"] | None = None
    note: str | None = None
    answer_preview: str | None = None
    promote_badcase: bool = Field(default=True, description="负面标签是否写入 pending")


class FeedbackResponse(BaseModel):
    request_id: str
    ok: bool
    feedback_id: str
    label: str
    badcase_promoted: bool
    sample: dict[str, Any] | None = None


def _resolve_under_root(raw: str | None, default: Path) -> Path:
    candidate = Path(raw).expanduser() if raw else default
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    root = ROOT.resolve()
    if not str(candidate).startswith(str(root)):
        raise ValueError(f"path must be under project root: {root}")
    return candidate


def _require_admin(request: Request) -> JSONResponse | None:
    """鉴权关闭时不拦；开启时必须 admin。"""
    if not get_settings().api_auth_enabled:
        return None
    auth = get_request_auth(request)
    if auth is None:
        return JSONResponse(
            status_code=401,
            content=build_error_body(
                request_id=str(uuid.uuid4()),
                code="UNAUTHORIZED",
                message="missing auth context",
            ),
        )
    try:
        require_role(auth, min_role=ROLE_ADMIN)
    except AuthError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_body(
                request_id=str(uuid.uuid4()),
                code=exc.code,
                message=exc.message,
            ),
        )
    return None


@router.get("/health", response_model=HealthResponse)
def health_v1() -> HealthResponse:
    return HealthResponse(status="ok", version=get_settings().app_version)


@router.get("/dataset")
def dataset_info_v1(request: Request) -> dict[str, Any]:
    """查看当前 dataset_version（reader 可读）。"""
    return active_dataset_info()


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"description": "INVALID_ARGUMENT"},
        401: {"description": "UNAUTHORIZED"},
        403: {"description": "FORBIDDEN"},
        429: {"description": "RATE_LIMITED"},
        503: {"description": "INDEX_NOT_READY"},
    },
)
def ask_v1(
    body: AskRequest,
    request: Request,
    mode: AskMode | None = Query(default=None),
) -> AskResponse | JSONResponse:
    return ask(body, request, mode=mode)


@router.post("/feedback", response_model=FeedbackResponse)
def feedback_v1(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Day25：标记有用/无用/引用错误/幻觉；负面可进 badcase 队列。"""
    auth = get_request_auth(request)
    pack = record_feedback(
        label=body.label,
        request_id=body.request_id,
        query=body.query,
        mode=body.mode,
        note=body.note,
        answer_preview=body.answer_preview,
        tenant_id=auth.tenant_id if auth else None,
        user_id=auth.user_id if auth else None,
        promote_badcase=body.promote_badcase,
    )
    return FeedbackResponse(
        request_id=str(uuid.uuid4()),
        ok=True,
        feedback_id=str(pack["feedback_id"]),
        label=str(pack["label"]),
        badcase_promoted=bool(pack["badcase_promoted"]),
        sample=pack.get("sample"),
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_v1(body: IngestRequest, request: Request) -> IngestResponse | JSONResponse:
    """Day23 入库 / 回滚；需要 admin 角色（鉴权开启时）。"""
    denied = _require_admin(request)
    if denied is not None:
        return denied

    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    settings = get_settings()

    if body.action == "rollback":
        if not body.dataset_version:
            return JSONResponse(
                status_code=400,
                content=build_error_body(
                    request_id=request_id,
                    code=CODE_INVALID_ARGUMENT,
                    message="rollback requires dataset_version",
                ),
            )
        try:
            report = rollback_dataset(body.dataset_version)
        except FileNotFoundError as exc:
            return JSONResponse(
                status_code=400,
                content=build_error_body(
                    request_id=request_id,
                    code=CODE_INVALID_ARGUMENT,
                    message=str(exc),
                ),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return IngestResponse(
            request_id=request_id,
            ok=True,
            latency_ms=latency_ms,
            report=report,
            meta={"api_version": "v1", "action": "rollback"},
        )

    if body.action == "rebuild_chunks":
        try:
            chunks_path = _resolve_under_root(body.chunks_path, DEFAULT_CHUNKS)
            default_index = Path(settings.kb_index_dir)
            if not default_index.is_absolute():
                default_index = ROOT / default_index
            index_dir = _resolve_under_root(body.index_dir, default_index.resolve())
            dim = int(body.dim or DEFAULT_DIM)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content=build_error_body(
                    request_id=request_id,
                    code=CODE_INVALID_ARGUMENT,
                    message=str(exc),
                ),
            )
        if not chunks_path.exists():
            return JSONResponse(
                status_code=400,
                content=build_error_body(
                    request_id=request_id,
                    code=CODE_INVALID_ARGUMENT,
                    message=f"chunks not found: {chunks_path}",
                ),
            )
        report = build_index_from_chunks_file(
            chunks_path, index_dir=index_dir, dim=dim, progress=False
        )
        clear_index_cache()
        latency_ms = int((time.perf_counter() - started) * 1000)
        return IngestResponse(
            request_id=request_id,
            ok=True,
            latency_ms=latency_ms,
            report=report,
            meta={"api_version": "v1", "action": "rebuild_chunks"},
        )

    # action=ingest（默认）
    try:
        default_docs = Path(settings.kb_docs_path)
        if not default_docs.is_absolute():
            default_docs = ROOT / default_docs
        docs_path = _resolve_under_root(body.docs_path, default_docs.resolve())
        dim = int(body.dim or DEFAULT_DIM)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content=build_error_body(
                request_id=request_id,
                code=CODE_INVALID_ARGUMENT,
                message=str(exc),
            ),
        )
    if not docs_path.exists():
        return JSONResponse(
            status_code=400,
            content=build_error_body(
                request_id=request_id,
                code=CODE_INVALID_ARGUMENT,
                message=f"docs not found: {docs_path}",
            ),
        )

    log_event(
        logger,
        "ingest_start",
        request_id=request_id,
        docs_path=str(docs_path),
        incremental=body.incremental,
        hint="Day23 /v1/ingest：增量入库开始",
    )
    try:
        report = run_ingest_from_docs(
            docs_path,
            incremental=body.incremental,
            dim=dim,
            dataset_version=body.dataset_version,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            logger,
            "ingest_error",
            level=logging.WARNING,
            request_id=request_id,
            latency_ms=latency_ms,
            error=f"{type(exc).__name__}: {exc}",
            hint="入库失败",
        )
        return JSONResponse(
            status_code=500,
            content=build_error_body(
                request_id=request_id,
                code="INGEST_FAILED",
                message=f"ingest failed: {type(exc).__name__}",
            ),
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    log_event(
        logger,
        "ingest_success",
        request_id=request_id,
        latency_ms=latency_ms,
        dataset_version=report.get("dataset_version"),
        chunks_rebuilt=report.get("chunks_rebuilt"),
        hint=(
            f"入库完成 version={report.get('dataset_version')} "
            f"rebuilt={report.get('chunks_rebuilt')} reused={report.get('chunks_reused')}"
        ),
    )
    return IngestResponse(
        request_id=request_id,
        ok=True,
        latency_ms=latency_ms,
        report=report,
        meta={
            "api_version": "v1",
            "action": "ingest",
            "versions": [v["dataset_version"] for v in list_versions()],
        },
    )


@router.post("/eval/run", response_model=EvalRunResponse)
def eval_run_v1(
    body: EvalRunRequest,
    request: Request,
) -> EvalRunResponse | JSONResponse:
    denied = _require_admin(request)
    if denied is not None:
        return denied

    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        samples_path = _resolve_under_root(body.samples_path, DEFAULT_SAMPLES)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content=build_error_body(
                request_id=request_id,
                code=CODE_INVALID_ARGUMENT,
                message=str(exc),
            ),
        )
    if not samples_path.exists():
        return JSONResponse(
            status_code=400,
            content=build_error_body(
                request_id=request_id,
                code=CODE_INVALID_ARGUMENT,
                message=f"samples not found: {samples_path}",
            ),
        )

    ask_caller = None
    if not body.offline:

        def ask_caller(query: str, mode: str) -> dict[str, Any]:
            ask_body = AskRequest(query=query, mode=mode)  # type: ignore[arg-type]
            result = ask(ask_body, request, mode=mode)  # type: ignore[arg-type]
            if isinstance(result, JSONResponse):
                payload = result.body
                try:
                    import json

                    data = json.loads(payload.decode("utf-8")) if payload else {}
                except Exception:  # noqa: BLE001
                    data = {}
                return {
                    "ok": False,
                    "answer": "",
                    "meta": {},
                    "latency_ms": None,
                    "status_code": result.status_code,
                    "error_code": data.get("code"),
                }
            return {
                "ok": True,
                "answer": result.answer,
                "meta": result.meta or {},
                "latency_ms": result.latency_ms,
                "status_code": 200,
                "error_code": None,
            }

    try:
        pack = run_eval_batch(
            samples_path=samples_path,
            limit=body.limit,
            suite=body.suite,
            offline=body.offline,
            mode_override=body.mode_override,
            ask_caller=ask_caller,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return JSONResponse(
            status_code=500,
            content=build_error_body(
                request_id=request_id,
                code="EVAL_FAILED",
                message=f"eval failed: {type(exc).__name__}: {exc}",
            ),
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    report = pack["report"]
    summary = {
        k: report.get(k)
        for k in (
            "total",
            "by_suite",
            "task_success_rate",
            "clarify_correct_rate",
            "safety_pass_rate",
            "ok_rate",
            "p50_latency_ms",
            "p95_latency_ms",
            "label",
            "generated_at",
        )
        if k in report
    }
    return EvalRunResponse(
        request_id=request_id,
        ok=True,
        latency_ms=latency_ms,
        report=summary,
        meta={
            "api_version": "v1",
            "offline": body.offline,
            "limit": body.limit,
            "suite": body.suite,
            "mode_override": body.mode_override,
            "count": pack.get("count"),
            **active_dataset_info(),
        },
    )
