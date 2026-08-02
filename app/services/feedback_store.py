"""Day25：用户反馈 → badcase 沉淀（供评测回归）。"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.core.config import get_settings
from app.core.logging import query_sha256_8

FeedbackLabel = Literal["useful", "useless", "wrong_citation", "hallucination"]

BAD_LABELS = frozenset({"useless", "wrong_citation", "hallucination"})
_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def feedback_path() -> Path:
    return _path(get_settings().feedback_jsonl_path)


def badcases_pending_path() -> Path:
    return _path(get_settings().badcases_pending_path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with _lock:
        with path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


def label_to_suite(label: str) -> str:
    if label == "useless":
        return "clarify"
    if label in {"wrong_citation", "hallucination"}:
        return "fact_qa"
    return "fact_qa"


def build_eval_sample_from_feedback(
    *,
    feedback_id: str,
    query: str,
    label: str,
    mode: str | None,
    note: str | None,
) -> dict[str, Any]:
    """把 badcase 转成 eval v2 样条（可人工再改 must_include）。"""
    suite = label_to_suite(label)
    sample: dict[str, Any] = {
        "id": f"fb_{feedback_id[:8]}",
        "suite": suite,
        "query": query,
        "mode": mode or "rag",
        "source": "feedback",
        "feedback_label": label,
        "note": note or "",
        "expect_clarify": suite == "clarify",
        "expect_refusal": False,
    }
    if suite == "fact_qa":
        # 最小约束：禁止空答；人工审核可补 must_include
        sample["must_include"] = []
        sample["must_include_any"] = [["根据", "资料", "引用", "步骤", "建议"]]
        sample["reject_phrases"] = ["我编造", "不确定但我觉得"]
    return sample


def record_feedback(
    *,
    label: FeedbackLabel,
    request_id: str | None = None,
    query: str | None = None,
    mode: str | None = None,
    note: str | None = None,
    answer_preview: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    promote_badcase: bool = True,
) -> dict[str, Any]:
    """写入反馈；负面标签可沉淀到 pending badcase。"""
    feedback_id = str(uuid.uuid4())
    q = (query or "").strip()
    row: dict[str, Any] = {
        "feedback_id": feedback_id,
        "ts": _utc_now(),
        "label": label,
        "request_id": request_id,
        "query_len": len(q) if q else None,
        "query_sha256_8": query_sha256_8(q) if q else None,
        # badcase 需要原文才能进评测；仅反馈库保留，requests.jsonl 仍不落问句
        "query": q or None,
        "mode": mode,
        "note": (note or "").strip() or None,
        "answer_preview": (answer_preview or "")[:500] or None,
        "tenant_id": tenant_id,
        "user_id": user_id,
    }
    append_jsonl(feedback_path(), row)

    promoted = None
    if promote_badcase and label in BAD_LABELS and q:
        sample = build_eval_sample_from_feedback(
            feedback_id=feedback_id,
            query=q,
            label=label,
            mode=mode,
            note=note,
        )
        pending = {
            "feedback_id": feedback_id,
            "ts": _utc_now(),
            "status": "pending_review",
            "sample": sample,
        }
        append_jsonl(badcases_pending_path(), pending)
        promoted = sample

    return {
        "feedback_id": feedback_id,
        "label": label,
        "badcase_promoted": promoted is not None,
        "sample": promoted,
    }


def promote_pending_to_eval(
    *,
    out_path: Path | str,
    limit: int = 50,
    reviewed_only: bool = False,
) -> dict[str, Any]:
    """把 pending badcase 写入 eval JSONL（默认全部 pending；可只收 reviewed）。"""
    src = badcases_pending_path()
    out = Path(out_path)
    if not src.exists():
        return {"promoted": 0, "out": str(out), "message": "no pending file"}

    rows: list[dict[str, Any]] = []
    with src.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    selected: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "pending_review")
        if reviewed_only and status != "reviewed":
            continue
        sample = row.get("sample")
        if isinstance(sample, dict) and sample.get("query"):
            selected.append(sample)
        if limit and len(selected) >= limit:
            break

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        for sample in selected:
            fp.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return {"promoted": len(selected), "out": str(out.resolve())}
