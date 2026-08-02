"""Day25：反馈接口 + badcase 沉淀 + 回归趋势脚本形状。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.feedback_store import promote_pending_to_eval, record_feedback
from scripts.run_regression_daily import load_trend, merge_samples, run_once


@pytest.fixture
def fb_client(tmp_path, monkeypatch):
    monkeypatch.setenv("REQUESTS_JSONL_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("API_AUTH_ENABLED", "false")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("FEEDBACK_JSONL_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setenv("BADCASES_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        yield client, tmp_path
    get_settings.cache_clear()


def test_record_feedback_promotes_badcase(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FEEDBACK_JSONL_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setenv("BADCASES_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    from app.core.config import get_settings

    get_settings.cache_clear()
    pack = record_feedback(
        label="wrong_citation",
        query="ANR traces 在哪？",
        mode="rag",
        note="路径错",
    )
    assert pack["badcase_promoted"] is True
    pending = (tmp_path / "pending.jsonl").read_text(encoding="utf-8")
    assert "ANR traces" in pending
    get_settings.cache_clear()


def test_feedback_http(fb_client) -> None:
    client, tmp_path = fb_client
    resp = client.post(
        "/v1/feedback",
        json={
            "label": "hallucination",
            "query": "OOM 和 ANR 一样吗？",
            "mode": "rag",
            "note": "概念混淆",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["badcase_promoted"] is True
    assert (tmp_path / "feedback.jsonl").exists()


def test_promote_and_regression_offline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FEEDBACK_JSONL_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setenv("BADCASES_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    from app.core.config import get_settings

    get_settings.cache_clear()
    for i in range(3):
        record_feedback(
            label="useless",
            query=f"又卡了{i}",
            mode="rag",
        )
    out = tmp_path / "fb_eval.jsonl"
    promo = promote_pending_to_eval(out_path=out, limit=10)
    assert promo["promoted"] == 3

    v2 = Path("eval_samples_v2.jsonl")
    merged = tmp_path / "merged.jsonl"
    n = merge_samples([v2, out], merged)
    assert n >= 80
    row = run_once(samples=merged, label="test")
    assert "task_success_rate" in row
    assert "clarify_correct_rate" in row

    trend = tmp_path / "trend.jsonl"
    for _ in range(3):
        r = run_once(samples=merged, label="t")
        with trend.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = load_trend(trend)
    assert len(rows) == 3
    get_settings.cache_clear()


def test_feedback_seed_file_exists() -> None:
    path = Path("eval_samples_feedback.jsonl")
    assert path.exists()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 5
