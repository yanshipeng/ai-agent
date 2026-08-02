"""Day23：增量入库 + dataset_version 回滚。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.kb.ingest_pipeline import rollback_dataset, run_ingest_from_docs
from app.kb.jsonl_io import write_jsonl


def _doc(doc_id: str, text: str, sha: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": doc_id,
        "url": f"https://example.com/{doc_id}",
        "category": "test",
        "category_name": "test",
        "tags": [],
        "content": text,
        "content_sha256_8": sha,
        "source": "unit",
    }


@pytest.fixture
def versioned_kb(tmp_path, monkeypatch):
    docs_path = tmp_path / "docs.jsonl"
    versions = tmp_path / "versions"
    monkeypatch.setenv("KB_VERSIONS_DIR", str(versions))
    monkeypatch.setenv("KB_DOCS_PATH", str(docs_path))
    monkeypatch.chdir(tmp_path)
    from app.core.config import get_settings

    get_settings.cache_clear()
    write_jsonl(
        docs_path,
        [
            _doc("d1", "# A\n\nANR 主线程卡死排查步骤一二三四五六七八九十。" * 3, "sha11111"),
            _doc("d2", "# B\n\nOOM 内存泄漏分析要点一二三四五六七八九十。" * 3, "sha22222"),
        ],
    )
    yield docs_path, versions
    get_settings.cache_clear()


def test_incremental_second_ingest_rebuilds_less(versioned_kb) -> None:
    docs_path, versions = versioned_kb
    r1 = run_ingest_from_docs(docs_path, incremental=True, dim=128)
    assert r1["docs_total"] == 2
    assert r1["vectors_embedded"] >= 1
    assert r1["docs_unchanged"] == 0
    v1 = r1["dataset_version"]
    embedded_first = r1["vectors_embedded"]

    # 第二次：相同 docs → 应全部 unchanged，几乎不再 embed
    r2 = run_ingest_from_docs(docs_path, incremental=True, dim=128)
    assert r2["incremental"] is True
    assert r2["docs_unchanged"] == 2
    assert r2["docs_added"] == 0
    assert r2["docs_changed"] == 0
    assert r2["vectors_embedded"] == 0
    assert r2["chunks_reused"] == r1["chunks_total"]
    assert r2["vectors_embedded"] < embedded_first
    # 版本号含毫秒；若极端撞名，至少 parent 指向上一版
    assert r2["dataset_version"] != v1 or r2.get("parent_version") == v1

    # 改一条 → 只重建 changed
    rows = [
        json.loads(line)
        for line in docs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["content"] = rows[0]["content"] + "\n额外变更段落。"
    rows[0]["content_sha256_8"] = "sha1changed"
    write_jsonl(docs_path, rows)
    r3 = run_ingest_from_docs(docs_path, incremental=True, dim=128)
    assert r3["docs_changed"] == 1
    assert r3["docs_unchanged"] == 1
    assert r3["vectors_embedded"] >= 1
    assert r3["vectors_embedded"] < embedded_first


def test_rollback_switches_current(versioned_kb) -> None:
    docs_path, _versions = versioned_kb
    r1 = run_ingest_from_docs(docs_path, incremental=False, dim=128)
    r2 = run_ingest_from_docs(docs_path, incremental=True, dim=128)
    assert r1["dataset_version"] != r2["dataset_version"]

    rolled = rollback_dataset(r1["dataset_version"])
    assert rolled["ok"] is True
    assert rolled["dataset_version"] == r1["dataset_version"]
    from app.kb.dataset_registry import read_current

    cur = read_current()
    assert cur is not None
    assert cur["dataset_version"] == r1["dataset_version"]
