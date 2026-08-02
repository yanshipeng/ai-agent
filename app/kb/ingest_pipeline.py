"""Day23：入库流水线 v2 — 增量 chunk/embed + dataset_version + 回滚。

输入：docs.jsonl（或已有 docs；URL 列表可先走 cleaner 生成 docs）
流程：
  1) 读 docs，与上一版本 fingerprints 对比 → added/changed/unchanged/removed
  2) unchanged：复用上一版本 chunks + vectors
  3) added/changed：重新 chunk + embed
  4) 写出新 version 目录并切换 current.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.kb.chunker import chunk_docs
from app.kb.dataset_registry import (
    active_dataset_info,
    list_versions,
    new_dataset_version,
    read_current,
    resolve_active_index_dir,
    version_dir,
    write_current,
)
from app.kb.embedder import DEFAULT_DIM
from app.kb.index_store import build_index, load_index, save_index
from app.kb.jsonl_io import load_jsonl, utc_now_iso, write_jsonl
from app.kb.retriever import clear_index_cache


def _doc_fingerprint(doc: dict[str, Any]) -> str:
    return str(
        doc.get("content_sha256_8")
        or doc.get("dedupe_key")
        or doc.get("doc_id")
        or ""
    )


def _load_fingerprints(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def _save_fingerprints(path: Path, fps: dict[str, str]) -> None:
    path.write_text(json.dumps(fps, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prev_version_dir() -> Path | None:
    cur = read_current()
    if not cur:
        return None
    ver = cur.get("dataset_version")
    if not ver:
        return None
    path = version_dir(str(ver))
    return path if path.exists() else None


def run_ingest_from_docs(
    docs_path: Path | str,
    *,
    incremental: bool = True,
    dim: int = DEFAULT_DIM,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    """从 docs.jsonl 构建新 dataset_version。"""
    docs_path = Path(docs_path)
    docs = load_jsonl(docs_path)
    version = dataset_version or new_dataset_version()
    out_dir = version_dir(version)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_out = out_dir / "index"

    prev_dir = _prev_version_dir() if incremental else None
    prev_fps = _load_fingerprints(prev_dir / "docs_fingerprints.json") if prev_dir else {}
    prev_index = None
    if prev_dir and (prev_dir / "index" / "manifest.json").exists():
        try:
            prev_index = load_index(prev_dir / "index")
        except Exception:  # noqa: BLE001
            prev_index = None

    new_fps = {str(d.get("doc_id")): _doc_fingerprint(d) for d in docs if d.get("doc_id")}
    added: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    for doc_id, fp in new_fps.items():
        old = prev_fps.get(doc_id)
        if old is None:
            added.append(doc_id)
        elif old != fp:
            changed.append(doc_id)
        else:
            unchanged.append(doc_id)
    removed = [doc_id for doc_id in prev_fps if doc_id not in new_fps]

    # 需要重算的 docs
    rebuild_ids = set(added) | set(changed)
    if not incremental or prev_index is None:
        rebuild_ids = set(new_fps.keys())
        unchanged = []
        added = list(new_fps.keys())
        changed = []

    docs_by_id = {str(d.get("doc_id")): d for d in docs if d.get("doc_id")}
    rebuild_docs = [docs_by_id[i] for i in sorted(rebuild_ids) if i in docs_by_id]

    new_chunks: list[dict[str, Any]] = []
    if rebuild_docs:
        new_chunks, _chunk_report = chunk_docs(rebuild_docs)

    # 复用 unchanged：从旧 index 按 doc_id 抽 meta+vector
    reused_meta: list[dict[str, Any]] = []
    reused_vectors: list[list[float]] = []
    if prev_index is not None and unchanged:
        keep = set(unchanged)
        for meta, vec in zip(prev_index["meta"], prev_index["vectors"]):
            if str(meta.get("doc_id")) in keep:
                reused_meta.append(meta)
                reused_vectors.append(vec)

    rebuilt_index = (
        build_index(new_chunks, dim=dim, progress=False) if new_chunks else {
            "dim": dim,
            "size": 0,
            "meta": [],
            "vectors": [],
        }
    )

    merged_meta = list(reused_meta) + list(rebuilt_index["meta"])
    merged_vectors = list(reused_vectors) + list(rebuilt_index["vectors"])
    merged = {
        "dim": int(rebuilt_index.get("dim") or dim),
        "size": len(merged_meta),
        "meta": merged_meta,
        "vectors": merged_vectors,
    }
    save_index(merged, index_out)

    # 落盘 chunks（完整集：复用旧 chunks.jsonl 中 unchanged + 新 chunks）
    all_chunks = list(new_chunks)
    if prev_dir and unchanged:
        old_chunks_path = prev_dir / "chunks.jsonl"
        if old_chunks_path.exists():
            keep = set(unchanged)
            for row in load_jsonl(old_chunks_path):
                if str(row.get("doc_id")) in keep:
                    all_chunks.append(row)
    write_jsonl(out_dir / "chunks.jsonl", all_chunks)
    _save_fingerprints(out_dir / "docs_fingerprints.json", new_fps)

    report = {
        "dataset_version": version,
        "docs_path": str(docs_path),
        "index_dir": str(index_out.resolve()),
        "incremental": bool(incremental and prev_index is not None),
        "docs_total": len(docs),
        "docs_added": len(added),
        "docs_changed": len(changed),
        "docs_unchanged": len(unchanged),
        "docs_removed": len(removed),
        "chunks_rebuilt": len(new_chunks),
        "chunks_reused": len(reused_meta),
        "chunks_total": len(merged_meta),
        "vectors_embedded": len(rebuilt_index["meta"]),
        "dim": merged["dim"],
        "parent_version": prev_dir.name if prev_dir else None,
        "generated_at": utc_now_iso(),
    }
    (out_dir / "ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_current(dataset_version=version, index_dir=index_out, extra={"docs_total": len(docs)})
    clear_index_cache()
    return report


def rollback_dataset(dataset_version: str) -> dict[str, Any]:
    """切换 current 到旧 version（需目录仍存在）。"""
    path = version_dir(dataset_version)
    index_dir = path / "index"
    if not (index_dir / "manifest.json").exists():
        raise FileNotFoundError(f"version index not found: {index_dir}")
    cur = write_current(
        dataset_version=dataset_version,
        index_dir=index_dir,
        extra={"rolled_back_at": utc_now_iso()},
    )
    clear_index_cache()
    report_path = path / "ingest_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return {
        "ok": True,
        "action": "rollback",
        "dataset_version": dataset_version,
        "index_dir": str(index_dir.resolve()),
        "current": cur,
        "ingest_report": report,
        "versions": [v["dataset_version"] for v in list_versions()],
    }


__all__ = [
    "active_dataset_info",
    "rollback_dataset",
    "run_ingest_from_docs",
]
