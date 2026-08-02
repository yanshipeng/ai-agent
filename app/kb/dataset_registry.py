"""Day23：知识库 dataset_version 注册表（active 指针 + 回滚）。

目录结构：
  data/stability_kb/versions/
    current.json          {"dataset_version": "v...", "index_dir": "..."}
    vYYYYMMDD_HHMMSS/
      docs_fingerprints.json
      chunks.jsonl
      index/  (meta.jsonl + vectors.jsonl + manifest.json)
      ingest_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.kb.jsonl_io import utc_now_iso

CURRENT_FILENAME = "current.json"


def versions_root() -> Path:
    settings = get_settings()
    root = Path(settings.kb_versions_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def current_path() -> Path:
    return versions_root() / CURRENT_FILENAME


def version_dir(dataset_version: str) -> Path:
    return versions_root() / dataset_version


def read_current() -> dict[str, Any] | None:
    path = current_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_current(
    *,
    dataset_version: str,
    index_dir: Path | str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    versions_root().mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_version": dataset_version,
        "index_dir": str(Path(index_dir).resolve()),
        "updated_at": utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    current_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def list_versions() -> list[dict[str, Any]]:
    root = versions_root()
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("v"):
            continue
        report = child / "ingest_report.json"
        item: dict[str, Any] = {"dataset_version": child.name, "path": str(child)}
        if report.exists():
            try:
                item["report"] = json.loads(report.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        rows.append(item)
    return rows


def resolve_active_index_dir() -> Path:
    """优先 current.json；否则回落 Settings.kb_index_dir。"""
    cur = read_current()
    if cur and cur.get("index_dir"):
        return Path(str(cur["index_dir"])).resolve()
    settings = get_settings()
    path = Path(settings.kb_index_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def new_dataset_version(*, prefix: str = "v") -> str:
    """版本号含毫秒，避免同一秒内两次入库撞名。"""
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}{stamp}"


def active_dataset_info() -> dict[str, Any]:
    """当前 active 版本摘要（供 /v1/dataset 与 eval meta）。"""
    cur = read_current()
    return {
        "current": cur,
        "active_index_dir": str(resolve_active_index_dir()),
        "versions": [v["dataset_version"] for v in list_versions()],
    }
