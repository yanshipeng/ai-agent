"""知识库 JSONL 读写与通用小工具。

==========================================================================
为什么用 JSONL 而不是单个大 JSON 数组
==========================================================================
1) 可追加、可流式读；崩溃时前面的行仍在。
2) wc -l 就能数条数，和报告对照方便。
3) 一行坏了最多丢一条，不必整个文件 parse 失败。

sha256_8：去重指纹够用且短，日志/字段里好展示（完整 sha 太长）。
==========================================================================
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """当前 UTC 时间的 ISO 字符串，用于报告/时间戳字段。"""
    return datetime.now(timezone.utc).isoformat()


def sha256_8(text: str) -> str:
    """正文指纹：取 SHA256 前 8 位，用于去重（content_sha256_8）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """读取 JSONL：跳过空行，返回 dict 列表。"""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> None:
    """整文件覆盖写入 JSONL（先确保父目录存在）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
