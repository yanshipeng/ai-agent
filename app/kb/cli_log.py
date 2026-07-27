"""CLI 人话进度输出（给新手看懂脚本在干什么）。"""

from __future__ import annotations

import sys


def step(title: str, detail: str = "") -> None:
    """打印一步标题。"""
    line = f"==> {title}"
    if detail:
        line = f"{line}: {detail}"
    print(line, flush=True)


def ok(msg: str) -> None:
    print(f"[ok] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr, flush=True)


def fail(msg: str) -> None:
    print(f"[fail] {msg}", file=sys.stderr, flush=True)
