#!/usr/bin/env python3
"""Day25：一键回归 + 趋势（至少可连续跑 3 次看曲线）。

默认离线跑：
  - eval/eval_samples_v2.jsonl（基线）
  - eval/eval_samples_feedback.jsonl（反馈 badcase，若存在）

结果追加到 reports/regression_trend.jsonl，并打印最近 N 次趋势。

【常用】
  python scripts/run_regression_daily.py
  python scripts/run_regression_daily.py --repeat 3
  python scripts/run_regression_daily.py --promote-pending
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.eval_v2_service import run_eval_batch  # noqa: E402
from app.services.feedback_store import promote_pending_to_eval  # noqa: E402
from scripts.run_eval_v2 import load_samples  # noqa: E402

DEFAULT_V2 = ROOT / "eval" / "eval_samples_v2.jsonl"
DEFAULT_FB = ROOT / "eval" / "eval_samples_feedback.jsonl"
DEFAULT_TREND = ROOT / "reports" / "regression_trend.jsonl"
DEFAULT_MERGED = ROOT / "reports" / "eval_regression_samples.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def merge_samples(paths: list[Path], out: Path) -> int:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for sample in load_samples(path):
            sid = str(sample.get("id") or "")
            if sid and sid in seen:
                continue
            if sid:
                seen.add(sid)
            rows.append(sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def append_trend(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_trend(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def print_trend(rows: list[dict[str, Any]], *, last_n: int = 10) -> None:
    tail = rows[-last_n:]
    print("--- trend (recent) ---")
    if not tail:
        print("(empty)")
        return
    for row in tail:
        print(
            f"{row.get('ts')} total={row.get('total')} "
            f"task={row.get('task_success_rate')} "
            f"clarify={row.get('clarify_correct_rate')} "
            f"safety={row.get('safety_pass_rate')} "
            f"label={row.get('label')}"
        )


def run_once(*, samples: Path, label: str) -> dict[str, Any]:
    pack = run_eval_batch(samples_path=samples, limit=0, offline=True)
    # limit=0 means all? Check eval_v2_service - `if limit and limit > 0` so 0 = all. Good.
    report = pack["report"]
    return {
        "ts": utc_now(),
        "label": label,
        "total": report.get("total"),
        "task_success_rate": report.get("task_success_rate"),
        "clarify_correct_rate": report.get("clarify_correct_rate"),
        "safety_pass_rate": report.get("safety_pass_rate"),
        "ok_rate": report.get("ok_rate"),
        "by_suite": report.get("by_suite"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Day25 daily regression + trend")
    parser.add_argument("--v2", default=str(DEFAULT_V2))
    parser.add_argument("--feedback", default=str(DEFAULT_FB))
    parser.add_argument("--merged", default=str(DEFAULT_MERGED))
    parser.add_argument("--trend", default=str(DEFAULT_TREND))
    parser.add_argument("--repeat", type=int, default=1, help="连续跑几次（演示趋势）")
    parser.add_argument(
        "--promote-pending",
        action="store_true",
        help="先把 pending badcase 写入 eval_samples_feedback.jsonl",
    )
    parser.add_argument("--last-n", type=int, default=10)
    args = parser.parse_args()

    fb_path = Path(args.feedback)
    if args.promote_pending:
        promo = promote_pending_to_eval(out_path=fb_path, limit=100)
        print("promote_pending:", promo)

    paths = [Path(args.v2), fb_path]
    n = merge_samples(paths, Path(args.merged))
    print(f"merged_samples={n} -> {args.merged}")
    if n == 0:
        print("no samples", file=sys.stderr)
        return 2

    for i in range(max(1, args.repeat)):
        label = f"offline:merged#{i+1}"
        row = run_once(samples=Path(args.merged), label=label)
        append_trend(Path(args.trend), row)
        print(
            f"[{i+1}/{args.repeat}] task={row['task_success_rate']} "
            f"clarify={row['clarify_correct_rate']} safety={row['safety_pass_rate']}"
        )

    print_trend(load_trend(Path(args.trend)), last_n=args.last_n)
    print(f"trend -> {args.trend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
