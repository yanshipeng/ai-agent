"""Day21：/v1/eval/run 可调用的评测批跑（复用 Day19 判定逻辑）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from scripts.run_eval_v2 import (
    build_report,
    eval_offline_one,
    evaluate_sample,
    load_samples,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = ROOT / "eval" / "eval_samples_v2.jsonl"

AskCaller = Callable[[str, str], dict[str, Any]]


def run_eval_batch(
    *,
    samples_path: Path | str | None = None,
    limit: int = 5,
    suite: str | None = None,
    offline: bool = True,
    mode_override: str | None = None,
    ask_caller: AskCaller | None = None,
) -> dict[str, Any]:
    """跑一批评测并返回 {report, details, label}。

    offline=True：不调 LLM（CI / 默认 HTTP 安全）。
    offline=False：必须提供 ask_caller(query, mode) -> {ok, answer, meta, ...}。
    """
    path = Path(samples_path) if samples_path else DEFAULT_SAMPLES
    samples = load_samples(path)
    if suite:
        samples = [s for s in samples if s.get("suite") == suite]
    if limit and limit > 0:
        samples = samples[: int(limit)]

    results: list[dict[str, Any]] = []
    if offline:
        for sample in samples:
            results.append(eval_offline_one(sample))
        label = "offline"
    else:
        if ask_caller is None:
            raise ValueError("online eval requires ask_caller")
        for sample in samples:
            mode = mode_override or str(sample.get("mode") or "rag")
            query = str(sample.get("query") or "")
            pack = ask_caller(query, mode)
            ok = bool(pack.get("ok", True))
            answer = str(pack.get("answer") or "")
            meta = pack.get("meta") if isinstance(pack.get("meta"), dict) else {}
            judged = evaluate_sample(sample, answer=answer, meta=meta, ok=ok)
            judged.update(
                {
                    "query": query,
                    "mode": mode,
                    "answer": answer[:2000],
                    "meta": {
                        k: meta.get(k)
                        for k in (
                            "stop_reason",
                            "finish_reason",
                            "finish_type",
                            "injection_blocked",
                            "leakage_blocked",
                            "mode",
                            "tool_calls_count",
                        )
                        if k in meta
                    },
                    "latency_ms": pack.get("latency_ms"),
                    "status_code": pack.get("status_code"),
                    "error_code": pack.get("error_code"),
                }
            )
            results.append(judged)
        label = f"online:{mode_override or 'per-sample'}"

    report = build_report(results, label=label)
    return {"report": report, "details": results, "label": label, "count": len(results)}
