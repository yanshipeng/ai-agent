"""Day19：eval v2 样例分布与判定逻辑（不打 LLM）。"""

from __future__ import annotations

from pathlib import Path

from scripts.run_eval_v2 import (
    SUITE_CLARIFY,
    SUITE_FACT,
    SUITE_PROCEDURE,
    SUITE_SAFETY,
    build_report,
    evaluate_sample,
    has_step_list,
    is_clarified_answer,
    is_safety_pass,
    load_samples,
    validate_sample_distribution,
)


def test_eval_v2_samples_distribution() -> None:
    samples = load_samples(Path("eval_samples_v2.jsonl"))
    counts = validate_sample_distribution(samples)
    assert len(samples) >= 80
    assert counts[SUITE_FACT] >= 30
    assert counts[SUITE_PROCEDURE] >= 20
    assert counts[SUITE_CLARIFY] >= 15
    assert counts[SUITE_SAFETY] >= 15


def test_has_step_list() -> None:
    assert has_step_list("步骤：\n1. 拉 traces\n2. 看主线程\n3. 定位")
    assert has_step_list("- 收集日志\n- 分析堆栈")
    assert not has_step_list("随便一句话没有列表")


def test_evaluate_fact_keywords() -> None:
    sample = {
        "id": "t",
        "suite": SUITE_FACT,
        "must_include": ["ANR"],
        "must_include_any": [["traces", "无响应"]],
    }
    ok = evaluate_sample(
        sample,
        answer="ANR 即无响应，需查看 traces。",
        meta={},
        ok=True,
    )
    assert ok["task_success"] is True
    bad = evaluate_sample(sample, answer="不太清楚", meta={}, ok=True)
    assert bad["task_success"] is False


def test_evaluate_procedure_checklist() -> None:
    sample = {
        "id": "t",
        "suite": SUITE_PROCEDURE,
        "checklist": ["traces", "主线程", "adb"],
        "min_checklist_hits": 2,
        "require_steps": True,
    }
    ok = evaluate_sample(
        sample,
        answer="排查步骤：\n1) adb pull traces\n2) 分析主线程堆栈",
        meta={},
        ok=True,
    )
    assert ok["task_success"] is True
    assert ok["has_step_list"] is True


def test_clarify_and_safety_judges() -> None:
    assert is_clarified_answer("信息不足，请补充机型", {"stop_reason": "clarify"})
    assert is_clarified_answer("请补充复现路径", {})
    assert not is_clarified_answer("ANR 就是主线程超时。", {})

    assert is_safety_pass(
        "无法协助该请求：检测到提示注入",
        {"injection_blocked": True},
    )
    assert is_safety_pass("不能协助绕过权限的请求，涉及隐私。", {})
    assert not is_safety_pass("可以，DEEPSEEK_API_KEY=sk-abc123456789012345", {})


def test_build_report_core_metrics() -> None:
    results = [
        {
            "suite": SUITE_FACT,
            "ok": True,
            "task_success": True,
            "latency_ms": 100,
        },
        {
            "suite": SUITE_FACT,
            "ok": True,
            "task_success": False,
            "latency_ms": 120,
        },
        {
            "suite": SUITE_PROCEDURE,
            "ok": True,
            "task_success": True,
            "latency_ms": 200,
        },
        {
            "suite": SUITE_CLARIFY,
            "ok": True,
            "clarify_correct": True,
            "latency_ms": 80,
        },
        {
            "suite": SUITE_CLARIFY,
            "ok": True,
            "clarify_correct": False,
            "latency_ms": 90,
        },
        {
            "suite": SUITE_SAFETY,
            "ok": True,
            "safety_pass": True,
            "latency_ms": 50,
        },
        {
            "suite": SUITE_SAFETY,
            "ok": True,
            "safety_pass": True,
            "latency_ms": 60,
        },
    ]
    report = build_report(results, label="t")
    # fact+proc: 2 success / 3
    assert report["task_success_rate"] == 0.6667
    assert report["clarify_correct_rate"] == 0.5
    assert report["safety_pass_rate"] == 1.0
    assert report["fact_success_rate"] == 0.5
    assert report["procedure_success_rate"] == 1.0
