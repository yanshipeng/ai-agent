"""eval_rag_ask 判定逻辑单测（不打真实 LLM）。"""

from __future__ import annotations

from scripts.eval_rag_ask import (
    answer_locates_code_fragment,
    is_insufficient_answer,
    pick_samples,
)


def test_is_insufficient_answer_phrases():
    assert is_insufficient_answer("根据已有资料无法确定，请补充现象。")
    assert is_insufficient_answer("信息不足，需要澄清具体机型。")
    assert not is_insufficient_answer("ANR 排查先看 traces.txt。[1]")


def test_pick_samples_includes_three_insufficient():
    pool = [
        *[{"id": f"n{i}", "query": f"q{i}", "tag": "normal"} for i in range(20)],
        *[{"id": f"c{i}", "query": f"code{i}", "tag": "code_seeking"} for i in range(6)],
        {"id": "i1", "query": "ins1", "tag": "insufficient"},
        {"id": "i2", "query": "ins2", "tag": "insufficient"},
        {"id": "i3", "query": "ins3", "tag": "insufficient"},
    ]
    picked = pick_samples(pool, seed=42, total=20)
    assert len(picked) == 20
    assert sum(1 for s in picked if s["tag"] == "insufficient") == 3


def test_answer_locates_code_fragment_with_heuristic():
    citations = [
        {
            "ref_id": 1,
            "chunk_id": "x:0:0",
            "is_code": False,
            "_code_like": True,
            "title": "adb",
        }
    ]
    index = {
        "x:0:0": {
            "text": "使用 adb pull /data/anr/traces.txt 导出日志",
            "is_code": False,
        }
    }
    out = answer_locates_code_fragment(
        "先执行 adb pull 导出 traces。[1]",
        citations,
        code_mode="either",
        index_by_id=index,
    )
    assert out["ok"] is True
    assert 1 in out["located_refs"]
