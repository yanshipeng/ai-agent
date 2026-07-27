#!/usr/bin/env python3
"""核对 /ask?mode=rag 响应：citations 是否全部来自本地 index。

【用途】
  证明「这次回答走了本地 RAG」，并把 citations 与 data/stability_kb/index 对账。

【常用命令】
  # 先保存响应，再核对
  curl -s 'http://127.0.0.1:8000/ask?mode=rag' \\
    -H 'Content-Type: application/json' \\
    -d '{"query":"Android ANR 怎么排查","top_k":5}' \\
    | tee /tmp/ask_rag.json >/dev/null
  python scripts/verify_ask_rag.py --response /tmp/ask_rag.json

  # 管道直接验
  curl -s '...' -d '...' | python scripts/verify_ask_rag.py --response -

  # 只查 requests.jsonl 指标（无 chunk 对账）
  python scripts/verify_ask_rag.py 8ef4bbc6-38b6-4306-80aa-e22183e1057a

【判定】
  - mode 必须是 rag
  - citations 非空，且每条含 ref_id/chunk_id/url/title/section_path/is_code
  - 每个 chunk_id 都能在本地 index/meta.jsonl 找到
  - url/title（若本地有）与 index 一致（允许截断标题）
  - 可选：回答中出现的 [n] 引用号 ⊆ citations.ref_id
  - 与 requests.jsonl 同 request_id 时交叉核对 mode/top_k/citations_count
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INDEX_DIR = ROOT / "data" / "stability_kb" / "index"
DEFAULT_METRICS = ROOT / "requests.jsonl"
CITATION_KEYS = ("ref_id", "chunk_id", "url", "title", "section_path", "is_code")
REF_IN_ANSWER_RE = re.compile(r"\[(\d+)\]")


def _ok(name: str, detail: str = "") -> dict[str, Any]:
    return {"name": name, "status": "pass", "detail": detail}


def _fail(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "fail", "detail": detail}


def _warn(name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "warn", "detail": detail}


def load_json_source(path: str) -> dict[str, Any]:
    """读响应 JSON：文件路径或 '-'（stdin）。"""
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    raw = raw.strip()
    if not raw:
        raise ValueError("empty response JSON")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("response JSON must be an object")
    return data


def find_metric_row(metrics_path: Path, request_id: str) -> dict[str, Any] | None:
    """在 requests.jsonl 里按 request_id 找最后一条。"""
    if not metrics_path.exists():
        return None
    found: dict[str, Any] | None = None
    with metrics_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or request_id not in line:
                continue
            row = json.loads(line)
            if row.get("request_id") == request_id:
                found = row
    return found


def load_index_meta(index_dir: Path) -> dict[str, dict[str, Any]]:
    """chunk_id → meta 行。"""
    meta_path = index_dir / "meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"index meta missing: {meta_path}")
    by_id: dict[str, dict[str, Any]] = {}
    with meta_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row.get("chunk_id")
            if cid:
                by_id[str(cid)] = row
    return by_id


def _normalize_title(value: str | None) -> str:
    text = (value or "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _title_compatible(cite_title: str | None, local_title: str | None) -> bool:
    """响应 title 可能截断/空格差异；做归一化前缀兼容。"""
    a = (cite_title or "").strip()
    b = (local_title or "").strip()
    if not a or not b:
        return True
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if longer.startswith(shorter) or shorter in longer:
        return True
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return True
    shorter_n, longer_n = (na, nb) if len(na) <= len(nb) else (nb, na)
    return longer_n.startswith(shorter_n) or shorter_n in longer_n


def check_metrics(row: dict[str, Any] | None, request_id: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if row is None:
        checks.append(_warn("metrics.row", f"requests.jsonl 未找到 request_id={request_id}"))
        return checks
    checks.append(_ok("metrics.row", f"found request_id={request_id}"))
    mode = row.get("mode")
    if mode != "rag":
        checks.append(_fail("metrics.mode", f"mode={mode!r}（期望 rag）"))
    else:
        checks.append(_ok("metrics.mode", "rag"))
    for key in ("retrieve_ms", "context_chunks", "citations_count", "top_k"):
        if key not in row:
            checks.append(_warn(f"metrics.{key}", "字段缺失"))
        else:
            checks.append(_ok(f"metrics.{key}", str(row[key])))
    return checks


def check_response_shape(resp: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    meta = resp.get("meta") or {}
    mode = meta.get("mode")
    if mode != "rag":
        checks.append(_fail("response.mode", f"meta.mode={mode!r}（期望 rag）"))
    else:
        checks.append(_ok("response.mode", "rag"))

    for key in ("retrieve_ms", "context_chunks", "citations_count", "top_k"):
        if key not in meta:
            checks.append(_warn(f"response.meta.{key}", "缺失"))
        else:
            checks.append(_ok(f"response.meta.{key}", str(meta[key])))

    citations = resp.get("citations")
    if not isinstance(citations, list) or not citations:
        checks.append(_fail("response.citations", f"citations 必须非空 list，实际={citations!r}"))
        return checks
    checks.append(_ok("response.citations_nonempty", f"n={len(citations)}"))

    bad: list[str] = []
    for i, c in enumerate(citations):
        if not isinstance(c, dict):
            bad.append(f"[{i}] not object")
            continue
        miss = [k for k in CITATION_KEYS if k not in c]
        if miss:
            bad.append(f"[{i}]缺{miss}")
    if bad:
        checks.append(_fail("response.citation_fields", "; ".join(bad[:5])))
    else:
        checks.append(_ok("response.citation_fields", ",".join(CITATION_KEYS)))
    return checks


def check_citations_against_index(
    citations: list[dict[str, Any]],
    index_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """返回 (checks, matched, total)。"""
    checks: list[dict[str, Any]] = []
    matched = 0
    total = len(citations)
    for c in citations:
        ref_id = c.get("ref_id")
        cid = str(c.get("chunk_id") or "")
        name = f"cite.[{ref_id}]"
        local = index_by_id.get(cid)
        if local is None:
            checks.append(_fail(name, f"chunk_id={cid!r} 不在本地 index"))
            continue
        matched += 1
        problems: list[str] = []
        warnings: list[str] = []
        local_url = local.get("url")
        if c.get("url") and local_url and c.get("url") != local_url:
            problems.append(f"url不一致 cite={c.get('url')} local={local_url}")
        if not _title_compatible(c.get("title"), local.get("title")):
            # chunk_id 已命中本地；title 差异多为截断/空格，降级为警告
            warnings.append("title 与本地略有差异（chunk_id 已命中）")
        cite_section = c.get("section_path") or ""
        local_section = local.get("section_path") or ""
        if cite_section != local_section:
            warnings.append(
                f"section_path cite={cite_section!r} local={local_section!r}"
            )
        if bool(c.get("is_code")) != bool(local.get("is_code")):
            problems.append("is_code 不一致")
        preview = (local.get("text") or "").replace("\n", " ").strip()[:100]
        if problems:
            checks.append(_fail(name, f"chunk_id={cid}; " + "; ".join(problems)))
        elif warnings:
            checks.append(
                _warn(name, f"chunk_id={cid}; " + "; ".join(warnings) + f" | {preview}…")
            )
        else:
            checks.append(
                _ok(
                    name,
                    f"chunk_id={cid} url_ok title_ok | {preview}…",
                )
            )

    rate = f"{matched}/{total}"
    if matched == total and total > 0:
        checks.append(_ok("cite.match_rate", f"{rate} 全部来自本地 index"))
    else:
        checks.append(_fail("cite.match_rate", f"{rate} 未全部命中本地 index"))
    return checks, matched, total


def check_answer_ref_ids(answer: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """回答里的 [n] 应落在 citations.ref_id 集合内。"""
    checks: list[dict[str, Any]] = []
    used = {int(x) for x in REF_IN_ANSWER_RE.findall(answer or "")}
    valid = {int(c["ref_id"]) for c in citations if isinstance(c.get("ref_id"), int)}
    if not used:
        checks.append(_warn("answer.refs", "回答中未出现 [n] 引用号（模型可能未遵守）"))
        return checks
    unknown = sorted(used - valid)
    if unknown:
        checks.append(_fail("answer.refs", f"出现未下发的引用号 {unknown}；合法={sorted(valid)}"))
    else:
        checks.append(_ok("answer.refs", f"使用引用 {sorted(used)} ⊆ citations"))
    return checks


def check_answer_overlap_with_cited_chunks(
    answer: str,
    citations: list[dict[str, Any]],
    index_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """弱校验：回答里用到的 [n] 对应 chunk，与回答是否有字面重叠。"""
    checks: list[dict[str, Any]] = []
    used = sorted({int(x) for x in REF_IN_ANSWER_RE.findall(answer or "")})
    if not used:
        return checks
    by_ref = {int(c["ref_id"]): c for c in citations if isinstance(c.get("ref_id"), int)}
    # 从回答抽若干较长 token（中文连续字 / 英文词）
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_./-]{3,}|[\u4e00-\u9fff]{3,}", answer)
    # 去重保序
    seen: set[str] = set()
    uniq_tokens: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        uniq_tokens.append(t)
        if len(uniq_tokens) >= 40:
            break

    hit_refs = 0
    details: list[str] = []
    for ref in used:
        cite = by_ref.get(ref)
        if not cite:
            continue
        local = index_by_id.get(str(cite.get("chunk_id") or ""))
        if not local:
            continue
        blob = str(local.get("text") or "")
        overlap = [t for t in uniq_tokens if t in blob][:5]
        if overlap:
            hit_refs += 1
            details.append(f"[{ref}] overlap={overlap}")
        else:
            details.append(f"[{ref}] 无明显字面重叠（仍可能语义改写）")

    name = "answer.chunk_overlap"
    detail = f"{hit_refs}/{len(used)} 个引用块有字面重叠; " + "; ".join(details[:6])
    if hit_refs > 0:
        checks.append(_ok(name, detail))
    else:
        checks.append(_warn(name, detail))
    return checks


def cross_check_meta_vs_metrics(
    resp: dict[str, Any],
    metric: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not metric:
        return []
    checks: list[dict[str, Any]] = []
    meta = resp.get("meta") or {}
    citations = resp.get("citations") or []
    pairs = [
        ("mode", meta.get("mode"), metric.get("mode")),
        ("top_k", meta.get("top_k"), metric.get("top_k")),
        ("context_chunks", meta.get("context_chunks"), metric.get("context_chunks")),
        ("citations_count", meta.get("citations_count"), metric.get("citations_count")),
    ]
    for key, a, b in pairs:
        if a is None or b is None:
            continue
        if a != b:
            checks.append(_fail(f"cross.{key}", f"response={a} metrics={b}"))
        else:
            checks.append(_ok(f"cross.{key}", str(a)))
    if metric.get("citations_count") is not None and len(citations) != metric["citations_count"]:
        checks.append(
            _fail(
                "cross.citations_len",
                f"len(citations)={len(citations)} metrics.citations_count={metric['citations_count']}",
            )
        )
    elif citations:
        checks.append(_ok("cross.citations_len", str(len(citations))))
    return checks


def run_verify(
    *,
    resp: dict[str, Any] | None,
    request_id: str | None,
    index_dir: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    rid = request_id or (resp or {}).get("request_id")
    metric = find_metric_row(metrics_path, rid) if rid else None
    if rid:
        checks.extend(check_metrics(metric, rid))

    matched = 0
    total = 0
    if resp is None:
        checks.append(
            _warn(
                "response.missing",
                "未提供 --response，仅完成 metrics 检查；"
                "请用 --response /tmp/ask_rag.json 做 chunk 对账",
            )
        )
    else:
        checks.extend(check_response_shape(resp))
        citations = resp.get("citations") if isinstance(resp.get("citations"), list) else []
        hard_fail_shape = any(
            c["status"] == "fail" and c["name"].startswith("response.") for c in checks
        )
        if citations and not hard_fail_shape:
            index_by_id = load_index_meta(index_dir)
            cite_checks, matched, total = check_citations_against_index(citations, index_by_id)
            checks.extend(cite_checks)
            checks.extend(check_answer_ref_ids(str(resp.get("answer") or ""), citations))
            checks.extend(
                check_answer_overlap_with_cited_chunks(
                    str(resp.get("answer") or ""),
                    citations,
                    index_by_id,
                )
            )
            checks.extend(cross_check_meta_vs_metrics(resp, metric))

    failed = sum(1 for c in checks if c["status"] == "fail")
    warned = sum(1 for c in checks if c["status"] == "warn")
    passed = sum(1 for c in checks if c["status"] == "pass")
    return {
        "ok": failed == 0,
        "request_id": rid,
        "matched_citations": matched,
        "total_citations": total,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="核对 /ask?mode=rag 的 citations 是否来自本地 index",
    )
    p.add_argument(
        "request_id",
        nargs="?",
        default=None,
        help="可选；不传则从 --response 的 request_id 读取",
    )
    p.add_argument(
        "--response",
        default=None,
        help="Ask 成功响应 JSON 路径，或 '-' 表示 stdin",
    )
    p.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p.add_argument("--metrics", default=str(DEFAULT_METRICS))
    p.add_argument("--json", action="store_true", help="输出 JSON 报告")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.request_id and not args.response:
        print(
            "ERROR: 请提供 request_id 和/或 --response（curl 响应 JSON）",
            file=sys.stderr,
        )
        return 2

    resp: dict[str, Any] | None = None
    if args.response:
        try:
            resp = load_json_source(args.response)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: 无法读取 response: {exc}", file=sys.stderr)
            return 2

    try:
        report = run_verify(
            resp=resp,
            request_id=args.request_id,
            index_dir=Path(args.index_dir),
            metrics_path=Path(args.metrics),
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for c in report["checks"]:
            mark = {"pass": "PASS", "fail": "FAIL", "warn": "WARN"}[c["status"]]
            print(f"[{mark}] {c['name']}: {c['detail']}")
        print(
            f"\nsummary: ok={report['ok']} "
            f"citations={report['matched_citations']}/{report['total_citations']} "
            f"pass={report['passed']} fail={report['failed']} warn={report['warned']} "
            f"request_id={report['request_id']}"
        )
        if report["ok"] and report["total_citations"] > 0:
            print("结论：citations 全部能在本地 index 对上，本次为本地 RAG 检索增强回答。")
        elif report["ok"]:
            print("结论：metrics 侧为 rag；请补 --response 做 chunk 对账。")
        else:
            print("结论：存在 FAIL，请根据上面条目排查。")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
