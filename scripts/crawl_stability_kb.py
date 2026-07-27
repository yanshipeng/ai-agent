#!/usr/bin/env python3
"""独立脚本：采集「线上排障与稳定性」公开资料（国内平台优先）。

采集策略（domestic_first）：
  1) 精选国内种子（美团/阿里云/腾讯云/高德/android.google.cn 等）
  2) 掘金 Search API、CSDN Search API 按关键词发现
  3) 必应中国（cn.bing.com）补链（百度对脚本常出安全验证，故不用作主通道）
  4) 每类数量仍不足时，才使用 seeds_foreign_fallback

主题 A–G：ANR/卡顿、Crash/白屏、内存、网络、地图定位、WebView/JSBridge、推送/IM。

用法：
  python scripts/crawl_stability_kb.py
  python scripts/crawl_stability_kb.py --domestic-only
  python scripts/crawl_stability_kb.py --category A,E --target 20
  python scripts/crawl_stability_kb.py --discover-only --limit 40
  python scripts/crawl_stability_kb.py --resume

输出（默认 data/stability_kb/）：
  - discovered_seeds.jsonl  发现阶段候选
  - articles.jsonl          抓取正文摘要
  - crawl_report.json       汇总

依赖：httpx；建议 beautifulsoup4：
  pip install beautifulsoup4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse, unquote

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = Path(__file__).resolve().parent / "stability_kb_seeds.json"
DEFAULT_OUT_DIR = ROOT / "data" / "stability_kb"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_DELAY_SEC = 0.6
DEFAULT_TIMEOUT_SEC = 25.0
MAX_TEXT_CHARS = 6000
MAX_SUMMARY_CHARS = 500
DEFAULT_TARGET_PER_CATEGORY = 15

# 国内优先域名（含 android/firebase 中国镜像、国内技术社区与开放平台）
DOMESTIC_HOST_SUFFIXES = (
    "juejin.cn",
    "juejin.im",
    "csdn.net",
    "zhihu.com",
    "cnblogs.com",
    "segmentfault.com",
    "aliyun.com",
    "tencent.com",
    "meituan.com",
    "amap.com",
    "android.google.cn",
    "firebase.google.cn",
    "developers.google.cn",
    "chrome.google.cn",
    "source.android.google.cn",
    "jianshu.com",
    "oschina.net",
    "51cto.com",
    "infoq.cn",
    "cn.bing.com",
)

try:
    from bs4 import BeautifulSoup  # type: ignore

    HAS_BS4 = True
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore
    HAS_BS4 = False


@dataclass
class Candidate:
    id: str
    category: str
    title: str
    url: str
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    discover_from: str = "seed"
    region: str = "domestic"  # domestic | foreign


@dataclass
class ArticleRecord:
    id: str
    category: str
    category_name: str
    title: str
    url: str
    source: str
    tags: list[str]
    notes: str
    summary: str
    text_excerpt: str
    status: str
    http_status: int | None
    fetched_at: str
    discover_from: str = "seed"
    region: str = "domestic"
    error: str | None = None
    content_sha256_8: str | None = None


class _TitleMetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self.title = ""
        self.description = ""
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if lower == "title":
            self._in_title = True
            self._buf = []
        elif lower == "meta":
            name = (attr_map.get("name") or attr_map.get("property") or "").lower()
            if name in {"description", "og:description"} and attr_map.get("content"):
                self.description = attr_map["content"].strip()

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title" and self._in_title:
            self._in_title = False
            self.title = "".join(self._buf).strip()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._buf.append(data)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_html_tags(text: str) -> str:
    return normalize_ws(re.sub(r"<[^>]+>", "", text or ""))


def sha256_8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def is_domestic_host(url: str) -> bool:
    host = host_of(url)
    return any(host == s or host.endswith("." + s) for s in DOMESTIC_HOST_SUFFIXES)


def canonicalize_url(url: str) -> str:
    """去追踪参数，便于去重。"""
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip()
    # CSDN / 搜索引擎追踪参数
    drop_keys = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "ops_request_misc",
        "request_id",
        "biz_id",
        "spm",
        "from",
    }
    query = parse_qs(parsed.query, keep_blank_values=False)
    kept = {k: v for k, v in query.items() if k not in drop_keys}
    # 重建 query（仅保留必要）
    flat = []
    for k, values in kept.items():
        for v in values:
            flat.append(f"{k}={v}")
    new_query = "&".join(flat)
    path = parsed.path or "/"
    return urlunparse(
        (parsed.scheme, parsed.netloc.lower(), path, "", new_query, "")
    )


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "categories" not in data:
        raise ValueError(f"invalid seeds config: {path}")
    return data


def parse_categories_arg(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    cats = {c.strip().upper() for c in raw.split(",") if c.strip()}
    return cats or None


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
        follow_redirects=True,
        timeout=DEFAULT_TIMEOUT_SEC,
        http2=False,
    )


def discover_juejin(
    client: httpx.Client,
    keyword: str,
    *,
    category: str,
    limit: int,
) -> list[Candidate]:
    url = "https://api.juejin.cn/search_api/v1/search/"
    payload = {
        "key_word": keyword,
        "cursor": "0",
        "limit": max(limit, 10),
        "search_type": 0,
        "sort_type": 0,
    }
    try:
        resp = client.post(url, json=payload, timeout=DEFAULT_TIMEOUT_SEC)
        resp.raise_for_status()
        payload_json = resp.json()
    except Exception:  # noqa: BLE001
        return []

    out: list[Candidate] = []
    for index, item in enumerate(payload_json.get("data") or []):
        model = item.get("result_model") or {}
        info = model.get("article_info") or model
        article_id = str(info.get("article_id") or model.get("article_id") or "")
        title = strip_html_tags(str(info.get("title") or ""))
        if not article_id or not title:
            continue
        article_url = f"https://juejin.cn/post/{article_id}"
        brief = strip_html_tags(str(info.get("brief_content") or ""))
        out.append(
            Candidate(
                id=f"{category}-JJ-{article_id[-8:]}-{index}",
                category=category,
                title=title,
                url=article_url,
                tags=["掘金", keyword],
                notes=brief[:200],
                discover_from="juejin",
                region="domestic",
            )
        )
        if len(out) >= limit:
            break
    return out


def discover_csdn(
    client: httpx.Client,
    keyword: str,
    *,
    category: str,
    limit: int,
) -> list[Candidate]:
    url = "https://so.csdn.net/api/v2/search"
    params = {"q": keyword, "t": "blog", "p": 1}
    try:
        resp = client.get(url, params=params, timeout=DEFAULT_TIMEOUT_SEC)
        resp.raise_for_status()
        payload_json = resp.json()
    except Exception:  # noqa: BLE001
        return []

    out: list[Candidate] = []
    for index, item in enumerate(payload_json.get("result_vos") or []):
        title = strip_html_tags(str(item.get("title") or ""))
        raw_url = str(item.get("url") or "")
        if not title or not raw_url:
            continue
        clean_url = canonicalize_url(raw_url)
        if "blog.csdn.net" not in host_of(clean_url):
            continue
        body = strip_html_tags(str(item.get("body") or ""))
        slug = sha256_8(clean_url)
        out.append(
            Candidate(
                id=f"{category}-CSDN-{slug}-{index}",
                category=category,
                title=title,
                url=clean_url,
                tags=["CSDN", keyword],
                notes=body[:200],
                discover_from="csdn",
                region="domestic",
            )
        )
        if len(out) >= limit:
            break
    return out


def discover_bing_cn(
    client: httpx.Client,
    keyword: str,
    *,
    category: str,
    limit: int,
) -> list[Candidate]:
    """百度常出安全验证，用必应中国做公开网页补链。"""
    query = f"{keyword} (site:juejin.cn OR site:blog.csdn.net OR site:zhuanlan.zhihu.com OR site:cloud.tencent.com OR site:developer.aliyun.com OR site:tech.meituan.com)"
    try:
        resp = client.get(
            "https://cn.bing.com/search",
            params={"q": query, "count": 20},
            timeout=DEFAULT_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        html = resp.text
    except Exception:  # noqa: BLE001
        return []

    hrefs = re.findall(r'href="(https?://[^"]+)"', html)
    out: list[Candidate] = []
    seen: set[str] = set()
    for index, href in enumerate(hrefs):
        # bing 跳转链
        if "bing.com/ck/a" in href and "u=a1" in href:
            # 尝试从查询串还原
            continue
        url = canonicalize_url(unquote(href))
        if not is_domestic_host(url):
            continue
        if any(
            bad in url
            for bad in ("login", "passport", "javascript:", "cn.bing.com")
        ):
            continue
        if url in seen:
            continue
        seen.add(url)
        host = host_of(url)
        out.append(
            Candidate(
                id=f"{category}-BING-{sha256_8(url)}-{index}",
                category=category,
                title=f"{keyword} · {host}",
                url=url,
                tags=["必应", keyword],
                notes="来自必应中国搜索补链",
                discover_from="bing_cn",
                region="domestic",
            )
        )
        if len(out) >= limit:
            break
    return out


def relevance_score(text: str, focus_terms: list[str]) -> int:
    hay = text or ""
    hay_lower = hay.lower()
    score = 0
    for term in focus_terms:
        if not term:
            continue
        if term.lower() in hay_lower or term in hay:
            score += 2
    return score


def is_relevant(title: str, notes: str, focus_terms: list[str], *, min_score: int = 2) -> bool:
    if not focus_terms:
        return True
    return relevance_score(f"{title} {notes}", focus_terms) >= min_score


def seed_to_candidate(seed: dict[str, Any], *, region: str) -> Candidate:
    url = str(seed["url"])
    return Candidate(
        id=str(seed["id"]),
        category=str(seed["category"]),
        title=str(seed.get("title") or ""),
        url=url,
        tags=list(seed.get("tags") or []),
        notes=str(seed.get("notes") or ""),
        discover_from="seed",
        region=region if region else ("domestic" if is_domestic_host(url) else "foreign"),
    )


def merge_candidates(
    buckets: dict[str, list[Candidate]],
    incoming: list[Candidate],
    *,
    target: int,
    focus_terms: list[str] | None = None,
    require_relevant: bool = False,
) -> None:
    for cand in incoming:
        cat = cand.category
        existing = buckets[cat]
        if len(existing) >= target:
            return
        if require_relevant and focus_terms and not is_relevant(
            cand.title, cand.notes, focus_terms
        ):
            continue
        urls = {canonicalize_url(c.url) for c in existing}
        if canonicalize_url(cand.url) in urls:
            continue
        existing.append(cand)


def build_candidate_pool(
    client: httpx.Client,
    config: dict[str, Any],
    *,
    categories: set[str] | None,
    target_per_category: int,
    domestic_only: bool,
    enable_bing: bool,
) -> list[Candidate]:
    cats_meta = config.get("categories") or {}
    selected_cats = [
        c for c in cats_meta.keys() if categories is None or c in categories
    ]
    buckets: dict[str, list[Candidate]] = defaultdict(list)

    # 1) 精选国内种子
    for seed in config.get("seeds_domestic") or []:
        if categories and seed.get("category") not in categories:
            continue
        merge_candidates(
            buckets,
            [seed_to_candidate(seed, region="domestic")],
            target=target_per_category,
        )

    # 2) 掘金 + CSDN 发现
    discover_cfg = config.get("discover") or {}
    for cat in selected_cats:
        focus_terms = list((cats_meta.get(cat) or {}).get("focus") or [])
        keywords = (discover_cfg.get(cat) or {}).get("keywords") or []
        for kw in keywords:
            if len(buckets[cat]) >= target_per_category:
                break
            need = target_per_category - len(buckets[cat])
            # 多拉一些再按主题词过滤，避免地图类搜到「Android Studio」
            jj = discover_juejin(client, kw, category=cat, limit=max(need * 3, 15))
            merge_candidates(
                buckets,
                jj,
                target=target_per_category,
                focus_terms=focus_terms,
                require_relevant=True,
            )
            time.sleep(0.25)
            if len(buckets[cat]) >= target_per_category:
                break
            csdn = discover_csdn(client, kw, category=cat, limit=max(need * 3, 15))
            merge_candidates(
                buckets,
                csdn,
                target=target_per_category,
                focus_terms=focus_terms,
                require_relevant=True,
            )
            time.sleep(0.25)

    # 若过滤后不足，放宽相关性再补一轮掘金
    for cat in selected_cats:
        if len(buckets[cat]) >= target_per_category:
            continue
        keywords = (discover_cfg.get(cat) or {}).get("keywords") or []
        for kw in keywords:
            if len(buckets[cat]) >= target_per_category:
                break
            need = target_per_category - len(buckets[cat])
            jj = discover_juejin(client, kw, category=cat, limit=max(need * 2, 10))
            merge_candidates(buckets, jj, target=target_per_category)
            time.sleep(0.2)

    # 3) 必应中国补链
    if enable_bing:
        for cat in selected_cats:
            if len(buckets[cat]) >= target_per_category:
                continue
            keywords = (discover_cfg.get(cat) or {}).get("keywords") or []
            for kw in keywords[:2]:
                need = target_per_category - len(buckets[cat])
                if need <= 0:
                    break
                bing = discover_bing_cn(client, kw, category=cat, limit=need + 5)
                merge_candidates(buckets, bing, target=target_per_category)
                time.sleep(0.35)

    # 4) 国外兜底
    if not domestic_only:
        for seed in config.get("seeds_foreign_fallback") or []:
            cat = str(seed.get("category"))
            if categories and cat not in categories:
                continue
            if len(buckets[cat]) >= target_per_category:
                continue
            merge_candidates(
                buckets,
                [seed_to_candidate(seed, region="foreign")],
                target=target_per_category,
            )

    pooled: list[Candidate] = []
    for cat in selected_cats:
        pooled.extend(buckets[cat][:target_per_category])
    return pooled


def extract_with_bs4(html: str) -> tuple[str, str, str]:
    assert BeautifulSoup is not None
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = normalize_ws(soup.title.string)
    description = ""
    meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if meta and meta.get("content"):
        description = normalize_ws(str(meta["content"]))
    if not description:
        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"):
            description = normalize_ws(str(og["content"]))
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=re.compile(r"(article|content|post|markdown)", re.I))
        or soup.body
        or soup
    )
    text = normalize_ws(main.get_text(" ", strip=True)) if main else ""
    return title, description, text


def extract_without_bs4(html: str) -> tuple[str, str, str]:
    parser = _TitleMetaParser()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001
        pass
    no_script = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    text = normalize_ws(re.sub(r"(?is)<[^>]+>", " ", no_script))
    return parser.title, parser.description, text


def extract_content(html: str) -> tuple[str, str, str]:
    if HAS_BS4:
        return extract_with_bs4(html)
    return extract_without_bs4(html)


def build_summary(description: str, text: str, notes: str) -> str:
    if description:
        return description[:MAX_SUMMARY_CHARS]
    if text:
        return text[:MAX_SUMMARY_CHARS]
    return (notes or "")[:MAX_SUMMARY_CHARS]


def crawl_one(
    client: httpx.Client,
    cand: Candidate,
    categories_meta: dict[str, Any],
    *,
    timeout: float,
    save_raw_dir: Path | None,
) -> ArticleRecord:
    cat_name = (categories_meta.get(cand.category) or {}).get("name", cand.category)
    fetched_at = utc_now_iso()
    try:
        resp = client.get(cand.url, timeout=timeout)
        resp.raise_for_status()
        try:
            html = resp.text
        except Exception:  # noqa: BLE001
            html = resp.content.decode("utf-8", errors="replace")
        title, description, text = extract_content(html)
        if not title:
            title = cand.title
        excerpt = text[:MAX_TEXT_CHARS]
        summary = build_summary(description, text, cand.notes)
        if save_raw_dir is not None:
            save_raw_dir.mkdir(parents=True, exist_ok=True)
            (save_raw_dir / f"{cand.id}.html").write_text(
                html, encoding="utf-8", errors="replace"
            )
        return ArticleRecord(
            id=cand.id,
            category=cand.category,
            category_name=cat_name,
            title=title or cand.title,
            url=cand.url,
            source=host_of(cand.url),
            tags=cand.tags,
            notes=cand.notes,
            summary=summary,
            text_excerpt=excerpt,
            status="ok",
            http_status=resp.status_code,
            fetched_at=fetched_at,
            discover_from=cand.discover_from,
            region=cand.region,
            content_sha256_8=sha256_8(excerpt) if excerpt else None,
        )
    except Exception as exc:  # noqa: BLE001
        return ArticleRecord(
            id=cand.id,
            category=cand.category,
            category_name=cat_name,
            title=cand.title,
            url=cand.url,
            source=host_of(cand.url),
            tags=cand.tags,
            notes=cand.notes,
            summary=(cand.notes or "")[:MAX_SUMMARY_CHARS],
            text_excerpt="",
            status="seed_only",
            http_status=None,
            fetched_at=fetched_at,
            discover_from=cand.discover_from,
            region=cand.region,
            error=f"{type(exc).__name__}: {exc}",
        )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(articles_path: Path) -> set[str]:
    done: set[str] = set()
    if not articles_path.exists():
        return done
    with articles_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") and row.get("status") == "ok":
                done.add(str(row["id"]))
    return done


def build_report(rows: list[dict[str, Any]], started_at: str) -> dict[str, Any]:
    by_cat = Counter(str(r.get("category")) for r in rows)
    by_status = Counter(str(r.get("status") or "unknown") for r in rows)
    by_discover = Counter(str(r.get("discover_from") or "unknown") for r in rows)
    by_region = Counter(str(r.get("region") or "unknown") for r in rows)
    by_source = Counter(str(r.get("source") or "unknown") for r in rows)
    total = len(rows)
    ok_n = by_status.get("ok", 0)
    return {
        "generated_at": utc_now_iso(),
        "started_at": started_at,
        "strategy": "domestic_first",
        "total": total,
        "ok": ok_n,
        "seed_only": by_status.get("seed_only", 0),
        "ok_rate": round(ok_n / total, 4) if total else 0.0,
        "by_category": dict(sorted(by_cat.items())),
        "by_status": dict(by_status),
        "by_discover_from": dict(by_discover),
        "by_region": dict(by_region),
        "top_sources": by_source.most_common(15),
        "failures": [
            {"id": r.get("id"), "url": r.get("url"), "error": r.get("error")}
            for r in rows
            if r.get("status") != "ok"
        ][:50],
        "bs4_enabled": HAS_BS4,
        "note": "百度对自动化访问常出安全验证，发现通道以掘金/CSDN/必应中国为主",
    }


def run(args: argparse.Namespace) -> int:
    config = load_config(Path(args.seeds))
    out_dir = Path(args.out_dir)
    articles_path = out_dir / "articles.jsonl"
    discovered_path = out_dir / "discovered_seeds.jsonl"
    report_path = out_dir / "crawl_report.json"
    raw_dir = out_dir / "raw" if args.save_raw else None

    category_filter = parse_categories_arg(args.category)
    target = args.target or int(
        config.get("target_per_category") or DEFAULT_TARGET_PER_CATEGORY
    )

    started_at = utc_now_iso()
    with make_client() as client:
        candidates = build_candidate_pool(
            client,
            config,
            categories=category_filter,
            target_per_category=target,
            domestic_only=args.domestic_only,
            enable_bing=not args.no_bing,
        )
        if args.limit is not None:
            candidates = candidates[: args.limit]

        # 落盘发现列表
        if discovered_path.exists() and not args.resume and not args.append:
            discovered_path.unlink()
        for cand in candidates:
            append_jsonl(discovered_path, asdict(cand))

        print(f"config     : {args.seeds}")
        print(f"strategy   : domestic_first domestic_only={args.domestic_only}")
        print(f"candidates : {len(candidates)}")
        print(
            "by_cat     :",
            dict(Counter(c.category for c in candidates)),
        )
        print(
            "by_discover:",
            dict(Counter(c.discover_from for c in candidates)),
        )
        print(f"bs4        : {HAS_BS4}")

        if args.discover_only:
            report = build_report([asdict(c) | {"status": "discovered"} for c in candidates], started_at)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"discovered => {discovered_path}")
            print(f"report     => {report_path}")
            return 0

        done_ids = load_done_ids(articles_path) if args.resume else set()
        if not args.resume and articles_path.exists() and not args.append:
            articles_path.unlink()

        todo = [c for c in candidates if c.id not in done_ids]
        print(f"to crawl   : {len(todo)} (resume_skip={len(done_ids)})")

        records: list[ArticleRecord] = []
        categories_meta = config.get("categories") or {}
        for index, cand in enumerate(todo, start=1):
            record = crawl_one(
                client,
                cand,
                categories_meta,
                timeout=args.timeout,
                save_raw_dir=raw_dir,
            )
            records.append(record)
            append_jsonl(articles_path, asdict(record))
            mark = "OK" if record.status == "ok" else "SEED"
            print(
                f"[{index}/{len(todo)}] {mark} {record.id} "
                f"{record.region}/{record.discover_from} "
                f"{record.source} title={record.title[:40]!r}"
            )
            if index < len(todo) and args.delay > 0:
                time.sleep(args.delay)

    # 汇总：resume 时扫全量 jsonl
    rows: list[dict[str, Any]] = []
    if articles_path.exists():
        with articles_path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        rows = [asdict(r) for r in records]

    report = build_report(rows, started_at)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"\ndone: total={report['total']} ok={report['ok']} "
        f"seed_only={report['seed_only']} ok_rate={report['ok_rate']:.1%}"
    )
    print(f"region     : {report['by_region']}")
    print(f"discover   : {report['by_discover_from']}")
    print(f"articles   => {articles_path}")
    print(f"discovered => {discovered_path}")
    print(f"report     => {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Domestic-first crawler for stability troubleshooting KB",
    )
    parser.add_argument("--seeds", default=str(DEFAULT_SEEDS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--category", default=None, help="如 A 或 A,E,G")
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="每类目标条数（默认读配置，约 15）",
    )
    parser.add_argument("--limit", type=int, default=None, help="全局最多抓取条数")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SEC)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument(
        "--domestic-only",
        action="store_true",
        help="禁止国外兜底种子",
    )
    parser.add_argument(
        "--no-bing",
        action="store_true",
        help="关闭必应中国补链",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="只发现候选链接，不抓正文",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
