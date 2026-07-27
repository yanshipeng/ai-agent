"""文档清洗核心：把原始采集（articles）变成干净文档（docs）。

==========================================================================
大白话
==========================================================================
网上抓回来的文章像「没拆塑封、夹着广告纸的书」。
本模块：拆塑封、撕广告、统一成可上架的书（Markdown + 固定字段）。

==========================================================================
为什么必须有这一层（不能直接用 articles）
==========================================================================
1) 采集结果质量参差：有的只有链接（seed_only），有的夹导航/登录墙。
2) 字段不统一：后续切块 / 建索引需要稳定 schema。
3) 去重：同一文多源转载会污染检索；优先 content_sha256_8，否则 url+title。
4) 可选 refetch：本地 excerpt 够用就 --no-refetch（快）；要代码围栏再联网慢抓。

输入：articles 列表 → 输出：docs 列表
调用方：scripts/build_stability_docs.py；也可 from app.kb import build_docs_from_articles
==========================================================================
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any
from urllib.parse import urlparse

import httpx

from app.kb.jsonl_io import sha256_8, utc_now_iso

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
TIMEOUT_SEC = 25.0
DEFAULT_DELAY_SEC = 0.35
MIN_CONTENT_CHARS = 280
MAX_CONTENT_CHARS = 80_000

BAD_PAGE_MARKERS = (
    "呃，似乎你迷路了",
    "此页面神不知鬼不觉地丢失",
    "页面不存在",
    "404 not found",
    "访问的页面不存在",
    "验证码",
    "请完成安全验证",
    "百度安全验证",
)
NOISE_LINE_PATTERNS = (
    re.compile(r"^关注\s*$"),
    re.compile(r"^点赞\s*$"),
    re.compile(r"^收藏\s*$"),
    re.compile(r"^分享\s*$"),
    re.compile(r"^推荐阅读"),
    re.compile(r"^相关推荐"),
    re.compile(r"^你可能感兴趣"),
    re.compile(r"^热门文章"),
    re.compile(r"^广告"),
    re.compile(r"^赞助"),
    re.compile(r"^版权声明"),
    re.compile(r"^原文链接"),
    re.compile(r"^作者[：:]"),
    re.compile(r"^编辑[：:]"),
    re.compile(r"^转载请注明"),
    re.compile(r"^欢迎关注"),
    re.compile(r"^扫码关注"),
    re.compile(r"^阅读\d+分钟"),
    re.compile(r"^\d{4}-\d{2}-\d{2}"),
    re.compile(r"^掘金$"),
    re.compile(r"^登录$"),
    re.compile(r"^注册$"),
)
CONTENT_SELECTORS = (
    ".markdown-body",
    ".article-viewer",
    ".article-content",
    ".blog-content-box",
    "#content_views",
    ".postBody",
    ".post-content",
    ".entry-content",
    "article",
    "main",
)

try:
    from bs4 import BeautifulSoup
    from bs4.element import NavigableString, Tag

    HAS_BS4 = True
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore
    Tag = object  # type: ignore
    NavigableString = str  # type: ignore
    HAS_BS4 = False


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def make_http_client(*, timeout: float = TIMEOUT_SEC) -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
        follow_redirects=True,
        timeout=timeout,
        http2=False,
    )


def fetch_html(client: httpx.Client, url: str) -> str:
    resp = client.get(url)
    resp.raise_for_status()
    try:
        return resp.text
    except Exception:  # noqa: BLE001
        return resp.content.decode("utf-8", errors="replace")


def pick_content_root(soup: Any) -> Any:
    for sel in CONTENT_SELECTORS:
        node = soup.select_one(sel)
        if node and normalize_ws(node.get_text(" ", strip=True)):
            return node
    return soup.body or soup


def remove_noise_nodes(root: Any) -> None:
    bad_tags = ("script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside")
    for tag in root.find_all(bad_tags):
        tag.decompose()
    noise_class = re.compile(
        r"(recommend|related|advert|ads?|sidebar|comment|footer|header|"
        r"login|share|toolbar|breadcrumb|author-info|follow)",
        re.I,
    )
    for tag in list(root.find_all(True)):
        attrs = getattr(tag, "attrs", None) or {}
        classes = " ".join(attrs.get("class") or [])
        tid = attrs.get("id") or ""
        if noise_class.search(classes) or noise_class.search(str(tid)):
            tag.decompose()


def node_to_markdown(node: Any, *, list_depth: int = 0) -> str:
    if isinstance(node, NavigableString):
        text = str(node)
        if not text or text.isspace():
            return ""
        return normalize_ws(text)

    if not isinstance(node, Tag):
        return ""

    name = (node.name or "").lower()
    if name in {"script", "style", "noscript", "svg", "iframe"}:
        return ""
    if name == "br":
        return "\n"

    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(name[1])
        inner = inline_children(node)
        return f"\n{'#' * level} {inner}\n\n" if inner else ""

    if name == "p":
        inner = inline_children(node)
        return f"\n{inner}\n\n" if inner else ""

    if name in {"ul", "ol"}:
        parts: list[str] = []
        index = 1
        for child in node.children:
            if getattr(child, "name", None) != "li":
                continue
            item = list_item_to_md(
                child, ordered=(name == "ol"), index=index, depth=list_depth
            )
            if item:
                parts.append(item)
                index += 1
        return ("\n".join(parts) + "\n\n") if parts else ""

    if name == "li":
        return list_item_to_md(node, ordered=False, index=1, depth=list_depth)

    if name == "pre":
        code = node.get_text("", strip=False)
        if not code.strip():
            code = node.get_text("\n", strip=False)
        code = code.replace("\r\n", "\n").strip("\n")
        lang = ""
        code_tag = node.find("code")
        if code_tag and code_tag.get("class"):
            for cls in code_tag.get("class"):
                if cls.startswith("language-"):
                    lang = cls.replace("language-", "", 1)
                    break
                if cls.startswith("lang-"):
                    lang = cls.replace("lang-", "", 1)
                    break
        return f"\n```{lang}\n{code}\n```\n\n"

    if name == "code":
        if node.parent and (node.parent.name or "").lower() == "pre":
            return ""
        return f"`{node.get_text()}`"

    if name == "blockquote":
        inner = block_children(node, list_depth=list_depth).strip()
        if not inner:
            return ""
        quoted = "\n".join(f"> {line}" if line else ">" for line in inner.splitlines())
        return f"\n{quoted}\n\n"

    if name in {"strong", "b"}:
        inner = inline_children(node)
        return f"**{inner}**" if inner else ""
    if name in {"em", "i"}:
        inner = inline_children(node)
        return f"*{inner}*" if inner else ""
    if name == "a":
        inner = inline_children(node)
        href = node.get("href") or ""
        if inner and href and not href.startswith("javascript:"):
            return f"[{inner}]({href})"
        return inner
    if name in {"div", "section", "article", "main", "span", "figure", "figcaption"}:
        return block_children(node, list_depth=list_depth)
    if name in {"table", "thead", "tbody", "tr", "td", "th"}:
        return block_children(node, list_depth=list_depth)
    return block_children(node, list_depth=list_depth)


def inline_children(node: Any) -> str:
    parts: list[str] = []
    for child in node.children:
        if getattr(child, "name", None) in {
            "ul",
            "ol",
            "pre",
            "p",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            parts.append(normalize_ws(child.get_text(" ", strip=True)))
            continue
        piece = node_to_markdown(child)
        if piece:
            parts.append(piece)
    return normalize_ws("".join(parts))


def block_children(node: Any, *, list_depth: int) -> str:
    parts: list[str] = []
    for child in node.children:
        piece = node_to_markdown(child, list_depth=list_depth)
        if piece:
            parts.append(piece)
    return "".join(parts)


def list_item_to_md(node: Any, *, ordered: bool, index: int, depth: int) -> str:
    indent = "  " * depth
    prefix = f"{index}. " if ordered else "- "
    text_parts: list[str] = []
    nested: list[str] = []
    for child in node.children:
        cname = getattr(child, "name", None)
        if cname in {"ul", "ol"}:
            nested.append(node_to_markdown(child, list_depth=depth + 1).rstrip())
        else:
            piece = node_to_markdown(child, list_depth=depth)
            if piece:
                text_parts.append(piece)
    head = normalize_ws(" ".join(text_parts))
    lines = [f"{indent}{prefix}{head}"] if head else []
    lines.extend([n for n in nested if n])
    return "\n".join(lines)


def clean_markdown(md: str) -> str:
    """最小化清洗：删噪声行、压空白、去重复段，保留代码块。"""
    if not md:
        return ""

    parts = re.split(r"(```[\s\S]*?```)", md)
    cleaned_parts: list[str] = []
    seen_paras: set[str] = set()

    for part in parts:
        if part.startswith("```") and part.endswith("```"):
            cleaned_parts.append(part.strip() + "\n\n")
            continue

        lines_out: list[str] = []
        for raw_line in part.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                if lines_out and lines_out[-1] != "":
                    lines_out.append("")
                continue
            if any(p.search(stripped) for p in NOISE_LINE_PATTERNS):
                continue
            if len(stripped) <= 1 and stripped not in {"-", "*"}:
                continue
            lines_out.append(line)

        buf: list[str] = []
        para: list[str] = []
        for line in lines_out + [""]:
            if line.strip():
                para.append(line)
                continue
            block = "\n".join(para).strip()
            para = []
            if not block:
                continue
            key = re.sub(r"\s+", " ", block)
            if key in seen_paras and len(key) > 40:
                continue
            seen_paras.add(key)
            buf.append(block)
        text = "\n\n".join(buf).strip()
        if text:
            cleaned_parts.append(text + "\n\n")

    result = "".join(cleaned_parts)
    result = re.sub(r"\n{3,}", "\n\n", result).strip() + "\n"
    if len(result) > MAX_CONTENT_CHARS:
        result = result[:MAX_CONTENT_CHARS].rstrip() + "\n"
    return result


def html_to_markdown(html: str) -> tuple[str, str]:
    """返回 (title, markdown_content)。"""
    if not HAS_BS4:
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        text = normalize_ws(re.sub(r"(?is)<[^>]+>", " ", text))
        return "", text

    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = normalize_ws(soup.title.string)
    root = pick_content_root(soup)
    remove_noise_nodes(root)
    md = clean_markdown(node_to_markdown(root))
    return title, md


def is_bad_content(content: str, *, min_chars: int = MIN_CONTENT_CHARS) -> bool:
    low = content.lower()
    if len(content.strip()) < min_chars:
        return True
    return any(marker.lower() in low for marker in BAD_PAGE_MARKERS)


def dedupe_key(
    content: str,
    url: str,
    title: str,
    existing_hash: str | None = None,
) -> str:
    body = content.strip()
    if body:
        return f"sha:{sha256_8(body)}"
    if existing_hash:
        return f"sha:{existing_hash}"
    return f"ut:{sha256_8((url or '').strip() + '|' + (title or '').strip())}"


def build_doc_from_article(
    article: dict[str, Any],
    *,
    client: httpx.Client | None = None,
    refetch: bool = False,
) -> dict[str, Any] | None:
    """把单条原始采集转成一条干净 doc。

    返回值约定：
      - 正常：完整 doc 字典
      - 应跳过：{"_skip": True, "doc_id": ..., "reason": ...}
      - 缺 id/url：None
    """
    doc_id = str(article.get("id") or "")
    url = str(article.get("url") or "")
    title = str(article.get("title") or "").strip()
    if not doc_id or not url:
        return None

    content = ""
    page_title = ""
    if refetch and client is not None:
        try:
            html = fetch_html(client, url)
            page_title, content = html_to_markdown(html)
        except Exception as exc:  # noqa: BLE001
            fallback = clean_markdown(str(article.get("text_excerpt") or ""))
            if not is_bad_content(fallback):
                content = fallback
            else:
                return {
                    "_skip": True,
                    "doc_id": doc_id,
                    "reason": f"refetch_failed:{type(exc).__name__}",
                }
    else:
        content = clean_markdown(str(article.get("text_excerpt") or ""))

    if page_title and (not title or title.endswith("- 掘金") or len(title) > 120):
        title = page_title
    title = re.sub(r"\s*[-_|]\s*掘金\s*$", "", title).strip()
    title = re.sub(r"\s*\|\s*Android Developers\s*$", "", title).strip()

    if is_bad_content(content):
        return {"_skip": True, "doc_id": doc_id, "reason": "bad_or_short_content"}

    return {
        "doc_id": doc_id,
        "category": article.get("category"),
        "category_name": article.get("category_name"),
        "title": title,
        "url": url,
        "source": article.get("source") or urlparse(url).netloc,
        "tags": list(article.get("tags") or []),
        "notes": article.get("notes") or None,
        "summary": article.get("summary") or None,
        "content": content,
        "created_at": article.get("fetched_at") or utc_now_iso(),
        "content_sha256_8": sha256_8(content),
    }


def build_docs_from_articles(
    articles: list[dict[str, Any]],
    *,
    refetch: bool = False,
    delay_sec: float = DEFAULT_DELAY_SEC,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """批量清洗去重。

    返回：(docs列表, report汇总字典)
    report 里有 docs/skipped/duplicates/by_category 等，方便写 docs_report.json。
    """
    docs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    dup_count = 0
    client: httpx.Client | None = None

    try:
        if refetch:
            client = make_http_client()

        for index, article in enumerate(articles, start=1):
            result = build_doc_from_article(
                article, client=client, refetch=refetch
            )
            if result is None:
                skipped.append(
                    {"doc_id": article.get("id"), "reason": "missing_id_or_url"}
                )
                continue
            if result.get("_skip"):
                skipped.append(
                    {"doc_id": result.get("doc_id"), "reason": result.get("reason")}
                )
                if progress:
                    print(
                        f"[{index}/{len(articles)}] SKIP "
                        f"{result.get('doc_id')} {result.get('reason')}"
                    )
                continue

            key = dedupe_key(
                result["content"],
                result["url"],
                result["title"],
                result.get("content_sha256_8"),
            )
            if key in seen_keys:
                dup_count += 1
                skipped.append({"doc_id": result["doc_id"], "reason": f"dup:{key}"})
                if progress:
                    print(f"[{index}/{len(articles)}] DUP  {result['doc_id']}")
                continue

            seen_keys.add(key)
            docs.append(result)
            if progress:
                has_code = "```" in result["content"]
                print(
                    f"[{index}/{len(articles)}] OK   {result['doc_id']} "
                    f"chars={len(result['content'])} code={has_code} "
                    f"title={result['title'][:40]!r}"
                )
            if refetch and index < len(articles) and delay_sec > 0:
                time.sleep(delay_sec)
    finally:
        if client is not None:
            client.close()

    report = {
        "generated_at": utc_now_iso(),
        "input_articles": len(articles),
        "docs": len(docs),
        "skipped": len(skipped),
        "duplicates": dup_count,
        "ok_rate": round(len(docs) / len(articles), 4) if articles else 0.0,
        "by_category": dict(Counter(d["category"] for d in docs)),
        "with_code_fence": sum(1 for d in docs if "```" in d["content"]),
        "avg_content_chars": int(sum(len(d["content"]) for d in docs) / len(docs))
        if docs
        else 0,
        "skip_reasons": dict(Counter(s.get("reason", "unknown") for s in skipped)),
        "bs4_enabled": HAS_BS4,
        "refetch": refetch,
    }
    return docs, report
