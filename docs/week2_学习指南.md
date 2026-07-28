# 第二周学习指南：稳定性知识库语料处理

> 面向刚入行同学。记录第二周「采集 → 清洗 → 切块 → Embedding/建索引/检索」做了什么、怎么操作、原理是什么、踩过哪些坑。  
> 第一周基础：见 [`week1_学习指南.md`](./week1_学习指南.md)。  
> 第三周 Agent / Tools：见 [`week3_学习指南.md`](./week3_学习指南.md)。  
> 服务日志怎么读：见 [`日志阅读指南.md`](./日志阅读指南.md)。  
> 核心代码注释：各模块文件头写了「做什么 + 为什么这么做」，建议从 `app/kb/`、`app/api.py` 读起。  
> 代码会变，以仓库当前实现为准；数字以你本机 `data/stability_kb/` 实际文件为准。

---

## 0. 第二周要达成什么

第一周（大致 Day1–Day5）已经有（详见 [Week1 指南](./week1_学习指南.md)）：

- FastAPI 服务：`/health`、`/ask`
- DeepSeek LLM 调用、错误映射、日志、`requests.jsonl`
- 契约测试、错误映射测试、回归评测脚本

第二周做 **RAG / 知识库**：先把语料链路跑通，再接上检索，最后才接到 `/ask`：

| 天数 | 主题 | 目标产物 |
|------|------|----------|
| Day 6 | 清洗 + 去重 + 统一 schema | `docs.jsonl` |
| Day 7 | Chunking v1（按标题/长度/代码块切） | `chunks.jsonl` |
| Day 8 | Embedding + 建索引 + `retrieve()` | `index/` + TopK 命中 |
| Day 9 | `/ask?mode=rag` + citations | 带引用的 RAG 回答 |
| Day 10 | 评测闭环（≥50）+ 单变量 A/B | `eval_report.json` / A/B 对比 |

一句话：

```text
网上公开文章
  → articles.jsonl   （原始采集，可能很脏）
  → docs.jsonl       （干净长文，可入库）
  → chunks.jsonl     （小段文字，检索单元）
  → index/           （向量 + 元数据，加速检索）
  → retrieve(query)  （按问题取 TopK 证据）
  → /ask?mode=rag    （拼 Context → LLM → 带 citations 回答）
```

---

## 1. 为什么要分三层？（入门必懂）

把整件事想成「建图书馆」：

| 层级 | 文件 | 像什么 | 给谁用 |
|------|------|--------|--------|
| 原始采集 | `articles.jsonl` | 刚拖回家、还没拆塑封的书 | 清洗脚本 |
| 干净文档 | `docs.jsonl` | 洗干净、统一格式的书 | 归档、再切块 |
| 检索单元 | `chunks.jsonl` | 按章节撕成的卡片 | Embedding / 检索 |
| 向量索引 | `index/` | 卡片的「坐标目录」 | `retrieve()` |

**为什么不能只用 articles？**  
网页有广告、导航、登录按钮；有的只有链接没有正文；格式也不统一。

**为什么还要再切成 chunks？**  
一篇可能几千字。用户问「ANR 怎么查 traces」，系统应找出相关**段落**塞给大模型，而不是整篇塞进去。

**为什么还要 Embedding + 索引？**  
有了几百张「卡片」还不够：用户随便问一句时，系统必须**快速找出最相关的几张**。把文字变成向量、事先建好索引，才能按相似度取 TopK，而不是每次翻遍全文或把整库塞给模型。

---

## 2. 目录与职责（第二周重点改动）

### 2.1 核心逻辑在 `app/kb/`（不是只放 scripts）

第二周重要架构决定：**清洗 / 切块是整条业务链路的前期数据处理，应放在核心包，脚本只做薄入口。**

```text
app/kb/                       ← 核心工具层（后厨）
  ├── __init__.py             ← 对外导出常用 API
  ├── cleaner.py              ← Day6：articles → docs
  ├── chunker.py              ← Day7：docs → chunks
  ├── embedder.py             ← 文本 → 向量（本地 hashing TF）
  ├── index_store.py          ← chunks → 索引落盘/加载
  ├── retriever.py            ← retrieve(query) → TopK
  ├── rag.py                  ← RAG prompt / citations
  └── jsonl_io.py             ← JSONL 读写、sha256_8 等

scripts/                      ← 薄 CLI（点餐员 / 遥控器）
  ├── crawl_stability_kb.py   ← 采集入口
  ├── build_stability_docs.py ← 清洗入口 → 调用 app.kb.cleaner
  ├── chunk_stability_docs.py ← 切块入口 → 调用 app.kb.chunker
  ├── build_kb_index.py       ← 建索引入口 → 调用 app.kb.index_store
  ├── retrieve_kb.py          ← 检索入口 → 调用 app.kb.retriever
  ├── verify_kb_retrieve.py   ← Day8 验收（索引 + 10 query 冒烟）
  ├── verify_ask_rag.py       ← 核对 RAG citations ↔ 本地 index
  ├── run_day6_7.sh           ← 一键串起来（默认可不扩采）
  └── stability_kb_seeds.json ← 采集关键词与精选种子
```

**「薄 CLI」是什么？**

- CLI = 你在终端敲的命令行程序
- 薄 = 自己几乎不干重活，只负责：读参数 → 调用核心 → 写文件 / 打印进度  
- 真正算法在 `app/kb/*`，以后 FastAPI 服务也能 `from app.kb import chunk_docs` 直接用

### 2.2 数据文件在 `data/stability_kb/`

```text
data/stability_kb/
  ├── discovered_seeds.jsonl  # 发现阶段候选链接
  ├── articles.jsonl          # 原始采集结果
  ├── crawl_report.json       # 采集汇总
  ├── docs.jsonl              # 干净文档
  ├── docs_report.json        # 清洗汇总
  ├── chunks.jsonl            # 检索单元
  ├── chunks_report.json      # 切块汇总
  └── index/                  # 向量索引目录
        ├── meta.jsonl
        ├── vectors.jsonl
        └── manifest.json
```

这些产物默认在 `.gitignore` 里（体积大、可重建），以本地为准。

---

## 3. Day 6：原始采集（补充回顾）

清洗之前，需要先有 `articles.jsonl`。第二周实际操作里采用了 **国内平台优先**。

### 3.1 采集策略

1. 精选国内种子（美团技术、阿里云/腾讯云、高德、`android.google.cn` 等）
2. 掘金 Search API、CSDN Search API 按 A–G 关键词发现
3. 必应中国补链（百度对脚本常出「安全验证」，不适合做主通道）
4. 每类数量仍不足时，才考虑国外兜底（可用 `--domestic-only` 关掉）

主题七类：

| 代号 | 名称 |
|------|------|
| A | 交互无响应 / 卡顿 / ANR |
| B | Crash / 启动失败 / 白屏 |
| C | 内存问题 |
| D | 网络与接口 |
| E | 地图 / 定位 / 导航 |
| F | Hybrid / WebView / JSBridge |
| G | 消息 / 推送 / IM |

### 3.2 怎么跑采集

```bash
source .venv/bin/activate
python scripts/crawl_stability_kb.py --domestic-only --target 30
```

常用参数：

| 参数 | 含义 |
|------|------|
| `--domestic-only` | 只要国内源 |
| `--target 30` | 每类目标约 30 条 |
| `--discover-only` | 只发现链接，不抓正文 |
| `--category A,E` | 只跑指定类别 |

### 3.3 articles 里一条大概有什么

- `id / category / category_name / title / url / source / tags`
- `text_excerpt`：当时抓到的正文草稿（可能不完美）
- `status`：`ok`（有正文）或 `seed_only`（基本只有链接）
- `content_sha256_8`：有正文时的指纹

**重要认知：** 采集条数 ≠ 最终可用文档数。很多会在清洗阶段被丢掉。

---

## 4. Day 6：清洗 → `docs.jsonl`

### 4.1 目标

把 articles 变成「可入库」的干净语料：

- 统一字段 schema
- 正文尽量是可读 Markdown（有标题、列表；代码在 ``` 里）
- 去重、去坏链、去过短垃圾页

### 4.2 核心做了什么（`app/kb/cleaner.py`）

1. **拿正文**  
   - `refetch=True`：重新打开网页，HTML → Markdown（保留结构）  
   - `refetch=False` / `--no-refetch`：只用本地 `text_excerpt`（更快，但不联网）

2. **最小化清洗**  
   - 删：导航、推荐阅读、广告、签名等噪声行  
   - 留：标题、分段、步骤列表、结论、代码块  
   - 去掉明显坏页（404、「迷路了」、验证码等）

3. **去重**  
   - 优先：正文哈希 `content_sha256_8`  
   - 没有可靠正文：`url + title`

4. **过滤**  
   - 正文太短（默认阈值约 280 字符）→ 丢弃

### 4.3 怎么操作

```bash
source .venv/bin/activate

# 推荐：不联网，只用本地已有正文
python scripts/build_stability_docs.py --no-refetch

# 需要更好 Markdown / 代码块时，才联网重抓（慢）
python scripts/build_stability_docs.py

# 调试（注意：--limit 会覆盖输出文件，正式跑不要随便加）
python scripts/build_stability_docs.py --no-refetch --limit 5
```

代码里直接调用（不经过 CLI）：

```python
from app.kb import build_docs_from_articles, load_jsonl, write_jsonl

articles = load_jsonl("data/stability_kb/articles.jsonl")
docs, report = build_docs_from_articles(articles, refetch=False)
write_jsonl("data/stability_kb/docs.jsonl", docs)
```

### 4.4 docs 一条字段说明

| 字段 | 含义 |
|------|------|
| `doc_id` | 文档 ID（常沿用采集 id） |
| `category` / `category_name` | 主题类 |
| `title` / `url` / `source` | 标题、链接、来源站 |
| `tags` | 标签数组 |
| `notes` / `summary` | 备注、摘要（可选） |
| `content` | **清洗后的正文（建议 Markdown）** |
| `created_at` | 时间（常用抓取时间） |
| `content_sha256_8` | 正文指纹，用于去重 |

### 4.5 如何验收（自学检查清单）

- [ ] `docs.jsonl` 能打开，每行是一个 JSON
- [ ] 随机抽几条：有 `doc_id/title/url/category/tags/content`
- [ ] `content` 大体可读（有分段更好）
- [ ] 看 `docs_report.json`：`docs`、`skipped`、`by_category` 是否合理
- [ ] 理解「候选很多 → 有效 docs 更少」是正常漏斗

---

## 5. Day 7：切块 → `chunks.jsonl`

### 5.1 目标

把每篇干净文档切成带引用元数据的小段，供后续检索。

### 5.2 切块规则 v1（写死在 `app/kb/chunker.py`）

1. **先按 Markdown 标题切**：`#` / `##` / `###`  
   - 生成 `section_path`，例如：`冷启动优化/核心回答`
2. **标题段内再按长度切**  
   - `chunk_size` 默认 1000 字符（建议区间 800–1200）  
   - `overlap` 默认 120（建议 100–150，防止答案卡在切割缝上）
3. **代码块单独成 chunk**  
   - 一段 ` ```...``` ` → 一个 chunk，`is_code=true`
4. **过滤过短非代码碎块**（默认短于约 40 字符的纯文字丢掉）

### 5.3 怎么操作

```bash
source .venv/bin/activate
python scripts/chunk_stability_docs.py --chunk-size 1000 --overlap 120
```

代码调用：

```python
from app.kb import chunk_docs, load_jsonl, write_jsonl

docs = load_jsonl("data/stability_kb/docs.jsonl")
chunks, report = chunk_docs(docs, chunk_size=1000, overlap=120)
write_jsonl("data/stability_kb/chunks.jsonl", chunks)
```

### 5.4 chunk 一条字段说明

| 字段 | 含义 |
|------|------|
| `chunk_id` | 如 `{doc_id}:{section_index}:{chunk_index}` |
| `doc_id` | 属于哪篇文档 |
| `category` / `category_name` | 主题 |
| `title` / `url` / `source` | 引用时展示来源 |
| `tags` | 标签 |
| `section_path` | 章节路径（小标题面包屑） |
| `text` | **真正用来检索/给模型看的文字** |
| `char_len` | 长度 |
| `is_code` | 是否代码块 |

### 5.5 自学对照练习（强烈建议做一遍）

1. 打开 `docs.jsonl`，记下某个 `doc_id`
2. 读它的 `content`（整篇）
3. 在 `chunks.jsonl` 里搜索同一个 `doc_id`
4. 观察：一篇如何变成多块；`section_path`、`is_code` 如何变化

这能把「切块」从抽象概念变成具体感觉。

### 5.6 如何验收

- [ ] `chunks.jsonl` 行数远大于 docs（一篇多块）
- [ ] 随机抽几条：`chunk_id/doc_id/title/url/text` 齐全
- [ ] 若正文含 Markdown 代码块：应能看到 `is_code=true` 的独立 chunk
- [ ] 看 `chunks_report.json`：`chunks`、`code_chunks`、`avg_char_len`

---

## 6. 一键脚本（Day6–7）

```bash
# 默认不扩采，只做清洗+切块（EXPAND 默认 0）
EXPAND=0 ./scripts/run_day6_7.sh

# 若确实要先扩采再清洗切块
EXPAND=1 TARGET=30 ./scripts/run_day6_7.sh
```

注意：扩采很耗时，且可能被网站限流；第二周后期约定「够用就停，不再盲目扩采」。

---

## 6.5 Day 8：Embedding + 建索引 + `retrieve()`

### 6.5.1 这一步要解决什么问题

前面已经有了 `chunks.jsonl`（约几百段小文字）。  
**问题：用户随便问一句，系统怎么从几百段里快速找出最相关的几段？**

没有这一步，常见两条坏路：

1. **整库塞给大模型** → 太贵、超长、还容易胡说  
2. **只靠关键词硬匹配** → 「ANR」能命中，「卡住无响应」就容易漏

所以要做检索（Retrieval）：把问题变成可比较的形式，取出 TopK 证据。  
接上大模型之后，才叫真正的 **RAG**（检索增强生成）。

```text
用户问题
   ↓ Embedding（变成向量）
与索引里每段 chunk 比相似度
   ↓
TopK 命中（带 score、标题、链接…）
   ↓（Day9）
拼进 Prompt → LLM 回答
```

### 6.5.2 做了哪三件事

| 步骤 | 核心模块 | 一句话 |
|------|----------|--------|
| Embedding | `app/kb/embedder.py` | 文字 → 固定长度的数字向量 |
| 建索引 | `app/kb/index_store.py` | 把所有 chunk 的向量提前算好落盘 |
| 检索 | `app/kb/retriever.py` | `retrieve(query, top_k=5)` 返回统一字段 + `retrieve_ms` |

薄 CLI（只传参、写文件）：

- `scripts/build_kb_index.py` → 建索引  
- `scripts/retrieve_kb.py` → 检索冒烟  

### 6.5.3 Embedding：文字 → 一串数字

**做什么：**  
把「Android ANR 怎么排查」和每一段 chunk 文本，都变成长度固定的向量（当前 `dim=1024`）。  
意思相近的句子，向量夹角更小、点积更大。

**当前怎么实现（不必一开始就上商业模型）：**

- 分词：英文单词 + 中文单字，再拼 bigram（对中文检索很有用）
- 哈希进 1024 个桶（Hashing Trick，省掉大词表）
- L2 归一化 → 查询时用**点积当余弦相似度**

**为什么这么做：**

| 考量 | 说明 |
|------|------|
| 先跑通链路 | Day 8 目标是「能检索」，不是立刻上最强语义模型 |
| 本地可复现 | 不依赖 Key、不下载大模型、不联网也能建索引 |
| 接口可替换 | 对外只有 `embed_texts` / `embed_query`；以后换成 BGE/OpenAI，检索层几乎不用改 |

**代价要心里有数：**  
这种向量偏「词面重合」，语义不如专用模型（「卡死」和「ANR」若正文都写了才更稳）。对当前排障语料够用；质量不够时再换 embedder，不必推翻 `retrieve()`。

### 6.5.4 建索引：把「慢活」提前做完

**做什么：**  
读 `chunks.jsonl`，对每段算向量，写到 `data/stability_kb/index/`：

| 文件 | 内容 |
|------|------|
| `vectors.jsonl` | 每行一个向量 |
| `meta.jsonl` | 与向量一一对应的 `chunk_id/title/url/text/...` |
| `manifest.json` | `dim`、`size`、生成时间等 |

建索引时拼进向量的不只是正文，还有 **title + section_path + category + tags**，  
让「标题里有 ANR、正文偏日志」的块也更容易被问到。

**为什么要离线建索引，而不是每次提问现算全部 chunk：**

- Embedding 相对贵；索引建一次，查询只算 **1 次**（用户问题）
- 查询时只做「问题向量 × 已有向量」的相似度排序
- meta 和向量分开存：换展示字段不必重算向量；排查时也好打开看

`size` 应与 `chunks.jsonl` 行数一致（例如本机约 781）。

### 6.5.5 `retrieve(query, top_k=5)`：查 TopK 并统一输出

**内部流程：**

1. 加载索引（进程内缓存，避免每次读盘）
2. `embed_query(query)`
3. 与全部 chunk 向量算余弦相似度
4. 按 `score` 降序取 TopK
5. 记录 `retrieve_ms`

**返回形状：**

```python
{
  "query": "...",
  "top_k": 5,
  "retrieve_ms": 95,
  "results": [
    {
      "chunk_id": "...",
      "score": 0.43,
      "title": "...",
      "url": "...",
      "section_path": "",
      "text_snippet": "...",
      "is_code": false
    }
  ]
}
```

**每个字段为什么要：**

| 字段 | 用途 |
|------|------|
| `chunk_id` | 唯一引用；评测、日志、去重都靠它 |
| `score` | 相关性；可做阈值、调试「为什么排第一」 |
| `title` / `url` | 回答里要能引用出处（citations） |
| `section_path` | 章节面包屑；有标题结构时更好定位 |
| `text_snippet` | 人眼快速看命中对不对；喂模型时再用更长正文 |
| `is_code` | 代码块可特殊展示，或优先给「怎么写」类问题 |
| 外层 `retrieve_ms` | 性能基线；以后换向量库/模型要对比延迟 |

### 6.5.6 为什么整体是「切块 → 向量检索」

1. **模型上下文有限** — 只能塞最相关几段（TopK，常见 3–8）  
2. **答案要可追溯** — 命中带 `url/title`，才能说「根据某某文章」  
3. **和前面分层一致** — articles / docs / chunks / index 各管一层；清洗错了检索救不了，索引坏了只重建索引即可  
4. **核心在 `app/kb/`** — 服务里可直接 `from app.kb import retrieve`，不必再跑脚本  

**这一步刻意没做的事（也是为什么）：**

- **没接 `/ask`**：先把检索质量看清楚，再拼 Prompt，避免「答得差」分不清是检索差还是模型差  
- **没用商业 Embedding**：优先可复现、零依赖；质量不够再换，接口已留好  
- **没用 Elasticsearch 关键词**：向量检索更能覆盖同义/近义，且和后续 RAG 主流路线一致  

一句话：**把「知识库」从静态文件，变成了「可按问题取 TopK 证据」的检索系统。**

### 6.5.7 直观例子

问：`Android ANR 怎么排查`

1. 问题变成 1024 维向量  
2. 和全部 chunk 向量比相似度  
3. 分数最高的几段往往是 ANR 系列里讲 traces / ActivityManager 的段落  
4. 你拿到 Top5 + score + 链接；下一步把这几段塞进 Prompt，让 LLM **基于材料**写排查步骤  

本机纯 Python 扫几百条，`retrieve_ms` 通常几十到一百毫秒量级；数据涨到十万级再考虑 FAISS / 向量库。

### 6.5.8 怎么操作

```bash
source .venv/bin/activate
# 建索引（读 chunks.jsonl → 写 data/stability_kb/index/）
python scripts/build_kb_index.py

# 检索冒烟
python scripts/retrieve_kb.py "Android ANR 怎么排查" --top-k 5
```

代码调用：

```python
from app.kb import retrieve

out = retrieve("OOM 内存泄漏怎么查", top_k=5)
# out["retrieve_ms"]
# out["results"][i]["chunk_id|score|title|url|section_path|text_snippet|is_code"]
```

### 6.5.9 自学对照练习

1. 打开 `index/manifest.json`，确认 `size` 与 `wc -l chunks.jsonl` 一致  
2. 跑一次 `retrieve_kb.py "WebView 白屏"`，看 Top1 的 `title/url/score`  
3. 用同一个 `chunk_id` 去 `chunks.jsonl` / `meta.jsonl` 里对照全文  
4. 换一个同义问法（如「页面一片白」），观察命中是否还合理 —— 体会当前 hashing 向量的边界  

### 6.5.10 如何验收（推荐：10 条 query）

#### 为什么用 10 条 query？

只测「ANR」一条，容易误判「碰巧能搜」。  
用 **10 条覆盖 A–F 主题**（卡顿/崩溃/内存/网络/定位/WebView/推送），能更快发现：

- 索引没建好 / 条数对不上  
- `retrieve` 字段或排序坏了  
- 某一类主题整体搜不准  

脚本：`scripts/verify_kb_retrieve.py`（问句列表写在脚本顶部的 `SMOKE_CASES`）。

#### 怎么跑

```bash
# 若还没建索引，先建
python scripts/build_kb_index.py

# Day8 验收：索引完整性 + retrieve 契约 + 10 query 主题冒烟
python scripts/verify_kb_retrieve.py

# 机器可读报告
python scripts/verify_kb_retrieve.py --json

# 主题冒烟不够稳时，先把主题失败降级为警告（索引/契约仍严格）
python scripts/verify_kb_retrieve.py --soft-warn
```

不依赖本机语料的单测（合成 3 条 chunk）：

```bash
pytest tests/test_kb_retrieve.py -q
```

#### 10 条 query 一览

| ID | Query | 期望类别 | 期望关键词（TopK 中至少命中一个） |
|----|--------|----------|----------------------------------|
| Q01 | Android ANR 怎么排查 | A | ANR / traces / 卡顿 / 无响应 |
| Q02 | 主线程卡顿怎么分析 | A | 卡顿 / 主线程 / ANR / 掉帧 |
| Q03 | Crash 堆栈怎么看 | B | Crash / 崩溃 / 堆栈 / 闪退 |
| Q04 | App 启动白屏原因 | B | 白屏 / 启动 / Splash |
| Q05 | OOM 内存泄漏怎么查 | C | OOM / 内存 / 泄漏 / Heap |
| Q06 | Bitmap 内存优化 | C | Bitmap / 内存 / OOM / 图片 |
| Q07 | 网络请求超时重试 | D | 网络 / 超时 / 重试 / HTTP / axios |
| Q08 | 定位不准或漂移怎么办 | E | 定位 / GPS / 经纬度 / 地图 |
| Q09 | WebView 白屏怎么办 | F | WebView / 白屏 / Hybrid / JSBridge |
| Q10 | 推送收不到消息怎么查 | G | 推送 / 消息 / Push / RocketMQ / MQTT |

> 说明：本机语料若没有 G 类（推送），Q10 仍做关键词冒烟；`category=G` 过滤检查会记 **WARN** 并自动跳过，不因此判整次失败。

#### 脚本检查什么

1. **索引完整性**  
   `manifest` / `meta.jsonl` / `vectors.jsonl` 存在；`meta` 行数 = `vectors` 行数 = `manifest.size`；并与 `chunks.jsonl` 行数一致。  
2. **retrieve 契约**（用 Q01 抽检）  
   外层有 `query / top_k / retrieve_ms / results`；命中含统一字段；`score` 降序；`retrieve_ms` 合法。  
3. **10 query 主题冒烟**  
   每条取 TopK（默认 5），在 `title / text_snippet / section_path / url` 里找期望关键词；有类语料时再测 `category` 过滤非空。  
4. **汇总**  
   终端最后应看到：`smoke.queries_summary: 10/10 queries 关键词命中`，且 `ok=True`、退出码 `0`。

成功时大致会看到：

```text
[PASS] smoke.Q01: q='Android ANR 怎么排查' ...
...
[PASS] smoke.Q10: q='推送收不到消息怎么查' ...
[WARN] smoke.Q10.category_filter: 索引无 category=G 语料，跳过过滤检查   # 可有可无
[PASS] smoke.queries_summary: 10/10 queries 关键词命中

summary: ok=True ...
验收通过：索引完整，retrieve 契约与主题冒烟 OK。
```

#### 人工核对清单

- [ ] `data/stability_kb/index/manifest.json` 存在，`size` 与 chunks 行数一致  
- [ ] `retrieve("ANR")` 返回 TopK，且 `score` 降序  
- [ ] 结果含统一字段；终端能看到 `retrieve_ms`  
- [ ] 能用大白话讲清：Embedding / 索引 / `retrieve` 各干什么、为什么要离线建索引  
- [ ] `python scripts/verify_kb_retrieve.py` 退出码为 0，且汇总为 **10/10**  

---

## 6.6 Day 9：接入 `/ask?mode=rag`（必须带 citations）

### 目标

把检索 TopK 拼成 Context，让 DeepSeek **基于引用**回答；响应里的 `citations[]` 必须填充。

```text
用户问题
  → retrieve TopK（含全文）
  → 拼 RAG prompt（只能基于 Context；不足则澄清/不知道；必须标 [1][2]）
  → LLMClient.chat
  → answer + citations[] + meta(retrieve_ms / context_chunks…)
```

### 核心模块

| 模块 | 作用 |
|------|------|
| `app/kb/rag.py` | 写死的 RAG system prompt、拼 Context、`hits_to_citations`、`run_rag_retrieve` |
| `app/api.py` | `/ask` 支持 `mode=llm\|rag`（query 优先于 body） |
| `app/services/metrics_store.py` | `requests.jsonl` 增加 RAG 最小字段 |

### RAG v1 Prompt 约束（写死）

1. 只能基于提供的 Context 回答  
2. Context 不足必须「澄清 / 不知道」  
3. 回答必须给出引用编号 `[1][2]...`  

### 怎么调用

```bash
# 先保证索引存在
python scripts/build_kb_index.py
./scripts/start_server.sh

# query 参数
curl -s 'http://127.0.0.1:8000/ask?mode=rag' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Android ANR 怎么排查","top_k":5}'

# 或 body.mode
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"OOM 内存泄漏怎么查","mode":"rag"}'
```

### 响应 citations 字段

| 字段 | 含义 |
|------|------|
| `ref_id` | 1..K，与 Context / 回答中的 `[n]` 对应 |
| `chunk_id` | 块 ID |
| `url` / `title` | 来源 |
| `section_path` | 章节路径 |
| `is_code` | 是否代码块 |

`meta` 额外：`mode`、`top_k`、`retrieve_ms`、`context_chunks`、`citations_count`。  
索引未建时返回 HTTP **503** / `INDEX_NOT_READY`。

### requests.jsonl 本周最小集（RAG）

在原有字段外增加：

| 字段 | 说明 |
|------|------|
| `mode` | `llm` / `rag` |
| `top_k` | 检索条数（llm 模式可无） |
| `retrieve_ms` | 检索耗时 |
| `context_chunks` | 实际塞进 Context 的条数 K |
| `citations_count` | citations 条数 |

### 如何验收

```bash
pytest tests/test_rag_ask.py tests/test_contract.py -q
```

- [ ] `mode=llm` 时 `citations == []`，行为与第一周一致  
- [ ] `mode=rag` 时 `citations` 非空且含 `ref_id..is_code`  
- [ ] `requests.jsonl` 能看到 `mode/top_k/retrieve_ms/context_chunks/citations_count`  
- [ ] 无索引时 `503 INDEX_NOT_READY`  

### 批量验收门槛（随机 20 query）

脚本：`scripts/eval_rag_ask.py`，样例池：`eval_rag_samples.jsonl`。

| 门槛 | 标准 |
|------|------|
| citations 非空 | 20 条里非空比例 **≥ 80%** |
| 代码定位 | 至少 **5** 条能引用到代码片段，且回答能对上关键片段 |
| 信息不足 | 固定 **3** 条「信息不足」问题能触发澄清/拒答（固定短语） |

```bash
# 服务已启动且已建索引
python scripts/eval_rag_ask.py --seed 42 --code-mode either
```

说明：

- 抽样：固定纳入 3 条 `insufficient`，其余随机；并优先塞入 `code_seeking` 问句。  
- **代码判定**：当前本地 index 的 `is_code=true` 可能为 0（`--no-refetch` 扁平正文无围栏）。  
  - `--code-mode is_code`：只认 `citations.is_code=true`（严格，现网语料常失败）  
  - `--code-mode heuristic`：正文像 `adb` / 堆栈 / 代码关键字也算  
  - `--code-mode either`（默认）：二者任一  
- 澄清判定短语含：`根据已有资料无法确定`、`信息不足`、`需要澄清`、`请补充` 等。  
- 报告：`reports/rag_eval_report_*.json` 与根目录 `rag_eval_report.json`。

### 如何确认「真的用了本地资料」

响应当场看：`meta.mode=rag`、`retrieve_ms`、`citations[].chunk_id`。  
落盘看：`requests.jsonl` 同 `request_id` 的 RAG 字段。  
对账脚本（把 curl 结果存成 JSON 再跑）：

```bash
curl -s 'http://127.0.0.1:8000/ask?mode=rag' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Android ANR 怎么排查","top_k":5}' \
  | tee /tmp/ask_rag.json >/dev/null

python scripts/verify_ask_rag.py --response /tmp/ask_rag.json
# 或只查指标：
python scripts/verify_ask_rag.py <request_id>
```

脚本会检查：citations 是否都在本地 `index/meta.jsonl`、url/title 是否一致、回答中的 `[n]` 是否合法，并与 `requests.jsonl` 交叉核对。

---

## 7. 第二周实际踩坑（务必记住）

### 7.1 百度不适合做主搜索通道

脚本访问百度常出「百度安全验证」页，几乎拿不到真实结果。  
实际主通道：**掘金 API + CSDN API + 必应中国补链**。

### 7.2 采集条数会远大于有效 docs

例如候选约 280，清洗后可能只剩 100+。常见原因：

- `seed_only`（没抓到正文）
- HTTP 404 / 超时 / 连接失败
- 正文太短或坏页
- 去重去掉重复文

这是**漏斗**，不是程序坏了。

### 7.3 `--limit` 调试会覆盖输出文件

`build_stability_docs.py --limit 2` 会把 `docs.jsonl` **整文件覆盖**成只含调试结果。  
正式数据请不要随便加 `--limit`；加了之后要用全量重新跑恢复。

### 7.4 后台长任务可能覆盖更好的结果

曾出现：联网重抓跑很久，大量 `PoolTimeout`，写出质量更差的 `docs.jsonl`，把之前较好的文件覆盖掉。  
经验：

- 长任务先小样本验证
- 重要产物可先复制备份：`cp docs.jsonl docs.jsonl.bak`
- 用户明确「不要再采」时，优先 `--no-refetch` 用本地数据

### 7.5 `--no-refetch` 与代码块质量

只用 `text_excerpt` 时，正文常常是「扁平纯文本」，可能没有漂亮的 Markdown 代码围栏，于是 `code_chunks` 会变少甚至为 0。  
若要高质量代码块，需要允许一次联网 `refetch`（更慢）。

### 7.6 国内优先 vs 国外文档

国外论坛可能要翻墙；第二周改为国内优先，更贴近日常可复现环境。  
`android.google.cn` / `firebase.google.cn` 这类中国镜像仍可算国内可访问官方文档。

---

## 8. 推荐学习路径（按顺序做）

### 第 1 步：把概念串起来（30 分钟）

读：

1. 本文第 1、2 节，以及第 **6.5** 节（Embedding / 索引 / 检索为什么这么做）  
2. `scripts/build_stability_docs.py`、`chunk_stability_docs.py` 顶部注释  
3. `scripts/build_kb_index.py`、`retrieve_kb.py`、`verify_kb_retrieve.py` 顶部注释  
4. `app/kb/cleaner.py`、`chunker.py`、`embedder.py`、`retriever.py` 模块说明

能回答：

- 为什么有 articles / docs / chunks / index 几层？  
- 薄 CLI 和 `app/kb` 各干什么？  
- 为什么要离线建索引，而不是每次提问现算全部 chunk？

### 第 2 步：看真实数据（30 分钟）

```bash
wc -l data/stability_kb/articles.jsonl \
      data/stability_kb/docs.jsonl \
      data/stability_kb/chunks.jsonl

cat data/stability_kb/docs_report.json
cat data/stability_kb/chunks_report.json
cat data/stability_kb/index/manifest.json   # 若已建索引
```

再做第 5.5 节「一篇 doc 对照多个 chunk」、第 6.5.9 节「检索命中对照」练习。

### 第 3 步：亲手跑通（1 小时）

在确认可以覆盖产物、或已备份的前提下：

```bash
cp data/stability_kb/docs.jsonl data/stability_kb/docs.jsonl.bak 2>/dev/null || true
cp data/stability_kb/chunks.jsonl data/stability_kb/chunks.jsonl.bak 2>/dev/null || true

python scripts/build_stability_docs.py --no-refetch
python scripts/chunk_stability_docs.py
python scripts/build_kb_index.py
python scripts/retrieve_kb.py "Android ANR 怎么排查" --top-k 5
```

看终端进度、report，以及检索 JSON 里的 `score` / `retrieve_ms`。

### 第 4 步：读核心函数（按兴趣深入）

清洗侧重点看：

- `clean_markdown`
- `build_doc_from_article`
- `build_docs_from_articles`
- `dedupe_key`

切块侧重点看：

- `parse_heading_sections`
- `split_by_fences` / `chunk_section_text`
- `split_text_with_overlap`
- `chunk_doc` / `chunk_docs`

检索侧重点看：

- `embed_texts` / `embed_query`
- `build_index` / `load_index`
- `retrieve`

---

## 9. Day 10：评测闭环（≥50）+ 单变量 A/B

旧版「Expo / iOS 权限 / 个推」题库不适合本仓库。  
本项目评测围绕 **稳定性知识库 A–G**（ANR / Crash / 内存 / 网络 / 定位 / WebView / 推送）。

### 目标

用指标证明「我改了什么，效果怎么变」，避免纯主观。

### 样例：`eval_samples_rag.jsonl`（≥50）

| tag | 条数 | 含义 |
|-----|------|------|
| `normal` | 30 | 知识库应能直接回答 |
| `insufficient` | 10 | 缺平台/日志/上下文 → 应澄清或拒答 |
| `sensitive` | 10 | 隐私绕过 / 违规 → 应合规拒答或安全提示 |

> 第一周的 `eval_samples.jsonl` 仍用于 Day5 `/ask`（mode=llm）契约回归，不要混用。

### 指标最小集（`eval_report.json`）

| 字段 | 含义 |
|------|------|
| `citation_coverage` | citations 非空比例 |
| `insufficient_handling_rate` | insufficient 是否澄清/拒答（固定短语） |
| `sensitive_handling_rate` | sensitive 是否合规拒答/安全提示 |
| `latency_ms_total.p50/p95` | 端到端耗时 |
| `retrieve_ms.p50/p95` | 检索耗时 |
| `top_errors` | 错误码 TopN |

### 怎么跑

```bash
./scripts/start_server.sh

# 全量 ≥50
python scripts/run_rag_eval.py

# 调试只跑前 5 条
python scripts/run_rag_eval.py --limit 5
```

### 单变量 A/B（一次只改一个）

**优先推荐：`top_k` 3 → 5**（无需重建索引）：

```bash
python scripts/run_rag_eval.py --ab-var top_k --a-value 3 --b-value 5
# 产出 reports/eval_report_A_*.json、B_*.json、eval_ab_compare.json
```

**备选：`chunk_size` 800 → 1200**（需重建索引并切换 `KB_INDEX_DIR` 重启服务）：

```bash
python scripts/run_rag_eval.py --prepare-chunk-ab --a-value 800 --b-value 1200
python scripts/run_rag_eval.py --ab-var chunk_size --a-value 800 --b-value 1200
# 交互提示下分别用 KB_INDEX_DIR=.../index_ab_800 与 index_ab_1200 重启后再评
```

对比已有报告：

```bash
python scripts/run_rag_eval.py --compare reports/eval_report_A_xxx.json reports/eval_report_B_xxx.json
```

看 `deltas`：`citation_coverage` / handling **越高越好**；latency **越低越好**。

### 与 Day9 小验收的关系

- `eval_rag_ask.py`：快速 20 条冒烟（含代码定位门槛）  
- `run_rag_eval.py`：正式闭环 ≥50 + A/B 指标  

---

## 10. 常用命令速查

```bash
# 环境
source .venv/bin/activate

# 采集（国内优先）
python scripts/crawl_stability_kb.py --domestic-only --target 15

# 清洗（不联网）
python scripts/build_stability_docs.py --no-refetch

# 切块
python scripts/chunk_stability_docs.py --chunk-size 1000 --overlap 120

# 建索引 + 检索
python scripts/build_kb_index.py
python scripts/retrieve_kb.py "Android ANR 怎么排查" --top-k 5
python scripts/verify_kb_retrieve.py

# RAG 响应对账（先 curl 存到 /tmp/ask_rag.json）
python scripts/verify_ask_rag.py --response /tmp/ask_rag.json

# Day10 评测闭环（≥50；调试可加 --limit 5）
python scripts/run_rag_eval.py
# python scripts/run_rag_eval.py --ab-var top_k --a-value 3 --b-value 5

# 一键 Day6–7（默认不扩采）
EXPAND=0 ./scripts/run_day6_7.sh

# 统计行数
wc -l data/stability_kb/*.jsonl
```

Python API：

```python
from app.kb import (
    load_jsonl,
    write_jsonl,
    build_docs_from_articles,
    chunk_docs,
    retrieve,
)
```

---

## 11. 第二周学习自检表

- [ ] 能用大白话讲清 articles / docs / chunks 的区别  
- [ ] 知道清洗、切块的核心代码在 `app/kb/`，scripts 只是薄 CLI  
- [ ] 会跑 `build_stability_docs.py` 与 `chunk_stability_docs.py`  
- [ ] 会跑 `build_kb_index.py` / `retrieve_kb.py`，理解 score 与 `retrieve_ms`  
- [ ] 会跑 `verify_kb_retrieve.py`，看到 **10/10** query 冒烟通过  
- [ ] 会调 `/ask?mode=rag`，看到非空 `citations` 与 `meta.retrieve_ms`  
- [ ] 会用 `verify_ask_rag.py --response` 把 citations 对上本地 index  
- [ ] 会跑 `eval_rag_ask.py`，看 citations≥80% / 代码定位≥5 / 澄清 3/3  
- [ ] 会跑 `run_rag_eval.py`（≥50）并看懂 `eval_report.json` / A/B `deltas`  
- [ ] 会用 `tee` + `trace_request.py` 读懂带 `hint` 的请求日志  
- [ ] 能讲清：为何要 Embedding、为何离线建索引、`retrieve` 返回字段各自给谁用  
- [ ] 会看 `docs_report.json` / `chunks_report.json` / `index/manifest.json`  
- [ ] 做过「一篇 doc ↔ 多个 chunk」对照  
- [ ] 知道为何候选很多但 docs 变少  
- [ ] 知道 `--limit` / 长任务覆盖文件的风险  
- [ ] 知道国内优先策略与百度验证码问题  

---

## 12. 变更记录（本周关键决策）

1. 知识库采集改为 **国内优先**（掘金/CSDN/必应中国）。  
2. Day6/Day7 核心逻辑下沉到 **`app/kb/`**，scripts 改为薄 CLI。  
3. 文件头注释用大白话写清「点餐员 vs 后厨」。  
4. 明确停止无意义扩采：现有数据够用就进入清洗/切块与后续检索设计。  
5. Day10 评测改为稳定性 A–G 题库（`eval_samples_rag.jsonl`），支持 `top_k` / `chunk_size` 单变量 A/B。  

---

文档维护：随实现变更继续往本文追加「操作步骤 + 命令 + 踩坑」。
