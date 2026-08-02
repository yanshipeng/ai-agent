# AI Start Agent

基于 **FastAPI + DeepSeek**（OpenAI 兼容）的稳定性知识库问答服务：
支持 **LLM / RAG / Agent** 三模式，并具备鉴权、限流、成本路由、评测与反馈闭环。
定位为可跑通的教学/交付型工程（Week1–5 / Day1–25），不是生产级向量检索平台。

## 已实现功能总览

### Week1 · 可问答、可观测、可回归

- FastAPI：`GET /health`、`POST /ask`（默认 `mode=llm`）
- DeepSeek 客户端：重试、上游错误映射、失败兜底
- JSON 结构化日志 + `requests.jsonl` 指标落盘
- 契约测试 + `eval/eval_samples.jsonl` 回归

### Week2 · 本地知识库 + RAG

- 流水线：采集 → 清洗 → 切块 → 建索引 → 检索
- `POST /ask?mode=rag`：检索上下文回答 + `citations`（无索引 → `503 INDEX_NOT_READY`）
- RAG 评测样例与报告脚本（`run_rag_eval.py`）

### Week3 · Agent Tools + 多轮 Session

- 真实 tool_calls：`kb_search` / `kb_get_chunk`（白名单、超时、可控错误码）
- Agent 状态机：Plan → Act → Observe → Final；超步可降级 RAG/澄清
- 进程内 `session_id` 多轮：滑窗 / 截断 / 可选摘要
- Agent 评测样例与 smoke（`run_agent_eval.py`）

### Week4 · 检索质量 / 安全 / 可观测

- Day16：向量 + BM25 混合检索、去重过滤、检索可观测字段
- Day17：注入预检拒答、引用一致性门禁、泄密扫描
- Day18：`agent_trace` + `traces.jsonl`、Token 预算压缩、Cache 命中计数

### Week5 · 评测加深 + 产品化闭环（无单独 Week6）

- Day19：评测 v2（任务/澄清/安全，`eval/eval_samples_v2.jsonl` ≥80）
- Day20：动态 TopK、chunk merge、flash/pro 成本路由
- Day21：API Contract v2；推荐 `/v1/ask` 等
- Day22：API Key + tenant/user/role 审计与 RBAC
- Day23：增量入库 / 版本化 / 回滚（`/v1/ingest`、`/v1/dataset`）
- Day24：按 tenant 限流（429）+ 单次 token 预算 + 含糊短问澄清
- Day25：用户反馈 → badcase → 日常回归趋势（`/v1/feedback`）

### 当前 API 一览

推荐优先用 `/v1/*`（字段更完整）；`/ask`、`/health` 是旧入口，逻辑基本一样。

| 路径 | 谁能调 | 干什么 | 常用参数 / 说明 |
|------|--------|--------|-----------------|
| `GET /health`、`GET /v1/health` | 任何人 | 探活：服务有没有起来 | 返回 `status` + `version`；**不走鉴权** |
| `POST /ask`、`POST /v1/ask` | 鉴权开启后需 Key | 主问答入口 | `query` 必填；`mode=llm\|rag\|agent`；可选 `session_id`、`top_k` |
| `GET /v1/dataset` | 鉴权开启后需 Key | 看当前知识库版本号 | 确认 ingest / 回滚有没有生效 |
| `POST /v1/ingest` | **admin** | 增量入库 / 回滚 / 重建 chunks | `action=ingest\|rollback\|rebuild_chunks` |
| `POST /v1/eval/run` | **admin** | 批量跑评测样例 | 默认 `offline=true`（不打真实 LLM，先练判定流水线） |
| `POST /v1/feedback` | 鉴权开启后需 Key | 给答案打标 | `useful` / `useless` / `wrong_citation` / `hallucination` |

`/ask` 三种 mode（新手先记这个）：

| mode | 白话理解 | 你会看到什么 |
|------|----------|--------------|
| `llm`（默认） | 直接问大模型，不查本地库 | `citations` 为空 |
| `rag` | 先从知识库检索，再带着资料回答 | 有 `citations`；没建索引会 `503` |
| `agent` | 模型自己决定要不要调用检索工具 | `meta` 里有 `agent_steps` / `agent_trace` |

## 能力边界（勿过称）——新手必读

这些不是「功能没写」，而是**教学项目故意做成简化版**。  
可以说「整条链路都打通了」；不要说「已经是生产级向量检索 / 多机系统」。

### 1. 检索：本地 hashing，不是云端语义模型

- **做了什么**：把文本哈希成向量，再和 BM25（关键词）混在一起排序。
- **适合**：术语重合高的问题，比如「Android ANR 怎么排查」。
- **不适合**：同义改写很强的语义检索（那是 BGE / OpenAI Embedding 的活）。
- **记住**：能跑通 RAG；**别拿效果跟商业向量库比**。

### 2. Session / 限流 / Cache：都在「当前进程内存」

- **做了什么**：多轮对话、限流计数、检索缓存都存在这个 Python 进程里。
- **影响**：重启服务 → 对话没了；开多个 worker → 彼此不共享。
- **记住**：单机演示够用；**不是 Redis 那种多机共享**。

### 3. 鉴权、限流：学习默认是关的

- **代码与 `.env.example` 默认**：`API_AUTH_ENABLED=false`、`RATE_LIMIT_ENABLED=false`（方便先跑通）。
- **你要演示鉴权/限流时**：在 `.env` 改成 `true`，并配置 `API_KEYS` / `RATE_LIMIT_RPM`。
- **记住**：上线前务必打开鉴权；学习阶段关着更省事。

### 4. Agent 工具：目前只有查知识库

- **真正能调**：`kb_search`、`kb_get_chunk`。
- **HITL**：代码预留了「需人工批准」错误码，但 shell/SQL **没有**挂成可用工具。
- **记住**：这是「会查 KB 的 Agent」；**不是通用运维机器人**。

### 5. Token 预算 / 模型路由：靠规则，不是机器学习

- Token：用字符长度等启发式估算，不是精确 tokenizer。
- 路由：按问题长短、检索分数在 flash/pro、TopK 间切换。
- **记住**：能演示成本权衡；**别说成自适应学习系统**。

### 6. 评测接口默认 offline

- 默认不调用真实大模型，快速出报告、省钱。
- 要测真实效果：`offline=false` + 服务已启动 + 有索引 + 有 Key。
- **记住**：默认是在练「评分流水线」，不是压测线上回答质量。

**怎么对外讲**

- ✅ 适合：从问答 → RAG → Agent → 鉴权/反馈，闭环都走通了  
- ❌ 别承诺：多机部署、强语义检索、完整 IAM、自动运维 Agent、计费级 token 统计  

**按周学习文档（详细操作与踩坑）：**

先看导航 [`docs/README.md`](docs/README.md)（任务节奏、鉴权提示、最小成功路径）。

| 文档 | 内容 |
|------|------|
| [`docs/README.md`](docs/README.md) | 新手总入口：读什么、怎么跟做、出事怎么退 |
| [`docs/week1_学习指南.md`](docs/week1_学习指南.md) | Day1–5：起服务、LLMClient、错误处理、日志、契约与回归 |
| [`docs/week2_学习指南.md`](docs/week2_学习指南.md) | Day6–10：采集 → 清洗 → 切块 → 索引/检索 → RAG → 评测 |
| [`docs/week3_学习指南.md`](docs/week3_学习指南.md) | Tools / Agent 状态机 / 多轮 session / 评测 |
| [`docs/week4_学习指南.md`](docs/week4_学习指南.md) | 检索质量 v2 / 安全抗注入 / 可观测 v2 |
| [`docs/week5_学习指南.md`](docs/week5_学习指南.md) | Day19–25（评测 v2 → 产品化闭环；无单独 Week6） |
| [`docs/日志阅读指南.md`](docs/日志阅读指南.md) | 怎么读服务日志（含 `hint`） |

---

## 1. 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 编辑 .env，填入 DEEPSEEK_API_KEY

./scripts/start_server.sh
```

| 地址 | 说明 |
|------|------|
| http://127.0.0.1:8000/v1/health | 探活（推荐） |
| http://127.0.0.1:8000/v1/ask | 问答（推荐，`mode=llm\|rag\|agent`） |
| http://127.0.0.1:8000/v1/ingest | 入库（docs 增量 / 回滚 / chunks 重建） |
| http://127.0.0.1:8000/v1/eval/run | 评测批跑（默认 offline） |
| http://127.0.0.1:8000/v1/feedback | 用户反馈（有用/无用/引用错误/幻觉） |
| http://127.0.0.1:8000/docs | Swagger 交互文档（浏览器点着试最友好） |
| http://127.0.0.1:8000/ask | 旧问答入口（仍可用） |

改端口：`PORT=8001 ./scripts/start_server.sh`。停止：`Ctrl+C`。

> **注意**：环境变量优先级是「shell 里 export 的 > `.env`」。若你之前 export 过 Key，可能盖住 `.env`。`start_server.sh` 会尽量以 `.env` 为准。

```bash
# 探活
curl -s http://127.0.0.1:8000/health

# 直连 LLM
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"用一句话解释什么是 API"}'

# RAG（需先建索引，见 §5）
curl -s 'http://127.0.0.1:8000/ask?mode=rag' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Android ANR 怎么排查","top_k":5}'

# Agent（真实 tool_calls；需索引）
curl -s 'http://127.0.0.1:8000/ask?mode=agent' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Android ANR 怎么排查"}'
```

> **Windows PowerShell**：`-d "{\"query\":...}"` 容易被壳弄坏 JSON。请写成文件后：  
> `curl.exe -s -X POST "http://127.0.0.1:8000/ask?mode=agent" -H "Content-Type: application/json" --data-binary "@ask.json"`

---

## 2. 项目结构（新手导读）

可以把它想成四层：

1. **`app/`**：真正提供 HTTP 服务的主程序（学习主线，优先读这里）
2. **`scripts/` / `tests/` / `eval/`**：帮你建库、冒烟、写样例、做断言（辅助，不是服务本体）
3. **`data/` / `reports/`**：运行后产生的数据与报告（一般不入库）
4. **`docs/`**：按周讲怎么做、怎么验收

```text
ai_start_anget/
├── app/                    # 【主程序】HTTP 服务 + 业务能力（学习主线）
│   ├── main.py             # 应用入口：创建 FastAPI、挂中间件、生命周期
│   ├── api.py              # 兼容路由：/health、/ask（三模式 + 安全门禁）
│   ├── api_v1.py           # 版本化路由：/v1/ask|ingest|eval|feedback|dataset
│   ├── core/               # 配置 / 日志 / 鉴权 / 安全（横切能力）
│   ├── services/           # LLM、会话、指标、路由、限流等业务服务
│   ├── kb/                 # 知识库：清洗→切块→索引→检索→RAG
│   └── agent/              # Agent：工具定义 + Tool Runner 状态机
│
├── scripts/                # 【运维脚本】建库 / 冒烟 / 评测 CLI（非主程序）
├── tests/                  # 【自动化测试】pytest，默认 mock，不打真实 Key
├── eval/                   # 【评测输入】可入库的 jsonl 样例（非运行产物）
├── reports/                # 【评测输出】报告/明细（gitignore，本地生成）
├── docs/                   # 【学习文档】week1–5 指南 + 日志阅读
├── data/
│   ├── stability_kb/       # 【知识库数据】语料、chunks、index、versions
│   ├── runtime/            # 【运行指标】requests.jsonl / traces.jsonl
│   └── feedback/           # 【反馈落盘】feedback / badcases（gitignore）
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板（复制为 .env）
├── pytest.ini              # pytest 配置（testpaths=tests）
└── README.md               # 本文件
```

### 2.1 `app/` — 主程序（先读这里）

一次请求大致这样走：

```text
浏览器/curl
  → main.py（鉴权、限流中间件）
  → api.py 或 api_v1.py（路由）
  → services/（调 LLM、记指标、管 session…）
  → kb/ 或 agent/（检索或工具循环）
  → 返回 JSON + 写日志/指标
```

| 文件 | 新手理解 |
|------|----------|
| `main.py` | **大门**：创建应用、挂中间件、启动时准备 `LLMClient` |
| `api.py` | **旧版柜台**：`/health`、`/ask`，三模式问答都在这实现 |
| `api_v1.py` | **新版柜台**：`/v1/*`，多了入库、评测、反馈等产品化接口 |

#### `app/core/` — 到处都会用到的基础能力

| 文件 | 新手理解 |
|------|----------|
| `config.py` | 读 `.env`：模型名、索引路径、要不要鉴权…… |
| `logging.py` | 打 JSON 日志；敏感字段脱敏；每条请求有 `request_id` |
| `auth.py` | 校验 API Key；区分 admin / reader |
| `audit_context.py` | 把 tenant/user/role 挂到当前请求上下文，方便写指标 |
| `safety.py` | 防提示注入、检查引用是否对得上、扫泄密 |

#### `app/services/` — 「做事」的服务模块

| 文件 | 新手理解 |
|------|----------|
| `llm_client.py` | 跟 DeepSeek 说话：重试、错误码、工具调用 |
| `metrics_store.py` | 把每次请求结果追加进 `requests.jsonl` / `traces.jsonl` |
| `session_store.py` | 记住多轮对话（内存里，重启会丢） |
| `conversation.py` | 历史太长时截断、滑窗、可选摘要 |
| `token_budget.py` | Agent 上下文太长时压缩，避免爆预算 |
| `cost_routing.py` | 简单问题少检索/用 flash；难问题可升 pro |
| `api_contract_v2.py` | 统一成功响应里 `meta` 必有哪些字段 |
| `eval_v2_service.py` | 批量评测：读样例 → 判定 → 出报告 |
| `rate_limit.py` | 谁调用太勤就返回 429 |
| `request_budget.py` | 单次请求的 token/TopK 预算；太含糊就先澄清 |
| `feedback_store.py` | 用户反馈落盘；差评可进 badcase 队列 |

#### `app/kb/` — 知识库从原文到能检索

可以按流水线理解：

```text
网上文章 → cleaner 清洗 → chunker 切块
  → embedder 变向量 → index_store 存盘
  → retriever 混合检索 → rag 拼 prompt + citations
```

| 文件 | 新手理解 |
|------|----------|
| `cleaner.py` | 去噪、去重，得到干净文档 |
| `chunker.py` | 切成小段，方便检索 |
| `embedder.py` | 文本 → 向量（教学用 hashing） |
| `index_store.py` | 读写本地索引文件 |
| `bm25.py` | 关键词检索 |
| `retriever.py` | 向量 + BM25 混排，并去重 |
| `rag.py` | 把检索结果塞进提示词，生成带引用的回答结构 |
| `ingest_pipeline.py` | 增量入库（不用每次全量重建） |
| `dataset_registry.py` | 记录当前知识库版本，支持回滚 |
| `jsonl_io.py` / `cli_log.py` | 读写 JSONL、CLI 日志小工具 |

#### `app/agent/` — 让模型「自己决定要不要查库」

| 文件 | 新手理解 |
|------|----------|
| `tools.py` | 声明工具：`kb_search`（搜）、`kb_get_chunk`（取一段全文） |
| `runner.py` | 循环：计划 → 调工具 → 观察结果 → 再问模型；步数太多可降级成普通 RAG |

### 2.2 `eval/` — 评测「考卷」（输入）

这里是**题目**，不是成绩单。文件可进 git，方便大家跑同一套样例。

| 文件 | 对应阶段 |
|------|----------|
| `eval_samples.jsonl` | Week1：先测直连 LLM |
| `eval_samples_rag.jsonl` / `eval_rag_samples.jsonl` | Week2：测 RAG |
| `agent_eval_samples.jsonl` | Week3：测 Agent |
| `eval_samples_injection.jsonl` | Week4：测注入防护 |
| `eval_samples_v2.jsonl` | Week5：综合评测 |
| `eval_samples_feedback.jsonl` | Week5：反馈转化的回归题 |

### 2.3 `data/` — 本地数据（多数不进 git）

| 路径 | 里面是什么 |
|------|------------|
| `data/stability_kb/` | 语料、切块、索引、版本目录（RAG 的「书库」） |
| `data/runtime/` | 服务跑起来后的 `requests.jsonl`、`traces.jsonl` |
| `data/feedback/` | 用户反馈、pending badcase |

### 2.4 `reports/` — 评测「成绩单」（输出）

跑脚本后生成，例如 `eval_v2_report.json`、`injection_eval_report.json`。  
一般 gitignore，避免把本机报告提交上去。

### 2.5 `docs/` — 按周说明书

| 文件 | 适合什么时候看 |
|------|----------------|
| `week1` … `week5_学习指南.md` | 跟着做、对照验收清单 |
| `日志阅读指南.md` | 服务日志看不懂时 |

### 2.6 `scripts/` — 命令行帮手（不是服务本身）

新手常见顺序：

1. `start_server.sh` 起服务  
2. Week2 用建库脚本把索引准备好  
3. 用 `smoke_*` / `run_*_eval.py` 做验收  
4. 用 `stats_requests.py` 看指标汇总  

| 分类 | 脚本 | 干什么 |
|------|------|--------|
| **运维** | `start_server.sh` | 启动服务 |
| | `stats_requests.py` | 汇总请求指标 |
| | `trace_request.py` | 按 `request_id` 回放日志事件 |
| **建知识库** | `crawl_stability_kb.py` → `build_stability_docs.py` → `chunk_stability_docs.py` → `build_kb_index.py` | 采集到建索引 |
| | `retrieve_kb.py` / `verify_kb_retrieve.py` | 不启服务也能试检索 |
| **冒烟/评测** | `smoke_*.py`、`run_eval.py`、`run_rag_eval.py`、`run_agent_eval.py` … | 对着真实服务或本机索引验收 |
| **Week4/5** | `run_injection_eval.py`、`run_eval_v2.py`、`run_day20_demo.py`、`run_regression_daily.py` 等 | 安全、成本、回归闭环 |

### 2.7 `tests/` — 自动化考试（默认不花 API 钱）

```bash
pytest -q
```

- **`tests/`**：用 mock，适合每次改代码都跑，CI 友好  
- **`scripts/`**：往往要真实服务/索引/Key，用来出报告、留验收痕迹  

测试文件名大致对应能力：`test_rag_ask`、`test_agent_tools`、`test_auth_day22`、`test_feedback_day25` 等，按 Day/周查阅即可。

---

## 3. 环境变量（先会这几项就够）

把 `.env.example` 复制成 `.env` 后，**新手最少改这一项**：

| 变量 | 必填 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | **是** | 没有 Key，问答会失败 |

其它常用项（有默认值，需要再改）：

| 变量 | 默认值 | 什么时候改 |
|------|--------|------------|
| `LLM_MODEL` / `LLM_MODEL_PRO` | flash / pro | Day20 成本路由 |
| `KB_INDEX_DIR` | `data/stability_kb/index` | 索引不在默认路径时 |
| `REQUESTS_JSONL_PATH` / `TRACES_JSONL_PATH` | `data/runtime/...` | 想换指标落盘位置 |
| `API_AUTH_ENABLED` / `API_KEYS` | `false` / 空 | 要演示鉴权时打开 |
| `RATE_LIMIT_ENABLED` / `RATE_LIMIT_RPM` | `false` / `60` | 要演示限流时打开 |
| `AGENT_MAX_STEPS` 等 | 见下表完整列表 | 调 Agent 行为 |
| `SESSION_*` | 见下表 | 调多轮上下文长短 |

### 完整变量表

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | - | DeepSeek API Key |
| `LLM_BASE_URL` | 否 | `https://api.deepseek.com` | API Base URL |
| `LLM_MODEL` | 否 | `deepseek-v4-flash` | 默认模型（Day20 flash） |
| `LLM_MODEL_PRO` | 否 | `deepseek-v4-pro` | Day20 高质量路由目标 |
| `LLM_TIMEOUT_SECONDS` | 否 | `30` | 超时（秒） |
| `LLM_MAX_TOKENS` | 否 | `2048` | 最大生成 token |
| `LLM_TEMPERATURE` | 否 | `0.2` | 温度 |
| `LLM_THINKING` | 否 | `disabled` | `disabled` / `enabled` |
| `REQUESTS_JSONL_PATH` | 否 | `./data/runtime/requests.jsonl` | 请求指标路径 |
| `TRACES_JSONL_PATH` | 否 | `./data/runtime/traces.jsonl` | Agent 逐步 Trace |
| `APP_VERSION` | 否 | `0.1.0ba` | `/health` 版本 |
| `KB_INDEX_DIR` | 否 | `data/stability_kb/index` | RAG 索引目录 |
| `RAG_TOP_K` | 否 | `5` | 默认检索条数 |
| `RAG_HYBRID_WEIGHT` | 否 | `0.6` | 混合检索里向量占比 |
| `RAG_MIN_SCORE` | 否 | `0.05` | 融合分低于此丢弃 |
| `RAG_ROUTE_PRO_MIN_SCORE` | 否 | `0.35` | Top1 低于此倾向走 pro |
| `API_AUTH_ENABLED` | 否 | `false` | 开启后除探活/docs 外要带 Key |
| `API_KEYS` | 否 | - | `key:role` 逗号分隔（admin/reader） |
| `KB_VERSIONS_DIR` | 否 | `data/stability_kb/versions` | 版本化索引根目录 |
| `KB_DOCS_PATH` | 否 | `data/stability_kb/docs.jsonl` | 入库 docs 路径 |
| `RATE_LIMIT_ENABLED` | 否 | `false` | 按 tenant/api_key 限流 |
| `RATE_LIMIT_RPM` | 否 | `60` | 窗口内最大请求数 |
| `REQUEST_TOKEN_BUDGET` | 否 | `1024` | 单次 completion token 硬顶 |
| `FEEDBACK_JSONL_PATH` | 否 | `data/feedback/feedback.jsonl` | 反馈落盘 |
| `AGENT_MAX_STEPS` | 否 | `5` | Agent 最大轮数 |
| `AGENT_MAX_TOTAL_TIME_MS` | 否 | `20000` | Agent 整单耗时上限 |
| `AGENT_ON_MAX_STEPS` | 否 | `rag` | 超步：`rag` / `clarify` / `error` |
| `AGENT_TOOL_TIMEOUT_SECONDS` | 否 | `10` | 单次工具超时 |
| `AGENT_MAX_CONTEXT_TOKENS` | 否 | `6000` | Agent context 预算 |
| `SESSION_MAX_TURNS` | 否 | `8` | 多轮保留最近 N 轮 |
| `SESSION_MAX_CHARS` | 否 | `20000` | 上下文字符预算 |
| `SESSION_TOOL_RESULT_MAX_CHARS` | 否 | `4000` | 单条 tool 结果截断 |
| `SESSION_CONTENT_MAX_CHARS` | 否 | `8000` | 单条消息截断 |
| `SESSION_ENABLE_SUMMARY` | 否 | `true` | 超预算是否摘要 |
| `SESSION_SUMMARY_USE_LLM` | 否 | `false` | `false`=抽取式；`true`=调 LLM |
| `SESSION_TTL_SECONDS` | 否 | `3600` | session 过期秒数 |

---

## 4. 接口怎么用（动手版）

### 4.1 先探活

```bash
curl -s http://127.0.0.1:8000/v1/health
```

期望类似：

```json
{"status": "ok", "version": "0.1.0ba"}
```

### 4.2 问答 `POST /ask` 或 `/v1/ask`

| 字段 | 必填 | 说明 |
|------|------|------|
| `query` | 是 | 问题文本；最大 2000 字符 |
| `mode` | 否 | `llm` / `rag` / `agent`；也可用 URL `?mode=` |
| `top_k` | 否 | 检索条数（rag/agent 有用） |
| `session_id` | 否 | 同一 id 会带上历史（进程内） |

三种模式再记一遍：

- **llm**：最快试通 Key 和网络  
- **rag**：要先有索引；回答应带引用  
- **agent**：模型可能先搜再答；看 `meta.agent_trace` 最有感觉  

统一错误体：`{"request_id","code","message"}`。  
常见码：`INDEX_NOT_READY`、`UNAUTHORIZED`、`FORBIDDEN`、`RATE_LIMITED`，以及工具侧 `TOOL_*`。细节见各周指南。

### 4.3 产品化接口（Week5）

| 路径 | 你在练什么 |
|------|------------|
| `/v1/ingest` | 知识库更新与回滚 |
| `/v1/eval/run` | 批跑评测（默认 offline） |
| `/v1/feedback` | 用户反馈 → badcase |
| `/v1/dataset` | 看当前数据集版本 |

浏览器打开 http://127.0.0.1:8000/docs 点「Try it out」通常比手写 curl 更省事。

---

## 5. 知识库 / RAG（最短路径）

没有索引时，`mode=rag/agent` 会失败或降级。最短建库：

```bash
python scripts/crawl_stability_kb.py --domestic-only --target 15
python scripts/build_stability_docs.py --no-refetch
python scripts/chunk_stability_docs.py
python scripts/build_kb_index.py
python scripts/verify_kb_retrieve.py
python scripts/retrieve_kb.py "Android ANR 怎么排查"
# 再启动服务，试 mode=rag
```

产物都在 `data/stability_kb/`：`articles` → `docs` → `chunks` → `index/`。  
更细步骤见 [Week2](docs/week2_学习指南.md)，混合检索见 [Week4 Day16](docs/week4_学习指南.md)。

---

## 6. Week4 工程化速查

```bash
# Day16：混合检索抽检
python scripts/spotcheck_retrieve_day16.py
python scripts/stats_requests.py --mode rag

# Day17：安全 / 注入
pytest -q tests/test_safety.py tests/test_injection_eval.py
python scripts/run_injection_eval.py --offline

# Day18：Trace / Budget / Cache（需服务 + 新的 agent 请求）
pytest -q tests/test_observability_day18.py
python scripts/stats_requests.py --mode agent
# 逐步轨迹：data/runtime/traces.jsonl
```

---

## 7. Week5 速查（Day19–25）

原先口头说的「Week6 产品化」已并进 Day21–25，没有单独第六周实现。

```bash
# Day19–20：评测 v2 + 成本路由 Demo
pytest -q tests/test_eval_v2.py tests/test_cost_routing_day20.py
python scripts/run_eval_v2.py --offline
python scripts/run_day20_demo.py              # 需服务

# Day21–23：契约 / 鉴权 / 增量入库
pytest -q tests/test_api_contract_v2.py tests/test_auth_day22.py tests/test_ingest_day23.py
python scripts/run_api_modes_batch.py         # 需服务
python scripts/run_day23_ingest_demo.py

# Day24–25：限流预算 / 反馈回归
pytest -q tests/test_rate_budget_day24.py tests/test_feedback_day25.py
python scripts/run_rate_limit_smoke.py --n 20 # 需 RATE_LIMIT_ENABLED=true
python scripts/run_regression_daily.py --repeat 3
```

成功响应的 `meta` 建议具备（Contract v2）：`model` / `mode` / `latency` / `finish_type` / `tool_calls_count`。

---

## 8. 常用命令速查

```bash
# 起服务 / 一键 Week1 回归
./scripts/start_server.sh
./scripts/run_day5.sh

# 看指标 / 回放某次请求
python scripts/stats_requests.py --path ./data/runtime/requests.jsonl
python scripts/trace_request.py <request_id> --log /tmp/app.log

# 各阶段评测（样例在 eval/，报告进 reports/）
python scripts/run_eval.py --samples ./eval/eval_samples.jsonl
python scripts/run_rag_eval.py
python scripts/run_agent_eval.py
python scripts/run_injection_eval.py --offline
python scripts/run_eval_v2.py --offline
python scripts/run_regression_daily.py --repeat 3

# 不花 HTTP 层、直接冒烟 LLM
python scripts/smoke_llm_client.py

# 自动化测试（推荐改代码后先跑）
pytest -q
```

**新手建议路径**：§1 起服务 → §5 建索引 → 浏览器 `/docs` 试三种 mode → `pytest -q` → 再按 `docs/week*_学习指南.md` 往下挖。
