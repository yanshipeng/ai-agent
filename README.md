# AI Start Agent

基于 **FastAPI + DeepSeek**（OpenAI 兼容 SDK）的问答服务，并带本地稳定性知识库的 **RAG** 能力。

| 能力 | 说明 |
|------|------|
| 探活 / 问答 | `GET /health`、`POST /ask` |
| 直连 LLM | `mode=llm`（默认）：错误映射、重试、兜底 |
| RAG | `mode=rag`：本地索引检索 → 带 citations 回答 |
| 可观测 | JSON 结构化日志、`requests.jsonl`、按 `request_id` 追踪 |
| 评测 | 契约/映射单测、Week1 回归、Week2 RAG 评测与 A/B |

**按周学习文档（详细操作与踩坑）：**

| 文档 | 内容 |
|------|------|
| [`docs/week1_学习指南.md`](docs/week1_学习指南.md) | Day1–5：起服务、LLMClient、错误处理、日志、契约与回归 |
| [`docs/week2_学习指南.md`](docs/week2_学习指南.md) | Day6–10：采集 → 清洗 → 切块 → 索引/检索 → RAG → 评测 |
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
| http://127.0.0.1:8000/health | 探活 |
| http://127.0.0.1:8000/docs | Swagger |
| http://127.0.0.1:8000/ask | 问答（默认 `mode=llm`） |

改端口：`PORT=8001 ./scripts/start_server.sh`。停止：`Ctrl+C`。

> **注意**：`pydantic-settings` 默认「环境变量 > `.env`」。若 shell 里已 `export DEEPSEEK_API_KEY`，会盖住 `.env`。`start_server.sh` 会强制以 `.env` 为准。

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
```

---

## 2. 项目结构

```text
ai_start_anget/
├── app/                              # 应用主代码
│   ├── main.py                       # FastAPI 创建、生命周期、uvicorn 入口
│   ├── api.py                        # 路由：GET /health、POST /ask
│   ├── core/
│   │   ├── config.py                 # 环境变量 / .env
│   │   └── logging.py                # JSON 结构化日志、脱敏
│   ├── services/
│   │   ├── llm_client.py             # DeepSeek 调用、重试、兜底
│   │   └── metrics_store.py          # 请求结束写入 requests.jsonl
│   └── kb/                           # 知识库核心（可被服务复用）
│       ├── cleaner.py                # 清洗去重：articles → docs
│       ├── chunker.py                # 切块：docs → chunks
│       ├── embedder.py               # 文本 → 向量
│       ├── index_store.py            # 建索引 / 加载索引
│       ├── retriever.py              # retrieve(query) → TopK
│       ├── rag.py                    # RAG prompt / citations
│       └── jsonl_io.py               # JSONL 读写工具
│
├── scripts/                          # 运维 / 流水线 / 冒烟验收（见下表分类）
├── tests/                            # pytest：契约 / 映射 / 单元测试（mock，不打网）
├── docs/                             # 学习与运维文档
│   ├── week1_学习指南.md
│   ├── week2_学习指南.md
│   └── 日志阅读指南.md
├── data/stability_kb/                # 语料与索引（本地数据，通常不入库）
├── eval_samples.jsonl                # Week1 回归样例（mode=llm）
├── eval_samples_rag.jsonl            # Week2 RAG 评测样例
├── requirements.txt
├── .env.example
└── README.md
```

### `scripts/` 按用途分类

| 分类 | 脚本 | 用途 |
|------|------|------|
| **运维启动** | `start_server.sh` | 停旧进程 → 清缓存 → 以 `.env` 为准启动 uvicorn |
| | `stats_requests.py` | 汇总 `requests.jsonl`（成功率、延迟分位、错误码等） |
| | `trace_request.py` | 按 `request_id` 从日志回放事件链 |
| **知识库流水线** | `crawl_stability_kb.py` | 采集公开资料 → `articles.jsonl` |
| | `stability_kb_seeds.json` | 采集种子 URL / 主题配置 |
| | `build_stability_docs.py` | 清洗去重 → `docs.jsonl` |
| | `chunk_stability_docs.py` | 切块 → `chunks.jsonl` |
| | `build_kb_index.py` | Embedding + 建索引 → `index/` |
| | `retrieve_kb.py` | 对索引做 TopK 检索（交互冒烟） |
| | `run_day6_7.sh` | 清洗 + 切块一键 |
| **冒烟 / 验收 / 回归** | `smoke_llm_client.py` | **不启 FastAPI**，连续调 DeepSeek 冒烟 |
| （会打真实服务或本机索引） | `run_day5.sh` | Week1 一键：pytest → 启服务 → `/ask` 回归 → 停服务 |
| | `run_eval.py` | Week1：对已启动服务批量 `POST /ask`，写报告 |
| | `verify_kb_retrieve.py` | 索引 + `retrieve` 契约 + 固定 query 冒烟 |
| | `verify_ask_rag.py` | 核对 RAG 响应的 citations 是否都来自本地 index |
| | `eval_rag_ask.py` | RAG 快速验收（约 20 query，需服务） |
| | `run_rag_eval.py` | Day10 评测闭环（≥50）+ 可选单变量 A/B |

### `tests/`（`pytest -q`，默认 mock，不依赖真实 Key）

| 文件 | 类型 | 用途 |
|------|------|------|
| `test_contract.py` | 接口契约 | `/health`、`/ask` 成功/失败字段（必须 mock LLM） |
| `test_error_mapping.py` | 接口契约 | 上游 401/429/timeout/5xx → `UPSTREAM_*` |
| `test_llm_client.py` | 单元测试 | `LLMClient` 重试/兜底（mock OpenAI） |
| `test_request_logging.py` | 单元测试 | 日志字段与脱敏 |
| `test_metrics_store.py` | 单元测试 | `requests.jsonl` 落盘字段 |
| `test_stats_requests.py` | 单元测试 | `stats_requests` 汇总逻辑 |
| `test_kb_retrieve.py` | 单元测试 | 索引 / `retrieve`（临时小索引，不依赖完整语料） |
| `test_rag_ask.py` | 接口契约 | RAG prompt、citations、`/ask?mode=rag` |
| `test_eval_rag_ask.py` | 单元测试 | `eval_rag_ask` 判定逻辑（不打 LLM） |
| `test_run_rag_eval.py` | 单元测试 | `run_rag_eval` 报告/分布逻辑（不打 LLM） |

> **怎么区分**：`tests/` = 自动化断言、CI 友好；`scripts/` 里的冒烟/回归 = 需要本机服务或真实索引/Key，用于验收与留档报告。详细步骤见对应周学习指南。

---

## 3. 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | - | DeepSeek API Key |
| `LLM_BASE_URL` | 否 | `https://api.deepseek.com` | API Base URL |
| `LLM_MODEL` | 否 | `deepseek-v4-flash` | 模型名 |
| `LLM_TIMEOUT_SECONDS` | 否 | `30` | 超时（秒） |
| `LLM_MAX_TOKENS` | 否 | `512` | 最大生成 token |
| `LLM_TEMPERATURE` | 否 | `0.2` | 温度 |
| `LLM_THINKING` | 否 | `disabled` | `disabled` / `enabled` |
| `REQUESTS_JSONL_PATH` | 否 | `./requests.jsonl` | 请求指标路径 |
| `APP_VERSION` | 否 | `0.1.0ba` | `/health` 版本 |
| `KB_INDEX_DIR` | 否 | `data/stability_kb/index` | RAG 索引目录 |
| `RAG_TOP_K` | 否 | `5` | RAG 默认检索条数 |

---

## 4. 接口速览

### `GET /health`

```json
{"status": "ok", "version": "0.1.0ba"}
```

### `POST /ask`

| 字段 | 必填 | 说明 |
|------|------|------|
| `query` | 是 | 非空；最大 2000 字符 |
| `session_id` / `client_tag` | 否 | 会话 / 来源 |
| `mode` | 否 | `llm`（默认）或 `rag`；也可用 `?mode=rag` |
| `top_k` | 否 | RAG 检索条数 |

- **mode=llm**：直连模型，`citations` 为空数组  
- **mode=rag**：先检索再回答，`citations` 必填；索引未建 → `503 INDEX_NOT_READY`

统一错误体：`{"request_id","code","message"}`。错误码、重试/兜底策略见 [Week1 学习指南](docs/week1_学习指南.md)。

---

## 5. 知识库 / RAG（最短路径）

核心在 `app/kb/`；`scripts/` 只是 CLI。完整说明见 [Week2 学习指南](docs/week2_学习指南.md)。

```bash
python scripts/crawl_stability_kb.py --domestic-only --target 15
python scripts/build_stability_docs.py --no-refetch
python scripts/chunk_stability_docs.py
python scripts/build_kb_index.py
python scripts/verify_kb_retrieve.py
# 服务启动后
python scripts/run_rag_eval.py
```

产物：`data/stability_kb/`（`articles` → `docs` → `chunks` → `index/`）。

---

## 6. 常用命令速查

```bash
./scripts/start_server.sh
./scripts/run_day5.sh
python scripts/smoke_llm_client.py
python scripts/stats_requests.py --path ./requests.jsonl
python scripts/trace_request.py <request_id> --log /tmp/app.log
python scripts/run_eval.py --samples ./eval_samples.jsonl
python scripts/run_rag_eval.py
pytest -q
```
