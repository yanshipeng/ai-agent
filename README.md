# AI Start Agent

基于 **FastAPI + DeepSeek**（OpenAI 兼容 SDK）的问答服务，并带本地稳定性知识库的 **RAG** 能力。

| 能力 | 说明 |
|------|------|
| 探活 / 问答 | `GET /health`、`POST /ask` |
| 直连 LLM | `mode=llm`（默认）：错误映射、重试、兜底 |
| RAG | `mode=rag`：混合检索（向量+BM25）→ 去重过滤 → 带 citations |
| Agent Tools | `mode=agent`：真实 tool_calls、状态机、超限降级、多轮 session |
| 安全 | 注入预检、引用门禁、泄密扫描、高风险工具 HITL |
| 可观测 | JSON 日志、`requests.jsonl`、`traces.jsonl`、Token Budget / Cache |
| 评测 | Week1 回归、Week2 RAG、Week3 Agent、Week4 注入评测 |

**按周学习文档（详细操作与踩坑）：**

| 文档 | 内容 |
|------|------|
| [`docs/week1_学习指南.md`](docs/week1_学习指南.md) | Day1–5：起服务、LLMClient、错误处理、日志、契约与回归 |
| [`docs/week2_学习指南.md`](docs/week2_学习指南.md) | Day6–10：采集 → 清洗 → 切块 → 索引/检索 → RAG → 评测 |
| [`docs/week3_学习指南.md`](docs/week3_学习指南.md) | **已完成**：Tools / Agent 状态机 / 多轮 session / 评测 |
| [`docs/week4_学习指南.md`](docs/week4_学习指南.md) | **已完成**：检索质量 v2 / 安全抗注入 / 可观测 v2 |
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

# Agent（真实 tool_calls；需索引）
curl -s 'http://127.0.0.1:8000/ask?mode=agent' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Android ANR 怎么排查"}'
```

> **Windows PowerShell**：`-d "{\"query\":...}"` 容易被壳弄坏 JSON。请写成文件后：  
> `curl.exe -s -X POST "http://127.0.0.1:8000/ask?mode=agent" -H "Content-Type: application/json" --data-binary "@ask.json"`

---

## 2. 项目结构

```text
ai_start_anget/
├── app/                              # 应用主代码
│   ├── main.py                       # FastAPI 创建、生命周期、uvicorn 入口
│   ├── api.py                        # 路由：GET /health、POST /ask（含安全门禁）
│   ├── core/
│   │   ├── config.py                 # 环境变量 / .env
│   │   ├── logging.py                # JSON 结构化日志、脱敏
│   │   └── safety.py                 # Day17：注入预检 / 引用门禁 / 泄密扫描
│   ├── services/
│   │   ├── llm_client.py             # DeepSeek 调用、重试、兜底、tool_calls
│   │   ├── metrics_store.py          # requests.jsonl + traces.jsonl
│   │   ├── token_budget.py           # Day18：token 估算与超预算压缩
│   │   ├── conversation.py           # 多轮：截断 / 滑窗 / 摘要
│   │   └── session_store.py          # 进程内 session
│   ├── kb/                           # 知识库核心（可被服务复用）
│   │   ├── cleaner.py / chunker.py / embedder.py / index_store.py
│   │   ├── bm25.py                   # Day16：Okapi BM25
│   │   ├── retriever.py              # hybrid 检索 + 去重过滤 + cache 计数
│   │   ├── rag.py                    # RAG prompt / citations
│   │   └── jsonl_io.py
│   └── agent/                        # Agent：工具 + Tool Runner（mode=agent）
│       ├── tools.py                  # kb_search / kb_get_chunk / HITL stub
│       └── runner.py                 # 状态机 + agent_trace + token 预算
│
├── scripts/                          # 运维 / 流水线 / 冒烟验收（见下表分类）
├── tests/                            # pytest：契约 / 映射 / 单元测试（mock，不打网）
├── docs/                             # 学习与运维文档（week1–4 + 日志指南）
├── data/stability_kb/                # 语料与索引（本地数据，通常不入库）
├── eval_samples.jsonl                # Week1 回归样例（mode=llm）
├── eval_samples_rag.jsonl            # Week2 RAG 评测样例
├── agent_eval_samples.jsonl          # Week3 Agent 评测样例（≥30）
├── eval_samples_injection.jsonl      # Week4 注入样例（≥10）
├── requirements.txt
├── .env.example
└── README.md
```

### `scripts/` 按用途分类

| 分类 | 脚本 | 用途 |
|------|------|------|
| **运维启动** | `start_server.sh` | 停旧进程 → 清缓存 → 以 `.env` 为准启动 uvicorn |
| | `stats_requests.py` | 汇总 `requests.jsonl`（含去重流 / `obs_v2` Trace·Budget·Cache） |
| | `trace_request.py` | 按 `request_id` 从日志回放事件链 |
| **知识库流水线** | `crawl_stability_kb.py` | 采集公开资料 → `articles.jsonl` |
| | `stability_kb_seeds.json` | 采集种子 URL / 主题配置 |
| | `build_stability_docs.py` | 清洗去重 → `docs.jsonl` |
| | `chunk_stability_docs.py` | 切块 → `chunks.jsonl` |
| | `build_kb_index.py` | Embedding + 建索引 → `index/` |
| | `retrieve_kb.py` | 对索引做 TopK 检索（交互冒烟） |
| | `run_day6_7.sh` | 清洗 + 切块一键 |
| **冒烟 / 验收 / 回归** | `smoke_llm_client.py` | **不启 FastAPI**，连续调 DeepSeek 冒烟 |
| （会打真实服务或本机索引） | `smoke_agent_tools.py` | Week3：5 次 `mode=agent`，验收真实 tool_calls |
| | `smoke_session_memory.py` | Week3：同 session 5 轮上下文记忆 |
| | `run_day5.sh` | Week1 一键：pytest → 启服务 → `/ask` 回归 → 停服务 |
| | `run_eval.py` | Week1：对已启动服务批量 `POST /ask`，写报告 |
| | `verify_kb_retrieve.py` | 索引 + `retrieve` 契约 + 固定 query 冒烟 |
| | `verify_ask_rag.py` | 核对 RAG 响应的 citations 是否都来自本地 index |
| | `eval_rag_ask.py` | RAG 快速验收（约 20 query，需服务） |
| | `run_rag_eval.py` | Day10 评测闭环（≥50）+ 可选单变量 A/B |
| | `run_agent_eval.py` | Week3 Agent 评测（≥30）→ `agent_eval_report.json` |
| **Week4 工程化** | `spotcheck_retrieve_day16.py` | 随机抽检 20 条 Top3（可 `--via-ask`） |
| | `run_injection_eval.py` | 注入评测（`--offline` / 在线）→ `reports/injection_eval_report.json` |

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
| `test_run_agent_eval.py` | 单元测试 | `run_agent_eval` 报告/样例分布（不打 LLM） |
| `test_agent_tools.py` | 单元/契约 | 工具校验、Tool Runner、`/ask?mode=agent` 指标 |
| `test_session_conversation.py` | 单元/契约 | 多轮 session 滑窗/截断/摘要 |
| `test_hybrid_retrieve.py` | 单元测试 | Day16 混合检索 / 去重过滤 |
| `test_safety.py` | 单元测试 | Day17 引用门禁 / 注入预检 / 泄密 |
| `test_injection_eval.py` | 验收测试 | Day17 拒答率 / 零泄露 |
| `test_observability_day18.py` | 单元测试 | Day18 Trace / Budget / Cache |

> **怎么区分**：`tests/` = 自动化断言、CI 友好；`scripts/` 里的冒烟/回归 = 需要本机服务或真实索引/Key，用于验收与留档报告。详细步骤见对应周学习指南。

---

## 3. 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | - | DeepSeek API Key |
| `LLM_BASE_URL` | 否 | `https://api.deepseek.com` | API Base URL |
| `LLM_MODEL` | 否 | `deepseek-v4-flash` | 模型名 |
| `LLM_TIMEOUT_SECONDS` | 否 | `30` | 超时（秒） |
| `LLM_MAX_TOKENS` | 否 | `2048` | 最大生成 token（亦作 Day18 `max_output_tokens`） |
| `LLM_TEMPERATURE` | 否 | `0.2` | 温度 |
| `LLM_THINKING` | 否 | `disabled` | `disabled` / `enabled` |
| `REQUESTS_JSONL_PATH` | 否 | `./requests.jsonl` | 请求指标路径 |
| `TRACES_JSONL_PATH` | 否 | `./traces.jsonl` | Day18：Agent 逐步 Trace |
| `APP_VERSION` | 否 | `0.1.0ba` | `/health` 版本 |
| `KB_INDEX_DIR` | 否 | `data/stability_kb/index` | RAG 索引目录 |
| `RAG_TOP_K` | 否 | `5` | RAG 默认检索条数 |
| `RAG_HYBRID_WEIGHT` | 否 | `0.6` | Day16：混合检索向量占比 |
| `RAG_MIN_SCORE` | 否 | `0.05` | Day16：融合分阈值 |
| `AGENT_MAX_STEPS` | 否 | `5` | Agent Plan 最大轮数 |
| `AGENT_MAX_TOTAL_TIME_MS` | 否 | `20000` | Agent 整单总耗时上限（ms） |
| `AGENT_ON_MAX_STEPS` | 否 | `rag` | 超步：`rag` / `clarify` / `error` |
| `AGENT_TOOL_TIMEOUT_SECONDS` | 否 | `10` | 单次工具执行超时 |
| `AGENT_MAX_CONTEXT_TOKENS` | 否 | `6000` | Day18：Agent context token 预算 |
| `SESSION_MAX_TURNS` | 否 | `8` | 多轮滑窗：保留最近 N 轮 |
| `SESSION_MAX_CHARS` | 否 | `20000` | 上下文总字符预算；超则摘要/再裁 |
| `SESSION_TOOL_RESULT_MAX_CHARS` | 否 | `4000` | 单条 tool 结果截断 |
| `SESSION_CONTENT_MAX_CHARS` | 否 | `8000` | 单条 user/assistant 截断 |
| `SESSION_ENABLE_SUMMARY` | 否 | `true` | 超预算时是否摘要旧对话 |
| `SESSION_SUMMARY_USE_LLM` | 否 | `false` | `false`=抽取式；`true`=调 LLM 摘要 |
| `SESSION_TTL_SECONDS` | 否 | `3600` | 进程内 session 过期秒数 |

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
| `session_id` / `client_tag` | 否 | 会话 / 来源；同 `session_id` 会带入历史（进程内） |
| `mode` | 否 | `llm`（默认）/ `rag` / `agent`；也可用 `?mode=` |
| `top_k` | 否 | RAG / Agent 检索条数 |

- **mode=llm**：直连模型，`citations` 为空数组  
- **mode=rag**：混合检索后再回答，`citations` 必填；索引未建 → `503 INDEX_NOT_READY`
- **mode=agent**：真实 `tool_calls` 多轮循环；`meta` 含 `agent_steps` / `agent_trace` / token 预算 / `cache_*`
- **安全（Week4）**：注入预检拒答；返回前引用一致性 + 泄密扫描；高风险工具 `TOOL_NEEDS_APPROVAL`
- **多轮**：同一 `session_id` 带历史；见 [Week3](docs/week3_学习指南.md)
- **指标**：`requests.jsonl` + `traces.jsonl`；`python scripts/stats_requests.py --mode rag|agent`

统一错误体：`{"request_id","code","message"}`。错误码见 [Week1](docs/week1_学习指南.md)。  
工具可控错误：`TOOL_INVALID_ARGS` / `TOOL_TIMEOUT` / `TOOL_NEEDS_APPROVAL`。

---

## 5. 知识库 / RAG（最短路径）

核心在 `app/kb/`；`scripts/` 只是 CLI。完整说明见 [Week2](docs/week2_学习指南.md) · 混合检索见 [Week4 Day16](docs/week4_学习指南.md)。

```bash
python scripts/crawl_stability_kb.py --domestic-only --target 15
python scripts/build_stability_docs.py --no-refetch
python scripts/chunk_stability_docs.py
python scripts/build_kb_index.py
python scripts/verify_kb_retrieve.py
python scripts/retrieve_kb.py "Android ANR 怎么排查"   # 看 hybrid + 去重流
# 服务启动后
python scripts/run_rag_eval.py
```

产物：`data/stability_kb/`（`articles` → `docs` → `chunks` → `index/`）。

---

## 6. Week4 工程化速查

详细步骤与踩坑见 [Week4 学习指南](docs/week4_学习指南.md)。

```bash
# Day16：混合检索抽检
python scripts/spotcheck_retrieve_day16.py
python scripts/stats_requests.py --mode rag

# Day17：安全 / 注入
pytest -q tests/test_safety.py tests/test_injection_eval.py
python scripts/run_injection_eval.py --offline

# Day18：Trace / Budget / Cache（需服务 + 新 agent 请求）
pytest -q tests/test_observability_day18.py
python scripts/stats_requests.py --mode agent
# 逐步轨迹：traces.jsonl
```

---

## 7. 常用命令速查

```bash
./scripts/start_server.sh
./scripts/run_day5.sh
python scripts/smoke_llm_client.py
python scripts/stats_requests.py --path ./requests.jsonl
python scripts/trace_request.py <request_id> --log /tmp/app.log
python scripts/run_eval.py --samples ./eval_samples.jsonl
python scripts/run_rag_eval.py
python scripts/run_agent_eval.py
python scripts/run_injection_eval.py --offline
python scripts/smoke_agent_tools.py --log /tmp/app.log
pytest -q
```
