# 第一周学习指南：FastAPI + DeepSeek 问答服务

> 面向刚入行同学。本周主题：**起服务 → 调 LLM → 错误处理 → 可观测 → 测试回归**。  
> 文档导航：[docs/README.md](./README.md) · 日志：[日志阅读指南.md](./日志阅读指南.md)  
> 下一周：[week2_学习指南.md](./week2_学习指南.md) · [week3_学习指南.md](./week3_学习指南.md)  
> 代码以仓库当前实现为准；本周默认 `/ask` 的 `mode=llm`（直连模型，`citations=[]`）。

---

## 0. 第一周要达成什么

| 天数（大致） | 主题 | 目标产物 |
|--------------|------|----------|
| Day 1–2 | FastAPI 骨架 + 配置 | `/health`、`/ask` 能通 |
| Day 3 | `LLMClient` + 错误映射 / 重试 / 兜底 | 上游失败可解释、可恢复 |
| Day 4 | 结构化日志 + `requests.jsonl` | 能按 `request_id` 追踪 |
| Day 5 | 契约测试 + 错误映射测试 + 回归样例 | 一键绿、可每周对比 |

### 为什么先做这些？（思考）

| 做法 | 动机 |
|------|------|
| 先 `/health` 再 `/ask` | 先确认进程活着，再查模型/Key 问题 |
| 统一错误码 + 重试/兜底 | 上游 429/超时很常见，不能直接把原始异常甩给用户 |
| JSON 日志 + `requests.jsonl` | 以后排查靠 `request_id`，不靠猜 |
| 契约测试 + 回归样例 | 改代码时不怕「悄悄把接口字段改坏」 |

一句话流程：

```text
用户 POST /ask
  → 校验 query
  → LLMClient.chat()（DeepSeek）
  → 成功：answer + citations=[] + meta
  → 失败：统一错误码 / 可重试则重试 / 耗尽可兜底
  → 全程：JSON 日志 + data/runtime/requests.jsonl
```

### 本周任务清单（自学请勾选）

- [ ] 虚拟环境 + `.env` 配好 Key，服务能起
- [ ] `/health`、`/ask`（mode=llm）跑通
- [ ] 会用 `request_id` 在日志里追一条请求
- [ ] `pytest -q` 相关测试通过（或至少契约/错误映射）
- [ ] `run_eval.py` 或 `run_day5.sh` 跑出报告到 `reports/`

---

## 1. 快速上手（本周最小路径）

**Linux / macOS：**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入 DEEPSEEK_API_KEY
# 学习阶段建议保持：
# API_AUTH_ENABLED=false
# RATE_LIMIT_ENABLED=false

./scripts/start_server.sh
```

**Windows PowerShell：**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY；鉴权/限流保持 false
python -m app.main
# 或：uvicorn app.main:app --host 127.0.0.1 --port 8000
```

| 地址 | 说明 |
|------|------|
| http://127.0.0.1:8000/health | 探活 |
| http://127.0.0.1:8000/docs | Swagger（推荐新手在这里点试） |
| http://127.0.0.1:8000/ask | 问答（默认 mode=llm） |

改端口：`PORT=8001 ./scripts/start_server.sh`。停止：`Ctrl+C`。

> **注意**：  
> 1) `pydantic-settings` 默认「环境变量 > `.env`」。若 shell 里已 `export DEEPSEEK_API_KEY`，会盖住 `.env`。  
> 2) 若 `/ask` 一律 **401**：多半开了鉴权。学习阶段设 `API_AUTH_ENABLED=false`，或请求头带 `X-Api-Key`（见 week5）。

手动启动等价：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 或
python -m app.main
```

验证：

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","version":"0.1.0ba"}
```

---

## 2. 本周相关目录（不含 RAG）

```text
app/
├── main.py                   # FastAPI 创建、生命周期、uvicorn 入口
├── api.py                    # GET /health、POST /ask
├── core/
│   ├── config.py             # 环境变量 / .env
│   └── logging.py            # JSON 结构化日志、脱敏
└── services/
    ├── llm_client.py         # DeepSeek 调用、重试、兜底
    └── metrics_store.py      # requests.jsonl

scripts/
├── start_server.sh
├── run_day5.sh               # Day 5 一键
├── smoke_llm_client.py
├── stats_requests.py
├── trace_request.py
└── run_eval.py

tests/
├── test_contract.py          # Day 5.1
├── test_error_mapping.py     # Day 5.2
├── test_llm_client.py
├── test_request_logging.py
├── test_metrics_store.py
└── test_stats_requests.py

eval/eval_samples.jsonl       # ≥20：10 normal / 5 clarify / 5 refuse
reports/eval_run_report.json  # 最近一次回归汇总（本地生成）
```

---

## 3. 环境变量（Week 1）

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | - | DeepSeek API Key |
| `LLM_BASE_URL` | 否 | `https://api.deepseek.com` | API Base URL |
| `LLM_MODEL` | 否 | `deepseek-v4-flash` | 模型名 |
| `LLM_TIMEOUT_SECONDS` | 否 | `30` | 超时（秒） |
| `LLM_MAX_TOKENS` | 否 | `2048` | 最大生成 token |
| `LLM_TEMPERATURE` | 否 | `0.2` | 温度 |
| `LLM_THINKING` | 否 | `disabled` | `disabled` / `enabled` |
| `REQUESTS_JSONL_PATH` | 否 | `./data/runtime/requests.jsonl` | 请求指标路径 |
| `APP_VERSION` | 否 | `0.1.0ba` | `/health` 版本 |

（第二周才用的 `KB_INDEX_DIR` / `RAG_TOP_K` 见 week2 指南。）

---

## 4. 接口：`/health` 与 `/ask`（mode=llm）

### `GET /health`

不依赖 DeepSeek。

```json
{"status": "ok", "version": "0.1.0ba"}
```

### `POST /ask`（本周：直连 LLM）

调用 `LLMClient.chat()`；`citations` 为空数组。第二周加 `mode=rag` 后行为见 week2。

**请求**

| 字段 | 必填 | 说明 |
|------|------|------|
| `query` | 是 | 非空；最大 2000 字符 |
| `session_id` | 否 | 多轮会话 ID |
| `client_tag` | 否 | 来源标识 |

```json
{"query": "中国的首都是哪里？", "session_id": "s-demo", "client_tag": "web"}
```

**成功响应**

```json
{
  "request_id": "uuid",
  "answer": "北京。",
  "citations": [],
  "latency_ms": 800,
  "model": "deepseek-v4-flash",
  "meta": {
    "finish_reason": "stop",
    "usage": {"prompt_tokens": 10, "completion_tokens": 4}
  }
}
```

**错误响应（统一结构）**

```json
{"request_id": "uuid", "code": "INVALID_ARGUMENT", "message": "..."}
```

| code | HTTP | 场景 |
|------|------|------|
| `INVALID_ARGUMENT` | 400 | 缺/空/超长 `query` |
| `UPSTREAM_UNAUTHORIZED` | 401 | 鉴权失败 / 缺 Key（不重试） |
| `UPSTREAM_BAD_REQUEST` | 400 | 上游认为请求非法（不重试） |
| `UPSTREAM_RATE_LIMITED` | 429 | 限流（可重试 → 可兜底） |
| `UPSTREAM_TIMEOUT` | 504 | 超时（可重试 → 可兜底） |
| `UPSTREAM_5XX` | 502 | 上游 5xx（可重试 → 可兜底） |
| `UPSTREAM_UNKNOWN` | 502 | 其它未知错误（可重试 → 可兜底） |

失败策略：

- **重试**：`RATE_LIMITED` / `TIMEOUT` / `5XX` / `UNKNOWN` 最多再试 2 次（共 3 次）
- **兜底**：可重试错误耗尽后返回 **HTTP 200** + 兜底文案（`model=fallback`，`meta.fallback=true`）
- **不兜底**：`UNAUTHORIZED` / `BAD_REQUEST` 直接返回对应 4xx

### 可复制 curl

```bash
# 成功 A
curl -s -i http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"中国的首都是哪里？","session_id":"s-demo","client_tag":"curl"}'

# 成功 B
curl -s -i http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"用一句话解释什么是 API","client_tag":"readme"}'

# 失败：空 query → 400
curl -s -i http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":""}'

# 失败：超长 query → 400
curl -s -i http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"$(python3 -c 'print(\"a\"*2001)')\"}"
```

成功核对：`request_id` / `answer` / `citations` / `latency_ms` / `model`  
失败核对：`request_id` / `code` / `message`

---

## 5. 可观测性

### 结构化日志（固定事件）

```text
request_start → llm_call_start → llm_call_end → request_success
request_start → … → request_error
validate_failed → request_error
```

常用字段：`ts`、`level`、`event`、`hint`、`request_id`、`path`、`method`、`status_code`、`ok`、`latency_ms_total`、`llm_provider`、`llm_model`、`finish_reason`、`retry_count`、`query_len`、`query_sha256_8`、`error_code`

**不记录** query 原文、API key。新手读日志请看 [`日志阅读指南.md`](./日志阅读指南.md)。

```bash
./scripts/start_server.sh 2>&1 | tee reports/app.log
python scripts/trace_request.py <request_id> --log reports/app.log
```

### requests.jsonl（每请求 1 行）

路径：`REQUESTS_JSONL_PATH`（默认 `./data/runtime/requests.jsonl`）。

字段：`ts, request_id, path, ok, status_code, latency_ms_total, latency_ms_llm, llm_model, retry_count, finish_reason, error_code, query_len, query_sha256_8`，以及能取到时的 `prompt_tokens / completion_tokens / total_tokens`。不写 query/answer 原文。

```bash
python scripts/stats_requests.py --path ./data/runtime/requests.jsonl
# 输出：total/ok/fail/ok_rate、p50/p95/max latency、retry_rate、top_error_codes、token 分位等
```

---

## 6. LLMClient

位置：`app/services/llm_client.py`。路由不要直接写 OpenAI SDK。

```python
from app.services.llm_client import LLMClient

client = LLMClient()
result = client.chat([{"role": "user", "content": "你好"}])
# result.answer / model / finish_reason / latency_ms / usage / retry_count
```

| 参数 | 值 |
|------|-----|
| `base_url` | `https://api.deepseek.com` |
| `model` | `deepseek-v4-flash` |
| `extra_body` | `{"thinking":{"type":"disabled"}}` |
| `max_tokens` | `512` |
| `stream` | `false` |
| `timeout` | 读自 `LLM_TIMEOUT_SECONDS`（默认 30s） |

```bash
# 不启 FastAPI，连续 10 次冒烟（成功率需 ≥ 9/10）
python scripts/smoke_llm_client.py
```

---

## 7. Day 5：测试 + 回归

### 一键跑（推荐）

```bash
./scripts/run_day5.sh
# 只跑契约/映射单测：
SKIP_EVAL=1 ./scripts/run_day5.sh
# 回归只跑前 N 条：
EVAL_LIMIT=5 ./scripts/run_day5.sh
```

流程：`pytest`（mock LLM）→ 启动 uvicorn → `run_eval.py` 打真实 `/ask` → 写报告 → 停服务。

### Day 5.1 契约 / Day 5.2 错误映射

```bash
pytest -q tests/test_contract.py tests/test_error_mapping.py
# 或全量
pytest -q
```

- `/health`、`/ask` 成功/失败契约（**必须 mock LLM**；`citations` 必须是 array）
- 上游 401/429/timeout/5xx → `UPSTREAM_*`，HTTP 策略统一；错误体含 `request_id/code/message`，**不含上游全文**

### Day 5.3 回归样例 + 批量报告

样例：`eval/eval_samples.jsonl`（20 条：10 `normal` / 5 `clarify` / 5 `refuse`）。

服务已启动时单独跑批：

```bash
python scripts/run_eval.py \
  --samples ./eval/eval_samples.jsonl \
  --base-url http://127.0.0.1:8000 \
  --results ./reports/eval_results.jsonl \
  --report ./reports/eval_run_report.json
```

报告字段：`total`、`ok_rate`、`p95_latency`、`top_errors`、各 tag 失败数。  
带时间戳副本在 `reports/eval_run_report_*.json`（`run_day5.sh` 生成），`reports/eval_run_report.json` 为最新一份，便于每周对比。

> 注意：第二周的 `eval/eval_samples_rag.jsonl` / `run_rag_eval.py` 是另一套评测，不要和本周 `eval/eval_samples.jsonl` 混用。

---

## 8. 对照代码读哪里

建议顺序（先主线后细节）：

1. `app/main.py` — 应用怎么创建  
2. `app/api.py` — `/health`、`/ask`  
3. `app/services/llm_client.py` — 重试 / 错误映射 / 兜底  
4. `app/core/logging.py` + `app/services/metrics_store.py` — 日志与指标  
5. `tests/test_contract.py`、`tests/test_error_mapping.py` — 接口不能乱改的约定  

---

## 9. 常见坑与回撤

| 现象 | 常见原因 | 怎么退 / 怎么修 |
|------|----------|-----------------|
| `/ask` 全是 401 | `.env` 开了 `API_AUTH_ENABLED` | 学习阶段改 `false`，或加 `X-Api-Key` |
| Key 明明写了 `.env` 却无效 | shell 里旧的 `export DEEPSEEK_API_KEY` 盖住了 | `start_server.sh` 启动，或先 `Remove-Item Env:DEEPSEEK_API_KEY`（PS） |
| 服务起不来 / 端口占用 | 旧进程未停 | 换 `PORT=8001`，或结束占用 8000 的进程 |
| 找不到 `requests.jsonl` | 还在根目录找 | 看 `data/runtime/requests.jsonl` |
| `run_day5.sh` 太慢或卡评测 | 想先只看测试 | 看脚本是否支持跳过评测；或先单独 `pytest -q` |
| 改坏了不知道回哪 | 本地乱改 | `git checkout -- <文件>` 恢复未提交改动；已提交用 `git log` / `git show` 对照 |

日志落盘建议（Windows 也通用）：

```bash
./scripts/start_server.sh 2>&1 | tee reports/app.log
python scripts/trace_request.py <request_id> --log reports/app.log
```

---

## 10. 常用命令速查

```bash
./scripts/start_server.sh
./scripts/run_day5.sh
python scripts/smoke_llm_client.py
python scripts/stats_requests.py --path ./data/runtime/requests.jsonl
python scripts/trace_request.py <request_id> --log reports/app.log
python scripts/run_eval.py --samples ./eval/eval_samples.jsonl
pytest -q
```

---

## 11. 验收清单（Week 1）

- [ ] `/health` 返回 `status` + `version`
- [ ] `/ask` 成功有 `request_id` / `answer` / `citations`（array）/ `latency_ms` / `model`
- [ ] 空/超长 query → `INVALID_ARGUMENT` + 400
- [ ] 上游错误映射到 `UPSTREAM_*`，策略统一
- [ ] 可重试错误会重试；耗尽可走兜底（200 + `model=fallback`）
- [ ] 日志可按 `request_id` 串起事件链；不落 query/key
- [ ] `data/runtime/requests.jsonl` 有每请求一行；`stats_requests.py` 能汇总
- [ ] `./scripts/run_day5.sh` 契约 + 映射 + 回归能跑通
