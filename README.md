# AI Start Agent

基于 **FastAPI + DeepSeek**（OpenAI 兼容 SDK）的问答服务。

能力概览：`/health` 探活、`/ask` 问答、错误映射与重试兜底、结构化日志、`requests.jsonl` 落盘、回归评测与契约测试。

---

## 1. 快速开始

```bash
# 首次：环境与依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 编辑 .env，填入 DEEPSEEK_API_KEY

# 推荐启动（停旧进程 → 清缓存 → 用 .env 覆盖环境变量 → 启动）
./scripts/start_server.sh
```

| 地址 | 说明 |
|------|------|
| http://127.0.0.1:8000/health | 探活 |
| http://127.0.0.1:8000/docs | Swagger |
| http://127.0.0.1:8000/ask | 问答 |

改端口：`PORT=8001 ./scripts/start_server.sh`。停止：`Ctrl+C`。

> **注意**：`pydantic-settings` 默认「环境变量 > `.env`」。若 shell 里已 `export DEEPSEEK_API_KEY`，会盖住 `.env`。`start_server.sh` 会强制以 `.env` 为准。

手动启动等价命令：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 或
python -m app.main
```

验证 health：

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","version":"0.1.0ba"}
```

---

## 2. 项目结构

```text
ai_start_anget/
├── app/                          # 应用主代码
│   ├── main.py                   # FastAPI 创建、生命周期、uvicorn 入口
│   ├── api.py                    # 路由：GET /health、POST /ask
│   ├── core/
│   │   ├── config.py             # 从环境变量 / .env 读取配置
│   │   └── logging.py            # JSON 结构化日志、固定事件、脱敏
│   └── services/
│       ├── llm_client.py         # LLMClient：DeepSeek 调用、重试、兜底
│       └── metrics_store.py      # 请求结束写入 requests.jsonl
├── scripts/                      # 运维 / 评测脚本（见下表）
│   ├── start_server.sh
│   ├── run_day5.sh               # Day 5 一键：pytest → 启服务 → 回归 → 停服务
│   ├── smoke_llm_client.py
│   ├── stats_requests.py
│   ├── trace_request.py
│   └── run_eval.py               # 调用 HTTP /ask，输出 eval_run_report.json
├── tests/                        # 自动化测试
│   ├── test_contract.py          # Day 5.1 接口契约（mock LLM）
│   ├── test_error_mapping.py     # Day 5.2 上游错误映射（mock）
│   ├── test_llm_client.py        # LLMClient 单元测试（mock，不打网）
│   ├── test_request_logging.py   # 日志字段与脱敏验收
│   ├── test_metrics_store.py     # requests.jsonl 落盘字段
│   └── test_stats_requests.py    # 统计脚本汇总逻辑
├── eval_samples.jsonl            # 回归样例集 v0（≥20：10 normal / 5 clarify / 5 refuse）
├── eval_run_report.json          # 最近一次回归汇总（留档对比用）
├── requirements.txt              # Python 依赖
├── .env.example                  # 环境变量模板（不含真 Key）
├── .env                          # 本地真实配置（勿提交）
└── README.md
```

### scripts 脚本说明

| 脚本 | 用途 | 常用命令 |
|------|------|----------|
| `start_server.sh` | 一键启动服务：停旧进程 → 清 `__pycache__` → 用 `.env` 覆盖环境变量 → 启动 uvicorn | `./scripts/start_server.sh` |
| `run_day5.sh` | Day 5 一键：契约+映射单测 → 启服务 → 跑回归 → 写报告 → 停服务 | `./scripts/run_day5.sh` |
| `smoke_llm_client.py` | **不启 FastAPI**，连续调用 `LLMClient.chat` 做冒烟（默认 10 次，成功率 ≥ 9/10） | `python scripts/smoke_llm_client.py` |
| `stats_requests.py` | 读取 `requests.jsonl`，输出 total/ok/fail、延迟分位、retry_rate、top 错误码、token 统计 | `python scripts/stats_requests.py --path ./requests.jsonl` |
| `trace_request.py` | 按 `request_id` 从日志中抽取固定事件链（`request_start` → … → `request_success/error`） | `python scripts/trace_request.py <request_id> --log /tmp/app.log` |
| `run_eval.py` | 对已启动服务逐条 `POST /ask`，写明细 + `eval_run_report.json`（单条失败不中断） | `python scripts/run_eval.py --samples ./eval_samples.jsonl` |

---

## 3. 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | 是 | - | DeepSeek API Key |
| `LLM_BASE_URL` | 否 | `https://api.deepseek.com` | API Base URL |
| `LLM_MODEL` | 否 | `deepseek-v4-flash` | 模型名 |
| `LLM_TIMEOUT_SECONDS` | 否 | `30` | 超时（秒）；`LLMClient` 会读取该值 |
| `LLM_MAX_TOKENS` | 否 | `512` | 最大生成 token |
| `LLM_TEMPERATURE` | 否 | `0.2` | 温度 |
| `LLM_THINKING` | 否 | `disabled` | thinking：`disabled` / `enabled` |
| `REQUESTS_JSONL_PATH` | 否 | `./requests.jsonl` | 请求指标路径 |
| `APP_VERSION` | 否 | `0.1.0ba` | `/health` 返回的版本 |

---

## 4. 接口

### `GET /health`

不依赖 DeepSeek。

```json
{"status": "ok", "version": "0.1.0ba"}
```

### `POST /ask`

调用 `LLMClient.chat()`；`citations` 暂为空数组。

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

常用字段：`ts`、`level`、`event`、`request_id`、`path`、`method`、`status_code`、`ok`、`latency_ms_total`、`llm_provider`、`llm_model`、`finish_reason`、`retry_count`、`query_len`、`query_sha256_8`、`error_code`

**不记录** query 原文、API key。

```bash
# 启动时把日志 tee 下来，再按 request_id 追踪
./scripts/start_server.sh 2>&1 | tee /tmp/app.log
python scripts/trace_request.py <request_id> --log /tmp/app.log
```

### requests.jsonl（每请求 1 行）

路径：`REQUESTS_JSONL_PATH`（默认 `./requests.jsonl`）。

字段：`ts, request_id, path, ok, status_code, latency_ms_total, latency_ms_llm, llm_model, retry_count, finish_reason, error_code, query_len, query_sha256_8`，以及能取到时的 `prompt_tokens / completion_tokens / total_tokens`。不写 query/answer 原文。

```bash
python scripts/stats_requests.py --path ./requests.jsonl
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

样例：`eval_samples.jsonl`（20 条：10 `normal` / 5 `clarify` / 5 `refuse`）。

服务已启动时单独跑批：

```bash
python scripts/run_eval.py \
  --samples ./eval_samples.jsonl \
  --base-url http://127.0.0.1:8000 \
  --results ./eval_results.jsonl \
  --report ./eval_run_report.json
```

报告字段：`total`、`ok_rate`、`p95_latency`、`top_errors`、各 tag 失败数。  
带时间戳副本在 `reports/eval_run_report_*.json`（`run_day5.sh` 生成），根目录 `eval_run_report.json` 为最新一份，便于每周对比。

---

## 8. 常用命令速查

脚本用途详见 [§2 项目结构](#2-项目结构)。

```bash
./scripts/start_server.sh
./scripts/run_day5.sh
python scripts/smoke_llm_client.py
python scripts/stats_requests.py --path ./requests.jsonl
python scripts/trace_request.py <request_id> --log /tmp/app.log
python scripts/run_eval.py --samples ./eval_samples.jsonl
pytest -q
```
