# 第三周学习指南：Tools + 真实 tool_calls（Agent）

> 面向刚入行同学。记录第三周「定义工具 → DeepSeek tool_calls → Tool Runner 状态机 → 多轮 session → `/ask?mode=agent` → Agent 评测」做了什么、怎么操作、原理是什么、怎么验收。  
> **状态：第三周内容已完成**（见文末「完成总结」）。  
> 第一周：[`week1_学习指南.md`](./week1_学习指南.md) · 第二周：[`week2_学习指南.md`](./week2_学习指南.md)  
> 服务日志怎么读：见 [`日志阅读指南.md`](./日志阅读指南.md)。  
> 代码会变，以仓库当前实现为准；Agent 依赖第二周建好的本地索引 `data/stability_kb/index/`。

---

## 0. 本次改了什么（变更清单 · 终版）

相对第二周结束时的代码，第三周落地如下能力。

### 新增文件

| 路径 | 作用 |
|------|------|
| `app/agent/__init__.py` | Agent 包对外导出 |
| `app/agent/tools.py` | 工具定义 + 执行：`kb_search` / `kb_get_chunk`（失败返回可控 error_code） |
| `app/agent/runner.py` | 状态机 Plan→Act→Observe→Final；`stop_reason` / 超步降级 / 超时兜底 |
| `app/services/session_store.py` | 进程内 `session_id → messages`（TTL） |
| `app/services/conversation.py` | 上下文：截断 → 滑窗 →（可选）摘要 |
| `eval/agent_eval_samples.jsonl` | Agent 评测样例 ≥30（默认 A=ANR：22 tool + 10 clarify） |
| `scripts/smoke_agent_tools.py` | 5 次 agent 冒烟，验收真实 `tool_call_start` |
| `scripts/smoke_session_memory.py` | 同 session 5 轮约束记忆 + 指标字段 |
| `scripts/run_agent_eval.py` | 批量评测 → `reports/agent_eval_report.json` |
| `tests/test_agent_tools.py` | 工具 / Runner / `/ask?mode=agent` |
| `tests/test_session_conversation.py` | 滑窗 / 摘要 / 多轮带历史 |
| `tests/test_run_agent_eval.py` | 评测报告指标形状（不打 LLM） |
| `docs/week3_学习指南.md` | 本文档 |

### 修改文件（行为变化）

| 路径 | 改动要点 |
|------|----------|
| `app/kb/retriever.py` | 新增 `get_chunk(chunk_id)`：按 id 取全文 |
| `app/services/llm_client.py` | `ToolCall` / `LLMTurnResult`；`chat_turn(..., tools=)` 真实 tool_calls |
| `app/api.py` | `mode=agent`；session 注入/落盘；`history_messages` / `history_chars` 指标 |
| `app/services/metrics_store.py` | agent 字段 + `session_id` / `history_*` |
| `app/core/config.py` | `AGENT_*` + `SESSION_*` |
| `app/core/logging.py` | `agent_step` / `agent_phase` / `agent_stop` / `tool_call_*` / `context_compact` |
| `scripts/stats_requests.py` | with/without session 的 p95；`--session-id` 过滤 |
| `.env.example` / `README.md` | 环境变量与用法 |

### 没有改动的边界

- 默认仍是 `mode=llm`（兼容 Week1 回归）
- `mode=rag` 行为不变（服务端先检索再拼 Context，**不是**模型主动调工具）
- 语料流水线仍在 `app/kb/`；Agent 只是**消费**已有索引
- session 存进程内存：多 worker / 重启会丢（教学足够）

---

## 1. 第三周要达成什么

前两周已有：

- Week1：`/ask` + LLMClient + 错误处理 + 日志/指标 + 契约回归
- Week2：知识库索引 + `mode=rag` + citations + Day10 评测

第三周做 **Agent / Tools**：让模型通过 DeepSeek 的 **function calling（tool_calls）** 真正调用本地工具，而不是由服务端替它先检索好。

| 主题 | 目标 |
|------|------|
| 定义 ≥2 个工具 | `kb_search` + `kb_get_chunk` |
| 真实 tool_calls | 模型返回 `tool_calls`，不是假脚本模拟 |
| Tool Runner 状态机 | Plan → Act → Observe → Final；超步/超时可控 |
| 多轮 session | 同 `session_id` 带历史；滑窗 / 截断 / 可选摘要 |
| 可观测 | 日志 `tool_call_*` / `agent_stop`；jsonl 含 agent + session 字段 |
| 稳健性 | 工具失败不崩：`TOOL_*`；`stop_reason` 六选一 |
| 评测 | `eval/agent_eval_samples.jsonl` ≥30 → `reports/agent_eval_report.json` |

一句话：

```text
用户 POST /ask?mode=agent
  → system 提示「必须先调工具查库」
  → LLMClient.chat_turn(messages, tools=[kb_search, kb_get_chunk])
  → 若有 tool_calls：执行工具 → messages 追加 assistant + tool
  → 再调模型……直到最终文本回答（或步数上限）
  → 写日志 + requests.jsonl（mode=agent, agent_steps, tool_calls_count, tools_used）
```

### `mode=rag` vs `mode=agent`（必懂）

| | `mode=rag` | `mode=agent` |
|--|------------|--------------|
| 谁决定检索 | **服务端**固定先 `retrieve` | **模型**决定是否/何时调 `kb_search` |
| 上下文怎么进模型 | 服务端拼好 Context 放进 user/system | 工具结果以 `role=tool` 回传 |
| citations | 来自当次 TopK | 来自本轮实际用过的工具命中 |
| 典型日志 | `retrieve_start` → `retrieve_end` → `llm_call_*` | `agent_step` → `llm_call_*` → `tool_call_*` → … |

---

## 2. 目录与职责

```text
app/agent/                    ← 第三周核心（后厨）
  ├── tools.py                ← 工具 schema + execute_tool
  └── runner.py               ← run_agent_loop（多轮循环）

app/services/
  ├── llm_client.py           ← chat_turn 支持 tools / tool_calls
  ├── session_store.py        ← 进程内 session_id → messages
  └── conversation.py         ← 滑窗 / 截断 /（可选）摘要

app/api.py                    ← mode=agent 入口 _ask_agent；session 注入
app/kb/retriever.py           ← retrieve + get_chunk（被工具调用）

scripts/smoke_agent_tools.py  ← 薄 CLI：5 次真实验收
scripts/run_agent_eval.py     ← Agent 评测 → reports/agent_eval_report.json
tests/test_agent_tools.py     ← mock LLM，不打网
tests/test_session_conversation.py  ← 多轮上下文单测
tests/test_run_agent_eval.py  ← 评测报告逻辑单测
```

**原则与 Week2 一样**：主逻辑在 `app/`；`scripts/` 只做人机入口。线上一次 `/ask` **不会** import smoke 脚本。

---

## 3. 两个工具分别干什么

### 3.1 `kb_search(query, top_k?)`

- **做什么**：对本地索引做向量检索，返回 TopK（含 `chunk_id` / `score` / `title` / `url` / `text_snippet`）
- **为什么先有它**：模型需要「先找候选卡片」，不必一次塞全文
- **底层**：复用 `app.kb.retriever.retrieve`

### 3.2 `kb_get_chunk(chunk_id)`

- **做什么**：按 `chunk_id` 从索引取**全文** + 元数据
- **为什么单独拆**：search 只给 snippet，细节不够时再按需取全文，控制 token
- **底层**：新增的 `app.kb.retriever.get_chunk`

### 3.3 OpenAI / DeepSeek tools schema

工具声明在 `TOOL_SPECS`，经 `openai_tools_schema()` 传给：

```python
client.chat.completions.create(..., tools=..., tool_choice="auto")
```

模型若需要工具，响应里会出现 `message.tool_calls[]`（含 `id` / `function.name` / `function.arguments`）。

---

## 4. Tool Runner：Plan → Act → Observe → Final

入口：`app.agent.runner.run_agent_loop(query, client=...)`。

```text
              ┌─────────┐
              │  PLAN   │  调 LLM（可带 tools）
              └────┬────┘
         tool_calls│        无 tool_calls → FINAL（终答）
                   ▼
              ┌─────────┐
              │   ACT   │  执行 kb_search / kb_get_chunk
              └────┬────┘
                   ▼
              ┌─────────┐
              │ OBSERVE │  role=tool 写回 messages
              └────┬────┘
                   └──► 回到 PLAN
```

日志事件：`agent_phase`（plan/act/observe/final）+ 原有 `agent_step` / `tool_call_*`。

---

## 4.1 多轮 session + 上下文控制

传同一 `session_id`，服务会把历史消息带入下一轮（进程内内存；重启清空）。

```text
请求带 session_id
  → get_session_messages
  → compact（截断 → 滑窗 → 可选摘要）
  → 拼进本轮 messages
  → 成功后写回 session
```

| 机制 | 默认 | 说明 |
|------|------|------|
| 滑窗 | `SESSION_MAX_TURNS=8` | 只保留最近 N 个 user 轮（建议 6–10） |
| 截断 | tool≤4k / 正文≤8k | 工具结果与长文限长；Observe 写 `role=tool` 时也会截断 |
| 摘要 | `SESSION_ENABLE_SUMMARY=true` | 总字符超 `SESSION_MAX_CHARS`（默认 20k）时，把旧轮压成一条 memory |
| LLM 摘要 | `SESSION_SUMMARY_USE_LLM=false` | 默认抽取式；打开后才多一次 LLM 调用 |

存储约定：

- **不存**主 system prompt（每次请求重新注入）
- **llm/rag** 只存短 `user`/`assistant`（RAG 的 Context 大段不进 session）
- **agent** 存截断后的完整轨迹（含 tool），下次 agent 可续用

日志 / `meta` / `requests.jsonl`：

| 字段 | 说明 |
|------|------|
| `session_id` | 会话 ID（jsonl / meta） |
| `history_messages` | 本轮注入的历史消息条数 |
| `history_chars` | 历史总字符（可选，已落盘） |
| `session_id_sha256_8` | 指纹（日志也用，避免到处打原文） |

```bash
# 同 session 5 轮记忆 + 指标 + P95
./scripts/start_server.sh 2>&1 | tee /tmp/app.log
python scripts/smoke_session_memory.py
python scripts/stats_requests.py --path ./data/runtime/requests.jsonl --session-id <上一步打印的 session_id>
```

手动两轮：

```bash
curl -s 'http://127.0.0.1:8000/ask' \
  -H 'Content-Type: application/json' \
  -d '{"query":"只讨论 Android 推送","session_id":"demo-1","mode":"llm"}'

curl -s 'http://127.0.0.1:8000/ask' \
  -H 'Content-Type: application/json' \
  -d '{"query":"约束是什么？","session_id":"demo-1","mode":"llm"}'
```

### 护栏

| 护栏 | 默认 | 超限行为 |
|------|------|----------|
| `AGENT_MAX_STEPS` | **5** | 见 `AGENT_ON_MAX_STEPS`：`rag` / `clarify` / `error` |
| `AGENT_MAX_TOTAL_TIME_MS` | **20000**（15–30s 建议区间） | 超时兜底：文案含 `request_id`，`stop_reason=timeout` |
| 单工具超时 | `AGENT_TOOL_TIMEOUT_SECONDS=10` | 工具结果里 `TOOL_TIMEOUT`，不崩服务 |

验收硬约束：

1. **任意请求 `agent_steps <= max_steps`**
2. **终止原因 `stop_reason`**（日志事件 `agent_stop` + `requests.jsonl` + 响应 `meta`）：

| stop_reason | 含义 |
|-------------|------|
| `final_answer` | 正常终答 |
| `clarify` | 超步后澄清 |
| `degraded_to_rag` | 超步后降级 RAG |
| `max_steps` | 超步硬终止（`error` 策略，或无法降级） |
| `timeout` | 总耗时超限 |
| `upstream_error` | 上游 LLM 失败 / 空回答 |

另：`agent_phase_trace` / `degraded_to`（兼容旧字段）仍可看。

### System Prompt 要点

写死在 `AGENT_SYSTEM_PROMPT`：排障类问题必须先 `kb_search`；不足再换关键词或 `kb_get_chunk`；禁止假装已调工具。

---

## 5. LLMClient 改动（`chat_turn`）

| 类型/方法 | 说明 |
|-----------|------|
| `ToolCall` | `id` / `name` / `arguments`（arguments 为 JSON 字符串） |
| `LLMTurnResult` | 一轮结果：`content` 和/或 `tool_calls` |
| `chat_turn(messages, tools=...)` | 单轮；有 tools 时 `tool_choice=auto`；允许空 content 只要有 tool_calls |
| `chat(messages)` | 仍给 `mode=llm/rag` 用；内部调用 `chat_turn`，不允许意外 tool_calls |

日志：`llm_call_end` 在有 tool_calls 时会带 `tool_calls_count` / `tools_called`，方便对照「是不是真实触发了工具」。

---

## 6. HTTP：`/ask?mode=agent`

### 请求

```bash
curl -s 'http://127.0.0.1:8000/ask?mode=agent' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Android ANR 怎么排查"}'
```

也可用 body：`{"query":"...","mode":"agent"}`（query `?mode=` 优先）。

### 成功响应（关注 meta）

```json
{
  "request_id": "...",
  "answer": "...",
  "citations": [{"ref_id": 1, "chunk_id": "...", "title": "...", ...}],
  "latency_ms": 1234,
  "model": "deepseek-v4-flash",
  "meta": {
    "mode": "agent",
    "agent_steps": 2,
    "tool_calls_count": 1,
    "tools_used": ["kb_search"],
    "finish_reason": "stop"
  }
}
```

### 可控错误码（工具/Agent，不崩进程）

| code | HTTP（大致） | 场景 |
|------|--------------|------|
| `TOOL_INVALID_ARGS` | 400 | 参数非法（也可先作为 tool 结果回传给模型） |
| `TOOL_TIMEOUT` | 504 | 单次工具执行超时 |
| `TOOL_NOT_FOUND` | 400 | 未知工具名 |
| `AGENT_MAX_STEPS` | 504 | 轮数用尽仍无终答 |
| `AGENT_NO_ANSWER` | 502 | 模型结束但没有文本 |

上游 LLM 错误仍走原有 `UPSTREAM_*`。

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `AGENT_MAX_STEPS` | `5` | Agent Plan 最大轮数 |
| `AGENT_MAX_TOTAL_TIME_MS` | `20000` | 整单总耗时上限（建议 15–30s） |
| `AGENT_ON_MAX_STEPS` | `rag` | 超步数：`rag` 降级 / `clarify` 澄清 |
| `AGENT_TOOL_TIMEOUT_SECONDS` | `10` | 单次工具超时（秒） |
| `KB_INDEX_DIR` | `data/stability_kb/index` | 工具读的索引目录 |

---

## 7. 可观测性

### 事件链（mode=agent）

```text
request_start
  → agent_step
  → llm_call_start → llm_call_end（可能带 tools_called）
  → tool_call_start → tool_call_end   # 每个真实 tool call 一对
  → agent_step → llm_call_* → …
  → request_success
```

用 `request_id` 回放：

```bash
./scripts/start_server.sh 2>&1 | tee /tmp/app.log
# 发请求后
python scripts/trace_request.py <request_id> --log /tmp/app.log
```

重点 grep：`tool_call_start` —— 出现它才说明**模型真的发起了 tool_calls**（不是服务端静默 retrieve）。

### `requests.jsonl` 新增字段（Agent）

| 字段 | 含义 |
|------|------|
| `mode` | `agent` |
| `agent_steps` | 实际 Plan 轮数（**≤ max_steps**） |
| `max_steps` | 配置上限 |
| `stop_reason` | 终止原因（六选一，见 §4） |
| `tool_calls_count` | 累计工具调用次数 |
| `tools_used` | 用过的工具名列表（去重保序） |

仍不写 query/answer 原文。

---

## 8. 怎么跑通（操作步骤）

### 前置

1. `.env` 已配置 `DEEPSEEK_API_KEY`  
2. 第二周索引已建好：

```bash
python scripts/build_kb_index.py   # 若还没有 index/
ls data/stability_kb/index/manifest.json
```

### 单次试用

```bash
./scripts/start_server.sh 2>&1 | tee /tmp/app.log

curl -s 'http://127.0.0.1:8000/ask?mode=agent' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Android ANR 怎么排查","client_tag":"week3"}'
```

核对：响应 `meta.tool_calls_count >= 1`；日志有 `tool_call_start`。

### 验收冒烟（5 次里 ≥3 次真实 tool_calls）

```bash
python scripts/smoke_agent_tools.py --log /tmp/app.log
```

脚本会：

1. 连续 5 次 `POST /ask?mode=agent`（稳定性相关问题）  
2. 统计日志里有多少 `request_id` 出现过 `tool_call_start`  
3. 检查本批 `requests.jsonl` 是否含 `mode/agent_steps/tool_calls_count/tools_used`

### Agent 批量评测（≥30，默认 A=ANR）

样例：`eval/agent_eval_samples.jsonl`（22 条 `tool` + 10 条 `clarify`）。

```bash
./scripts/start_server.sh 2>&1 | tee /tmp/app.log
python scripts/run_agent_eval.py
# 冒烟只跑前 5 条
python scripts/run_agent_eval.py --limit 5 --skip-validate
```

产出：

- `reports/agent_eval_report.json`：汇总指标
- `reports/agent_eval_details.jsonl`：每条样例明细

| 指标 | 含义 |
|------|------|
| `tool_call_rate` | `tool_calls_count >= 1` 的比例 |
| `citation_coverage` | `citations` 非空比例 |
| `clarify_rate` | 该澄清的样例里，实际澄清（`stop_reason=clarify` 或澄清话术）比例 |
| `latency_ms_total.p50/p95` | 端到端延迟 |
| `tool_fail_rate` + `top_errors` | 工具/Agent 失败码占比与 Top 错误 |
| `avg_steps` / `p95_steps` | `meta.agent_steps` 均值与 P95 |

### 单测（不打真实 Key）

```bash
pytest -q tests/test_agent_tools.py tests/test_session_conversation.py tests/test_run_agent_eval.py
# 或全量
pytest -q
```

单测用 mock `chat_turn` 伪造一轮 `tool_calls` + 一轮终答，覆盖：参数错误、超时、Runner、metrics 落盘；session 单测覆盖滑窗/摘要与同 `session_id` 带历史；`test_run_agent_eval` 覆盖报告指标形状。

---

## 9. 代码怎么读（建议顺序）

1. `app/agent/tools.py` —— 两个工具的 schema 与 `execute_tool`  
2. `app/agent/runner.py` —— `run_agent_loop` 循环  
3. `app/services/llm_client.py` —— `chat_turn` / `_parse_chat_turn`  
4. `app/services/conversation.py` + `session_store.py` —— 多轮上下文  
5. `app/api.py` —— `_ask_agent` / session 注入  
6. `tests/test_agent_tools.py` / `test_session_conversation.py`  
7. `scripts/smoke_agent_tools.py` —— 真实验收标准

---

## 10. 常见坑

| 现象 | 可能原因 | 怎么办 |
|------|----------|--------|
| `tool_calls_count=0` | 模型没选工具；或索引空导致模型直接「无法确定」 | 加强 system prompt；确认索引有数据；问题用 ANR/OOM 等排障词 |
| `TOOL_INDEX_NOT_READY` | 没建索引 | `python scripts/build_kb_index.py` |
| `TOOL_TIMEOUT` | 工具超时太短 / 机器慢 | 调大 `AGENT_TOOL_TIMEOUT_SECONDS` |
| `AGENT_MAX_STEPS` | 模型反复调工具不收口 | 调 `AGENT_MAX_STEPS`；检查 tool 返回是否可读 |
| 和 `mode=rag` 搞混 | rag 日志是 `retrieve_*`，没有 `tool_call_*` | 看 `mode` 字段；agent 才有 tool_call 事件 |
| 冒烟失败但单测绿 | 单测 mock 了 LLM；冒烟才打真实 DeepSeek | 查 Key、网络、模型是否支持 tools |

---

## 11. 验收清单（Week 3）

- [x] 至少 2 个工具：`kb_search`、`kb_get_chunk`，schema 可传给 DeepSeek  
- [x] 任意请求 `agent_steps <= max_steps`  
- [x] `stop_reason` 可区分：`final_answer` / `clarify` / `degraded_to_rag` / `max_steps` / `timeout` / `upstream_error`  
- [x] Tool Runner 状态机：Plan → Act → Observe → Final  
- [x] `max_steps` + `AGENT_ON_MAX_STEPS`（rag / clarify / error）  
- [x] `max_total_time_ms`：超时兜底文案含 `request_id`  
- [x] `/ask?mode=agent`；`meta` 含 steps / stop_reason / phase_trace  
- [x] 冒烟：`smoke_agent_tools.py`（真实 tool_calls）  
- [x] 工具失败可控：`TOOL_INVALID_ARGS` / `TOOL_TIMEOUT` 等  
- [x] 多轮 `session_id`：滑窗 / 截断 / 摘要；`smoke_session_memory.py`  
- [x] `requests.jsonl`：`session_id` / `history_messages` / `history_chars`  
- [x] Agent 评测：`eval/agent_eval_samples.jsonl` ≥30 + `run_agent_eval.py`  
- [x] 单测：`test_agent_tools` / `test_session_conversation` / `test_run_agent_eval`  

---

## 12. 和前后周的关系

```text
Week1  能问答、能观测、能回归
Week2  有本地知识库 + rag 固定检索增强
Week3  模型自己调工具查库 + 多轮 + Agent 评测  ← 本周已完成
Week4  工程化可交付（检索质量 v2 / 安全 / 可观测 v2）见 [`week4_学习指南.md`](./week4_学习指南.md)（**已完成**）
```

默认分支约定：`week-1` / `week-2` / `week-3` 分周；`main` 保持最全。

---

## 13. 第三周完成总结（给自己复盘用）

### 你现在应该能讲清楚的三句话

1. **rag vs agent**：rag 是服务端先检索再问模型；agent 是模型自己发 `tool_calls`，工具结果以 `role=tool` 回传。  
2. **状态机为什么存在**：Plan/Act/Observe/Final 把「决策、执行、回写、收口」拆开，方便打日志、卡 `max_steps` / 总时长、做降级。  
3. **多轮靠什么不炸上下文**：session 存历史 → 进模型前截断 + 滑窗 +（可选）摘要；jsonl 只记条数/字符，不记原文。

### 建议亲手再跑一遍的命令

```bash
pytest -q tests/test_agent_tools.py tests/test_session_conversation.py tests/test_run_agent_eval.py
./scripts/start_server.sh 2>&1 | tee /tmp/app.log
python scripts/smoke_agent_tools.py --log /tmp/app.log
python scripts/smoke_session_memory.py
python scripts/run_agent_eval.py --limit 5 --skip-validate   # 全量去掉 --limit
```

### 读代码顺序（复习）

1. `tools.py` → 2. `runner.py` → 3. `llm_client.chat_turn` → 4. `conversation.py` / `session_store.py` → 5. `api._ask_agent` → 6. `run_agent_eval.py`
