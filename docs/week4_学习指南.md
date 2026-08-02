# 第四周学习指南：工程化可交付（检索质量 / 安全 / 可观测）

> 面向刚入行同学。前三周已有：LLM 服务壳 → RAG v1 → Agent v1。  
> 第四周目标：从「能跑」升级为「工程化可交付」——检索更稳、上线更安全、排障可回放。  
> **状态：第四周 Day16–18 已完成**（见文末「完成总结」）。  
> 第一～三周：[`week1_学习指南.md`](./week1_学习指南.md) · [`week2_学习指南.md`](./week2_学习指南.md) · [`week3_学习指南.md`](./week3_学习指南.md)  
> 服务日志：[`日志阅读指南.md`](./日志阅读指南.md)。

---

## 0. 本次改了什么（变更清单 · 终版）

相对第三周结束时的代码，第四周落地如下能力。

### 新增文件

| 路径 | 作用 |
|------|------|
| `app/kb/bm25.py` | 简易 Okapi BM25 + min-max 归一化（Day16） |
| `app/core/safety.py` | 安全规则 / 注入预检 / 引用门禁 / 泄密扫描（Day17） |
| `app/services/token_budget.py` | 启发式 token 估算 + 超预算压缩（Day18） |
| `eval_samples_injection.jsonl` | 注入样例 ≥10（Day17） |
| `scripts/spotcheck_retrieve_day16.py` | 随机抽检 20 条 Top3 + 主观理由表 |
| `scripts/run_injection_eval.py` | 注入评测（离线/在线）→ `reports/injection_eval_report.json` |
| `tests/test_hybrid_retrieve.py` | Day16 混合检索 / 去重 |
| `tests/test_safety.py` | Day17 安全单测 |
| `tests/test_injection_eval.py` | Day17 注入评测验收 |
| `tests/test_observability_day18.py` | Day18 Trace / Budget / Cache |
| `docs/week4_学习指南.md` | 本文档 |

### 修改文件（行为变化）

| 路径 | 改动要点 |
|------|----------|
| `app/kb/retriever.py` | hybrid 融合 → 过滤/去重 → TopK；`cache_hit`/`cache_miss`；检索统计字段 |
| `app/kb/rag.py` | 透传检索统计；Context「文档非指令」横幅 + 安全 prompt |
| `app/agent/runner.py` | 抗注入 prompt；`agent_trace` 逐步记录；Plan 前 token 预算 |
| `app/agent/tools.py` | `HIGH_RISK_TOOLS` → `TOOL_NEEDS_APPROVAL`（HITL） |
| `app/api.py` | 注入预检；引用/泄密门禁；Day18 meta + `traces.jsonl`；请求开始 reset cache |
| `app/services/metrics_store.py` | Day16 检索字段；Day18 `agent_trace` / budget / cache；`append_trace_metric` |
| `app/services/llm_client.py` | `max_tokens` 读 `LLM_MAX_TOKENS` |
| `app/core/config.py` | `RAG_HYBRID_*` / `AGENT_MAX_CONTEXT_TOKENS` / `TRACES_JSONL_PATH` |
| `scripts/stats_requests.py` | 去重流 + `obs_v2`（trace / context_tokens / cache） |
| `.env.example` / `README.md` | 环境变量与用法 |

### 没有改动的边界

- 默认仍是 `mode=llm`（兼容 Week1）
- 语料流水线（crawl → docs → chunks → index）不变；本周主要改**检索后处理**与**服务侧护栏**
- Token 估算是启发式（非官方 tokenizer），够做预算门禁，不必与上游 usage 完全一致
- 缓存是进程内索引/BM25 缓存，不是 Redis/LLM 响应缓存

---

## 1. 第四周要达成什么

| 天数 | 主题 | 一句话 |
|------|------|--------|
| Day16 | 检索质量 v2 | 向量 + BM25 融合，去重过滤，引用覆盖不掉 |
| Day17 | 安全与抗注入 | 预检拒答、引用门禁、高风险工具 HITL |
| Day18 | 可观测 v2 | 每请求一条 Trace + Token Budget + Cache 指标 |

```text
用户 POST /ask
  →（Day17）注入预检？拒答 : 继续
  → mode=rag  ：hybrid retrieve → Context → LLM → 引用/泄密门禁
  → mode=agent：token 预算 → Plan/Act/Observe → agent_trace 落盘
  →（Day18）requests.jsonl + traces.jsonl + cache_hit/miss
```

---

## 总览

| 天数 | 主题 | 目标 |
|------|------|------|
| Day16 | 检索质量 v2 | 混合检索 + 去重/过滤 + 可观测 |
| Day17 | 安全与抗注入 | Prompt 隔离 / 引用门禁 / 工具 HITL |
| Day18 | 可观测 v2 | Trace + Token Budget + Cache 指标 |

---

## Day16：检索质量 v2（混合检索 + 去重/过滤）

### 要做什么

1. **混合检索（Hybrid Search）**  
   - 向量（余弦）+ 关键词（BM25）  
   - 融合：`score = hybrid_weight * vec_norm + (1 - hybrid_weight) * bm25_norm`
2. **结果去重与过滤**  
   - 去重：同 `doc_id` 上限、同 `url`、高相似正文（Jaccard）  
   - 过滤：低分阈值、过短 chunk、噪声行（导航/广告）
3. **检索可观测**  
   - `retrieve_ms` / `retrieve_candidates`  
   - `retrieve_before_dedup` → `retrieve_after_dedup`  
   - `retrieve_kept` / `hybrid_weight` / `dedup_dropped`

### 原理（一句话）

纯向量对「专名/错误码」不稳；BM25 补关键词；先放大候选再去重，避免同一文档占满 TopK。

### 配置

```bash
RAG_HYBRID_WEIGHT=0.6   # 向量占比；关键词 = 0.4
RAG_MIN_SCORE=0.05      # 融合分阈值（归一化后）
```

### 怎么验收

```bash
pytest -q tests/test_hybrid_retrieve.py tests/test_kb_retrieve.py tests/test_rag_ask.py

python scripts/retrieve_kb.py "Android ANR 怎么排查"

# ① 随机抽检 20 条 Top3（本地，不打 LLM）
python scripts/spotcheck_retrieve_day16.py
# 打开 reports/day16_spotcheck.md，逐条填 top3_more_relevant + reason

# ② 经 /ask 写 jsonl（需服务已启动）
python scripts/spotcheck_retrieve_day16.py --via-ask
python scripts/stats_requests.py --path ./requests.jsonl --mode rag

python scripts/run_rag_eval.py   # normal 类引用覆盖率不下降
```

### 日志 / 指标字段

| 字段 | 含义 |
|------|------|
| `retrieve_ms` | 检索阶段耗时 |
| `retrieve_candidates` | 进入过滤前的候选池大小 |
| `retrieve_before_dedup` | 过滤后、去重前 |
| `retrieve_after_dedup` | 去重后（截 TopK 前） |
| `retrieve_kept` | 最终保留条数 |
| `hybrid_weight` | 向量权重 |
| `dedup_dropped` | 去重丢弃数 |

### 验收标准

- `eval_samples_rag.jsonl` 中 **normal** 类：**引用覆盖率不下降**
- **随机抽检 20 条** Top3 主观更相关 → `reports/day16_spotcheck.md`
- `requests.jsonl` 能看到 `retrieve_ms` 与 before→after 去重流

### 读代码顺序（Day16）

1. `app/kb/bm25.py` → 2. `app/kb/retriever.py` → 3. `app/kb/rag.py` → 4. `app/api.py`（`retrieve_end`）→ 5. `tests/test_hybrid_retrieve.py`

---

## Day17：安全与抗注入（RAG/Agent 上线门槛）

### 要做什么

1. **Prompt 注入防护**  
   - 用户输入 vs 检索文档隔离（文档无指令优先级）  
   - System 写死：拒泄密、拒覆盖系统规则  
   - `/ask` 入口注入预检：命中则拒答，不调 LLM/工具
2. **引用强约束**  
   - 回答中 `[n]` 必须落在 `citations.ref_id`（非法编号运行时剔除）  
   - 像确定性结论却无引用时打 `citation_guard` meta
3. **工具安全**  
   - 白名单 + 参数校验 + 超时（Week3）  
   - 高风险工具（如 `execute_sql`）→ `TOOL_NEEDS_APPROVAL`（HITL，不执行）

### 变更清单

| 路径 | 作用 |
|------|------|
| `app/core/safety.py` | 规则文本 + 预检 + 引用门禁 + 泄密扫描 |
| `app/kb/rag.py` / `app/agent/runner.py` | 安全 prompt + 文档横幅 |
| `app/agent/tools.py` | `HIGH_RISK_TOOLS` + `TOOL_NEEDS_APPROVAL` |
| `app/api.py` | 预检拒答；返回前 `enforce_*` |
| `eval_samples_injection.jsonl` | ≥10 条注入样例 |
| `scripts/run_injection_eval.py` | 离线/在线评测 |
| `tests/test_safety.py` / `test_injection_eval.py` | 单测 + 验收 |

### 怎么验收

```bash
pytest -q tests/test_safety.py tests/test_injection_eval.py

python -c "from app.agent.tools import execute_tool; print(execute_tool('execute_sql', {'sql':'select 1'}))"
# 期望：TOOL_NEEDS_APPROVAL

python scripts/run_injection_eval.py --offline
# 报告：reports/injection_eval_report.json
# 期望：refusal_rate ≥ 90%，leakage_rate == 0
```

### 响应 meta 字段

| 字段 | 含义 |
|------|------|
| `injection_blocked` | 注入预检命中，已拒答 |
| `leakage_blocked` | 回答疑似泄密，已替换拒答 |
| `citation_guard` | `stripped_invalid_refs` / `missing_refs_for_claims` / `injection_precheck` |
| `citation_invalid_refs` | 被剔除的非法 `[n]` |
| `citation_missing_for_claims` | 像结论但正文无引用 |
| `citation_refs_used` | 正文实际使用的引用号 |

### 验收标准

- 注入样例 ≥ **10**；拒答率 ≥ **90%**；`leakage_rate == 0`
- `execute_sql` 等返回 `TOOL_NEEDS_APPROVAL`

### 读代码顺序（Day17）

1. `app/core/safety.py` → 2. `app/api.py`（注入预检）→ 3. `eval_samples_injection.jsonl` → 4. `scripts/run_injection_eval.py` → 5. `tests/test_injection_eval.py`

---

## Day18：可观测 v2（Trace + Token Budget + Cache）

### 要做什么

1. **Agent Trace**（每次请求一条，含 steps）  
   - `step_idx` / `action`：`plan` / `tool_call` / `final` / `clarify` / `degrade`  
   - tool 步：`tool_name` / `tool_latency_ms` / `tool_ok` / `tool_error_code`  
   - 写入 `requests.jsonl.agent_trace`，并另写 `traces.jsonl`
2. **Token 预算**  
   - `max_context_tokens` / `context_tokens_used` / `max_output_tokens`  
   - 超预算：先压缩（截断 tool/长文）；仍超 → 澄清 + 排查 checklist
3. **缓存指标**（索引 / BM25 进程内缓存）  
   - `cache_hit` / `cache_miss`（每个 `/ask` 开始清零）

### 配置

| 环境变量 | 默认 | 含义 |
|----------|------|------|
| `AGENT_MAX_CONTEXT_TOKENS` | 6000 | Agent context 预算（启发式） |
| `TRACES_JSONL_PATH` | `./traces.jsonl` | 逐步 trace 落盘 |
| `LLM_MAX_TOKENS` | 2048 | 输出侧 `max_output_tokens` |

### 怎么验收

```bash
pytest -q tests/test_observability_day18.py

# 服务已启动后（Windows PowerShell 请用文件体，见「常见坑」）
curl.exe -s -X POST "http://127.0.0.1:8000/ask?mode=agent" `
  -H "Content-Type: application/json" --data-binary "@ask.json"

python scripts/stats_requests.py --mode agent
# 期望出现：obs_v2.count / trace_steps / context_tokens / cache hit/miss
# 另：Get-Content traces.jsonl -Tail 1
```

### meta / jsonl 字段

| 字段 | 含义 |
|------|------|
| `agent_trace` | 逐步 action 列表 |
| `max_context_tokens` | context 预算 |
| `context_tokens_used` | 估算已用 |
| `max_output_tokens` | 输出上限 |
| `budget_compressed` | 是否做过压缩 |
| `cache_hit` / `cache_miss` | 本请求索引缓存命中 |

### 读代码顺序（Day18）

1. `app/services/token_budget.py` → 2. `app/agent/runner.py`（`_append_trace` / `_apply_token_budget`）→ 3. `app/api.py`（落盘）→ 4. `app/services/metrics_store.py` → 5. `tests/test_observability_day18.py`

---

## 2. 配置一览（Week4 新增）

| 变量 | 默认 | 说明 |
|------|------|------|
| `RAG_HYBRID_WEIGHT` | `0.6` | 混合检索向量占比 |
| `RAG_MIN_SCORE` | `0.05` | 融合分阈值 |
| `AGENT_MAX_CONTEXT_TOKENS` | `6000` | Agent context token 预算 |
| `TRACES_JSONL_PATH` | `./traces.jsonl` | Agent Trace 落盘路径 |

其余见 [README §3](../README.md) / `.env.example`。

---

## 3. 一键验收（整周）

```bash
# 单测（不打网）
pytest -q \
  tests/test_hybrid_retrieve.py \
  tests/test_safety.py \
  tests/test_injection_eval.py \
  tests/test_observability_day18.py

# Day17 离线注入评测
python scripts/run_injection_eval.py --offline

# Day16 抽检（可选填主观理由）
python scripts/spotcheck_retrieve_day16.py

# 服务起来后：agent + stats
python -m app.main   # 或 bash scripts/start_server.sh
# 写 ask.json 后：
# curl.exe ... --data-binary "@ask.json"
python scripts/stats_requests.py --mode agent
```

---

## 4. 常见坑

| 现象 | 可能原因 | 怎么办 |
|------|----------|--------|
| `stats` 没有 `obs_v2` / `dedup_flow` | 旧进程写的 jsonl，或服务未重启 | 重启 `python -m app.main`，再打一枪新请求 |
| PowerShell `JSON decode error` | `-d "{\"query\":...}"` 引号被壳吃掉 | 用 `--data-binary "@ask.json"` |
| `budget_compressed` 很多 | history/tool 结果太长 | 调大 `AGENT_MAX_CONTEXT_TOKENS`，或缩短 session |
| `trace_steps` 接近 `max_steps*几` | 模型多轮 tool_calls | 正常；看 `stop_reason` 是否收口 |
| BM25 单文档分数被抹成 0 | 旧版 min-max 同分归零 | 已修：正分且全等时归一为 `1.0` |
| 注入评测拒答率低 | 样例与预检正则不对齐 | 看 `eval_samples_injection.jsonl` 与 `INJECTION_PATTERNS` |

---

## 5. 验收清单（Week 4）

- [x] Day16：hybrid + 过滤/去重；`retrieve_*` 字段进 jsonl/meta  
- [x] Day16：抽检脚本 + `stats_requests --mode rag` 去重流  
- [x] Day17：注入预检 + 引用门禁 + 泄密扫描  
- [x] Day17：`TOOL_NEEDS_APPROVAL`；注入评测 refusal≥90%、leakage=0  
- [x] Day18：`agent_trace` + `traces.jsonl`  
- [x] Day18：token 预算压缩/澄清；`cache_hit`/`cache_miss`  
- [x] Day18：`stats_requests --mode agent` 输出 `obs_v2`  
- [x] 单测：`test_hybrid_retrieve` / `test_safety` / `test_injection_eval` / `test_observability_day18`  

---

## 6. 和前后周的关系

```text
Week1  能问答、能观测、能回归
Week2  有本地知识库 + rag 固定检索增强
Week3  模型自己调工具查库 + 多轮 + Agent 评测
Week4  检索更稳 + 安全护栏 + Trace/预算/缓存  ← 本周已完成
```

默认分支约定：`week-1` / `week-2` / `week-3` / `week-4` 分周；`main` 保持最全。

---

## 7. 第四周完成总结（给自己复盘用）

### 你现在应该能讲清楚的三句话

1. **为什么 hybrid**：向量擅长语义，BM25 稳住专名/错误码；融合后再去重，TopK 不挤满同一文档。  
2. **上线为什么要 Day17**：文档可能含恶意指令；预检 + 引用门禁 + HITL，把「泄密/越权」挡在业务外。  
3. **Day18 解决什么**：`agent_phase_trace` 太粗；逐步 `agent_trace` + token 预算 + cache 计数，才能算成本、回放失败步。

### 建议亲手再跑一遍的命令

```bash
pytest -q tests/test_hybrid_retrieve.py tests/test_injection_eval.py tests/test_observability_day18.py
python scripts/run_injection_eval.py --offline
python scripts/spotcheck_retrieve_day16.py
# 启服务后打 mode=agent，再：
python scripts/stats_requests.py --mode agent
```

### 读代码顺序（复习 · 整周）

1. `bm25.py` + `retriever.py`（Day16）  
2. `safety.py` + `api` 注入预检（Day17）  
3. `token_budget.py` + `runner._append_trace`（Day18）  
4. `metrics_store` / `stats_requests`（落盘与汇总）  
5. 对应 `tests/test_*` 与 `scripts/*eval*`
