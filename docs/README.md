# 学习文档导航（给新手）

本目录是**按周跟做**的说明书。仓库业务代码在 `app/`，这里讲「做什么、为什么、怎么验收、出事怎么退」。

更短的项目总览见根目录 [`README.md`](../README.md)。

---

## 建议阅读顺序

| 顺序 | 文档 | 你在练什么 |
|------|------|------------|
| 0 | [日志阅读指南.md](./日志阅读指南.md) | 看懂终端 JSON / `request_id`（随时可查） |
| 1 | [week1_学习指南.md](./week1_学习指南.md) | 起服务、调 DeepSeek、错误处理、日志、回归 |
| 2 | [week2_学习指南.md](./week2_学习指南.md) | 建本地知识库 + RAG |
| 3 | [week3_学习指南.md](./week3_学习指南.md) | Agent 工具循环 + 多轮 session |
| 4 | [week4_学习指南.md](./week4_学习指南.md) | 混合检索、安全门禁、可观测 |
| 5 | [week5_学习指南.md](./week5_学习指南.md) | 评测加深 + 鉴权/入库/限流/反馈（含原 Week6） |

> **没有单独 Week6 文档**：产品化内容已并入 week5 的 Day21–25。

---

## 每篇文档里你要找什么

跟做时，尽量按这个节奏读（各周详略不同）：

| 板块 | 作用 |
|------|------|
| **本周目标 / 总览** | 先知道「做成什么样算过关」 |
| **为什么这样做** | 理解动机，避免只会抄命令 |
| **怎么做（步骤）** | 可复制命令；Windows 用户对照文内 PowerShell 提示 |
| **对照代码读哪里** | 知道改哪几个文件 |
| **怎么验收** | 文首「本周任务清单」+ 文末细清单（均为 `[ ]`，请自己勾） |
| **常见坑与回撤** | 401、索引坏了、回滚版本、先关掉鉴权再学等 |

样例输入在 [`eval/`](../eval/)；报告在 `reports/`；运行指标在 `data/runtime/`。  
日志文件建议写到 `reports/app.log`（不要依赖 Linux 专用的 `/tmp`）。

---

## 本地学习前 30 秒（很重要）

1. 复制环境变量：`cp .env.example .env`，填入 `DEEPSEEK_API_KEY`。  
2. **学习阶段建议先关掉鉴权/限流**（避免 curl 无 Key 一直 401）：

```env
API_AUTH_ENABLED=false
RATE_LIMIT_ENABLED=false
```

3. 需要练鉴权/限流时，再改回 `true`，并配置 `API_KEYS`（见 week5）。  
4. Windows：用 `.\.venv\Scripts\Activate.ps1` 激活虚拟环境；日志不要依赖 `/tmp`，可写到项目下 `reports/app.log`。

---

## 最小成功路径（不想一次读完时）

```text
Week1：起服务 → /health → /ask?mode=llm → pytest -q 部分契约测试
Week2：建索引 → /ask?mode=rag → 看到 citations
Week3：/ask?mode=agent → meta 里有 tool / agent_steps
Week4：跑注入 offline 评测；看 data/runtime/traces.jsonl
Week5：/v1/* + 反馈/回归；需要时再开鉴权
```

卡住时优先：[`日志阅读指南.md`](./日志阅读指南.md) → 对应周「常见坑」→ 根目录 README「能力边界」。
