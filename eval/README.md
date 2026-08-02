# 评测样例（输入）

本目录只放**可入库的评测样例**，不放运行产物。

| 文件 | 用途 |
|------|------|
| `eval_samples.jsonl` | Week1：LLM 回归（≥20） |
| `eval_samples_rag.jsonl` | Week2：RAG 评测（≥50） |
| `eval_rag_samples.jsonl` | Week2：RAG 快速验收 |
| `agent_eval_samples.jsonl` | Week3：Agent 评测（≥30） |
| `eval_samples_injection.jsonl` | Week4：注入安全样例（≥10） |
| `eval_samples_v2.jsonl` | Week5：评测 v2（≥80） |
| `eval_samples_feedback.jsonl` | Week5：反馈 badcase 回归 |

运行产物（报告 / 明细 / 指标）统一写到：

- `reports/`：评测报告与明细
- `data/runtime/`：`requests.jsonl`、`traces.jsonl` 等运行时指标
