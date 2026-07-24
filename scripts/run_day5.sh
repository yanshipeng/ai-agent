#!/usr/bin/env bash
# =============================================================================
# Day 5 一键验证：契约/映射单测 → 启动服务 → 回归跑批 → 停服务
#
# 用法：
#   ./scripts/run_day5.sh
#   SKIP_EVAL=1 ./scripts/run_day5.sh          # 只跑 pytest，不跑真实 /ask 回归
#   EVAL_LIMIT=5 ./scripts/run_day5.sh         # 回归只跑前 N 条
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
REPORT_DIR="${REPORT_DIR:-./reports}"
TS="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="${REPORT_DIR}/day5_server_${TS}.log"
REPORT_PATH="${REPORT_DIR}/eval_run_report_${TS}.json"
RESULTS_PATH="${REPORT_DIR}/eval_results_${TS}.jsonl"

mkdir -p "$REPORT_DIR"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

echo "======== Day 5.1 + 5.2: pytest (contract + mapping, mock LLM) ========"
pytest -q tests/test_contract.py tests/test_error_mapping.py
echo "[ok] contract + mapping tests passed"

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "SKIP_EVAL=1, skip regression runner"
  exit 0
fi

echo "======== start server on :${PORT} ========"
# 复用 start_server 的停旧进程逻辑，但后台跑以便继续回归
PIDS="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "${PIDS}" ]]; then
  kill ${PIDS} 2>/dev/null || true
  sleep 0.5
  STILL="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${STILL}" ]]; then
    kill -9 ${STILL} 2>/dev/null || true
  fi
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

uvicorn app.main:app --host 0.0.0.0 --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[start] uvicorn pid=$SERVER_PID log=$SERVER_LOG"

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[stop] killing uvicorn pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 0.5
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "======== wait /health ========"
READY=0
for i in $(seq 1 60); do
  if curl -sf "$BASE_URL/health" >/dev/null; then
    READY=1
    echo "[ok] health ready (${i}s)"
    break
  fi
  sleep 1
done
if [[ "$READY" != "1" ]]; then
  echo "[fail] server not healthy; see $SERVER_LOG" >&2
  exit 1
fi

echo "======== Day 5.3: regression runner ========"
EVAL_CMD=(
  python scripts/run_eval.py
  --samples ./eval_samples.jsonl
  --base-url "$BASE_URL"
  --results "$RESULTS_PATH"
  --report "$REPORT_PATH"
)
if [[ -n "$EVAL_LIMIT" ]]; then
  EVAL_CMD+=(--limit "$EVAL_LIMIT")
fi
"${EVAL_CMD[@]}"

# 同步一份固定文件名，便于「当前最新报告」查看
cp "$REPORT_PATH" ./eval_run_report.json
cp "$RESULTS_PATH" ./eval_results.jsonl

echo "======== Day 5 done ========"
echo "report (timestamped): $REPORT_PATH"
echo "report (latest)     : ./eval_run_report.json"
echo "results             : ./eval_results.jsonl"
echo "server log          : $SERVER_LOG"
