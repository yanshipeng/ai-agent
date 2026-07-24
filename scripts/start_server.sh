#!/usr/bin/env bash
# =============================================================================
# 启动 FastAPI（推荐入口）
#
# 步骤：
#   1) 停止占用 PORT 的旧进程
#   2) 清理 __pycache__ / .pyc（新进程会重建 get_settings 缓存）
#   3) 用 .env 覆盖 shell 已 export 的同名变量（避免环境变量盖住 .env）
#   4) 前台启动 uvicorn；Ctrl+C 停止
#
# 用法：
#   ./scripts/start_server.sh
#   PORT=8001 ./scripts/start_server.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
APP_MODULE="${APP_MODULE:-app.main:app}"

echo "[start] project=$ROOT"
echo "[start] target=${HOST}:${PORT}"

# 1) 停止占用端口的旧进程
PIDS="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "${PIDS}" ]]; then
  echo "[start] stopping old process on :$PORT -> PID(s): $PIDS"
  kill ${PIDS} 2>/dev/null || true
  sleep 0.5
  STILL="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${STILL}" ]]; then
    echo "[start] force kill PID(s): $STILL"
    kill -9 ${STILL} 2>/dev/null || true
    sleep 0.3
  fi
else
  echo "[start] no listener on :$PORT"
fi

# 2) 清缓存（新进程会重建 get_settings 的 lru_cache；顺带清 pyc）
echo "[start] clearing python caches"
find "$ROOT" \
  \( -path "$ROOT/.venv" -o -path "$ROOT/.git" \) -prune -o \
  \( -type d -name '__pycache__' -print \) 2>/dev/null \
  | while read -r d; do rm -rf "$d"; done
find "$ROOT" \
  \( -path "$ROOT/.venv" -o -path "$ROOT/.git" \) -prune -o \
  \( -type f \( -name '*.pyc' -o -name '*.pyo' \) -print \) 2>/dev/null \
  | while read -r f; do rm -f "$f"; done
rm -rf "$ROOT/.pytest_cache" 2>/dev/null || true

# 3) 激活 venv（若存在）
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  echo "[start] using venv: $ROOT/.venv"
else
  echo "[start] WARNING: .venv not found, using system python"
fi

# 4) 强制让 .env 覆盖 shell 里已 export 的同名变量
#    pydantic-settings 默认：环境变量 > .env 文件；若不覆盖，改 .env 不生效
if [[ -f "$ROOT/.env" ]]; then
  if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[start] shell DEEPSEEK_API_KEY was set (will be overridden by .env; value not printed)"
  fi
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
  if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[start] DEEPSEEK_API_KEY loaded from .env (configured=yes, value not printed)"
  else
    echo "[start] WARNING: DEEPSEEK_API_KEY empty after loading .env"
  fi
else
  echo "[start] WARNING: .env not found"
fi

# 5) 启动（前台，方便看日志；Ctrl+C 停止）
echo "[start] starting uvicorn ${APP_MODULE}"
echo "[start] health: http://127.0.0.1:${PORT}/health"
echo "[start] docs:   http://127.0.0.1:${PORT}/docs"
exec uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT"
