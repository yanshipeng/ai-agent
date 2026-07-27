#!/usr/bin/env bash
# Day 6–7：扩采（可选）→ docs.jsonl → chunks.jsonl
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

TARGET="${TARGET:-30}"
EXPAND="${EXPAND:-0}"

if [[ "$EXPAND" == "1" ]]; then
  echo "======== expand crawl target=${TARGET}/category ========"
  python scripts/crawl_stability_kb.py --domestic-only --target "$TARGET" --delay 0.35
fi

echo "======== Day 6: build docs.jsonl ========"
python scripts/build_stability_docs.py --delay 0.3

echo "======== Day 7: chunk docs → chunks.jsonl ========"
python scripts/chunk_stability_docs.py --chunk-size 1000 --overlap 120

echo "======== done ========"
wc -l data/stability_kb/docs.jsonl data/stability_kb/chunks.jsonl
