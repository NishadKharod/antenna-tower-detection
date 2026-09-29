#!/usr/bin/env bash
# Quick-start production server (Linux/macOS)
# Usage: ./run_server.sh path/to/best.pt
set -e
WEIGHTS="${1:-runs/tower_detection/yolo11_tower_detector/weights/best.pt}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-2}"

cd "$(dirname "$0")"
exec gunicorn -w "$WORKERS" -b "$HOST:$PORT" \
  --chdir src \
  --timeout 120 \
  --env ATD_WEIGHTS="$WEIGHTS" \
  "app:create_app('$WEIGHTS')"
