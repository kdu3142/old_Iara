#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script supports macOS only."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Apple Silicon (arm64) is required."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required. Install Node.js (https://nodejs.org/) and retry."
  exit 1
fi

PYTHON_BIN="python3"
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Python 3.12+ required. Install python@3.12 and retry.")
PY

if [[ ! -d "$ROOT_DIR/client/node_modules" ]]; then
  echo "Installing client dependencies..."
  (cd "$ROOT_DIR/client" && npm install)
fi

if [[ ! -f "$ROOT_DIR/server/.env" ]]; then
  cp "$ROOT_DIR/server/env.example" "$ROOT_DIR/server/.env"
  echo "Created server/.env from env.example."
  echo "Update API keys in server/.env if needed."
fi

SERVER_CMD=""

if command -v uv >/dev/null 2>&1; then
  echo "Setting up server environment with uv..."
  (cd "$ROOT_DIR/server" && uv sync --prerelease=allow)
  (cd "$ROOT_DIR/server" && uv pip install espeakng_loader)
  SERVER_CMD="uv run bot.py"
else
  if [[ ! -d "$ROOT_DIR/server/.venv" ]]; then
    echo "Creating server virtual environment..."
    (cd "$ROOT_DIR/server" && "$PYTHON_BIN" -m venv .venv)
  fi

  echo "Installing server dependencies..."
  SERVER_VENV_PY="$ROOT_DIR/server/.venv/bin/python"
  if ! "$SERVER_VENV_PY" - <<'PY' >/dev/null 2>&1; then
import sys
raise SystemExit(0)
PY
    echo "Server virtual environment looks broken. Recreating..."
    (cd "$ROOT_DIR/server" && rm -rf .venv)
    (cd "$ROOT_DIR/server" && "$PYTHON_BIN" -m venv .venv)
    SERVER_VENV_PY="$ROOT_DIR/server/.venv/bin/python"
  fi
  "$SERVER_VENV_PY" -m pip install -r "$ROOT_DIR/server/requirements.txt"
  "$SERVER_VENV_PY" -m pip install espeakng_loader

  SERVER_CMD="$SERVER_VENV_PY bot.py"
fi

TTS_VENV="$ROOT_DIR/server/.venv-qwen"
if [[ ! -d "$TTS_VENV" ]]; then
  echo "Creating TTS worker virtual environment..."
  (cd "$ROOT_DIR/server" && "$PYTHON_BIN" -m venv ".venv-qwen")
fi
echo "Installing TTS worker dependencies..."
TTS_VENV_PY="$TTS_VENV/bin/python"
if ! "$TTS_VENV_PY" - <<'PY' >/dev/null 2>&1; then
import sys
raise SystemExit(0)
PY
  echo "TTS virtual environment looks broken. Recreating..."
  (cd "$ROOT_DIR/server" && rm -rf .venv-qwen)
  (cd "$ROOT_DIR/server" && "$PYTHON_BIN" -m venv ".venv-qwen")
  TTS_VENV_PY="$TTS_VENV/bin/python"
fi
"$TTS_VENV_PY" -m pip install --upgrade pip
"$TTS_VENV_PY" -m pip install --pre -U "mlx-audio>=0.3.1"
"$TTS_VENV_PY" -m pip install -U espeakng_loader
export TTS_WORKER_PYTHON="$TTS_VENV/bin/python"
export QWEN_TTS_PYTHON="$TTS_VENV/bin/python"

cleanup() {
  if [[ -n "${CLIENT_PID:-}" ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
    kill "$CLIENT_PID"
  fi
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
  fi
  wait "${CLIENT_PID:-}" "${SERVER_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting server..."
(
  cd "$ROOT_DIR/server"
  $SERVER_CMD
) &
SERVER_PID=$!

echo "Starting client..."
(
  cd "$ROOT_DIR/client"
  npm run dev
) &
CLIENT_PID=$!

echo "Press Ctrl+C to stop."
wait "$CLIENT_PID" "$SERVER_PID"
