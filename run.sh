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

# Export server env vars so the client proxy uses the same host/port.
if [[ -f "$ROOT_DIR/server/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/server/.env"
  set +a
fi

IARA_SERVER_HOST="${IARA_SERVER_HOST:-localhost}"
IARA_SERVER_PORT="${IARA_SERVER_PORT:-7860}"

is_port_free() {
  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

if ! is_port_free "$IARA_SERVER_PORT"; then
  echo "Port $IARA_SERVER_PORT is in use. Searching for a free port..."
  for p in $(seq 7861 7899); do
    if is_port_free "$p"; then
      IARA_SERVER_PORT="$p"
      echo "Using port $IARA_SERVER_PORT."
      break
    fi
  done
  if ! is_port_free "$IARA_SERVER_PORT"; then
    echo "No free port found in range 7861-7899. Set IARA_SERVER_PORT in server/.env."
    exit 1
  fi
fi

export IARA_SERVER_HOST IARA_SERVER_PORT

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
"$TTS_VENV_PY" -m pip install -U misaki
"$TTS_VENV_PY" -m pip install -U num2words
"$TTS_VENV_PY" -m pip install -U spacy
"$TTS_VENV_PY" -m pip install -U phonemizer
"$TTS_VENV_PY" -m pip install -U espeakng_loader
export TTS_WORKER_PYTHON="$TTS_VENV/bin/python"
export QWEN_TTS_PYTHON="$TTS_VENV/bin/python"

cleanup() {
  # Kill process groups to ensure child processes (e.g. uvicorn, workers) exit too.
  if [[ -n "${CLIENT_PID:-}" ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
    kill -TERM "-$CLIENT_PID" 2>/dev/null || kill -TERM "$CLIENT_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  fi
  # Give processes a moment to exit cleanly.
  for _ in {1..20}; do
    if { [[ -z "${CLIENT_PID:-}" ]] || ! kill -0 "$CLIENT_PID" 2>/dev/null; } && \
       { [[ -z "${SERVER_PID:-}" ]] || ! kill -0 "$SERVER_PID" 2>/dev/null; }; then
      break
    fi
    sleep 0.1
  done
  # Force kill if still running.
  if [[ -n "${CLIENT_PID:-}" ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
    kill -KILL "-$CLIENT_PID" 2>/dev/null || kill -KILL "$CLIENT_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -KILL "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
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
