#!/usr/bin/env bash
set -e

# ANSI Color Codes
CYAN="\033[1;36m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${ROOT_DIR}/backend"
FRONTEND_DIR="${ROOT_DIR}/frontend"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5174}"

# Helper to kill any existing process occupying a target port
kill_port() {
    local port="$1"
    local name="$2"
    if command -v fuser >/dev/null 2>&1; then
        local pids
        pids=$(fuser "${port}/tcp" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo -e "${YELLOW}[SENTRIX]${RESET} Freeing port ${port} (${name})..."
            fuser -k -9 "${port}/tcp" >/dev/null 2>&1 || true
            sleep 0.3
        fi
    elif command -v lsof >/dev/null 2>&1; then
        local pids
        pids=$(lsof -ti tcp:"${port}" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo -e "${YELLOW}[SENTRIX]${RESET} Freeing port ${port} (${name})..."
            echo "$pids" | xargs -r kill -9 2>/dev/null || true
            sleep 0.3
        fi
    fi
}

# Cleanup function to cleanly stop all processes on Ctrl+C (SIGINT) / SIGTERM / EXIT
cleanup() {
    trap - INT TERM EXIT
    echo ""
    echo -e "${YELLOW}[SENTRIX]${RESET} Shutting down servers..."
    if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${YELLOW}[SENTRIX]${RESET} Stopping backend server (PID: $BACKEND_PID)..."
        kill -TERM "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "${YELLOW}[SENTRIX]${RESET} Stopping frontend server (PID: $FRONTEND_PID)..."
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    fi
    # Also kill child background jobs and ensure ports are freed
    jobs -p | xargs -r kill 2>/dev/null || true
    kill_port "$BACKEND_PORT" "Backend"
    kill_port "$FRONTEND_PORT" "Frontend"
    wait 2>/dev/null || true
    echo -e "${YELLOW}[SENTRIX]${RESET} All servers stopped."
}

trap cleanup INT TERM EXIT

echo -e "${YELLOW}[SENTRIX]${RESET} Starting Sentrix AI Review Analytics Platform..."

# Ensure target ports are free before starting
kill_port "$BACKEND_PORT" "Backend"
kill_port "$FRONTEND_PORT" "Frontend"

# Prefix helper functions (line-buffered real-time streaming)
prefix_backend() {
    while IFS= read -r line || [ -n "$line" ]; do
        echo -e "${CYAN}[BACKEND]${RESET} $line"
    done
}

prefix_frontend() {
    while IFS= read -r line || [ -n "$line" ]; do
        echo -e "${GREEN}[FRONTEND]${RESET} $line"
    done
}

# Start Backend
echo -e "${CYAN}[BACKEND]${RESET} Starting on http://localhost:${BACKEND_PORT}..."
(
    cd "$BACKEND_DIR"
    export PYTHONPATH=src
    export PYTHONUNBUFFERED=1
    uv run uvicorn src.main:app --port "$BACKEND_PORT" --reload 2>&1 | prefix_backend
) &
BACKEND_PID=$!

# Start Frontend
echo -e "${GREEN}[FRONTEND]${RESET} Starting on http://localhost:${FRONTEND_PORT}..."
(
    cd "$FRONTEND_DIR"
    npm run dev 2>&1 | prefix_frontend
) &
FRONTEND_PID=$!

echo -e "${YELLOW}[SENTRIX]${RESET} Servers running. Press ${RED}Ctrl+C${RESET} to stop both."

# Wait for background processes
wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
