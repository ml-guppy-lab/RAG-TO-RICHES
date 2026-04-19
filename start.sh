#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🎬 Starting RAG to Riches..."

# activate venv and start the FastAPI backend
cd "$SCRIPT_DIR/backend"
source .venv/bin/activate

echo "▶ Starting backend API on http://127.0.0.1:8000"
uvicorn api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# wait for the API to be ready
sleep 3

# start the Gradio frontend (uses the same activated venv)
echo "▶ Starting frontend at http://127.0.0.1:7860"
cd "$SCRIPT_DIR/frontend"
python3 app.py

# cleanup on exit
kill $API_PID 2>/dev/null
