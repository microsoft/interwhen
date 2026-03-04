#!/bin/bash

# Cleanup script to kill all vLLM processes and Python instances

echo "Stopping all vLLM processes..."
pkill -9 -f "vllm.entrypoints.openai.api_server"

echo "Stopping Python processes..."
pkill -9 -f "bestofk_baseline.py"

sleep 2

echo "Verifying all processes stopped..."
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "WARNING: Some vLLM processes still running"
else
    echo "✓ All vLLM processes stopped"
fi

if pgrep -f "bestofk_baseline.py" > /dev/null; then
    echo "WARNING: Some Python processes still running"
else
    echo "✓ All Python processes stopped"
fi

echo "Cleanup complete"
