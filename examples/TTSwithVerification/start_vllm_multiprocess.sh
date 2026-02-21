#!/bin/bash

# Start 4 vLLM processes with explicit GPU assignment
# Process 1: GPUs 0-1, Port 8000
# Process 2: GPUs 2-3, Port 8001
# Process 3: GPUs 4-5, Port 8002
# Process 4: GPUs 6-7, Port 8003

MODEL="Qwen/QwQ-32B"
GPU_MEMORY=0.4
TENSOR_PARALLEL=2

echo "Killing any existing vLLM processes..."
pkill -9 -f "vllm.entrypoints.openai.api_server"
sleep 2

echo "Starting 4 vLLM processes..."

# Process 1 - GPUs 0,1
(
    export CUDA_VISIBLE_DEVICES=0,1
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port 8000 \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --gpu-memory-utilization $GPU_MEMORY \
        --disable-log-requests \
        > /tmp/vllm_8000.log 2>&1
) &
PID1=$!
echo "Started Process 1 (GPUs 0-1, Port 8000) - PID: $PID1"

sleep 5

# Process 2 - GPUs 2,3
(
    export CUDA_VISIBLE_DEVICES=2,3
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port 8001 \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --gpu-memory-utilization $GPU_MEMORY \
        --disable-log-requests \
        > /tmp/vllm_8001.log 2>&1
) &
PID2=$!
echo "Started Process 2 (GPUs 2-3, Port 8001) - PID: $PID2"

sleep 5

# Process 3 - GPUs 4,5
(
    export CUDA_VISIBLE_DEVICES=4,5
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port 8002 \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --gpu-memory-utilization $GPU_MEMORY \
        --disable-log-requests \
        > /tmp/vllm_8002.log 2>&1
) &
PID3=$!
echo "Started Process 3 (GPUs 4-5, Port 8002) - PID: $PID3"

sleep 5

# Process 4 - GPUs 6,7
(
    export CUDA_VISIBLE_DEVICES=6,7
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port 8003 \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --gpu-memory-utilization $GPU_MEMORY \
        --disable-log-requests \
        > /tmp/vllm_8003.log 2>&1
) &
PID4=$!
echo "Started Process 4 (GPUs 6-7, Port 8003) - PID: $PID4"

echo ""
echo "All 4 vLLM processes started successfully."
echo "Process PIDs: $PID1 $PID2 $PID3 $PID4"
echo ""
echo "Log files:"
echo "  /tmp/vllm_8000.log - Process 1"
echo "  /tmp/vllm_8001.log - Process 2"
echo "  /tmp/vllm_8002.log - Process 3"
echo "  /tmp/vllm_8003.log - Process 4"
echo ""
echo "To stop all processes, run:"
echo "  pkill -9 -f 'vllm.entrypoints.openai.api_server'"
echo ""
echo "Waiting for processes to initialize (this may take 60-120 seconds)..."
echo ""

# Wait for all processes
wait $PID1 $PID2 $PID3 $PID4
