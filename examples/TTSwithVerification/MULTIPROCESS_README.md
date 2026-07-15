# Multi-Process vLLM Setup for Best-of-K Baseline

This directory contains scripts and code for running the best-of-K baseline with multi-process vLLM serving.

## Setup

### 1. Start vLLM with 4 processes (2 GPUs each)

```bash
bash start_vllm_multiprocess.sh
```

This launches 4 vLLM OpenAI-compatible API servers:
- **Process 1**: GPUs 0-1, Port 8000
- **Process 2**: GPUs 2-3, Port 8001  
- **Process 3**: GPUs 4-5, Port 8002
- **Process 4**: GPUs 6-7, Port 8003

Each process uses `tensor-parallel-size 2` for distributed inference.

### 2. Run the baseline

In a separate terminal:

```bash
# Test with 1 example
python bestofk_baseline.py --task game24 --num_examples 1 --k 4 --use_critic

# Run on maze dataset
python bestofk_baseline.py --task maze --num_examples 10 --k 4

# Run on spatialmap dataset
python bestofk_baseline.py --task spatialmap --num_examples 5 --k 4
```

Or use the test script:
```bash
bash run_multiprocess_test.sh game24 5
```

## Load Balancing

- Requests are distributed **round-robin** across the 4 vLLM instances
- Each generation request goes to the next available port (8000 → 8001 → 8002 → 8003 → 8000 ...)
- Critic evaluation requests use separate round-robin tracking (independent counter)
- This ensures even load distribution across all 4 GPU pairs

## Stopping vLLM

```bash
pkill -9 -f "vllm.entrypoints.openai.api_server"
```

## Configuration

Edit `start_vllm_multiprocess.sh` to change:
- `MODEL`: Model name (default: `Qwen/QwQ-32B`)
- `MAX_TOKENS`: Maximum sequence length (default: 8192)
- `GPU_MEMORY`: GPU memory utilization (default: 0.4)
- `TENSOR_PARALLEL`: Must be ≤ 2 for this 8-GPU setup

## Benefits

- **Better throughput**: 4 independent processes handle requests in parallel
- **Fault tolerance**: If one process crashes, others continue
- **GPU utilization**: Balanced load across all 8 GPUs (2 GPUs per process)
- **Reduced latency**: Each process has dedicated GPU resources
