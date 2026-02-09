# Early Stopping Monitors

Early stopping monitors reduce inference cost by detecting when a reasoning model has reached sufficient confidence in its answer and then terminating generation early. Instead of waiting for the model to exhaust its token budget, these monitors analyze intermediate signals (entropy, confidence, answer stability) and inject a `</think>` token to force the model to emit its final answer.

All early stopping monitors are used via `stream_completion`:

```python
from interwhen import stream_completion
from interwhen.monitors import EATMonitor, DEERMonitor, KstableAnswerMCQMonitor, KstableAnswerGame24Monitor

answer = await stream_completion(
    prompt,
    llm_server=llm_server,
    monitors=(your_monitor,),
    add_delay=False,
    termination_requires_validation=False,
    async_execution=True
)
```

---

## EAT (Entropy After Think)

Uses entropy-based early stopping. At each `\n\nWait` token boundary, the monitor computes the entropy of the next token and tracks the exponential moving average (EMA) variance. When the EMA variance drops below a threshold `delta` (after a minimum number of steps), the model's uncertainty has stabilized — meaning it is confident enough to answer — and generation is stopped by appending `</think>`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | — | Unique identifier for this monitor |
| `model_name` | str | — | HuggingFace model used to compute token entropy (e.g., `"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"`) |
| `alpha` | float | `0.2` | EMA smoothing factor — higher values weight recent entropy more |
| `delta` | float | `0.0001` | EMA variance threshold — generation stops when variance drops below this |
| `min_steps` | int | `4` | Minimum number of steps before early stopping can trigger |
| `answer_start_token` | str | `"</think>"` | Token that marks the transition from reasoning to answer |
| `async_execution` | bool | `True` | Whether to run verification asynchronously |

### Usage

```python
EATMonitor(
    name="EAT_monitor",
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    alpha=0.2,
    delta=0.0002,
    min_steps=4,
    answer_start_token="</think>",
    async_execution=True
)
```

---

## DEER (Dynamic Early Exit in Reasoning)

Uses answer confidence to decide when to stop. At each `\n\nWait` token boundary, the monitor appends a </think> token and sends it to an LLM server to compute the geometric mean confidence of the generated answer tokens. When confidence exceeds a threshold, generation is stopped.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | — | Unique identifier for this monitor |
| `model_name` | dict | — | LLM server configuration for confidence computation |
| `threshold` | float | `0.995` | Geometric mean confidence threshold — stops when exceeded |
| `answer_start_token` | str | `"</think>"` | Token that marks the transition from reasoning to answer |
| `async_execution` | bool | `True` | Whether to run verification asynchronously |
| `logprobs` | int | `20` | Number of log-probabilities to request from the server |

### Usage

```python
DEERMonitor(
    name="DEER_monitor",
    model_name=earlystop_model,
    threshold=0.80,
    answer_start_token="</think>",
    async_execution=True
)
```

---

## K-Stable Answer Monitor (MCQ)

Detects when the model has converged on a multiple-choice answer by monitoring its reasoning trace. When the same normalized answer appears `k` consecutive times in lines containing the word "answer", the monitor concludes the model has stabilized and triggers early stop.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | — | Unique identifier for this monitor |
| `k` | int | — | Number of consecutive identical answers required to trigger stop |
| `options` | dict | — | MCQ option mapping, e.g., `{"A": "Yes", "B": "No", "C": "2", "D": "4"}` |
| `answer_start_token` | str | `"</think>"` | Token that marks the transition from reasoning to answer |

### Usage

```python
KstableAnswerMCQMonitor(
    name="maze_kstable",
    k=3,
    options=options,
    answer_start_token="</think>"
)
```
---

## K-Stable Answer Monitor (Game of 24)

Same concept as the MCQ variant, but specialized for Game of 24. Monitors the reasoning trace for arithmetic expressions and triggers early stop when the same normalized equation appears `k` consecutive times. Optionally validates that the equation uses exactly the expected input numbers.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | — | Unique identifier for this monitor |
| `k` | int | — | Number of consecutive identical equations required to trigger stop |
| `expected_nums` | list or None | `None` | If set, validates the equation uses exactly these numbers |
| `answer_start_token` | str | `"</think>"` | Token that marks the transition from reasoning to answer |

### Usage

```python
KstableAnswerGame24Monitor(
    name="game24_kstable",
    k=3,
    expected_nums=nums,
    answer_start_token="</think>"
)
```

---

## Example Scripts

Each example script runs a full evaluation loop: loading a dataset, building prompts, running inference with an early stopping monitor, and computing accuracy/token statistics.

```bash
# Game of 24 with EAT early stopping
python ./examples/EarlyStopping/game24_example.py -n 1

# Maze MCQ with K-Stable answer monitor
python ./examples/EarlyStopping/maze_example.py -n 1

# SpatialMap MCQ with K-Stable answer monitor
python ./examples/EarlyStopping/spatialmap_example.py -n 1
```

### Common arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-n`, `--num_examples` | Number of examples to run | varies by task |
| `-m`, `--monitor` | Enable monitor (early stopping) | `True` |
| `-d`, `--debug` | Enable debug logging | `False` |
| `--main_model` | Main generation model | `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| `--earlystop_model` | Auxiliary model for EAT/DEER | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
