# Test-Time Search with Step Verification Monitors

Step verification monitors improve reasoning accuracy by checking the model's intermediate steps in real time and providing corrective feedback when errors are detected. Unlike early stopping monitors (which only decide *when* to stop), these monitors actively verify *what* the model is producing and steer it back on track when it goes wrong.

Each monitor uses domain-specific verifiers — arithmetic checks for Game of 24, grid navigation for Maze, and Z3 constraint solving for SpatialMap — to catch mistakes as they happen rather than after the full response has been generated.

All step verification monitors are used via `stream_completion`:

```python
from interwhen import stream_completion
from interwhen.monitors import StepVerifierGame24Monitor, StepVerifierMazeMonitor, StepVerifierSpatialMapMonitor

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

## Game of 24 Step Verifier

Verifies Game of 24 solutions step by step. The model is prompted to output a structured format where each step declares available numbers, a suggested arithmetic operation, and the remaining numbers. The monitor verifies that each operation is mathematically correct, uses valid available numbers, and produces correct remaining numbers. On error, it appends feedback prompting the model to retry that step.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | — | Unique identifier for this monitor |
| `answer_start_token` | str | — | Token marking the transition from reasoning to answer (typically `"</think>"`) |
| `original_numbers` | list | — | The four input numbers for the Game of 24 problem |
| `max_corrections` | int | `5` | Maximum number of correction attempts before giving up |
| `async_execution` | bool | `True` | Whether to run verification asynchronously |

### Usage

```python
StepVerifierGame24Monitor(
    name="game24_step_verifier",
    answer_start_token="</think>",
    original_numbers=[1, 2, 6, 8],
    max_corrections=5,
)
```

### Expected model output format

The model is prompted to produce structured steps using a meta prompt:

```
>Step1
available numbers: [1, 2, 6, 8]
suggested operation: 8 / 2 = 4
remaining numbers: [4, 1, 6]

>Step2
available numbers: [4, 1, 6]
suggested operation: 6 * 4 = 24
remaining numbers: [24, 1]
```
---

## Maze Step Verifier

Verifies maze navigation step by step against the actual maze grid. The model outputs structured steps declaring movement direction, from/to positions, turn type, and running turn counts. The monitor checks all of these against the ground-truth maze grid — verifying that moves are valid, positions are walkable, turn classifications are correct, and running counts are accurate.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | — | Unique identifier for this monitor |
| `answer_start_token` | str | — | Token marking the transition from reasoning to answer |
| `grid` | list | — | The maze grid (2D list of characters) |
| `start_pos` | tuple | — | Starting position `(row, col)` |
| `exit_pos` | tuple | — | Exit position `(row, col)` |
| `max_corrections` | int | `5` | Maximum correction attempts per example |
| `question_type` | str | `"right_turns"` | One of `"right_turns"`, `"total_turns"`, or `"relative_position"` |
| `async_execution` | bool | `True` | Whether to run verification asynchronously |

### Factory method

You can also create the monitor directly from a prompt using the factory method:

```python
monitor = StepVerifierMazeMonitor.from_prompt(
    prompt_text=user_prompt,
    max_corrections=5,
    name="maze_step_verifier"
)
```

This automatically parses the maze grid, start/exit positions, and auto-detects the question type from the prompt text.

### Usage

```python
from interwhen.utils.maze_verifier import parse_maze_from_prompt

grid, start_pos, exit_pos = parse_maze_from_prompt(user_prompt)
question_type = StepVerifierMazeMonitor.detect_question_type(user_prompt)

StepVerifierMazeMonitor(
    name="maze_step_verifier",
    answer_start_token="</think>",
    grid=grid,
    start_pos=start_pos,
    exit_pos=exit_pos,
    max_corrections=5,
    question_type=question_type,
)
```

### Question types

| Type | What it verifies |
|------|-----------------|
| `"right_turns"` | Full step-by-step navigation: direction, positions, walkability, turn type, and right-turn count |
| `"total_turns"` | Same as above, but tracks total turns (left + right) |
| `"relative_position"` | Only verifies the LOCATE section where the model identifies S and E positions |

---

## SpatialMap Step Verifier

Verifies spatial/directional reasoning using Z3 constraint solving. The model reasons about objects on a map and their relative positions (e.g., "A is northwest of B"). The monitor initializes Z3 constraints from the problem description, then incrementally verifies each directional claim the model makes. Valid claims strengthen the constraint set; contradictory claims trigger feedback.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | — | Unique identifier for this monitor |
| `answer_start_token` | str | — | Token marking the transition from reasoning to answer |
| `problem_text` | str | — | The spatial reasoning problem description |
| `max_corrections` | int | `5` | Maximum correction attempts per example |
| `async_execution` | bool | `True` | Whether to run verification asynchronously |

### Factory method (recommended)

```python
monitor = StepVerifierSpatialMapMonitor.from_prompt(
    problem_text=user_prompt,
    max_corrections=5,
    name="spatialmap_step_verifier"
)
```

This parses the spatial relationships from the problem text and initializes the Z3 solver automatically.

### Supported directions

The Z3 solver handles diagonal directions (`Northwest`, `Northeast`, `Southwest`, `Southeast`) and cardinal directions (`North`, `South`, `East`, `West`), including transitivity (if A is NW of B and B is NW of C, then A is NW of C) and reversibility (if A is NW of B, then B is SE of A).

---

# Best-of-K Baseline

A simple best-of-K baseline that generates K independent reasoning traces per example and selects the best based on:
1. **Ground-truth matching** (default): Greedy selection of first correct answer among K samples
2. **Critic model evaluation** (optional): Use a separate critic LLM to evaluate correctness without access to ground truth

This baseline demonstrates that with sufficient sampling, even simple CoT can achieve good performance.

## Usage

```bash
# Best-of-K with ground-truth evaluation
python ./examples/TTSwithVerification/bestofk_baseline.py --task game24 -n 10 --k 4

# Best-of-K with critic model evaluation
python ./examples/TTSwithVerification/bestofk_baseline.py --task game24 -n 10 --k 4 --use_critic --critic_model Qwen/Qwen3-30B-A3B-Thinking-2507 --critic_port 8001
```

### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Task: `game24`, `maze`, or `spatialmap` | required |
| `--k` | Number of samples per example | `4` |
| `--use_critic` | Use critic model for evaluation instead of ground truth | `False` |
| `--critic_model` | Model to use for critic evaluation | MAIN_MODEL |
| `--critic_port` | vLLM server port for critic model | `8001` |
| `--num_examples`, `-n` | Number of examples to run | varies |
| `--main_model` | Model for generation | `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| `--port` | vLLM server port for main model | `8000` |

---

## Example Scripts

Each script runs a full evaluation: loading a dataset, building structured prompts, running inference with step verification, and computing accuracy/token statistics.

```bash
# Game of 24 with step verification
python ./examples/TTSwithVerification/game24_stepverifier.py -n 1

# Maze with step verification
python ./examples/TTSwithVerification/maze_stepverifier.py -n 1

# SpatialMap with step verification
python ./examples/TTSwithVerification/spatialmap_stepverifier.py -n 1

# Best-of-K baseline (standard CoT, no monitors)
python ./examples/TTSwithVerification/bestofk_baseline.py --task game24 -n 1 --k 4
python ./examples/TTSwithVerification/bestofk_baseline.py --task maze -n 1 --k 4
python ./examples/TTSwithVerification/bestofk_baseline.py --task spatialmap -n 1 --k 4

# Best-of-K with critic model evaluation
python ./examples/TTSwithVerification/bestofk_baseline.py --task game24 -n 1 --k 4 --use_critic
```

### Common arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-n`, `--num_examples` | Number of examples to run | varies by task |
| `--max_corrections` | Max correction attempts per example | `5` |
| `-d`, `--debug` | Enable debug logging | `False` |
| `--model` / `--main_model` | Main generation model | `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| `--port` | vLLM server port | `8000` |
