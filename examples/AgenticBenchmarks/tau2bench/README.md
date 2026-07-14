# Running interwhen on tau2-bench
---

**Note:** The code provided in this folder is built on top of the original code for tau2bench, found at https://github.com/sierra-research/tau2-bench. In each file, we have mentioned the changes we have made, and the code we have used verbatim, relative to the same file in the original Tau2Bench repo.

## 0. Install `uv`

The upstream project uses [`uv`](https://docs.astral.sh/uv/) to manage its
Python environment. Install it once (skip if `uv --version` already works):

```bash
# Install uv via the official standalone installer.
# Download the script first (inspect it if you like), then run it — safer than
# piping a remote script straight into a shell.
curl -LsSf https://astral.sh/uv/install.sh -o uv-install.sh
sh uv-install.sh

# Make uv available in the current shell (new shells pick it up automatically)
source $HOME/.local/bin/env

# Verify
uv --version
```

## 1. Clone the upstream tau2-bench repo

```bash
git clone TAU2_BENCH_URL
cd tau2-bench
uv sync
```

This README and the flat files alongside it are an overlay on top of the
upstream `main` branch.

## 2. Overlay the modified files

The overlay files live in subfolders of this directory (`verifiers/`) as well as
a few flat files at the root. The `Source` column below is the path **relative
to this folder**.

| Source (relative to this folder) | Destination in clone | Purpose |
|---|---|---|
| `verifiers/verifier_python/verifier.py` | `src/tau2/verifier/verifier.py` | `PolicyVerifier` — wires Python pre/post rules + SLM helper (uses `telecom_policy_spec`; Lean glue only when `TAU2_USE_AUTO_GLUE` is set) |
| `verifiers/verifier_lean/telecom_glue_spec.py` | `src/tau2/verifier/telecom_glue_spec.py` | HGlue mapping tau2 tool calls + DB state to Lean `check_all` requests |
| `verifiers/verifier_python/telecom_policy_spec.py` | `src/tau2/verifier/telecom_policy_spec.py` | Curated Python rules and Policy spec object used by the verifier to know which rules apply where |
| `verifiers/verifier_lean/telecom_python_rules.py` | `src/tau2/verifier/telecom_python_rules.py` | Python pre/post rules (escalation, phone normalization, ticket reasoning, etc.) run after lean rules |
| `verifiers/verifier_lean/slm_helper.py` | `src/tau2/verifier/slm_helper.py` | `slm_extract` — calls a small LLM to extract structured fields from free-form tool args; required by many Python rules |
| `orchestrator.py` | `src/tau2/orchestrator/orchestrator.py` | Drop-in replacement that self-instantiates the verifier when `TAU2_VERIFIER=1` (the default). Also skips `[VERIFIER]` user messages during checkpoint replay. |
| `llm_agent.py` | `src/tau2/agent/llm_agent.py` | Agent updates |
| `user_simulator.py` | `src/tau2/user/user_simulator.py` | User simulator updates |
| `llm_utils.py` | `src/tau2/utils/llm_utils.py` | `litellm.suppress_debug_info = True`, less noisy errors |
| `environment_top.py` | `src/tau2/environment/environment.py` | Adds `[VERIFIER]` message skip during checkpoint replay |
| `environment_telecom.py` | `src/tau2/domains/telecom/environment.py` | Adds `get_tasks_solo()` / `get_tasks_solo_split()` loaders |
| `utils_telecom.py` | `src/tau2/domains/telecom/utils.py` | Adds `TELECOM_TASK_SET_SOLO_PATH` |
| `registry.py` | `src/tau2/registry.py` | Registers the `telecom_solo` task set |
| `batch.py` | `src/tau2/runner/batch.py` | Auto-swaps `telecom` -> `telecom_solo` when the chosen agent has `solo_mode=True` metadata |
| `cli.py` | `src/tau2/cli.py` | CLI updates |
| `simulation.py` | `src/tau2/data_model/simulation.py` | Simulation data-model updates |
| `build.py` | `src/tau2/runner/build.py` | Runner build updates |
| `tasks_solo.json` | `data/tau2/domains/telecom/tasks_solo.json` | 114 telecom tasks rewritten so each ticket is self-contained (works without a user simulator) |
| `split_tasks_solo.json` | `data/tau2/domains/telecom/split_tasks_solo.json` | Task-id splits (`base`, `train`, `test`, …) for the solo task set |

Copy them in:

```bash
SRC=/path/to/this/readme/folder   # this folder (contains verifiers/ and the root files)
DST=/path/to/upstream/tau2-bench

# Source path (relative to SRC) -> destination path (relative to DST)
declare -A MAP=(
  [verifiers/verifier_python/verifier.py]=src/tau2/verifier/verifier.py
  [verifiers/verifier_lean/telecom_glue_spec.py]=src/tau2/verifier/telecom_glue_spec.py
  [verifiers/verifier_python/telecom_policy_spec.py]=src/tau2/verifier/telecom_policy_spec.py
  [verifiers/verifier_lean/telecom_python_rules.py]=src/tau2/verifier/telecom_python_rules.py
  [verifiers/verifier_lean/slm_helper.py]=src/tau2/verifier/slm_helper.py
  [orchestrator.py]=src/tau2/orchestrator/orchestrator.py
  [llm_agent.py]=src/tau2/agent/llm_agent.py
  [user_simulator.py]=src/tau2/user/user_simulator.py
  [llm_utils.py]=src/tau2/utils/llm_utils.py
  [environment_top.py]=src/tau2/environment/environment.py
  [environment_telecom.py]=src/tau2/domains/telecom/environment.py
  [utils_telecom.py]=src/tau2/domains/telecom/utils.py
  [registry.py]=src/tau2/registry.py
  [batch.py]=src/tau2/runner/batch.py
  [cli.py]=src/tau2/cli.py
  [simulation.py]=src/tau2/data_model/simulation.py
  [build.py]=src/tau2/runner/build.py
  [tasks_solo.json]=data/tau2/domains/telecom/tasks_solo.json
  [split_tasks_solo.json]=data/tau2/domains/telecom/split_tasks_solo.json
)

for src in "${!MAP[@]}"; do
  dst="${MAP[$src]}"
  mkdir -p "$DST/$(dirname "$dst")"
  cp "$SRC/$src" "$DST/$dst"
done
```

### (Optional) Build the Lean policy-checker binary

The default run uses the **Python verifier** (`verifiers/verifier_python/`), which
needs no Lean binary — so `policychecker_telecom` is **not shipped** and **not
required** for the reference run below. Only build it if you want the **Lean**
verifier variant (`verifiers/verifier_lean/`, activated by setting
`TAU2_USE_AUTO_GLUE=1`).

To generate `bin/policychecker_telecom`:

```bash
# 1. Install elan (the Lean toolchain manager). The pinned toolchain in
#    verifiers/verifier_lean/lean-toolchain (leanprover/lean4:v4.30.0-rc2) is
#    fetched automatically by lake on first build.
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
source "$HOME/.elan/env"

# 2. From the Lean project, pull the prebuilt mathlib cache (large download;
#    avoids compiling mathlib v4.30.0-rc2 from source) and build the binary.
cd "$SRC/verifiers/verifier_lean"
lake exe cache get      # downloads prebuilt mathlib artifacts
lake build              # produces the `policychecker` executable (~387 MB)

# 3. Drop the binary into the clone and point the verifier at it.
mkdir -p "$DST/bin"
cp .lake/build/bin/policychecker "$DST/bin/policychecker_telecom"
chmod +x "$DST/bin/policychecker_telecom"
export TAU2_LEAN_BINARY="$DST/bin/policychecker_telecom"
export TAU2_USE_AUTO_GLUE=1   # tell the verifier to use the Lean glue path
```

> Note: building mathlib without the cache can take a long time and significant
> disk/RAM. If you only need the Python verifier, skip this entire section.

## 3. Python environment

Follow the upstream tau2 README. In short:

```bash
cd $DST
uv sync
```

## 4. Model servers

The reference run uses Qwen3-30B-A3B-Thinking-2507 as the **agent** and Qwen2.5-3B-Instruct as the **SLM** (the small model that backs `slm_helper.slm_extract`). Two
OpenAI-compatible vLLM servers are needed.

### 4a. Agent vLLM (port 8000)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --served-model-name Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### 4b. SLM vLLM (port 8001)

You can serve the same Qwen3 weights again on a second port, or point at any
smaller model that handles structured extraction. The verifier reads
`SLM_API_BASE` to find it.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8001 \
    --tensor-parallel-size 1 \
    --max-model-len 32768
```

## 5. Required environment variables

```bash
export TAU2_VERIFIER=1                           # default; set 0 to disable verifier
export TAU2_POLICY_TODAY=2025-02-25              # date the Lean policy treats as "today"
export TAU2_VERIFIER_STATS_DIR="$DST/data/simulations/_stats"
mkdir -p "$TAU2_VERIFIER_STATS_DIR"

# SLM endpoint used by slm_helper.slm_extract
export SLM_API_BASE="http://localhost:8001/v1"
# unset auto-glue so the curated telecom_policy_spec containing python rules is used ( and not Lean)
unset TAU2_USE_AUTO_GLUE
```

## 6. Run

The reference command (114 telecom tasks, 1 trial, solo agent, dummy user,
verifier on, Qwen3 as agent):

```bash
# export OPENAI_API_KEY=<key>   # used for evaluation purposes
SLM_API_BASE=http://localhost:8001/v1 \
uv run tau2 run \
    --domain telecom \
    --agent llm_agent_solo \
    --user dummy_user \
    --agent-llm openai/Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --agent-llm-args '{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "api_base": "http://localhost:8000/v1"}' \
    --num-trials 1 \
    --num-tasks 114 \
    --enable-tool-call-verifier
```

Flags:

| Flag | Meaning |
|---|---|
| `--domain telecom` | Picks the telecom domain. With `--agent llm_agent_solo` and the batch.py overlay, the runner auto-swaps to the `telecom_solo` task set. |
| `--agent llm_agent_solo` | Solo-mode agent: no user simulator turn; full ticket text is in the opening message.  |
| `--user dummy_user` | No-op user ; issues a single opening turn and never speaks again. |
| `--agent-llm openai/...` | LiteLLM model id. `openai/<hf_id>` routes through the OpenAI-compatible client. |
| `--agent-llm-args` | JSON dict forwarded to LiteLLM; sets sampling params and points at the local vLLM (`api_base`). |
| `--num-tasks 114` | Run the whole solo task set. |