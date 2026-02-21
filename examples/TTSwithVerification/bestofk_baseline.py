import argparse
import asyncio
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import aiohttp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from interwhen import stream_completion

# ============== MODEL CONFIGURATION ==============
MAIN_MODEL = "Qwen/QwQ-32B"
# Multi-process vLLM configuration
VLLM_PORTS = [8000, 8001, 8002, 8003]  # 4 instances with tensor-parallel-size 2 each
REQUEST_COUNTER = {"main": 0, "critic": 0}  # Track request count for round-robin load balancing


logger = logging.getLogger(__name__)


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@dataclass
class SampleResult:
    output: str
    correct: bool
    extracted: Optional[str]
    message: str
    tokens: int
    critic_correct: Optional[bool] = None
    critic_feedback: Optional[str] = None


def get_model_short_name(model_name: str) -> str:
    short_name = model_name.split("/")[-1]
    return short_name.replace(" ", "_").replace(":", "-")


def get_next_port(server_type: str = "main") -> int:
    """Get next vLLM port in round-robin fashion."""
    global REQUEST_COUNTER
    port = VLLM_PORTS[REQUEST_COUNTER[server_type] % len(VLLM_PORTS)]
    REQUEST_COUNTER[server_type] += 1
    return port


def get_output_dirs(task: str, main_model: str, use_critic: bool, critic_early_stop: bool, base_dir: str = "../../b-pchanda/Outputs_TTS/BestOfKResults"):
    model_short_name = get_model_short_name(main_model)
    critic_status = "on" if use_critic else "off"
    earlystop_status = "on" if critic_early_stop else "off"
    output_base = os.path.join(base_dir, task, model_short_name, f"critic_{critic_status}", f"earlystop_{earlystop_status}")
    dirs = {
        "base": output_base,
        "reasoning": os.path.join(output_base, "Reasoning_output"),
        "critic": os.path.join(output_base, "Critic_output") if use_critic else None,
    }
    for dir_path in dirs.values():
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    return dirs


def init_llm_server(model_name, max_tokens=32768, port=8000, temperature=0.6, seed=42):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "do_sample": True,
        "temperature": temperature,
        "stream": False,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed": seed,
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text, with fallback to character count."""
    try:
        if not text or len(text.strip()) == 0:
            return 0
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Tokenization failed: {e}, using character count estimate")
        # Rough estimate: ~4 characters per token
        return max(1, len(text) // 4)


def save_outputs(idx: int, outputs: List[SampleResult], best_idx: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"output_{idx}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"BEST_INDEX={best_idx}\n")
        for i, result in enumerate(outputs):
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"SAMPLE {i}\n")
            f.write(f"CORRECT={result.correct}\n")
            f.write(f"CRITIC_CORRECT={result.critic_correct}\n")
            f.write(f"EXTRACTED={result.extracted}\n")
            f.write(f"TOKENS={result.tokens}\n")
            f.write(f"MESSAGE={result.message}\n")
            if result.critic_feedback:
                f.write(f"CRITIC_FEEDBACK={result.critic_feedback}\n")
            f.write("\n")
            f.write(result.output)
            f.write("\n")
    # logger.info(f"Saved outputs to {filepath}")


# --------------------- Game24 helpers ---------------------

def build_game24_prompt(nums):
    a, b, c, d = nums
    boxed = r"\\boxed{}"
    base_prompt = f"""
You are solving the Game of 24.

You are given four numbers: {a}, {b}, {c}, {d}

Your job is to produce a valid arithmetic expression using:
- ALL four numbers exactly once
- ONLY +, -, *, /
- The expression must evaluate to exactly 24.

Please reason step by step, and put your final answer containing only the expression within {boxed}.
""".strip()
    return base_prompt


def extract_solution_game24(text):
    boxed_pattern = r"\\boxed\{"
    matches = list(re.finditer(boxed_pattern, text))
    if not matches:
        return None
    last_match = matches[-1]
    start = last_match.end()
    brace_count = 1
    end = start
    while end < len(text) and brace_count > 0:
        if text[end] == "{":
            brace_count += 1
        elif text[end] == "}":
            brace_count -= 1
        end += 1
    expr = text[start:end - 1].strip()

    frac_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
    while re.search(frac_pattern, expr):
        expr = re.sub(frac_pattern, r"(\1/\2)", expr)

    replacements = {
        r"\times": "*",
        r"\cdot": "*",
        r"\div": "/",
    }
    for latex, op in replacements.items():
        expr = expr.replace(latex, op)

    expr = expr.replace(r"\\,", "").replace(r"\\ ", "")
    expr = re.sub(r"\)\s*\(", ")*(", expr)
    expr = re.sub(r"\)\s*(\d)", r")*\1", expr)
    expr = re.sub(r"(\d)\s*\(", r"\1*(", expr)

    return expr


def extract_numbers_from_expr(expr):
    numbers = re.findall(r"\d+\.?\d*", expr)
    return [int(float(n)) if float(n).is_integer() else float(n) for n in numbers]


def validate_numbers_used(expr, expected_nums):
    used_nums = extract_numbers_from_expr(expr)
    return sorted(used_nums) == sorted(expected_nums)


def evaluate_expression(expr, expected_nums=None):
    try:
        if expected_nums is not None and not validate_numbers_used(expr, expected_nums):
            return False
        value = eval(expr, {"__builtins__": None}, {})
        return abs(value - 24) < 1e-6
    except Exception:
        return False


def evaluate_game24_answer(answer, nums):
    expr = extract_solution_game24(answer)
    if not expr:
        return False, None, "No expression found"
    if evaluate_expression(expr, expected_nums=nums):
        return True, expr, "Correct solution (evaluates to 24 using exactly the given numbers)"
    used_nums = extract_numbers_from_expr(expr)
    if sorted(used_nums) != sorted(nums):
        return False, expr, f"Incorrect: Expression uses {used_nums}, expected {nums}"
    return False, expr, "Expression does not evaluate to 24"


# --------------------- Maze/SpatialMap helpers ---------------------

def remove_last_paragraph(s: str) -> str:
    return s[:-143] if len(s) > 143 else s


def build_maze_prompt(example):
    pre_prompt = (
        "You are an expert problem solver. Carefully read the following multiple-choice question "
        "and think through the solution step-by-step before providing your final answer. "
        "Provide your final answer option by enclosing it within \\boxed{A/B/C/D}.:"
    )
    description = remove_last_paragraph(str(example.get("prompt")))
    return pre_prompt, description


def build_spatialmap_prompt(example):
    pre_prompt = (
        "You are an expert problem solver. Carefully read the following multiple-choice question "
        "and think through the solution step-by-step before providing your final answer." 
        "Provide your final answer option by enclosing it within \\boxed{A/B/C/D}.:"
    )
    description = remove_last_paragraph(str(example.get("prompt")))
    return pre_prompt, description


def extract_solution_mcq(text):
    """Extract MCQ solution from model output."""
    # Try multiple boxed patterns
    patterns = [
        r"\\boxed\{([^}]*)\}",  # \boxed{...}
        r"boxed\{([^}]*)\}",     # boxed{...} without escape
        r"\*\*([A-D])\*\*",      # **A** format
        r"answer[:\s]*([A-D])",  # answer: A format
        r"(?:^|\n)([A-D])(?:\s|$|\.)",  # Standalone letter
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            expr = matches[-1].strip()
            choice_match = re.search(r"\b([ABCD])\b", expr, flags=re.IGNORECASE)
            if choice_match:
                return choice_match.group(1).upper()
    
    # Last resort: look for any standalone A, B, C, or D
    standalone = re.findall(r"\b([ABCD])\b", text)
    if standalone:
        return standalone[-1].upper()
    
    return None


def extract_options_from_prompt(prompt_text, target_options):
    pattern = r"\b([A-D])\.\s*(.*?)(?=\s*[A-D]\.\s*|$)"
    raw = re.findall(pattern, prompt_text, flags=re.DOTALL)
    options = {k: v.strip().rstrip(".") for k, v in raw}
    if target_options:
        options = {k: v for k, v in options.items() if k in target_options}
    return options


def evaluate_mcq_answer(answer, options, ground_truth):
    sol = extract_solution_mcq(answer)
    gt_sol = str(ground_truth).strip()
    if not sol:
        return False, None, "No expression found"
    sol = sol.strip()
    if sol in options:
        if options[sol] == gt_sol:
            return True, sol, f"Correct: option {sol} -> {options[sol]}"
        return False, sol, f"Incorrect: expected '{gt_sol}', got '{options[sol]}' (option {sol})"
    if sol.lower() == gt_sol.lower():
        return True, sol, f"Correct: answer text matches ground truth: {sol}"
    for opt_letter, opt_value in options.items():
        if sol.lower() == opt_value.lower():
            if opt_value == gt_sol:
                return True, sol, f"Correct: answer text {sol} (option {opt_letter})"
            return False, sol, f"Incorrect: expected '{gt_sol}', got '{opt_value}' (option {opt_letter})"
    return False, sol, f"Solution '{sol}' not found in options or ground truth"


def build_full_prompt(task, example, nums=None):
    if task == "game24":
        prompt = build_game24_prompt(nums)
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if task == "maze":
        system_prompt, user_prompt = build_maze_prompt(example)
    else:
        system_prompt, user_prompt = build_spatialmap_prompt(example)
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_dataset_for_task(task):
    if task == "game24":
        return load_dataset("nlile/24-game", split="train")
    if task == "maze":
        return load_dataset("microsoft/VISION_LANGUAGE", "maze_text_only", split="val")
    if task == "spatialmap":
        return load_dataset("microsoft/VISION_LANGUAGE", "spatial_map_text_only", split="val")
    raise ValueError(f"Unsupported task: {task}")


def resolve_indices(task, dataset_len, args):
    if args.indices:
        return [int(x.strip()) for x in args.indices.split(",")]
    if args.xrange:
        parts = args.xrange.split("-")
        if len(parts) == 2:
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                return range(start, end)
            except ValueError:
                raise ValueError(f"Invalid xrange format: {args.xrange}. Use 'start-end'")
    if args.num_examples:
        return np.linspace(0, dataset_len - 1, args.num_examples, dtype=int)
    # Default: use full range
    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else dataset_len
    return range(start, end)


def batch_generate_samples(prompt, llm_server, k, seed, quiet=True):
    """Generate k samples using vLLM batch processing via API across multiple instances."""
    payload_template = llm_server["payload"].copy()
    headers = llm_server["headers"]
    
    # Create k requests with different seeds
    batch_payloads = []
    for i in range(k):
        payload = payload_template.copy()
        payload["prompt"] = prompt
        payload["seed"] = seed + i
        batch_payloads.append(payload)
    
    # Send requests to vLLM instances in parallel (true concurrency)
    async def _fetch_one(session, sem, idx, url, payload):
        async with sem:
            try:
                async with session.post(url, json=payload, headers=headers, timeout=300) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        logger.warning(f"HTTP error for seed {seed + idx} on {url}: {resp.status} - {text[:200]}")
                        return idx, ""
                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON for seed {seed + idx} on {url}")
                        return idx, ""
            except Exception as e:
                logger.warning(f"Batch generation failed for seed {seed + idx} on {url}: {e}")
                return idx, ""

            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if isinstance(choice, dict):
                    output_text = choice.get("text") or choice.get("message", {}).get("content", "")
                else:
                    output_text = str(choice)
                if output_text and len(output_text.strip()) > 0:
                    return idx, output_text
                logger.warning(f"Empty output for seed {seed + idx} on {url}")
                return idx, ""

            logger.warning(f"No choices in response for seed {seed + idx} on {url}: {result.keys() if isinstance(result, dict) else type(result)}")
            return idx, ""

    async def _run_parallel():
        sem = asyncio.Semaphore(len(VLLM_PORTS))
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, payload in enumerate(batch_payloads):
                port = get_next_port(server_type="main")
                url = f"http://localhost:{port}/v1/completions"
                tasks.append(asyncio.create_task(_fetch_one(session, sem, i, url, payload)))
            results = await asyncio.gather(*tasks)
            return results

    if quiet:
        with suppress_output():
            results = asyncio.run(_run_parallel())
    else:
        results = asyncio.run(_run_parallel())

    outputs = [""] * k
    for idx, output_text in results:
        outputs[idx] = output_text
        if output_text and not quiet:
            print(f"[Generated sample {idx}] {len(output_text)} chars, {len(output_text.split())} words")

    return outputs


# --------------------- Critic model helpers ---------------------

def build_game24_critic_prompt(nums, reasoning_output):
    """Build critic prompt to evaluate Game of 24 solution and provide reasoning."""
    return f"""You are a math verifier. Evaluate the following Game of 24 solution.

Numbers: {nums}
Target: 24

Student's reasoning and answer:
{reasoning_output}

Verify:
1. Does it use ALL four numbers exactly once?
2. Does each step follow correct arithmetic?
3. Does the final expression evaluate to exactly 24?

Respond in the following format:
VERDICT: CORRECT or INCORRECT
REASONING: Your detailed explanation

If CORRECT, briefly explain why.
If INCORRECT, explain what went wrong and how to fix it.
"""


def build_mcq_critic_prompt(task, task_description, reasoning_output):
    """Build critic prompt to evaluate MCQ solution and provide reasoning."""
    task_name = "Maze" if task == "maze" else "Spatial Reasoning"
    return f"""You are an expert {task_name} verifier. Evaluate the following solution.

Task:
{task_description}

Student's reasoning and answer:
{reasoning_output}

Verify the correctness of the step-by-step reasoning and final answer.

Respond in the following format:
VERDICT: CORRECT or INCORRECT
REASONING: Your detailed explanation

If CORRECT, briefly explain why.
If INCORRECT, explain what went wrong and suggest the correct approach.
"""


def batch_evaluate_with_critic(outputs_df, task, example, critic_llm_server, tokenizer, nums=None, quiet=True):
    """Batch evaluate outputs using vLLM API across multiple instances. Outputs_df should have columns: 'output', 'seed_idx'"""
    payload_template = critic_llm_server["payload"].copy()
    headers = critic_llm_server["headers"]

    async def _fetch_one(session, sem, idx, url, payload):
        async with sem:
            try:
                async with session.post(url, json=payload, headers=headers, timeout=300) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        logger.warning(f"HTTP error for critic sample {idx} on {url}: {resp.status} - {text[:200]}")
                        return idx, "", False
                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON for critic sample {idx} on {url}")
                        return idx, "", False
            except Exception as e:
                logger.warning(f"Critic evaluation failed for sample {idx} on {url}: {e}")
                return idx, "", False

            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                critic_output = choice.get("text") or choice.get("message", {}).get("content", "")
            else:
                critic_output = ""

            is_correct = "CORRECT" in critic_output.upper()
            reasoning = ""
            if "REASONING:" in critic_output:
                reasoning = critic_output.split("REASONING:", 1)[1].strip()
            elif "VERDICT:" not in critic_output:
                reasoning = critic_output

            return idx, reasoning, is_correct

    async def _run_parallel():
        sem = asyncio.Semaphore(len(VLLM_PORTS))
        async with aiohttp.ClientSession() as session:
            tasks = []
            for idx, row in outputs_df.iterrows():
                output_text = row["output"]
                if task == "game24":
                    critic_prompt = build_game24_critic_prompt(nums, output_text)
                else:
                    if task == "maze":
                        _, task_desc = build_maze_prompt(example)
                    else:
                        _, task_desc = build_spatialmap_prompt(example)
                    critic_prompt = build_mcq_critic_prompt(task, task_desc, output_text)

                critic_system = "You are a strict academic verifier."
                full_prompt = f"<|im_start|>system\n{critic_system}<|im_end|>\n<|im_start|>user\n{critic_prompt}<|im_end|>\n<|im_start|>assistant\n"

                payload = payload_template.copy()
                payload["prompt"] = full_prompt
                payload["seed"] = row.get("critic_seed", idx)

                port = get_next_port(server_type="critic")
                url = f"http://localhost:{port}/v1/completions"
                tasks.append(asyncio.create_task(_fetch_one(session, sem, idx, url, payload)))

            return await asyncio.gather(*tasks)

    if quiet:
        with suppress_output():
            results = asyncio.run(_run_parallel())
    else:
        results = asyncio.run(_run_parallel())

    rows = []
    for sample_idx, reasoning, is_correct in results:
        rows.append({
            "sample_idx": sample_idx,
            "critic_correct": is_correct,
            "critic_feedback": reasoning,
        })

    return pd.DataFrame(rows)


def run_k_samples_with_critic(
    prompt,
    llm_server,
    critic_llm_server,
    k,
    seed,
    task,
    example,
    tokenizer,
    eval_fn,
    nums=None,
    early_stop=False,
    quiet=True,
):
    """Run k samples with critic evaluation using vLLM batching."""
    # Generate k samples
    outputs = batch_generate_samples(prompt, llm_server, k, seed, quiet=quiet)
    
    # Create dataframe with outputs
    df_samples = pd.DataFrame({
        "sample_idx": range(k),
        "output": outputs,
        "seed": [seed + i for i in range(k)],
    })
    
    # If early stop mode, stop at first critic-correct
    if early_stop:
        sample_results = []
        for idx, row in df_samples.iterrows():
            output = row["output"]
            
            # Evaluate with critic
            df_critic = batch_evaluate_with_critic(
                pd.DataFrame([{"output": output, "seed_idx": idx}]),
                task, example, critic_llm_server, tokenizer, nums=nums, quiet=quiet
            )
            critic_correct = df_critic.iloc[0]["critic_correct"] if len(df_critic) > 0 else False
            critic_feedback = df_critic.iloc[0]["critic_feedback"] if len(df_critic) > 0 else ""
            
            # Evaluate with ground truth
            is_correct, extracted, message = eval_fn(output)
            token_count = count_tokens(output, tokenizer)
            
            sample_results.append(SampleResult(
                output=output,
                correct=is_correct,
                extracted=extracted,
                message=f"Critic verdict: {'CORRECT' if critic_correct else 'INCORRECT'} | {message}",
                tokens=token_count,
                critic_correct=critic_correct,
                critic_feedback=critic_feedback,
            ))
            
            if critic_correct:
                break
        
        return sample_results
    else:
        # Batch critic evaluation
        df_critic = batch_evaluate_with_critic(
            df_samples, task, example, critic_llm_server, tokenizer, nums=nums, quiet=quiet
        )
        
        # Merge critic results
        df_samples = df_samples.merge(df_critic, left_index=True, right_on="sample_idx", how="left")
        
        # Process all results
        sample_results = []
        for idx, row in df_samples.iterrows():
            output = row["output"]
            critic_correct = row.get("critic_correct", False)
            critic_feedback = row.get("critic_feedback", "")
            
            is_correct, extracted, message = eval_fn(output)
            token_count = count_tokens(output, tokenizer)
            
            sample_results.append(SampleResult(
                output=output,
                correct=is_correct,
                extracted=extracted,
                message=f"Critic verdict: {'CORRECT' if critic_correct else 'INCORRECT'} | {message}",
                tokens=token_count,
                critic_correct=critic_correct,
                critic_feedback=critic_feedback,
            ))
        
        return sample_results





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Best-of-K baseline (standard CoT) for TTSwithVerification datasets")
    parser.add_argument("--task", type=str, required=True, choices=["game24", "maze", "spatialmap"],
                        help="Task to run")
    parser.add_argument("--k", type=int, default=4, help="Number of samples per example")
    parser.add_argument("--num_examples", "-n", type=int, default=None,
                        help="Number of examples to run (overrides start/end)")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices to run")
    parser.add_argument("--xrange", type=str, default=None,
                        help="Range of indices to run (format: 'start-end')")
    parser.add_argument("--start", type=int, default=None, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--main_model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--use_critic", action="store_true", help="Use critic model for evaluation instead of ground truth")
    parser.add_argument("--critic_model", type=str, default=MAIN_MODEL, help="Critic model to use for evaluation")
    parser.add_argument("--critic_port", type=int, default=8000, help="vLLM server port for critic model (default: same as main model port)")
    parser.add_argument("--critic_early_stop", action="store_true", help="Stop sampling after first critic-correct trace")
    parser.add_argument("--critic_feedback_baseline", action="store_true", help="Use critic feedback as a separate baseline for post-hoc correction")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.ERROR
    logging.basicConfig(level=log_level, format="%(message)s")

    quiet_mode = not args.debug

    if quiet_mode:
        with suppress_output():
            dataset = load_dataset_for_task(args.task)
    else:
        dataset = load_dataset_for_task(args.task)
    indices = resolve_indices(args.task, len(dataset), args)

    llm_server = init_llm_server(
        args.main_model,
        max_tokens=args.max_tokens,
        port=args.port,
        temperature=args.temperature,
        seed=args.seed,
    )

    critic_llm_server = None
    if args.use_critic:
        critic_llm_server = init_llm_server(
            args.critic_model,
            max_tokens=512,
            port=args.critic_port,
            temperature=0.2,
            seed=args.seed,
        )
        # logger.info(f"Using critic model: {args.critic_model} on port {args.critic_port}")

    # logger.info(f"Loading tokenizer for {args.main_model}...")
    if quiet_mode:
        with suppress_output():
            tokenizer = AutoTokenizer.from_pretrained(args.main_model, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.main_model, trust_remote_code=True)
    # logger.info("Tokenizer loaded successfully.")

    output_dirs = get_output_dirs(args.task, args.main_model, args.use_critic, args.critic_early_stop)

    total_examples = 0
    total_correct = 0
    total_correct_samples = 0
    total_samples = 0
    critic_correct_samples = 0
    critic_total_samples = 0
    total_tokens = 0
    total_tokens_all_samples = 0
    results = []

    for idx in tqdm(indices, desc="Processing examples", unit="example"):
        example = dataset[int(idx)]
        if args.task == "game24":
            nums = example["numbers"]
            prompt = build_full_prompt(args.task, example, nums=nums)
            eval_fn = lambda output: evaluate_game24_answer(output, nums)
            options = None
        else:
            prompt = build_full_prompt(args.task, example)
            gt = str(example.get("ground_truth", "")).strip()
            if gt == "Q4":
                target_options = ["A", "B"]
            else:
                target_options = ["A", "B", "C", "D"]
            if args.task == "maze":
                _, user_prompt = build_maze_prompt(example)
            else:
                _, user_prompt = build_spatialmap_prompt(example)
            options = extract_options_from_prompt(user_prompt, target_options)
            eval_fn = lambda output: evaluate_mcq_answer(output, options, gt)

        #logger.info(f"---- Example {idx} ----")
        
        quiet_mode = not args.debug
        
        if args.use_critic:
            sample_results = run_k_samples_with_critic(
                prompt, llm_server, critic_llm_server, args.k, args.seed,
                args.task, example, tokenizer, eval_fn, nums=(nums if args.task == "game24" else None),
                early_stop=args.critic_early_stop, quiet=quiet_mode
            )
        else:
            outputs = batch_generate_samples(prompt, llm_server, args.k, args.seed, quiet=quiet_mode)
            sample_results = []
            for output in outputs:
                is_correct, extracted, message = eval_fn(output)
                token_count = count_tokens(output, tokenizer)
                sample_results.append(SampleResult(
                    output=output,
                    correct=is_correct,
                    extracted=extracted,
                    message=message,
                    tokens=token_count,
                    critic_correct=None,
                ))

        if args.use_critic:
            best_idx = next((i for i, r in enumerate(sample_results) if r.critic_correct), 0)
        else:
            best_idx = next((i for i, r in enumerate(sample_results) if r.correct), 0)
        best_result = sample_results[best_idx]
        any_correct = any(r.correct for r in sample_results)
        correct_samples = sum(1 for r in sample_results if r.correct)
        critic_correct_samples_example = sum(1 for r in sample_results if r.critic_correct)

        save_outputs(idx, sample_results, best_idx, output_dirs["reasoning"])

        total_examples += 1
        if any_correct:
            total_correct += 1
        total_correct_samples += correct_samples
        total_samples += len(sample_results)
        critic_correct_samples += critic_correct_samples_example
        critic_total_samples += len(sample_results)
        total_tokens += best_result.tokens
        total_tokens_all_samples += sum(r.tokens for r in sample_results)

        results.append({
            "idx": int(idx),
            "best_idx": best_idx,
            "any_correct": any_correct,
            "best_correct": best_result.correct,
            "best_critic_correct": best_result.critic_correct,
            "best_extracted": best_result.extracted,
            "best_message": best_result.message,
            "best_critic_feedback": best_result.critic_feedback,
            "best_tokens": best_result.tokens,
            "all_tokens": [r.tokens for r in sample_results],
            "all_correct": [r.correct for r in sample_results],
            "all_critic_correct": [r.critic_correct for r in sample_results],
            "all_critic_feedback": [r.critic_feedback for r in sample_results],
            "options": options,
        })

        #logger.info(f"Best sample: {best_idx} | Correct in K: {any_correct}")
        #logger.info(f"Best message: {best_result.message}")

    accuracy = total_correct / total_examples if total_examples else 0
    avg_best_tokens = total_tokens / total_examples if total_examples else 0
    avg_all_tokens = total_tokens_all_samples / total_examples if total_examples else 0

    summary = {
        "task": args.task,
        "model": args.main_model,
        "k": args.k,
        "use_critic": args.use_critic,
        "total_examples": total_examples,
        "correct": total_correct,
        "correct_samples": total_correct_samples,
        "total_samples": total_samples,
        "critic_correct_samples": critic_correct_samples,
        "critic_total_samples": critic_total_samples,
        "critic_accuracy": (critic_correct_samples / critic_total_samples) if critic_total_samples else 0,
        "accuracy": accuracy,
        "avg_best_tokens": avg_best_tokens,
        "avg_all_tokens": avg_all_tokens,
        "total_tokens_best": total_tokens,
        "total_tokens_all_samples": total_tokens_all_samples,
        "results": results,
    }
    
    if args.use_critic:
        summary["critic_model"] = args.critic_model
        summary["critic_port"] = args.critic_port
        summary["critic_early_stop"] = args.critic_early_stop
        summary["critic_feedback_baseline"] = args.critic_feedback_baseline

    summary_path = os.path.join(output_dirs["base"], "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # logger.info(f"Saved summary to {summary_path}")
