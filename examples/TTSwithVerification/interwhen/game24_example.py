"""
Game of 24 experiment with thinking-phase step verification.

Uses ThinkingPhaseStepVerifierGame24Monitor which:
  - Verifies the model's intermediate expressions during <think> via side-streams
  - Injects expression extraction after </think>
  - Verifies the final \\boxed{} expression for correctness
"""

import argparse
import asyncio
import logging
import os
import re
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer

from interwhen import stream_completion
from interwhen.monitors import ThinkingPhaseStepVerifierGame24Monitor

# ============== MODEL CONFIGURATION ==============
MAIN_MODEL = "Qwen/QwQ-32B"
# =================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Walk up to find the repo root (contains pyproject.toml), output to its parent
_dir = _SCRIPT_DIR
while _dir != os.path.dirname(_dir) and not os.path.isfile(os.path.join(_dir, "pyproject.toml")):
    _dir = os.path.dirname(_dir)
_OUTPUT_ROOT = os.path.dirname(_dir)

def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name

def get_output_dirs(main_model: str, base_dir: str = None):
    """Create and return output directory paths based on model name."""
    if base_dir is None:
        base_dir = os.path.join(_OUTPUT_ROOT, "Outputs_TTS", "Gameof24results")
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    
    dirs = {
        "base": output_base,
        "reasoning": os.path.join(output_base, "Reasoning_output"),
        "csv_saved": os.path.join(output_base, "csv_saved"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_log_filename(main_model: str, num_examples: int, base_dir: str = None) -> str:
    """Generate log filename based on model name."""
    if base_dir is None:
        base_dir = os.path.join(_OUTPUT_ROOT, "Outputs_TTS", "Gameof24results")
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.log")

def save_prompt(idx, prompt_with_answer, reason_dir):
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)

logger = logging.getLogger(__name__)


def init_llm_server(modelname, max_tokens=32768, port=8000):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": modelname,
        "max_tokens": max_tokens,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "do_sample": True,
        "temperature": 0.6,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed": 42
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


def build_prompt(nums):
    a, b, c, d = nums
    boxed = r"\boxed{}"
    base_prompt = f"""
    You are solving the Game of 24.
    
    You are given four numbers: {a}, {b}, {c}, {d}
    
    Your job is to produce a valid arithmetic expression using:
    - ALL four numbers exactly once
    - ONLY +, -, *, /
    - The expression must evaluate to exactly 24.
    
    Please reason step by step, and put your final answer containing only the expression within {boxed}.""".strip()
    return base_prompt


def count_tokens(text: str, tokenizer) -> int:
    """Count the total number of tokens in the generated text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def extract_solution(text):
    
    # Only search for \boxed{} AFTER </think> to avoid grabbing unverified
    # expressions from inside the thinking trace.
    # If model opened <think> but never closed it (hit token limit), there is
    # no final answer — return None.
    if '</think>' in text:
        search_text = text[text.rfind('</think>'):]
    elif '<think>' in text:
        # Model started thinking but never finished — no verified answer
        return None
    else:
        search_text = text

    # Use a more robust extraction that handles nested braces in \boxed{}
    # Find \boxed{ and then match braces properly
    boxed_pattern = r"\\boxed\{"
    matches = list(re.finditer(boxed_pattern, search_text))
    if not matches:
        return None
    
    # Get the last \boxed{} content by matching braces
    last_match = matches[-1]
    start = last_match.end()  # Position right after \boxed{
    brace_count = 1
    end = start
    while end < len(search_text) and brace_count > 0:
        if search_text[end] == '{':
            brace_count += 1
        elif search_text[end] == '}':
            brace_count -= 1
        end += 1
    
    expr = search_text[start:end-1].strip()  # -1 to exclude the closing brace

    # Skip empty \boxed{} (e.g., from verifier feedback "Wrap in \boxed{}.")
    if not expr:
        return None

    # 1. Convert \frac{a}{b} to (a/b)
    frac_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
    while re.search(frac_pattern, expr):
        expr = re.sub(frac_pattern, r"(\1/\2)", expr)

    # 2. Replace LaTeX operators
    replacements = {
        r"\times": "*",
        r"\cdot": "*",
        r"\div": "/",
    }
    for latex, op in replacements.items():
        expr = expr.replace(latex, op)

    # 2b. Replace Unicode math operators (QwQ frequently uses these)
    expr = expr.replace('\u00d7', '*').replace('\u00f7', '/').replace('\u2212', '-')
    expr = expr.replace('\u2013', '-').replace('\u2014', '-')  # en-dash, em-dash

    # 3. Cleanup (remove LaTeX formatting artifacts)
    expr = expr.replace(r"\,", "").replace(r"\ ", "")
    expr = expr.replace(r"\left", "").replace(r"\right", "")

    # 3b. Strip trailing "= <number>" (e.g., "10 - 8/8 * 1 = 24" -> "10 - 8/8 * 1")
    expr = re.sub(r'\s*=\s*[\d.]+\s*$', '', expr)

    # 4. Handle implicit multiplication (e.g., "(11+1)(1+1)" -> "(11+1)*(1+1)")
    # Insert * between: )( , )number, number(, )(
    expr = re.sub(r'\)\s*\(', ')*(', expr)  # )( -> )*(
    expr = re.sub(r'\)\s*(\d)', r')*\1', expr)  # )number -> )*number
    expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)  # number( -> number*(

    return expr

def extract_numbers_from_expr(expr):
    """Extract all numbers (including decimals) from an expression."""
    # Match integers and decimals
    numbers = re.findall(r'\d+\.?\d*', expr)
    return [int(float(n)) if float(n).is_integer() else float(n) for n in numbers]

def validate_numbers_used(expr, expected_nums):
    """Check if the expression uses exactly the given numbers (each exactly once)."""
    used_nums = extract_numbers_from_expr(expr)
    # Sort both lists to compare
    return sorted(used_nums) == sorted(expected_nums)

def evaluate_expression(expr, expected_nums=None):
    try:
        # First check if expression uses exactly the given numbers
        if expected_nums is not None:
            if not validate_numbers_used(expr, expected_nums):
                return False
        
        value = eval(expr, {"__builtins__": None}, {})
        return abs(value - 24) < 1e-6
    except Exception:
        return False

def evaluate_game24_answer(answer, nums):
    """
    Evaluate a Game24 answer and return (is_correct, expr, error_message).
    
    Args:
        answer: Raw model output
        nums: Expected numbers to use
        
    Returns:
        Tuple of (is_correct, extracted_expression, error_message)
    """
    expr = extract_solution(answer)
    if not expr:
        return False, None, "No expression found"
    if evaluate_expression(expr, expected_nums=nums):
        return True, expr, "Correct solution (evaluates to 24 using exactly the given numbers)"
    else:
        used_nums = extract_numbers_from_expr(expr)
        if sorted(used_nums) != sorted(nums):
            return False, expr, f"Incorrect: Expression uses {used_nums}, expected {nums}"
        else:
            return False, expr, "Expression does not evaluate to 24"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Game of 24 step-by-step solver with monitors")
    parser.add_argument("--num_examples", "-n", type=int, default=1362, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs")
    parser.add_argument("--newline_threshold", type=int, default=20, help="Number of newlines in thinking before forcing step verification")
    parser.add_argument("--max_corrections", type=int, default=3, help="Maximum number of correction attempts per example")
    parser.add_argument("--warmup", type=int, default=4, help="Number of \\n to skip before starting side-chain verification")
    parser.add_argument("--model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    args = parser.parse_args()

    main_model = args.model

    output_dirs = get_output_dirs(main_model)
    logfile = get_log_filename(main_model, args.num_examples)
    reason_dir = output_dirs["reasoning"]

    log_level = logging.DEBUG if args.debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler()
        ],
        force=True,
    )

    logger.info(f"Main model: {main_model}")
    logger.info(f"Output directory: {output_dirs['base']}")
    logger.info(f"Newline threshold: {args.newline_threshold}")
    logger.info(f"Warmup: {args.warmup}")

    dataset = load_dataset("nlile/24-game", split="train")

    llm_server = init_llm_server(main_model, port=args.port)

    logger.info(f"Loading tokenizer for {main_model}...")
    tokenizer = AutoTokenizer.from_pretrained(main_model, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully.")

    num_correct = 0
    num_attempted = 0  # model produced a real answer (not "no solution" and not missing after </think>)
    num_excluded = 0   # excluded from soundness (no solution or token budget exceeded)
    N = args.num_examples
    total_generated_tokens = 0
    generated_token_counts = []

    indices = np.linspace(0, len(dataset)-1, N, dtype=int)

    for idx in indices:
        example = dataset[idx]
        nums = example["numbers"]
        prompt = build_prompt(nums)
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        monitor = ThinkingPhaseStepVerifierGame24Monitor(
            name="game24_verifier",
            original_numbers=nums,
            llm_server=llm_server,
            prompt=full_prompt,
            newline_threshold=args.newline_threshold,
            max_corrections=args.max_corrections,
            answer_start_token="</think>",
            warmup_newlines=args.warmup,
        )

        logger.info(f"---- Example {idx+1} ----")
        logger.info(f"Numbers: {nums}")

        try:
            answer = asyncio.run(stream_completion(
                full_prompt,
                llm_server=llm_server,
                monitors=(monitor,),
                add_delay=False,
                termination_requires_validation=False,
                async_execution=True
            ))
        except Exception as e:
            logger.error(f"Error running example {idx}: {e}")
            continue

        save_prompt(idx, answer, reason_dir)
        logger.info(f"Raw final output:\n{answer}")

        generated_tokens = count_tokens(answer, tokenizer)
        generated_token_counts.append(generated_tokens)
        total_generated_tokens += generated_tokens
        logger.info(f"Generated tokens in this example: {generated_tokens}")

        is_correct, expr, message = evaluate_game24_answer(answer, nums)
        # Attempted: model produced a real answer (not "no solution" and not missing after </think>)
        gave_no_solution = (expr is not None and "no solution" in expr.strip().lower())
        no_expr_found = (expr is None)
        attempted = not (gave_no_solution or no_expr_found)
        if attempted:
            num_attempted += 1
        else:
            num_excluded += 1

        if expr:
            logger.info(f"Extracted expression: {expr}")
        logger.info(message)
        if is_correct:
            num_correct += 1

    avg_generated_tokens = total_generated_tokens / N if N > 0 else 0
    accuracy = num_correct / N if N > 0 else 0
    soundness = num_correct / num_attempted if num_attempted > 0 else 0

    print(f"\nFinal Accuracy: {num_correct}/{N} ({accuracy:.2%})")
    print(f"Soundness: {num_correct}/{num_attempted} ({soundness:.2%})")
    print(f"Excluded from soundness (no solution / token budget exceeded): {num_excluded}")
    print(f"Average Generated Tokens: {avg_generated_tokens:.2f}")
    print(f"Total Generated Tokens: {total_generated_tokens}")

    results_file = logfile.replace('.log', '_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Game of 24 Evaluation Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {main_model}\n")
        f.write(f"Number of Examples: {N}\n\n")
        f.write(f"Results:\n")
        f.write(f"---------\n")
        f.write(f"Correct: {num_correct}/{N}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Soundness: {num_correct}/{num_attempted} = {soundness:.2%}\n")
        f.write(f"Excluded from soundness (no solution / token budget exceeded): {num_excluded}\n\n")
        f.write(f"Generated Token Statistics:\n")
        f.write(f"---------------------------\n")
        f.write(f"Total Generated Tokens: {total_generated_tokens}\n")
        f.write(f"Average Generated Tokens: {avg_generated_tokens:.2f}\n")
        if generated_token_counts:
            f.write(f"Min Generated Tokens: {min(generated_token_counts)}\n")
            f.write(f"Max Generated Tokens: {max(generated_token_counts)}\n")
            f.write(f"Std Dev: {np.std(generated_token_counts):.2f}\n")
    logger.info(f"Results saved to {results_file}")
    print(f"Results saved to {results_file}")
