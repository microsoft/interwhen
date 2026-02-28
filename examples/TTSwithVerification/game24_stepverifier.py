import argparse
import asyncio
import csv
import logging
import os
import re
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer

from interwhen import stream_completion
from interwhen.monitors import ThinkingPhaseStepVerifierGame24Monitor

# ============== MODEL CONFIGURATION ==============
# Change these model names to scale experiments easily
MAIN_MODEL = "microsoft/Phi-4-reasoning"
EARLYSTOP_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# =================================================

def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name

def get_output_dirs(main_model: str, base_dir: str = "../../Outputs_TTS/Gameof24results"):
    """Create and return output directory paths based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    
    dirs = {
        "base": output_base,
        "reasoning": os.path.join(output_base, "Reasoning_output"),
        "csv_saved": os.path.join(output_base, "csv_saved"),
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_log_filename(main_model: str, num_examples: int, base_dir: str = "../../Outputs_TTS/Gameof24_results") -> str:
    """Generate log filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.log")

def get_token_filename(main_model: str, num_examples: int, base_dir: str = "../../Outputs_TTS/Gameof24_results") -> str:
    """Generate token CSV filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.csv")

def save_prompt(idx, prompt_with_answer, reason_dir):
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)

logger = logging.getLogger(__name__)


def load_game24_dataset():
    ds = load_dataset("nlile/24-game", split="train")
    return ds

def init_llm_server(modelname, max_tokens=200, port=8001):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": modelname,
        "max_tokens": max_tokens,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed" : 42
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


def build_prompt(nums):
    """Build a simple prompt for Game of 24."""
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
    parser.add_argument("--thinking", "-t", action="store_true", help="Enable chain-of-thought output")
    parser.add_argument("--monitor", "-m", default = True, action="store_true", help="Enable step-by-step monitor")
    parser.add_argument("--num_examples", "-n", type=int, default=1362, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs")
    parser.add_argument("--thinking_verify", "-tv", action="store_true", default = True, help="Enable thinking-phase step verification (verify during <think> trace)")
    parser.add_argument("--newline_threshold", type=int, default=20, help="Number of newlines in thinking before forcing step verification (used with --thinking_verify)")
    parser.add_argument("--warmup", type=int, default=4, help="Number of \\n\\n to skip before starting side-chain verification (warmup period)")
    parser.add_argument("--main_model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--earlystop_model", type=str, default=EARLYSTOP_MODEL, help="Model to use for early stopping")
    args = parser.parse_args()

    # Use models from args (allows command-line override)
    main_model = args.main_model
    earlystop_model = args.earlystop_model

    # Setup output directories based on model name
    output_dirs = get_output_dirs(main_model)
    logfile = get_log_filename(main_model, args.num_examples)
    token_filename = get_token_filename(main_model, args.num_examples)
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
    logger.info(f"Early stop model: {earlystop_model}")
    logger.info(f"Output directory: {output_dirs['base']}")

    dataset = load_game24_dataset()

    llm_server = init_llm_server(main_model, max_tokens=22000)

    # Load tokenizer for accurate token counting
    logger.info(f"Loading tokenizer for {main_model}...")
    tokenizer = AutoTokenizer.from_pretrained(main_model, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully.")

    num_correct = 0
    num_attempted = 0  # examples where a \boxed{} answer was produced
    num_excluded = 0   # examples excluded from soundness (no solution or token budget exceeded)
    N = args.num_examples
    max_token_budget = llm_server["payload"]["max_tokens"]
    total_reasoning_tokens = 0
    reasoning_token_counts = []
    per_example_results = []  # list of dicts for CSV

    # total = len(dataset)
    indices = np.linspace(0, len(dataset)-1, N, dtype=int)

    for idx in indices:
        example = dataset[idx]
        nums = example["numbers"]

        prompt = build_prompt(nums)
        system_prompt = (
            "You are Phi, a language model trained by Microsoft to help users. "
            "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process "
            "before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle "
            "of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop "
            "well-considered thinking process. Please structure your response into two main sections: Thought and Solution "
            "using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, "
            "detail your reasoning process in steps. Each step should include detailed considerations such as analysing "
            "questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, "
            "refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, "
            "explorations, and reflections from the Thought section, systematically present the final solution that you "
            "deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed "
            "to reach the conclusion. Now, try to solve the following question through the above guidelines."
        )
        full_prompt = f"<|im_start|>system<|im_sep|>\n{system_prompt}<|im_end|>\n<|im_start|>user<|im_sep|>\n{prompt}<|im_end|>\n<|im_start|>assistant<|im_sep|>\n"

        if args.monitor:
            # ThinkingPhaseStepVerifierGame24Monitor handles both cases:
            # - With --thinking_verify: also verifies during the <think> phase
            # - Without: only injects structured prompt after </think> and verifies steps
            threshold = args.newline_threshold if args.thinking_verify else 999999
            monitors=(ThinkingPhaseStepVerifierGame24Monitor(
                name="game24_verifier",
                original_numbers=nums,
                llm_server=llm_server,
                prompt=full_prompt,
                newline_threshold=threshold,
                max_corrections=3,
                answer_start_token="</think>",
                warmup_newlines=args.warmup,
            ),)
        else:
            monitors = ()

        logger.info(f"---- length of monitors {len(monitors)} ----")
        logger.info(f"---- Example {idx+1} ----")
        logger.info(f"Numbers: {nums}")

        answer = asyncio.run(stream_completion(
            full_prompt,
            llm_server=llm_server,
            monitors=monitors,
            add_delay=False,
            termination_requires_validation=False,
            async_execution=True
        ))

        save_prompt(idx, answer, reason_dir)
        logger.info(f"Raw final output:\n{answer}")

        reasoning_tokens = count_tokens(answer, tokenizer)
        reasoning_token_counts.append(reasoning_tokens)
        total_reasoning_tokens += reasoning_tokens
        logger.info(f"Generated tokens in this example: {reasoning_tokens}")

        is_correct, expr, message = evaluate_game24_answer(answer, nums)
        # "attempted" = model produced a real \boxed{} answer (not "no solution")
        attempted = (expr is not None and expr.strip().lower() != "no solution")
        if attempted:
            num_attempted += 1

        # Determine if this example should be excluded from soundness:
        #   - answered "no solution" (gave up / max corrections)
        #   - no expression found (verifier never completed Phase 2)
        gave_no_solution = (expr is not None and "no solution" in expr.strip().lower())
        no_expr_found = (expr is None)
        excluded = gave_no_solution or no_expr_found
        if excluded:
            num_excluded += 1
        
        if expr:
            logger.info(f"Extracted expression: {expr}")
        logger.info(message)
        
        if is_correct:
            num_correct += 1

        per_example_results.append({
            "index": int(idx),
            "numbers": str(nums),
            "expression": expr if expr else "",
            "correct": is_correct,
            "attempted": attempted,
            "excluded": excluded,
            "tokens": reasoning_tokens,
            "message": message,
        })

    # Calculate final statistics
    avg_reasoning_tokens = total_reasoning_tokens / N if N > 0 else 0
    accuracy = num_correct / N if N > 0 else 0
    soundness_denom = N - num_excluded
    soundness = num_correct / soundness_denom if soundness_denom > 0 else 0  # correct / (total - excluded)
    
    print(f"\nFinal Accuracy: {num_correct}/{N} ({accuracy:.2%})")
    print(f"Soundness: {num_correct}/{soundness_denom} ({soundness:.2%})")
    print(f"Excluded from soundness (no solution / token budget exceeded): {num_excluded}")
    print(f"Average Reasoning Tokens: {avg_reasoning_tokens:.2f}")
    print(f"Total Reasoning Tokens: {total_reasoning_tokens}")

    # Save per-example CSV
    csv_file = os.path.join(output_dirs["csv_saved"], f"results_{N}examples.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["index", "numbers", "expression", "correct", "attempted", "excluded", "tokens", "message"])
        writer.writeheader()
        writer.writerows(per_example_results)
    logger.info(f"Per-example CSV saved to {csv_file}")
    
    # Save results summary to a text file
    results_file = logfile.replace('.log', '_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Game of 24 Evaluation Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {main_model}\n")
        f.write(f"Number of Examples: {N}\n")
        f.write(f"Monitor Enabled: {args.monitor}\n")
        f.write(f"Thinking Phase Verify: {args.thinking_verify}\n")
        if args.thinking_verify:
            f.write(f"Newline Threshold: {args.newline_threshold}\n")
        f.write(f"\n")
        f.write(f"Results:\n")
        f.write(f"---------\n")
        f.write(f"Correct: {num_correct}/{N}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Attempted (produced \\boxed answer): {num_attempted}/{N}\n")
        f.write(f"Excluded (no solution / token budget exceeded): {num_excluded}/{N}\n")
        f.write(f"Soundness (correct / (total - excluded)): {num_correct}/{soundness_denom} = {soundness:.2%}\n\n")
        f.write(f"Token Statistics:\n")
        f.write(f"---------------------------\n")
        f.write(f"Total Tokens: {total_reasoning_tokens}\n")
        f.write(f"Average Tokens: {avg_reasoning_tokens:.2f}\n")
        if reasoning_token_counts:
            f.write(f"Median Tokens: {float(np.median(reasoning_token_counts)):.0f}\n")
            f.write(f"Min Tokens: {min(reasoning_token_counts)}\n")
            f.write(f"Max Tokens: {max(reasoning_token_counts)}\n")
            f.write(f"Std Dev: {np.std(reasoning_token_counts):.2f}\n")
    
    logger.info(f"Results saved to {results_file}")
    print(f"Results saved to {results_file}")
