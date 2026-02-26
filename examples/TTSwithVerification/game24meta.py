import argparse
import asyncio
import csv
import json
import logging
import os
import re
import numpy as np

from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer

from interwhen import stream_completion
from interwhen.monitors import KstableAnswerGame24Monitor, StepVerifierGame24Monitor

# ============== MODEL CONFIGURATION ==============
# Change these model names to scale experiments easily
MAIN_MODEL = "Qwen/QwQ-32B"
EARLYSTOP_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# =================================================

def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name

def get_output_dirs(main_model: str, base_dir: str = "../../Outputs_TTS/Gameof24results/metaPrompt"):
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

def get_log_filename(main_model: str, num_examples: int, base_dir: str = "../../Outputs_TTS/Gameof24_results/metaPrompt") -> str:
    """Generate log filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.log")

def get_token_filename(main_model: str, num_examples: int, base_dir: str = "../../Outputs_TTS/Gameof24_results/metaPrompt") -> str:
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

def init_llm_server(modelname, max_tokens=200, port=8000):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": modelname,
        "max_tokens": max_tokens,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "temperature": 0.6,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed" : 42
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


def build_meta_prompt_from_example(nums):
    """Build the system and user prompts for Game of 24 with step verification format."""
    a, b, c, d = nums
    
    system_prompt = r"""You are solving the Game of 24.

GAME RULES:
- You are given four numbers
- Use ALL four numbers exactly once
- Use ONLY the operations: +, -, *, /
- The final expression must evaluate to exactly 24

OUTPUT FORMAT:
You must follow this EXACT structured format for your solution:

>Step1
available numbers: [a, b, c, d]
suggested operation: a * b = result1
remaining numbers: [result1, c, d]

>Step2
available numbers: [result1, c, d]
suggested operation: result1 + c = result2
remaining numbers: [result2, d]

>Step3
available numbers: [result2, d]
suggested operation: result2 - d = 24
remaining numbers: [24]

> Final expression: \boxed{expression using original numbers}

IMPORTANT RULES:
1. Each step MUST show the available numbers at the start
2. Each step MUST show the suggested operation with its result
3. Each step MUST show the remaining numbers after the operation
4. Continue until you reach exactly 24
5. The final expression inside \boxed{} must use the ORIGINAL numbers
6. If you receive VERIFIER FEEDBACK, immediately provide a corrected step - do NOT restart your thinking

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 1: Numbers [2, 3, 4, 5]
═══════════════════════════════════════════════════════════════════════════════

### Final Answer

>Step1
available numbers: [2, 3, 4, 5]
suggested operation: 5 + 3 = 8
remaining numbers: [8, 2, 4]

>Step2
available numbers: [8, 2, 4]
suggested operation: 8 - 2 = 6
remaining numbers: [6, 4]

>Step3
available numbers: [6, 4]
suggested operation: 6 * 4 = 24
remaining numbers: [24]

> Final expression: \boxed{(5 + 3 - 2) * 4}

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 2: Numbers [1, 5, 5, 5]
═══════════════════════════════════════════════════════════════════════════════

### Final Answer

>Step1
available numbers: [1, 5, 5, 5]
suggested operation: 1 / 5 = 0.2
remaining numbers: [0.2, 5, 5]

>Step2
available numbers: [0.2, 5, 5]
suggested operation: 5 - 0.2 = 4.8
remaining numbers: [4.8, 5]

>Step3
available numbers: [4.8, 5]
suggested operation: 4.8 * 5 = 24
remaining numbers: [24]

> Final expression: \boxed{(5 - 1/5) * 5}

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 3: Handling Verifier Feedback - Numbers [1, 2, 6, 8]
═══════════════════════════════════════════════════════════════════════════════

### Final Answer

>Step1
available numbers: [1, 2, 6, 8]
suggested operation: 8 / 2 = 4
remaining numbers: [4, 1, 6]

>Step2
available numbers: [4, 1, 6]
suggested operation: 4 - 1 = 3
remaining numbers: [3, 6]

[VERIFIER FEEDBACK for Step 2:
  ✗ Cannot reach 24 from remaining numbers [3, 6]. This path is a dead end.
The previous steps are correct. Please provide a corrected Step 2 and continue.]

>Step2
available numbers: [4, 1, 6]
suggested operation: 6 - 1 = 5
remaining numbers: [5, 4]

[VERIFIER FEEDBACK for Step 2:
  ✗ Cannot reach 24 from remaining numbers [4, 5]. This path is a dead end.
The previous steps are correct. Please provide a corrected Step 2 and continue.]

>Step2
available numbers: [4, 1, 6]
suggested operation: 6 * 1 = 6
remaining numbers: [6, 4]

>Step3
available numbers: [6, 4]
suggested operation: 6 * 4 = 24
remaining numbers: [24]

> Final expression: \boxed{(8 / 2) * 6 * 1}

═══════════════════════════════════════════════════════════════════════════════

Now solve the following Game of 24 problem using the EXACT same format."""

    user_prompt = f"""
Numbers: {a}, {b}, {c}, {d}

Find an arithmetic expression using these four numbers exactly once each with +, -, *, / that equals 24.

Use the structured step-by-step format shown in the examples above."""

    # Combine into a single prompt
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    return full_prompt


def count_tokens(text: str, tokenizer) -> int:
    """Count the total number of tokens in the generated text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def extract_solution(text):
    
    # Use a more robust extraction that handles nested braces in \boxed{}
    # Find \boxed{ and then match braces properly
    boxed_pattern = r"\\boxed\{"
    matches = list(re.finditer(boxed_pattern, text))
    if not matches:
        return None
    
    # Get the last \boxed{} content by matching braces
    last_match = matches[-1]
    start = last_match.end()  # Position right after \boxed{
    brace_count = 1
    end = start
    while end < len(text) and brace_count > 0:
        if text[end] == '{':
            brace_count += 1
        elif text[end] == '}':
            brace_count -= 1
        end += 1
    
    expr = text[start:end-1].strip()  # -1 to exclude the closing brace

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

    # 3. Cleanup (remove LaTeX spacing)
    expr = expr.replace(r"\,", "").replace(r"\ ", "")

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
    parser.add_argument("--monitor", "-m", default = False, action="store_true", help="Enable step-by-step monitor")
    parser.add_argument("--num_examples", "-n", type=int, default=1, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs")
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
    N = args.num_examples
    total_reasoning_tokens = 0
    reasoning_token_counts = []

    # total = len(dataset)
    indices = np.linspace(0, len(dataset)-1, N, dtype=int)

    for idx in indices: #for idx in indices:
        example = dataset[idx]
        nums = example["numbers"]

        prompt = build_meta_prompt_from_example(nums)

        if args.monitor:
            # Use StepVerifierGame24Monitor to detect when equation stabilizes k times
            monitors=(StepVerifierGame24Monitor(
                name="game24_kstable",
                answer_start_token = "</think>",
                original_numbers=nums,  # Validate equations use exactly these numbers
            ),)
        else:
            monitors = ()

        logger.info(f"---- length of monitors {len(monitors)} ----")
        logger.info(f"---- Example {idx+1} ----")
        logger.info(f"Numbers: {nums}")

        try:
            answer = asyncio.run(stream_completion(
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                llm_server=llm_server,
                monitors=monitors,
                add_delay=False,
                termination_requires_validation=False,
                async_execution=True
            ))
        except Exception as e:
            logger.error(f"Error running example {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

        save_prompt(idx, answer, reason_dir)
        logger.info(f"Raw final output:\n{answer}")

        reasoning_tokens = count_tokens(answer, tokenizer)
        reasoning_token_counts.append(reasoning_tokens)
        total_reasoning_tokens += reasoning_tokens
        logger.info(f"Generated tokens in this example: {reasoning_tokens}")

        is_correct, expr, message = evaluate_game24_answer(answer, nums)
        
        if expr:
            logger.info(f"Extracted expression: {expr}")
        logger.info(message)
        
        if is_correct:
            num_correct += 1

    # Calculate final statistics
    avg_reasoning_tokens = total_reasoning_tokens / N if N > 0 else 0
    accuracy = num_correct / N if N > 0 else 0
    
    print(f"\nFinal Accuracy: {num_correct}/{N} ({accuracy:.2%})")
    print(f"Average Reasoning Tokens: {avg_reasoning_tokens:.2f}")
    print(f"Total Reasoning Tokens: {total_reasoning_tokens}")
    
    # Save results to a text file
    results_file = logfile.replace('.log', '_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Game of 24 Evaluation Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {main_model}\n")
        f.write(f"Number of Examples: {N}\n")
        f.write(f"Monitor Enabled: {args.monitor}\n\n")
        f.write(f"Results:\n")
        f.write(f"---------\n")
        f.write(f"Correct: {num_correct}/{N}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n\n")
        f.write(f"Reasoning Token Statistics:\n")
        f.write(f"---------------------------\n")
        f.write(f"Total Reasoning Tokens: {total_reasoning_tokens}\n")
        f.write(f"Average Reasoning Tokens: {avg_reasoning_tokens:.2f}\n")
        if reasoning_token_counts:
            f.write(f"Min Reasoning Tokens: {min(reasoning_token_counts)}\n")
            f.write(f"Max Reasoning Tokens: {max(reasoning_token_counts)}\n")
            f.write(f"Std Dev: {np.std(reasoning_token_counts):.2f}\n")
    
    logger.info(f"Results saved to {results_file}")
    print(f"Results saved to {results_file}")